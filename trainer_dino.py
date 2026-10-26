"""DINO 预训练器 —— Caron et al., ICCV 2021, "Emerging Properties in Self-Supervised Vision Transformers".

独立文件，不 import trainer.py 的 BYOLTrainer，改动这里不会影响 BYOL / SimSiam 那两套基线。

与 BYOL 的关键差别（框架层面，不是调参）：
  1. loss 换成 teacher 分布 → student 分布的交叉熵（softmax 归一化后的"原型"分类问题），
     不是 BYOL 的 L2 归一化回归；也没有 predictor
  2. 防塌陷靠 teacher 的 centering + sharpening（center 缓冲 EMA 更新、teacher 温度 0.04→0.07
     warmup），不是 BYOL 的 predictor + 动量更新
  3. teacher 温度、EMA 动量都按余弦/warmup 调度，还有 freeze_last_layer（默认冻结第一轮 head 末层）

与 BYOL 保持一致的地方（保证两套结果可直接对比）：
  * 网络结构完全一致：ResNet18(backbone) + MLPHead(projection)，head 里不放 weight_norm
    （DINO 官方对末层做 weight_norm，但那会往 state_dict 里塞 weight_g/weight_v，
     fineTuning*.py 用 strict=True 加载 online_network_state_dict 会直接失败，所以此处不加）
  * teacher 只做参数的 EMA（BN 的 running stats 不做 EMA），与 trainer.py 的 BYOL 口径一致
  * checkpoint 里保留 'online_network_state_dict' 这个 key，fineTuning*.py 靠它加载预训练权重

另外支持：
  * resume_from：从已有 checkpoint（student/teacher/center）接着训，checkpoint 编号接着往下走，
    且续训时 teacher 温度不再重新 warmup（早就过了那段）
  * 可选的余弦学习率衰减 + 线性 warmup（长 schedule 必需；见 PreTrain_dino.py 的 --cosine）
  * 每 monitor_every_n_epochs 轮在固定一批样本上量 backbone 特征的有效秩并记进 wandb ——
    这是判断「欠训练 / 容量不够 / 塌陷」最直接的指标（实测：BYOL 200 epoch 约 106，DINO 同预算约 18）
"""

import math
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import _create_model_training_folder

# --- wandb 可视化。可用 use_wandb=False 或环境变量 WANDB_MODE=disabled / offline 关闭或转离线 ---
try:
    import wandb
except ImportError:
    wandb = None


class DINOTrainer:
    def __init__(self, student_network, teacher_network, optimizer, device, model_path, **params):
        self.student_network = student_network
        self.teacher_network = teacher_network
        self.optimizer = optimizer
        self.device = device
        self.model_path = model_path
        self.model_folder = params.get('model_folder', './SaveModel')
        self.max_epochs = params['max_epochs']
        self.batch_size = params['batch_size']
        self.num_workers = params.get('num_workers', 0)
        self.checkpoint_interval = params.get('checkpoint_interval', 0)
        self.n_views = params.get('n_views')
        self.multi_times = params.get('multi_times')

        # --- DINO 超参（默认值来自论文 Table 9 / 官方 main_dino.py）---
        self.t_s = params.get('temperature_student', 0.1)                 # student 温度，固定
        self.t_t_start = params.get('temperature_teacher', 0.04)          # teacher 温度 warmup 起点
        self.t_t_end = params.get('temperature_teacher_final', 0.07)      # teacher 温度 warmup 终点
        self.teacher_temp_warmup_epochs = params.get('teacher_temp_warmup_epochs', 30)
        self.center_momentum = params.get('center_momentum', 0.9)         # center 缓冲的 EMA 动量
        self.m = params['m']                                             # EMA 动量起点（config.yaml: 0.996）
        self.m_end = params.get('m_final', 1.0)                          # EMA 动量终点（论文余弦升到 1.0）
        self.freeze_last_layer_epochs = params.get('freeze_last_layer_epochs', 1)
        self.save_every_n_epochs = params.get('save_every_n_epochs', 20)

        # 这两个量每步都会被 _set_schedules 覆盖（调度里）
        self.momentum = self.m
        self.t_t = self.t_t_start

        out_dim = student_network.projection.net[-1].out_features
        # center 缓冲：teacher 输出的 batch 均值（不参与梯度，随后续 batch EMA 更新）
        self.center = torch.zeros(out_dim, device=device)
        # freeze_last_layer 只作用于投影头最后一层（Linear），按参数对象引用，不动 state_dict 结构
        self.last_layer_params = list(student_network.projection.net[-1].parameters())

        # --- 续训 / 调度 / 监控 ---
        # resume_from：从已有 checkpoint 接着训（student + teacher + center 都恢复）
        self.resume_from = params.get('resume_from')
        # start_epoch：续训时的轮次起点，只影响 checkpoint 文件名的编号，让 200 之后接着 220/240...
        self.start_epoch = params.get('start_epoch', 0)
        self.use_cosine_schedule = params.get('use_cosine_schedule', False)
        self.lr_warmup_epochs = params.get('lr_warmup_epochs', 0)
        self.base_lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0.0

        # 有效秩监控：这是判断「欠训练 / 容量不够 / 塌陷」最直接的一个量，
        # 训练时就记进 wandb，省得训完才发现还在涨。默认开（代价极小）
        self.monitor_eff_rank = params.get('monitor_eff_rank', True)
        self.monitor_every_n_epochs = params.get('monitor_every_n_epochs', 10)
        self._monitor_views = None   # 固定一批样本，保证各 epoch 之间可比

        # --- wandb 配置 ---
        self.use_wandb = params.get('use_wandb', True)
        self.wandb_project = params.get('wandb_project', 'swallow-pretrain-dino')
        self.wb_run = None
        self.wb_step = 0

        run_folder = os.path.join(self.model_folder, self.model_path)
        _create_model_training_folder(
            run_folder, files_to_same=["config.yaml", "PreTrain_dino.py", "trainer_dino.py"])

    # ---------------- teacher 初始化 / 更新 ----------------

    def initializes_teacher_network(self):
        """teacher 从 student 复制权重，且不参与梯度（与 BYOL 的 target network 一样）。"""
        for param_s, param_t in zip(self.student_network.parameters(), self.teacher_network.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False

    @torch.no_grad()
    def _update_teacher_network_parameters(self):
        """teacher 的 EMA 动量更新：param_t = momentum * param_t + (1 - momentum) * param_s。"""
        momentum = self.momentum
        for param_s, param_t in zip(self.student_network.parameters(), self.teacher_network.parameters()):
            param_t.data.mul_(momentum).add_(param_s.data, alpha=1. - momentum)

    @torch.no_grad()
    def _update_center(self, teacher_logits):
        """center 缓冲的 EMA：C = m * C + (1 - m) * mean_batch(teacher logits)。

        centering 把某些维度整体偏高的情况压掉，防止所有样本都塌到同一个原型上；
        sharpening（低 teacher 温度）则反过来防止输出过于均匀 —— 两者配合才是 DINO。
        """
        batch_center = torch.stack([logits.mean(dim=0) for logits in teacher_logits]).mean(dim=0)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1. - self.center_momentum)

    # ---------------- 调度 ----------------

    def _set_schedules(self, step, total_steps, t_t_warmup_steps):
        """按全局 step 更新 EMA 动量（余弦 0.996→1.0）与 teacher 温度（warmup 0.04→0.07）。"""
        progress = min(1.0, step / max(total_steps, 1))
        self.momentum = self.m_end - (self.m_end - self.m) * (math.cos(math.pi * progress) + 1) / 2

        if t_t_warmup_steps > 0:
            warmup_progress = min(1.0, step / t_t_warmup_steps)
        else:
            warmup_progress = 1.0
        self.t_t = self.t_t_start + (self.t_t_end - self.t_t_start) * warmup_progress

    def _set_lr(self, step, total_steps, warmup_steps):
        """按全局 step 设置学习率：线性 warmup（可选）+ 余弦衰减（可选）。

        长 schedule 必须配余弦衰减，否则后期 lr 一直很大，会像我们那次 epoch 160 的
        student 一样抖一下。续训时新开一段 schedule，所以这里是按「本段」的 step 算的。
        """
        if warmup_steps > 0 and step < warmup_steps:
            lr = self.base_lr * (step + 1) / warmup_steps
        elif self.use_cosine_schedule:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            lr = self.base_lr * 0.5 * (1. + math.cos(math.pi * min(1.0, progress)))
        else:
            lr = self.base_lr
        for group in self.optimizer.param_groups:
            group['lr'] = lr

    @torch.no_grad()
    def _eff_rank_of_monitor_batch(self):
        """固定一批样本，量 backbone(512-d) 特征的有效秩 —— 塌陷/欠训练的最直接指标。

        参照值：BYOL 训到 200 epoch 是 ~106，DINO 同预算下只有 ~18（还在涨），
        所以这个数一路上升说明还在学习，早早平在低位才是容量瓶颈。
        """
        if self._monitor_views is None:
            return None
        feats = self.student_network.encoder(self._monitor_views[0])
        feats = feats.view(feats.shape[0], -1).float()
        feats = feats - feats.mean(dim=0, keepdim=True)
        cov = (feats.t() @ feats) / max(1, feats.shape[0] - 1)
        lam = torch.linalg.eigvalsh(cov).clamp(min=0)
        return ((lam.sum() ** 2) / (lam.pow(2).sum() + 1e-12)).item()

    def cancel_gradients_last_layer(self, epoch):
        """前 freeze_last_layer_epochs 轮把投影头末层的梯度丢掉（DINO 的稳定性技巧）。

        只改 grad，不改结构 —— 这样 checkpoint 里的 state_dict 仍然和 BYOL/SimSiam 互通。
        """
        if epoch >= self.freeze_last_layer_epochs:
            return
        for param in self.last_layer_params:
            if param.grad is not None:
                param.grad = None

    # ---------------- 核心：DINO 的 loss ----------------

    def update(self, views):
        """算一个 batch 的 loss，返回 (loss, metrics)。views 是同一样本的 n_views 个增强视图。

        loss = -sum_{i != j} P_t(view_j) · log P_s(view_i)，对所有 (i, j) 配对数取平均：
          P_s = softmax(student_logits / t_s)                        # 高温，输出更平滑
          P_t = softmax((teacher_logits - C) / t_t)                  # 低温 sharpen + 减 center
        """
        n_views = len(views)

        # student：带梯度
        student_logits = [self.student_network(view) for view in views]

        # teacher：no_grad，且先更新 center 再用它做 centering
        with torch.no_grad():
            teacher_logits = [self.teacher_network(view) for view in views]
            self._update_center(teacher_logits)
            teacher_probs = [F.softmax((logits - self.center) / self.t_t, dim=-1) for logits in teacher_logits]

        student_log_probs = [F.log_softmax(logits / self.t_s, dim=-1) for logits in student_logits]

        loss = 0.0
        n_loss_terms = 0
        for i_q, q in enumerate(teacher_probs):
            for i_v, log_p_s in enumerate(student_log_probs):
                if i_v == i_q:
                    continue  # 同一个视图不配对（teacher 和 student 看同一个视图没意义）
                loss = loss - (q * log_p_s).sum(dim=-1).mean()
                n_loss_terms += 1
        loss = loss / max(n_loss_terms, 1)

        # --- collapse 监控 ---
        # 正常训练时 teacher 熵缓慢下降但不塌到 0；max_prob 长期贴在 1.0 说明已经塌陷，
        # entropy 长期贴着 log(dim) 说明 teacher 太均匀（温度过高）。
        with torch.no_grad():
            t_stack = torch.stack([p.detach() for p in teacher_probs])
            teacher_entropy = -(t_stack * torch.log(t_stack + 1e-8)).sum(dim=-1).mean().item()
            teacher_max_prob = t_stack.max(dim=-1).values.mean().item()
            s_probs = [F.softmax(logits / self.t_s, dim=-1) for logits in student_logits]
            s_stack = torch.stack([p.detach() for p in s_probs])
            student_entropy = -(s_stack * torch.log(s_stack + 1e-8)).sum(dim=-1).mean().item()
            feature_std = torch.stack([logits.detach().std(dim=0).mean() for logits in student_logits]).mean().item()

        metrics = {
            "teacher_entropy": teacher_entropy,
            "student_entropy": student_entropy,
            "teacher_max_prob": teacher_max_prob,
            "feature_std": feature_std,
            "t_teacher": self.t_t,
            "momentum": self.momentum,
            "center_norm": self.center.norm().item(),
        }
        return loss, metrics

    # ---------------- 训练循环（与 BYOLTrainer / SimSiamTrainer 同口径，方便在 wandb 里对比）----------------

    def train(self, train_loader):
        self.model_path = os.path.join(self.model_folder, self.model_path)
        os.makedirs(self.model_path, exist_ok=True)
        print(f"预训练方法: dino")
        print(f"模型将保存到: {self.model_path}")

        steps_per_epoch = len(train_loader)
        total_steps = max(1, self.max_epochs * steps_per_epoch)
        t_t_warmup_steps = self.teacher_temp_warmup_epochs * steps_per_epoch
        lr_warmup_steps = self.lr_warmup_epochs * steps_per_epoch
        end_epoch = self.start_epoch + self.max_epochs
        print(f"预训练阶段: epoch {self.start_epoch + 1} -> {end_epoch}"
              f"{'（续训）' if self.resume_from else ''}")

        # --- wandb：一个预训练任务（模型名）对应一个 run ---
        if self.use_wandb and wandb is not None:
            try:
                self.wb_run = wandb.init(
                    project=self.wandb_project,
                    name=(os.path.basename(self.model_path.rstrip('/'))
                          + (f"-resume{self.start_epoch}to{end_epoch}" if self.resume_from else "")),
                    config={
                        "method": "dino",
                        "resumed_from": self.resume_from,
                        "start_epoch": self.start_epoch,
                        "end_epoch": end_epoch,
                        "use_cosine_schedule": self.use_cosine_schedule,
                        "lr_warmup_epochs": self.lr_warmup_epochs,
                        "base_lr": self.base_lr,
                        "optimizer": type(self.optimizer).__name__,
                        "max_epochs": self.max_epochs,
                        "batch_size": self.batch_size,
                        "n_views": self.n_views,
                        "multi_times": self.multi_times,
                        "num_workers": self.num_workers,
                        "lr": self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else None,
                        "temperature_student": self.t_s,
                        "temperature_teacher": self.t_t_start,
                        "temperature_teacher_final": self.t_t_end,
                        "teacher_temp_warmup_epochs": self.teacher_temp_warmup_epochs,
                        "center_momentum": self.center_momentum,
                        "momentum_m": self.m,
                        "momentum_m_final": self.m_end,
                        "freeze_last_layer_epochs": self.freeze_last_layer_epochs,
                        "out_dim": self.center.shape[0],
                        "device": str(self.device),
                        "num_batches_per_epoch": steps_per_epoch,
                    },
                    reinit=True,
                )
            except Exception as e:
                print(f"[wandb] 初始化失败，本次跳过记录：{type(e).__name__}: {e}")
                print("[wandb] 如需启用，先执行登录： .venv/bin/python -m wandb login")
                self.wb_run = None

        # 续训：把 student / teacher / center 都恢复回来；否则 teacher 从 student 复制
        if self.resume_from:
            ck = torch.load(self.resume_from, map_location=self.device)
            self.student_network.load_state_dict(ck['online_network_state_dict'])
            if 'target_network_state_dict' in ck:
                self.teacher_network.load_state_dict(ck['target_network_state_dict'])
            else:
                self.initializes_teacher_network()
            if 'center' in ck:
                resumed_center = ck['center'].to(self.device)
                if resumed_center.shape != self.center.shape:
                    raise ValueError(
                        f"checkpoint 的 center 维度 {tuple(resumed_center.shape)} 与当前 "
                        f"{tuple(self.center.shape)} 不一致（--dim 是不是和预训练时不同？）")
                self.center.copy_(resumed_center)
            for param_t in self.teacher_network.parameters():
                param_t.requires_grad = False
            print(f"已续训：从 {self.resume_from} 载入 student/teacher/center，"
                  f"接着 epoch {self.start_epoch} 继续")
            print(f"[note] EMA 动量与 teacher 温度的 cosine 是按「本段」重新起算的，"
                  f"所以动量会从 {self.m:.4f} 重新爬到 {self.m_end:.4f}（不会重置 teacher 权重）")
        else:
            self.initializes_teacher_network()

        best_loss = float('inf')
        average_loss = None
        for epoch_counter in range(self.start_epoch, end_epoch):
            local_epoch = epoch_counter - self.start_epoch
            progress_bar = tqdm(
                enumerate(train_loader), total=steps_per_epoch,
                desc=f"Epoch {epoch_counter + 1}/{end_epoch}"
            )
            total_loss = 0.0
            num_batches = 0

            # --- wandb：批级平滑曲线（每 log_every_n_batches 个 batch 记一次均值）---
            log_every_n_batches = 10
            window_loss, window_batches = 0.0, 0
            window_metrics = {}

            for batch_idx, (batch_views, _) in progress_bar:
                # 数据是 (B, 2, T) 的上/下行双通道，reshape 成 (B, 1, 2, T) 喂给 resnet
                views = []
                for v in batch_views:
                    v = v.to(self.device).float()
                    views.append(v.reshape(v.shape[0], 1, v.shape[1], v.shape[2]))

                # 固定一批样本用于每 N 轮量有效秩（各 epoch 之间可比）
                if self.monitor_eff_rank and self._monitor_views is None:
                    self._monitor_views = [v.detach().clone() for v in views]

                step = local_epoch * steps_per_epoch + batch_idx
                self._set_schedules(step, total_steps, t_t_warmup_steps)
                self._set_lr(step, total_steps, lr_warmup_steps)

                loss, metrics = self.update(views)

                self.optimizer.zero_grad()
                loss.backward()
                self.cancel_gradients_last_layer(epoch_counter)
                self.optimizer.step()

                # 更新 teacher 的参数（EMA）
                self._update_teacher_network_parameters()

                total_loss += loss.item()
                num_batches += 1

                window_loss += loss.item()
                window_batches += 1
                for k, v in metrics.items():
                    window_metrics[k] = window_metrics.get(k, 0.0) + float(v)

                progress_bar.set_postfix(loss=f"{loss.item():.4f}", t_t=f"{self.t_t:.3f}")

                if self.wb_run is not None and window_batches >= log_every_n_batches:
                    log_dict = {
                        "train/batch_loss": window_loss / window_batches,
                        "train/lr": self.optimizer.param_groups[0]['lr'],
                        "epoch": epoch_counter + 1,
                    }
                    for k, v in window_metrics.items():
                        log_dict[f"train/{k}"] = v / window_batches
                    self.wb_run.log(log_dict, step=self.wb_step)
                    self.wb_step += 1
                    window_loss, window_batches = 0.0, 0
                    window_metrics = {}

            average_loss = total_loss / max(num_batches, 1)
            print("average_loss", average_loss)
            best_loss = min(best_loss, average_loss)

            # 有效秩监控（每 monitor_every_n_epochs 轮一次）
            eff_rank = None
            if self.monitor_eff_rank and (epoch_counter + 1) % self.monitor_every_n_epochs == 0:
                eff_rank = self._eff_rank_of_monitor_batch()
                if eff_rank is not None:
                    print(f"[monitor] epoch {epoch_counter + 1} backbone eff_rank = {eff_rank:.2f}")

            # --- wandb：每 epoch 的均值 ---
            if self.wb_run is not None:
                log_dict = {
                    "train/loss": average_loss,
                    "train/lr": self.optimizer.param_groups[0]['lr'],
                    "epoch": epoch_counter + 1,
                }
                if eff_rank is not None:
                    log_dict["train/eff_rank"] = eff_rank
                self.wb_run.log(log_dict, step=self.wb_step)
                self.wb_step += 1

            # 定期存；最后一轮无论是否整除都存一次（短跑 --epochs N 也能拿到权重）
            if (epoch_counter + 1) % self.save_every_n_epochs == 0 or (epoch_counter + 1) == end_epoch:
                self.save_model(os.path.join(self.model_path, f'model_epoch_{epoch_counter + 1}.pth'))

        # --- wandb：收尾 ---
        if self.wb_run is not None:
            self.wb_run.summary["best_epoch_loss"] = best_loss
            self.wb_run.summary["final_epoch_loss"] = average_loss
            self.wb_run.summary["model_path"] = self.model_path
            self.wb_run.finish()
            self.wb_run = None

    def save_model(self, PATH):
        torch.save({
            'online_network_state_dict': self.student_network.state_dict(),
            'target_network_state_dict': self.teacher_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'center': self.center.cpu(),
        }, PATH)