"""SimSiam 预训练器 —— Chen & He, CVPR 2021, "Exploring Simple Siamese Representation Learning".

独立文件，不 import trainer.py 的 BYOLTrainer，所以改动这里不会影响 BYOL 基线。

与 BYOL 的三处关键差别：
  1. 只有一个网络：两个 view 都过同一个 online_network（权重共享），没有 target network
  2. 另一支的 z 必须 detach()（stop-gradient）—— 这是 SimSiam 不塌陷的关键
  3. 没有 EMA 动量更新；loss 换成负余弦相似度 -cos(p, sg(z))，两支对称求和

predictor 结构沿用 models/mlp_head.py 的 MLPHead（Linear-BN-ReLU-Linear），与论文一致：
隐藏层的 BN 不能去掉（论文消融里 "predictor w/o BN" 明显更差）。

checkpoint 里保留 'online_network_state_dict' 这个 key，fineTuning*.py 靠它加载预训练权重。
"""

import glob
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


class SimSiamTrainer:
    def __init__(self, online_network, predictor, optimizer, device, model_path, **params):
        self.online_network = online_network
        self.predictor = predictor
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

        # --- wandb 配置 ---
        self.use_wandb = params.get('use_wandb', True)
        self.wandb_project = params.get('wandb_project', 'swallow-pretrain-simsiam')
        # SimSiam 官方仓库里的可选 cosine 衰减（论文默认设置是不开）
        self.use_cosine_schedule = params.get('use_cosine_schedule', False)
        self.save_every_n_epochs = params.get('save_every_n_epochs', 20)
        self.wb_run = None
        self.wb_step = 0

        run_folder = os.path.join(self.model_folder, self.model_path)
        _create_model_training_folder(
            run_folder, files_to_same=["config.yaml", "PreTrain_simsiam.py", "trainer_simsiam.py"])

    # ---------------- 核心：SimSiam 的 loss ----------------

    @staticmethod
    def cosine_similarity_loss(p, z):
        """负余弦相似度（SimSiam 的损失函数）。"""
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=-1).mean()

    def update(self, batch_view_1, batch_view_2):
        """算一个 batch 的 loss，返回 (loss, metrics)。

        注意 z 的 detach()：没有它整个 loss 会退化成一个常数（论文 Sec.2 的推导），
        表示必然塌陷 —— stop-gradient 是 SimSiam 的全部秘密。
        """
        z1 = self.online_network(batch_view_1)
        z2 = self.online_network(batch_view_2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        loss = 0.5 * (self.cosine_similarity_loss(p1, z2.detach())
                      + self.cosine_similarity_loss(p2, z1.detach()))

        # collapse 监控：embedding 在每个维度上的 batch 内标准差（塌陷时 → 0），
        # 以及 p 与 z 的余弦相似度（正常时应该从 ~0 慢慢升上去）
        with torch.no_grad():
            metrics = {
                "feature_std": 0.5 * (z1.std(dim=0).mean().item() + z2.std(dim=0).mean().item()),
                "pred_cos": 0.5 * (F.cosine_similarity(p1, z2, dim=1).mean().item()
                                   + F.cosine_similarity(p2, z1, dim=1).mean().item()),
            }
        return loss, metrics

    # ---------------- 训练循环（与 BYOLTrainer 同口径，方便在 wandb 里对比）----------------

    def train(self, train_loader):
        self.model_path = os.path.join(self.model_folder, self.model_path)
        os.makedirs(self.model_path, exist_ok=True)
        print(f"预训练方法: simsiam")
        print(f"模型将保存到: {self.model_path}")

        # 同名目录已有权重时提醒一声（避免覆盖 BYOL 的结果）
        existed = sorted(glob.glob(os.path.join(self.model_path, 'model_epoch_*.pth')))
        if existed:
            print(f"[提示] 该目录已有 {len(existed)} 个 checkpoint（如 {os.path.basename(existed[-1])}），"
                  f"本次训练会覆盖同名文件。")

        # --- wandb：一个预训练任务（模型名）对应一个 run ---
        if self.use_wandb and wandb is not None:
            try:
                self.wb_run = wandb.init(
                    project=self.wandb_project,
                    name=os.path.basename(self.model_path.rstrip('/')),
                    config={
                        "method": "simsiam",
                        "max_epochs": self.max_epochs,
                        "batch_size": self.batch_size,
                        "n_views": self.n_views,
                        "multi_times": self.multi_times,
                        "num_workers": self.num_workers,
                        "lr": self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else None,
                        "cosine_schedule": self.use_cosine_schedule,
                        "device": str(self.device),
                        "num_batches_per_epoch": len(train_loader),
                    },
                    reinit=True,
                )
            except Exception as e:
                print(f"[wandb] 初始化失败，本次跳过记录：{type(e).__name__}: {e}")
                print("[wandb] 如需启用，先执行登录： .venv/bin/python -m wandb login")
                self.wb_run = None

        scheduler = None
        if self.use_cosine_schedule:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, self.max_epochs * len(train_loader)))

        best_loss = float('inf')
        average_loss = None
        for epoch_counter in range(self.max_epochs):
            progress_bar = tqdm(
                enumerate(train_loader), total=len(train_loader),
                desc=f"Epoch {epoch_counter + 1}/{self.max_epochs}"
            )
            total_loss = 0.0
            num_batches = 0

            # --- wandb：批级平滑曲线（每 log_every_n_batches 个 batch 记一次均值）---
            log_every_n_batches = 10
            window_loss, window_batches = 0.0, 0
            window_metrics = {}

            for batch_idx, ((batch_view_1, batch_view_2), _) in progress_bar:

                batch_view_1 = batch_view_1.to(self.device).float()
                batch_view_1 = batch_view_1.reshape(batch_view_1.shape[0], 1, batch_view_1.shape[1],
                                                    batch_view_1.shape[2])
                batch_view_2 = batch_view_2.to(self.device).float()
                batch_view_2 = batch_view_2.reshape(batch_view_2.shape[0], 1, batch_view_2.shape[1],
                                                    batch_view_2.shape[2])

                loss, metrics = self.update(batch_view_1, batch_view_2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                total_loss += loss.item()
                num_batches += 1

                window_loss += loss.item()
                window_batches += 1
                for k, v in metrics.items():
                    window_metrics[k] = window_metrics.get(k, 0.0) + float(v)

                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

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

            # --- wandb：每 epoch 的均值 ---
            if self.wb_run is not None:
                self.wb_run.log({
                    "train/loss": average_loss,
                    "train/lr": self.optimizer.param_groups[0]['lr'],
                    "epoch": epoch_counter + 1,
                }, step=self.wb_step)
                self.wb_step += 1

            # 定期存；最后一轮无论是否整除都存一次（短跑 --epochs N 也能拿到权重）
            if (epoch_counter + 1) % self.save_every_n_epochs == 0 or (epoch_counter + 1) == self.max_epochs:
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
            'online_network_state_dict': self.online_network.state_dict(),
            'predictor_state_dict': self.predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)