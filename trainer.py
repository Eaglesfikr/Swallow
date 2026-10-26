import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import _create_model_training_folder

# --- wandb 可视化。可传 use_wandb=False 或环境变量 WANDB_MODE=disabled / offline 关闭或转离线 ---
try:
    import wandb
except ImportError:
    wandb = None


class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, device,model_path,**params):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.model_path = model_path
        self.model_folder = params.get('model_folder', './SaveModel')
        self.max_epochs = params['max_epochs']
        self.m = params['m']
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.checkpoint_interval = params['checkpoint_interval']
        self.n_views = params.get('n_views')
        self.multi_times = params.get('multi_times')

        # --- wandb 配置 ---
        self.use_wandb = params.get('use_wandb', True)
        self.wandb_project = params.get('wandb_project', 'swallow-pretrain')
        self.wb_run = None
        self.wb_step = 0

        run_folder = os.path.join(self.model_folder, self.model_path)
        _create_model_training_folder(run_folder, files_to_same=["config.yaml", "PreTrain.py", 'trainer.py'])

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_loader):

        self.model_path = os.path.join(self.model_folder, self.model_path)

        os.makedirs(self.model_path, exist_ok=True)
        print(f"模型将保存到: {self.model_path}")

        # --- wandb：一个预训练任务（模型名）对应一个 run ---
        if self.use_wandb and wandb is not None:
            try:
                self.wb_run = wandb.init(
                    project=self.wandb_project,
                    name=os.path.basename(self.model_path.rstrip('/')),
                    config={
                        "max_epochs": self.max_epochs,
                        "batch_size": self.batch_size,
                        "momentum_m": self.m,
                        "n_views": self.n_views,
                        "multi_times": self.multi_times,
                        "num_workers": self.num_workers,
                        "lr": self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else None,
                        "device": str(self.device),
                        "num_batches_per_epoch": len(train_loader),
                    },
                    reinit=True,
                )
            except Exception as e:
                print(f"[wandb] 初始化失败，本次跳过记录：{type(e).__name__}: {e}")
                print("[wandb] 如需启用，先执行登录： .venv/bin/python -m wandb login")
                self.wb_run = None

        self.initializes_target_network()

        niter = 0
        best_loss = float('inf')
        for epoch_counter in range(self.max_epochs):
            # 初始化 tqdm 进度条
            print("Starting training loop")
            progress_bar = tqdm(
                enumerate(train_loader), total=len(train_loader),
                desc=f"Epoch {epoch_counter + 1}/{self.max_epochs}"
            )
            total_loss = 0.0
            num_batches = 0
            # --- wandb：批级平滑曲线用（每 log_every_n_batches 个 batch 记一次均值）---
            log_every_n_batches = 10
            window_loss, window_batches = 0.0, 0

            for batch_idx, ((batch_view_1, batch_view_2), _) in progress_bar:

                batch_view_1 = batch_view_1.to(self.device).float()
                batch_view_1 = batch_view_1.reshape(batch_view_1.shape[0], 1, batch_view_1.shape[1],
                                                    batch_view_1.shape[2])
                batch_view_2 = batch_view_2.to(self.device).float()
                batch_view_2 = batch_view_2.reshape(batch_view_2.shape[0], 1, batch_view_2.shape[1],
                                                    batch_view_2.shape[2])
                loss = self.update(batch_view_1, batch_view_2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 更新目标网络的参数
                self._update_target_network_parameters()
                niter += 1
                total_loss += loss.item()
                num_batches += 1

                # --- wandb：批级 loss（滑动窗口均值），曲线比每 epoch 一个点细得多 ---
                window_loss += loss.item()
                window_batches += 1
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                if self.wb_run is not None:
                    if window_batches >= log_every_n_batches:
                        self.wb_run.log({
                            "train/batch_loss": window_loss / window_batches,
                            "train/lr": self.optimizer.param_groups[0]['lr'],
                            "epoch": epoch_counter + 1,
                        }, step=self.wb_step)
                        self.wb_step += 1
                        window_loss, window_batches = 0.0, 0

            average_loss = total_loss / num_batches
            print("average_loss", average_loss)
            best_loss = min(best_loss, average_loss)

            # --- wandb：每 epoch 的均值 ---
            if self.wb_run is not None:
                self.wb_run.log({
                    "train/loss": average_loss,
                    "epoch": epoch_counter + 1,
                }, step=self.wb_step)
                self.wb_step += 1

            if (epoch_counter + 1) % 20 == 0:
                self.save_model(os.path.join(self.model_path, f'model_epoch_{epoch_counter + 1}.pth'))

        #self.save_model(os.path.join(self.model_path, 'final_model.pth'))

        # --- wandb：收尾（记录最好 loss 与权重目录）---
        if self.wb_run is not None:
            self.wb_run.summary["best_epoch_loss"] = best_loss
            self.wb_run.summary["final_epoch_loss"] = average_loss
            self.wb_run.summary["model_path"] = self.model_path
            self.wb_run.finish()
            self.wb_run = None
    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)
            targets_to_view_1 = self.target_network(batch_view_2)

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()

    def save_model(self, PATH):

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)