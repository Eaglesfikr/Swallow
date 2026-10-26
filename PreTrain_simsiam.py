"""Swallow SimSiam 预训练脚本。

独立文件：不修改 BYOL 的 PreTrain.py / trainer.py / config.yaml，可以两套并存对比。
数据、增强（loader/dataAugmentation.py 的 Augmentor）、网络（ResNet18 + MLPHead）都与
BYOL 那套完全一致，唯一变量是自监督框架本身。

权重默认存到 `SaveModel/<模型名>-SimSiam/`，不会覆盖 BYOL 的 `SaveModel/<模型名>/`。
下游微调要加载它时，把 fineTuning-AE.py 里的 model_name 写成同名（例如 D1-Undefence-SimSiam）。

用法：
  .venv/bin/python PreTrain_simsiam.py                        # 跑 D1-Undefence / WTF-PAD / Front
  .venv/bin/python PreTrain_simsiam.py --models D1-Undefence   # 只跑一个
  .venv/bin/python PreTrain_simsiam.py --cos --lr 0.05         # SimSiam 论文常用设置（0.05@batch512）
  .venv/bin/python PreTrain_simsiam.py --epochs 1 --no-wandb   # 快速验证能否跑通
"""

import argparse
import logging
import os
import warnings

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from loader.dataAugmentation import Augmentor
from models.mlp_head import MLPHead
from models.resnet_base_network import ResNet18
from trainer_simsiam import SimSiamTrainer

torch.manual_seed(0)
warnings.filterwarnings("ignore")

# Suppress TensorFlow and logging warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)


class TrainData(Dataset):
    """与 PreTrain.py 里同名类一致：每个样本返回 n_views 个增强视图。"""

    def __init__(self, x_train, y_train, x_load_time, x_slow_duration, augmentor, n_views):
        self.x = x_train
        self.y = y_train
        self.x_load_time = x_load_time
        self.x_slow_duration = x_slow_duration
        self.augmentor = augmentor
        self.n_views = n_views

    def __getitem__(self, index):
        res = [self.augmentor.augment(self.x[index], self.x_load_time[index], self.x_slow_duration[index])
               for _ in range(self.n_views)]
        return res, self.y[index]

    def __len__(self):
        return len(self.x)


def load_config(config_path="config.yaml"):
    """Load configuration dictionary from YAML file."""
    print("Loading config...")
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def prepare_data(model_name, config, batch_size, num_workers):
    """Load dataset, apply augmentation, and create data loader."""
    print(f"\nProcessing model: {model_name}")
    data_path = config["path"]["pretrain_data"] + "%s-PreTrain.npy" % (model_name)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu", 0)
    print(f"Device: {device}")

    print("Loading data...")
    data = np.load(data_path, allow_pickle=True).item()
    x_train = data['dataset']
    y_train = data['label']
    x_load_time = data['load_time']
    x_slot_duration = data['slot_duration']

    num_classes = len(np.unique(y_train))
    print(f"Number of classes: {num_classes}")
    print(f'Train data shapes: {x_train.shape}, {y_train.shape}')

    # Create multiple augmented versions of the dataset（与 PreTrain.py 完全一致）
    print("Creating augmented datasets...")
    augmentor = Augmentor()
    multi_times = config['trainer']['multi_times']
    n_views = config['trainer']['n_views']

    datasets = []
    for i in range(multi_times):
        train_dataset_i = TrainData(x_train, y_train, x_load_time, x_slot_duration, augmentor, n_views)
        datasets.append(train_dataset_i)
    train_dataset = ConcatDataset(datasets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, device, n_views, multi_times


def build_model(config, device):
    """SimSiam 只需要一个网络 + predictor，没有 target network。"""
    print("Initializing networks...")
    online_network = ResNet18(**config['network']).to(device)

    # Load pretrained weights if specified（SimSiam 从零开始，这里默认不加载）
    pretrained_folder = config['network'].get('fine_tune_from', None)
    if pretrained_folder:
        try:
            checkpoints_folder = os.path.join('./runs', pretrained_folder, 'checkpoints')
            load_params = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=device)
            online_network.load_state_dict(load_params['online_network_state_dict'])
            print("Pre-trained weights loaded successfully.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    # predictor 结构与 projection 一致（Linear-BN-ReLU-Linear），隐藏层的 BN 按论文要保留
    predictor = MLPHead(in_channels=online_network.projection.net[-1].out_features,
                        **config['network']['projection_head']).to(device)

    optimizer = torch.optim.SGD(
        list(online_network.parameters()) + list(predictor.parameters()),
        **config['optimizer']['params']
    )

    return online_network, predictor, optimizer


def main(args):
    model_name_list = [m.strip() for m in args.models.split(",") if m.strip()]
    for model_name in model_name_list:
        config = load_config()

        batch_size = config['trainer']['batch_size']
        # 让 wandb 里记的 num_workers 与实际 DataLoader 用的一致
        config['trainer']['num_workers'] = args.workers

        # 传给 SimSiamTrainer 的开关
        config['trainer']['use_wandb'] = not args.no_wandb
        config['trainer']['wandb_project'] = args.wandb_project
        config['trainer']['use_cosine_schedule'] = args.cos
        if args.epochs is not None:
            config['trainer']['max_epochs'] = args.epochs
        if args.lr is not None:
            config['optimizer']['params']['lr'] = args.lr

        run_name = f"{model_name}{args.suffix}"
        print(f"\n{'=' * 60}\nSimSiam 预训练: {run_name}\n{'=' * 60}")
        print(f"  max_epochs={config['trainer']['max_epochs']} lr={config['optimizer']['params']['lr']} "
              f"batch_size={batch_size} cosine={config['trainer']['use_cosine_schedule']}")
        print(f"  [提醒] 下游微调要加载这套权重时，model_name 用 '{run_name}'")

        # 注意：数据文件名用的是 model_name（D1-Undefence-PreTrain.npy），
        # 带 -SimSiam 后缀的 run_name 只用于保存目录和 wandb
        train_loader, device, n_views, multi_times = prepare_data(model_name, config, batch_size, args.workers)
        online_net, predictor, optimizer = build_model(config, device)

        trainer = SimSiamTrainer(
            online_network=online_net,
            predictor=predictor,
            optimizer=optimizer,
            device=device,
            model_path=run_name,
            model_folder=config['path']['model_path'],
            **config['trainer']
        )
        trainer.train(train_loader)


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning, message="loaded more than 1 DLL from .libs")

    parser = argparse.ArgumentParser(description="Swallow SimSiam 预训练（独立于 BYOL 的那套，含 wandb）")
    parser.add_argument("--models", default="D1-Undefence,D1-WTF-PAD,D1-Front",
                        help="逗号分隔的预训练数据集名，默认与 PreTrain.py 相同")
    parser.add_argument("--suffix", default="-SimSiam",
                        help="保存目录后缀，默认 -SimSiam（存到 SaveModel/<模型名>-SimSiam，不覆盖 BYOL 权重）")
    parser.add_argument("--cos", action="store_true",
                        help="开启 cosine 学习率衰减（SimSiam 官方仓库可选开关，论文默认不开）")
    parser.add_argument("--epochs", type=int, default=None,
                        help="覆盖 config.yaml 里的 max_epochs，方便快速验证")
    parser.add_argument("--lr", type=float, default=None,
                        help="覆盖学习率（SimSiam 论文用 0.05 @ batch 512，config.yaml 现在是 0.03）")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker 数，默认 4（与 PreTrain.py 一致）")
    parser.add_argument("--no-wandb", action="store_true", help="关闭 wandb 记录")
    parser.add_argument("--wandb-project", default="swallow-pretrain-simsiam",
                        help="wandb 项目名，默认 swallow-pretrain-simsiam"
                             "（传 swallow-pretrain 可与 BYOL 曲线放同一个项目里对比）")
    args = parser.parse_args()
    print(f"wandb: {'已关闭' if args.no_wandb else '开启 -> ' + args.wandb_project}"
          f"{' | cosine 衰减: 开' if args.cos else ''}")

    main(args)