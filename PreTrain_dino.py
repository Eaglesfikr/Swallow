"""Swallow DINO 预训练脚本。

独立文件：不修改 BYOL 的 PreTrain.py / trainer.py / config.yaml，可以和 BYOL、SimSiam 三套并存对比。
数据、增强（loader/dataAugmentation.py 的 Augmentor）、网络（ResNet18 + MLPHead）都与 BYOL 那套
完全一致，唯一变量是自监督框架本身。

权重默认存到 `SaveModel/<模型名>-DINO/`，不会覆盖 BYOL 的 `SaveModel/<模型名>/`。
下游微调要加载它时，把 fineTuning-AE.py 里的 model_name 写成同名（例如 D1-Undefence-DINO）。

用法：
  .venv/bin/python PreTrain_dino.py                       # 默认只跑 D1-Undefence
  .venv/bin/python PreTrain_dino.py --models D1-Undefence,D1-WTF-PAD   # 想跑别的数据集再传
  .venv/bin/python PreTrain_dino.py --epochs 1 --no-wandb  # 快速验证能否跑通
  .venv/bin/python PreTrain_dino.py --dim 1024 --teacher-temp-warmup 0 --epochs 100 \
      --suffix -DINO-K1024
      # 加宽原型数 K（128 -> 1024）且固定 teacher 温度 0.07。head 变宽不影响下游：
      # fineTuning-AE-dino.py / -d2d6d7.py 加载时会先剔掉 projection.* 再 load_state_dict，
      # 所以这两个脚本不用跟着改配置。

续训（诊断结论：200 epoch 严重欠训练 —— 有效秩才 ~18 且还在涨，BYOL 同预算是 106；
student 在 epoch 160 还塌过一次，说明 SGD lr=0.03 对 DINO 太猛。所以续训建议换 AdamW + 余弦衰减）：

  .venv/bin/python PreTrain_dino.py --resume SaveModel/D1-Undefence-DINO \
      --optimizer adamw --cosine --epochs 600

  含义：从 SaveModel/D1-Undefence-DINO/ 里 epoch 最大的权重（200）接着训 600 轮，
        总到 800；新 checkpoint 编号接着 220/240/.../800 走，不覆盖旧的；
        student/teacher/center 全部恢复，teacher 温度不再重新 warmup、head 不再冻结；
        AdamW lr = 5e-4 * batch_size/256 = 1e-3、wd = 0.04，前 5 轮线性 warmup 后余弦衰减到 0。
  （旧 SGD 的动量/scaler 状态会被丢掉，这是有意为之：换优化器续训不该继承旧状态）
"""

import argparse
import glob
import logging
import os
import re
import sys
import warnings

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from loader.dataAugmentation import Augmentor
from models.resnet_base_network import ResNet18
from trainer_dino import DINOTrainer

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


def build_model(config, device, dim, args, batch_size):
    """DINO 需要 student + teacher 两个同结构网络（没有 predictor）。"""
    print("Initializing networks...")
    # DINO 的 head 输出维度就是"原型"个数 K（论文里 65536，config.yaml 默认 128）。
    # K 只影响预训练目标，下游微调会把整个 projection 丢掉，加载时也先剔掉 projection.*
    # 再 load_state_dict(strict=False)，所以这里改 --dim 不需要动微调脚本。
    if dim is not None and dim != config['network']['projection_head']['projection_size']:
        config['network']['projection_head']['projection_size'] = dim
        print(f"[注意] projection_head.projection_size 已改为 {dim}（K={dim}）")

    student_network = ResNet18(**config['network']).to(device)
    teacher_network = ResNet18(**config['network']).to(device)

    if args.optimizer == 'adamw':
        # DINO 论文的配方：AdamW，lr = 0.0005 * batch_size / 256，wd 0.04
        lr = args.lr if args.lr is not None else 5e-4 * batch_size / 256.0
        wd = args.weight_decay if args.weight_decay is not None else 0.04
        optimizer = torch.optim.AdamW(student_network.parameters(),
                                      lr=lr, betas=(0.9, 0.999), weight_decay=wd)
        print(f"优化器: AdamW lr={lr:.5f} weight_decay={wd}")
    else:
        optimizer = torch.optim.SGD(student_network.parameters(), **config['optimizer']['params'])
        print(f"优化器: SGD {config['optimizer']['params']}")

    return student_network, teacher_network, optimizer


def resolve_resume(resume_arg, config):
    """--resume 可以给目录（取其中 epoch 最大的）或直接给 .pth，返回 (路径, 起始 epoch)。"""
    path = resume_arg
    if os.path.isdir(path):
        candidates = glob.glob(os.path.join(path, 'model_epoch_*.pth'))
        if not candidates:
            raise FileNotFoundError(f"{path} 里没有 model_epoch_*.pth")
        path = max(candidates, key=lambda p: int(re.search(r'model_epoch_(\d+)', p).group(1)))
    if not os.path.exists(path):
        raise FileNotFoundError(f"续训权重不存在: {path}")
    m = re.search(r'model_epoch_(\d+)\.pth$', path)
    start_epoch = int(m.group(1)) if m else 0
    return path, start_epoch


def fix_suffix_argv(argv):
    """把 `--suffix -DINO` 改写成 `--suffix=-DINO`。

    argparse 看到以横线开头的值会当成另一个选项，直接报 "expected one argument"，
    所以空格写法对这类「值自带横线」的参数根本用不了；= 号写法则没问题。
    """
    out = []
    i = 0
    while i < len(argv):
        if argv[i] == '--suffix' and i + 1 < len(argv):
            out.append('--suffix=' + argv[i + 1])
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def main(args):
    model_name_list = [m.strip() for m in args.models.split(",") if m.strip()]
    for model_name in model_name_list:
        config = load_config()

        batch_size = config['trainer']['batch_size']
        # 让 wandb 里记的 num_workers 与实际 DataLoader 用的一致
        config['trainer']['num_workers'] = args.workers

        # 续训：定位权重文件，并把起始 epoch 带过去
        resume_path, start_epoch = (None, 0)
        if args.resume:
            resume_path, start_epoch = resolve_resume(args.resume, config)

        # 传给 DINOTrainer 的开关
        config['trainer']['use_wandb'] = not args.no_wandb
        config['trainer']['wandb_project'] = args.wandb_project
        config['trainer']['temperature_student'] = args.student_temp
        config['trainer']['temperature_teacher'] = args.teacher_temp
        config['trainer']['freeze_last_layer_epochs'] = (
            0 if resume_path else args.freeze_last_layer)   # 续训时 head 早就不冻了
        config['trainer']['use_cosine_schedule'] = args.cosine
        config['trainer']['lr_warmup_epochs'] = args.lr_warmup_epochs if args.optimizer == 'adamw' else 0
        config['trainer']['monitor_eff_rank'] = not args.no_monitor_eff_rank
        config['trainer']['monitor_every_n_epochs'] = args.monitor_every
        config['trainer']['resume_from'] = resume_path
        config['trainer']['start_epoch'] = start_epoch
        # 续训时 teacher 温度早就 warmup 完了，不能再退回 0.04 重新升温；
        # 没显式指定 --teacher-temp-warmup 的话就在续训时置 0
        if args.teacher_temp_warmup is None:
            config['trainer']['teacher_temp_warmup_epochs'] = 0 if resume_path else 30
        else:
            config['trainer']['teacher_temp_warmup_epochs'] = args.teacher_temp_warmup
        if args.epochs is not None:
            config['trainer']['max_epochs'] = args.epochs
        if args.lr is not None:
            config['optimizer']['params']['lr'] = args.lr

        run_name = f"{model_name}{args.suffix}"
        print(f"\n{'=' * 60}\nDINO 预训练: {run_name}\n{'=' * 60}")
        print(f"  max_epochs={config['trainer']['max_epochs']} batch_size={batch_size} "
              f"n_views={config['trainer']['n_views']} cosine={config['trainer']['use_cosine_schedule']}")
        print(f"  t_student={args.student_temp} t_teacher={args.teacher_temp} "
              f"freeze_last_layer={config['trainer']['freeze_last_layer_epochs']} dim={args.dim}")
        print(f"  teacher_temp_warmup_epochs={config['trainer']['teacher_temp_warmup_epochs']} "
              f"optimizer={args.optimizer}")
        if resume_path:
            print(f"  续训自: {resume_path}（起始 epoch {start_epoch}，"
                  f"本次再训 {config['trainer']['max_epochs']} 轮 -> {start_epoch + config['trainer']['max_epochs']}）")
        print(f"  [提醒] 下游微调要加载这套权重时，model_name 用 '{run_name}'")

        # 注意：数据文件名用的是 model_name（D1-Undefence-PreTrain.npy），
        # 带 -DINO 后缀的 run_name 只用于保存目录和 wandb
        train_loader, device, n_views, multi_times = prepare_data(model_name, config, batch_size, args.workers)
        student_net, teacher_net, optimizer = build_model(config, device, args.dim, args, batch_size)

        trainer = DINOTrainer(
            student_network=student_net,
            teacher_network=teacher_net,
            optimizer=optimizer,
            device=device,
            model_path=run_name,
            model_folder=config['path']['model_path'],
            **config['trainer']
        )
        trainer.train(train_loader)


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning, message="loaded more than 1 DLL from .libs")

    parser = argparse.ArgumentParser(description="Swallow DINO 预训练（独立于 BYOL/SimSiam 的那套，含 wandb）")
    parser.add_argument("--models", default="D1-Undefence",
                        help="逗号分隔的预训练数据集名，默认只跑 D1-Undefence（与 SimSiam/诊断脚本口径一致）")
    parser.add_argument("--suffix", default="-DINO",
                        help="保存目录后缀，默认 -DINO（存到 SaveModel/<模型名>-DINO，不覆盖其他方法的权重）")
    parser.add_argument("--epochs", type=int, default=None,
                        help="覆盖 config.yaml 里的 max_epochs，方便快速验证")
    parser.add_argument("--lr", type=float, default=None,
                        help="覆盖学习率：SGD 时改 config.yaml 的 0.03；"
                             "AdamW 时默认按论文公式 5e-4*bs/256，用这个可手动覆盖")
    parser.add_argument("--dim", type=int, default=None,
                        help="原型个数 K（投影头输出维度），默认沿用 config.yaml 的 projection_size(128)。"
                             "下游微调会剔掉 projection 再加载，所以改这个不用动微调脚本")
    parser.add_argument("--student-temp", type=float, default=0.1, help="student 温度，默认 0.1（论文值）")
    parser.add_argument("--teacher-temp", type=float, default=0.04,
                        help="teacher 温度 warmup 起点，默认 0.04（30 轮内升到 0.07，论文值）")
    parser.add_argument("--freeze-last-layer", type=int, default=1,
                        help="前多少轮冻结投影头末层梯度，默认 1（论文值）；0 表示不冻结；续训时自动置 0")
    parser.add_argument("--resume", default=None,
                        help="续训：给目录（取其中 epoch 最大的权重）或直接给 .pth。"
                             "会恢复 student/teacher/center，checkpoint 编号接着往下走，"
                             "且 teacher 温度不再重新 warmup")
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd",
                        help="优化器，默认 sgd（config.yaml 里的那套，与 BYOL 可比）；"
                             "DINO 论文用 adamw（lr=5e-4*bs/256, wd=0.04）")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="AdamW 的 weight_decay，默认 0.04（论文值）；SGD 时忽略")
    parser.add_argument("--cosine", action="store_true",
                        help="开启余弦学习率衰减（长 schedule 建议开）")
    parser.add_argument("--lr-warmup-epochs", type=int, default=5,
                        help="学习率线性 warmup 轮数，默认 5（仅 AdamW 生效）")
    parser.add_argument("--teacher-temp-warmup", type=int, default=None,
                        help="teacher 温度 warmup 轮数，默认 30（论文值）；续训时默认自动置 0")
    parser.add_argument("--no-monitor-eff-rank", action="store_true",
                        help="关闭每 N 轮的有效秩监控（默认开启，会记进 wandb 的 train/eff_rank）")
    parser.add_argument("--monitor-every", type=int, default=10,
                        help="每多少轮量一次有效秩，默认 10")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker 数，默认 4（与 PreTrain.py 一致）")
    parser.add_argument("--no-wandb", action="store_true", help="关闭 wandb 记录")
    parser.add_argument("--wandb-project", default="swallow-pretrain-dino",
                        help="wandb 项目名，默认 swallow-pretrain-dino"
                             "（传 swallow-pretrain 可与 BYOL 曲线放同一个项目里对比）")
    args = parser.parse_args(fix_suffix_argv(sys.argv[1:]))
    print(f"wandb: {'已关闭' if args.no_wandb else '开启 -> ' + args.wandb_project}")

    main(args)