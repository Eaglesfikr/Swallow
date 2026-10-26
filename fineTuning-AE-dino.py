"""Swallow 闭世界微调（Scenario#1-AE，目标域 D2/D6/D7）—— DINO 预训练权重。

独立文件，不修改 fineTuning-AE.py。只改两样东西：加载哪份预训练权重（D1-Undefence-DINO）、
结果写到哪个文件。训练超参（N-shot 列表、run 数、epoch 数、lr、batch、评估间隔）与
fineTuning-AE.py 完全一致，保证和 BYOL 基线（fineTuning-AE-d2d6d7.py）的差异
只来自自监督框架这一个变量。

微调逻辑本身不用为 DINO 改任何东西，原因：
  1. DINO 预训练的 checkpoint 里保留了 'online_network_state_dict'（student 主干+投影头），
     与 BYOL/SimSiam 的 key 名一致，encoder.load_state_dict(...) 直接能用
  2. 加载时先把 projection.* 整个剔掉再 load_state_dict(strict=False)：反正下面编码器
     本来就会丢掉投影头，剔掉之后 DINO 的原型个数 K 是多少都不影响加载
     （K=128 的老权重、K=1024 的新权重都能读；主干权重缺一个就立刻报错，不会静默拿随机权重跑）
  3. 微调时投影头整个丢掉、只留 backbone，所以 DINO head 的 centering/softmax 那些
     预训练目标不会影响下游；这一步对任何自监督框架都一样

与 fineTuning-AE.py 的差异：
  1. model_name 带后缀（默认 -DINO），权重从 SaveModel/D1-Undefence-DINO/ 加载；
     K=1024 那套用 --suffix -DINO-K1024，读 SaveModel/D1-Undefence-DINO-K1024/
  2. 目标域默认只有 D2,D6,D7：D2-D5 属于同一类「网络条件变化」的跨域，这里用 D2 作代表，
     再补上 D6、D7 两类
  3. 结果写到独立文件 results-Scenario#1-AE-D2D6D7-DINO.txt，不和 BYOL 的条目混在一起
  4. checkpoint 找不到时自动退回目录里 epoch 最大的那个；加载失败直接报错，
     而不是像原脚本那样静默跳过、拿随机初始化的编码器接着跑
  5. 新增可选 --freeze-encoder 与 --fix-bn-mode，默认都关闭，保持与 fineTuning-AE.py 同口径。
     注意 DINO 论文的评估口径是冻结主干做 linear probe，想按那套跑要加 --freeze-encoder，
     但要让对比成立，BYOL 那边也得加上同样的开关

用法：
  .venv/bin/python fineTuning-AE-dino.py                      # D1-Undefence-DINO -> D2,D6,D7
  .venv/bin/python fineTuning-AE-dino.py --targets D6         # 只跑一个目标域
  .venv/bin/python fineTuning-AE-dino.py --freeze-encoder     # 冻结主干，只训分类头
  .venv/bin/python fineTuning-AE-dino.py --suffix ""          # suffix 传空 = 跑 BYOL 基线，同口径对比
  .venv/bin/python fineTuning-AE-dino.py --suffix -DINO-K1024 # 跑 K=1024 那套预训练权重
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
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

from models.resnet_base_network import ResNet18

warnings.filterwarnings("ignore")

# --- wandb 可视化。可用 --no-wandb 或环境变量 WANDB_MODE=disabled / offline 关闭或转离线 ---
try:
    import wandb
except ImportError:
    wandb = None

USE_WANDB = True
WANDB_PROJECT = "swallow-finetune-ae-dino"
FIX_BN_MODE = False
FREEZE_ENCODER = False

# Set environment variables to suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)


def sample_traces(x, y, N):
    """Sample N traces per class from the dataset."""
    train_index = []
    labels = np.unique(y)
    for label in labels:
        idx = np.where(y == label)[0]
        idx = np.random.choice(idx, min(N, len(idx)), False)
        train_index.extend(idx)

    train_index = np.array(train_index)
    np.random.shuffle(train_index)
    return x[train_index], y[train_index]


class Data(Dataset):
    """Dataset class for training and testing data."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class LogisticRegression(nn.Module):
    """Logistic Regression classifier."""

    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def resolve_checkpoint(model_name, epoch):
    """定位预训练权重：优先取指定 epoch，找不到就退回目录里 epoch 最大的那个。"""
    model_dir = os.path.join(config['path']['model_path'], model_name)
    wanted = os.path.join(model_dir, f'model_epoch_{epoch}.pth')
    if os.path.exists(wanted):
        return wanted

    candidates = glob.glob(os.path.join(model_dir, 'model_epoch_*.pth'))
    if not candidates:
        raise FileNotFoundError(
            f"没找到预训练权重：{model_dir}/model_epoch_*.pth\n"
            f"  先确认预训练是否跑完，或用 --suffix 指定正确的目录后缀。")

    def _epoch_of(path):
        m = re.search(r'model_epoch_(\d+)\.pth$', path)
        return int(m.group(1)) if m else -1

    fallback = max(candidates, key=_epoch_of)
    print(f"[提示] {wanted} 不存在，改用 {os.path.basename(fallback)}")
    return fallback


def finetune(model_name, defence_name, scenario_type):
    # Load configuration
    # Load dataset
    fine_data_base_path = config["path"]["finetune_data"]
    data_path = fine_data_base_path + '%s-FineTune.npz' % (defence_name)
    data = np.load(data_path)
    x_train_total = data['X_tune_train']
    y_train_total = data['Y_tune_train']
    x_test_sup = data['X_tune_test']
    y_test_sup = data['Y_tune_test']

    # Map labels to consecutive integers
    unique_values_sorted = np.unique(y_train_total)
    mapping_dict = {value: idx for idx, value in enumerate(unique_values_sorted)}
    y_train_total = np.array([mapping_dict[value] for value in y_train_total]).astype(np.int64)
    y_test_sup = np.array([mapping_dict[value] for value in y_test_sup]).astype(np.int64)

    # Create test dataset and loader
    test_dataset = Data(x_test_sup, y_test_sup)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

    ckpt_path = resolve_checkpoint(model_name, checkpoint_epoch)
    print(f"预训练权重: {ckpt_path}")

    # --- wandb：一个迁移对（自己的预训练模型 -> 目标数据集）对应一个 run ---
    wb_run = None
    if USE_WANDB and wandb is not None:
        try:
            wb_run = wandb.init(
                project=WANDB_PROJECT,
                group=f"{scenario_type}{suffix}",
                name=f"{model_name}->{defence_name}",
                config={
                    "scenario": scenario_type,
                    "method": method_tag,
                    "model_name": model_name,
                    "defence_name": defence_name,
                    "checkpoint": ckpt_path,
                    "num_classes": int(len(unique_values_sorted)),
                    "N_shots": list(N_shots),
                    "num_runs": num_runs,
                    "num_epochs": num_epochs,
                    "eval_every_n_epochs": eval_every_n_epochs,
                    "batch_size": batch_size,
                    "lr": 1e-4,
                    "freeze_encoder": FREEZE_ENCODER,
                    "fix_bn_mode": FIX_BN_MODE,
                    "device": str(device),
                    "train_size_total": int(len(x_train_total)),
                    "test_size": int(len(x_test_sup)),
                },
                reinit=True,
            )
        except Exception as e:
            print(f"[wandb] 初始化失败，本次跳过记录：{type(e).__name__}: {e}")
            print("[wandb] 如需启用，先执行登录： .venv/bin/python -m wandb login")
            wb_run = None

    wb_step = 0        # 跨 N / run 单调递增，保证曲线连续
    result_rows = []   # 汇总表数据

    # Training loop for different N-shot settings
    for N in N_shots:
        accuracies = []
        print("%s -> %s %s" % (model_name, defence_name, N))
        for run in range(num_runs):
            # Load pre-trained encoder（靠 online_network_state_dict 取主干）
            encoder = ResNet18(**config['network']).to(device)
            load_params = torch.load(ckpt_path, map_location=device)

            if 'online_network_state_dict' not in load_params:
                raise KeyError(
                    f"{ckpt_path} 里没有 'online_network_state_dict'，"
                    f"实际包含的 key：{list(load_params.keys())}")
            # 先把投影头剔掉再加载：下面的 encoder 反正会丢掉 projection，
            # 而 DINO 的 K（projection_size）可以和 config.yaml 里的 128 不一样，
            # 剔掉就不会因为形状不匹配而在 strict 加载时报错。
            sd_all = load_params['online_network_state_dict']
            sd = {k: v for k, v in sd_all.items() if not k.startswith('projection.')}
            missing, unexpected = encoder.load_state_dict(sd, strict=False)
            bad_missing = [k for k in missing if not k.startswith('projection.')]
            if bad_missing:
                raise RuntimeError(f"主干权重没加载上，checkpoint 与 network 配置不匹配：{bad_missing}")
            if unexpected:
                print(f"[提示] checkpoint 里多出来的权重被忽略：{unexpected}")
            k_out = sd_all.get('projection.net.3.weight')
            extra = f"（已跳过投影头 {len(sd_all) - len(sd)} 个张量"
            extra += f"，该 checkpoint 的 K={k_out.shape[0]}）" if k_out is not None else "）"
            print(f"Parameters successfully loaded.{extra}")

            # Remove projection head
            output_feature_dim = encoder.projection.net[0].in_features
            encoder = nn.Sequential(*list(encoder.children())[:-1])

            # Sample training data
            x_train, y_train = sample_traces(x_train_total, y_train_total, N)
            train_dataset = Data(x_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

            # Initialize classifier
            logreg = LogisticRegression(output_feature_dim, len(unique_values_sorted)).to(device)

            # Setup optimizer and loss function
            # --freeze-encoder：冻结主干（同时停掉 BN 统计量更新），只训分类头
            if FREEZE_ENCODER:
                encoder.eval()
                for param in encoder.parameters():
                    param.requires_grad = False
                optimizer = torch.optim.Adam(logreg.parameters(), lr=1e-4)
            else:
                optimizer = torch.optim.Adam(list(encoder.parameters()) + list(logreg.parameters()), lr=1e-4)
            criterion = nn.CrossEntropyLoss()

            # Training loop
            best_acc = 0.0
            if FREEZE_ENCODER:
                encoder.eval()
            else:
                encoder.train()
            logreg.train()

            for epoch in range(num_epochs):
                # --fix-bn-mode：原脚本在第 0 轮评估后就不再切回 train 模式，
                # BN 统计量从此冻结在预训练值上；这里每轮开始重置一次
                # （冻结主干时会保持 eval，不能在训练途中切回 train 污染 BN 统计量）
                if FIX_BN_MODE and not FREEZE_ENCODER:
                    encoder.train()
                    logreg.train()

                # Training phase
                correct_train, total_train = 0, 0
                total_loss = 0

                for x, y in train_loader:
                    x = x.float().to(device)
                    y = y.to(device)
                    x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

                    # Forward pass
                    features = encoder(x).view(x.size(0), -1)
                    logits = logreg(features)
                    loss = criterion(logits, y)

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Calculate training accuracy
                    predictions = torch.argmax(logits, dim=1)
                    correct_train += (predictions == y).sum().item()
                    total_train += y.size(0)
                    total_loss += loss.item()

                # Evaluation phase
                test_acc = None
                if epoch % eval_every_n_epochs == 0:

                    # Test evaluation
                    encoder.eval()
                    logreg.eval()
                    correct_test, total_test = 0, 0
                    website_res = []

                    with torch.no_grad():
                        for x, y in test_loader:
                            x = x.float().to(device)
                            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
                            y = y.to(device)

                            features = encoder(x).view(x.size(0), -1)
                            logits = logreg(features)
                            predictions = torch.argmax(logits, dim=1)

                            correct_test += (predictions == y).sum().item()
                            total_test += y.size(0)

                            for i in range(len(predictions)):
                                website_res.append([y[i].item(), predictions[i].item()])
                    test_acc = 100 * correct_test / total_test
                    if test_acc > best_acc:
                        best_acc = test_acc

                    print(f"Epoch [{epoch}/{num_epochs - 1}], Testing accuracy: {test_acc:.2f}%")

                # --- wandb：每 epoch 记录训练指标；评估轮额外记录测试指标 ---
                if wb_run is not None:
                    log_dict = {
                        "train/loss": total_loss / max(len(train_loader), 1),
                        "train/accuracy": 100 * correct_train / max(total_train, 1),
                        "epoch": epoch,
                        "N_shot": N,
                        "run_index": run,
                    }
                    if test_acc is not None:
                        log_dict["test/accuracy"] = test_acc
                        log_dict["test/best_accuracy"] = best_acc
                    wb_run.log(log_dict, step=wb_step)
                wb_step += 1

            accuracies.append(best_acc)

        print(f"avg -> {np.mean(accuracies):.2f}, std -> {np.std(accuracies):.2f}")

        # 结果文件名带 suffix，避免和原脚本的 D2-D5 行、DINO 版互相干扰
        results_path = 'results/close/results-%s%s.txt' % (scenario_type, suffix)
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'a') as file:
            file.write(f'{model_name}-{defence_name} - {N} \n')
            file.write(f" avg -> {np.mean(accuracies):.2f}, std -> {np.std(accuracies):.2f}, "
                       f"max -> {np.max(accuracies):.2f}\n")

        # --- wandb：写入 summary（key 里带 N，多个 N 不会互相覆盖）---
        n_avg, n_std, n_max = float(np.mean(accuracies)), float(np.std(accuracies)), float(np.max(accuracies))
        result_rows.append([N, n_avg, n_std, n_max])
        if wb_run is not None:
            wb_run.summary[f"N={N}/avg"] = n_avg
            wb_run.summary[f"N={N}/std"] = n_std
            wb_run.summary[f"N={N}/max"] = n_max

    # --- wandb：一张表汇总所有 N，然后结束本次 run ---
    if wb_run is not None:
        if result_rows:
            wb_run.log({"results/summary": wandb.Table(
                columns=["N-shot", "avg", "std", "max"], data=result_rows)}, step=wb_step)
        wb_run.finish()


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Swallow 闭世界微调（DINO 预训练权重，Scenario#1-AE，目标域 D2/D6/D7）")
    parser.add_argument("--suffix", default="-DINO",
                        help="预训练目录后缀，默认 -DINO（读 SaveModel/D1-Undefence-DINO/）；"
                             "K=1024 那套传 -DINO-K1024；"
                             "传空字符串则回到 BYOL 基线 SaveModel/D1-Undefence/")
    parser.add_argument("--defences", default="Undefence",
                        help="逗号分隔的预训练防御类型，默认 Undefence（与 fineTuning-AE.py 一致）")
    parser.add_argument("--targets", default="D2,D6,D7",
                        help="逗号分隔的目标域，默认 D2,D6,D7；只跑一个域可传 --targets D2")
    parser.add_argument("--checkpoint-epoch", type=int, default=200,
                        help="加载哪个 epoch 的权重，默认 200；该文件缺失时自动退回目录里 epoch 最大的")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="冻结主干只训分类头（linear probe 口径）；默认关闭，保持与 fineTuning-AE.py 同口径")
    parser.add_argument("--fix-bn-mode", action="store_true",
                        help="每个 epoch 重置回 train 模式（原脚本第 0 轮评估后一直停在 eval，BN 冻结）")
    parser.add_argument("--no-wandb", action="store_true", help="关闭 wandb 记录")
    parser.add_argument("--wandb-project", default=WANDB_PROJECT,
                        help=f"wandb 项目名，默认 {WANDB_PROJECT}")
    args = parser.parse_args(fix_suffix_argv(sys.argv[1:]))
    USE_WANDB = not args.no_wandb
    WANDB_PROJECT = args.wandb_project
    FIX_BN_MODE = args.fix_bn_mode
    FREEZE_ENCODER = args.freeze_encoder
    suffix = args.suffix
    checkpoint_epoch = args.checkpoint_epoch
    method_tag = "byol" if not suffix else suffix.lstrip('-').lower()
    print(f"wandb: {'开启 -> ' + WANDB_PROJECT if USE_WANDB else '已关闭'}"
          f"{' | freeze-encoder: 开' if FREEZE_ENCODER else ''}"
          f"{' | fix-bn-mode: 开' if FIX_BN_MODE else ''}")

    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    # Training parameters（与 fineTuning-AE.py 完全一致，保证同口径对比）
    N_shots = [5, 10, 15, 20]
    num_runs = 5
    num_epochs = 41
    eval_every_n_epochs = 10
    batch_size = 16

    # Scenario #1: Different Locations of Guard Relays（目标域 D2/D6/D7）。
    # 结果写 results-Scenario#1-AE-D2D6D7<suffix>.txt，不碰原来的 results-Scenario#1-AE.txt
    defence_type_list = [d.strip() for d in args.defences.split(",") if d.strip()]
    data_list = [t.strip() for t in args.targets.split(",") if t.strip()]
    for defence in defence_type_list:
        model_name = f"D1-{defence}{suffix}"
        for data in data_list:
            defence_name = f"{data}-{defence}"
            print(f"{model_name} -> {defence_name}")
            finetune(model_name, defence_name, "Scenario#1-AE-D2D6D7")