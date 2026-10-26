import argparse
import torch
import yaml
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from models.resnet_base_network import ResNet18
import warnings
import os
import logging

warnings.filterwarnings("ignore")

# --- wandb 可视化。可用 --no-wandb 或环境变量 WANDB_MODE=disabled / offline 关闭或转离线 ---
try:
    import wandb
except ImportError:
    wandb = None

USE_WANDB = True
WANDB_PROJECT = "swallow-finetune-ae"

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

    # --- wandb：一个迁移对（自己的预训练模型 -> 目标数据集）对应一个 run ---
    wb_run = None
    if USE_WANDB and wandb is not None:
        try:
            wb_run = wandb.init(
                project=WANDB_PROJECT,
                group=scenario_type,
                name=f"{model_name}->{defence_name}",
                config={
                    "scenario": scenario_type,
                    "model_name": model_name,
                    "defence_name": defence_name,
                    "checkpoint": f"{config['path']['model_path']}/{model_name}/model_epoch_200.pth",
                    "num_classes": int(len(unique_values_sorted)),
                    "N_shots": list(N_shots),
                    "num_runs": num_runs,
                    "num_epochs": num_epochs,
                    "eval_every_n_epochs": eval_every_n_epochs,
                    "batch_size": batch_size,
                    "lr": 1e-4,
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
            # Load pre-trained encoder
            encoder = ResNet18(**config['network']).to(device)
            model_path = config['path']['model_path']
            load_params = torch.load(
                f'{model_path}/{model_name}/model_epoch_200.pth',
                map_location=device)

            if 'online_network_state_dict' in load_params:
                encoder.load_state_dict(load_params['online_network_state_dict'])
                print("Parameters successfully loaded.")

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
            optimizer = torch.optim.Adam(list(encoder.parameters()) + list(logreg.parameters()), lr=1e-4)
            criterion = nn.CrossEntropyLoss()

            # Training loop
            best_acc = 0.0
            encoder.train()
            logreg.train()

            for epoch in range(num_epochs):
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
        with open('results/close/results-%s.txt' % (scenario_type), 'a') as file:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Swallow 闭世界微调（使用自己预训练的 D1 模型，Scenario#1-AE）")
    parser.add_argument("--no-wandb", action="store_true", help="关闭 wandb 记录")
    parser.add_argument("--wandb-project", default=WANDB_PROJECT,
                        help=f"wandb 项目名，默认 {WANDB_PROJECT}")
    parser.add_argument("--targets", default="D2,D3,D4,D5",
                        help="逗号分隔的目标域，默认 D2,D3,D4,D5；只跑一个域可传 --targets D2")
    args = parser.parse_args()
    USE_WANDB = not args.no_wandb
    WANDB_PROJECT = args.wandb_project
    print(f"wandb: {'开启 -> ' + WANDB_PROJECT if USE_WANDB else '已关闭'}")

    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    # Training parameters
    N_shots = [5, 10, 15, 20]
    num_runs = 5
    num_epochs = 41
    eval_every_n_epochs = 10
    batch_size = 16

    # Scenario #1: Different Locations of Guard Relays.
    # defence_type_list = ["Undefence", "WTF-PAD", "Front"]
    defence_type_list = ["Undefence"]
    data_list = [t.strip() for t in args.targets.split(",") if t.strip()]
    for defence in defence_type_list:
        model_name = f"D1-{defence}"
        for data in data_list:
            defence_name = f"{data}-{defence}"
            print(f"{model_name} -> {defence_name}")
            finetune(model_name, defence_name, "Scenario#1-AE")





