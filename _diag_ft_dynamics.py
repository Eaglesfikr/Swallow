"""诊断：微调到底有没有把预训练表征毁掉（只读评测 + 复现微调循环，不碰任何训练脚本）。

已知事实（都在 D2/D6/D7 上，同一套 N-shot 协议）：
  - DINO-800 的冻结特征已经和 BYOL-200 打平（D2/D6）甚至更好（D7）
  - 但端到端微调后 DINO-800 反而比 BYOL 低 2~3 分，DINO-800 在 N=5 时甚至比自己
    的冻结特征还低 —— 说明差距是「微调过程」产生的，不是预训练权重本身

本脚本把 fineTuning-AE*.py 的微调循环原样复现一遍（同 bs=16、同 Adam lr=1e-4、同 N-shot），
但额外量几个微调脚本看不到的东西：
  1. ||ΔW|| / ||W0||  —— 编码器权重被推离预训练点的相对距离
  2. 微调后编码器再冻结做线性探针 —— 预训练结构还剩多少（掉了就说明被毁掉了）
  3. 梯度范数 —— 同样 lr 下谁被推得更猛

跑法：
  .venv/bin/python _diag_ft_dynamics.py                     # D2 N=5
  .venv/bin/python _diag_ft_dynamics.py --domain D6 --n 20
  .venv/bin/python _diag_ft_dynamics.py --bn-mode train      # 试试让 BN 统计量也跟着更新
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

from models.resnet_base_network import ResNet18


class Data(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)


class LogReg(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.linear(x)


@torch.no_grad()
def extract(enc, x, device, bs=256):
    outs = []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(np.asarray(x[i:i + bs])).float().to(device)
        xb = xb.reshape(xb.shape[0], 1, xb.shape[1], xb.shape[2])
        outs.append(enc(xb).view(xb.size(0), -1).cpu())
    return torch.cat(outs).numpy()


def frozen_probe(enc, x_tr, y_tr, x_te, y_te, n_cls, device, epochs=300):
    """在（可能已被微调过的）编码器上重新训一个线性头，看特征本身还剩多少可分性。"""
    ftr = extract(enc, x_tr, device)
    fte = extract(enc, x_te, device)
    torch.manual_seed(0)
    X = torch.from_numpy(ftr).float().to(device)
    Y = torch.from_numpy(y_tr).long().to(device)
    lin = nn.Linear(ftr.shape[1], n_cls).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    lin.train()
    for _ in range(epochs):
        opt.zero_grad()
        crit(lin(X), Y).backward()
        opt.step()
    lin.eval()
    with torch.no_grad():
        pred = lin(torch.from_numpy(fte).float().to(device)).argmax(1).cpu().numpy()
    return 100.0 * (pred == y_te).mean()


def sample_traces(x, y, N, seed):
    r = np.random.RandomState(seed)
    idx = []
    for lab in np.unique(y):
        cand = np.where(y == lab)[0]
        idx.extend(r.choice(cand, min(N, len(cand)), replace=False))
    return np.array(idx)


def load_encoder(path, config, device, which="student"):
    ck = torch.load(path, map_location=device)
    key = "target_network_state_dict" if which == "teacher" else "online_network_state_dict"
    if key not in ck:
        key = "online_network_state_dict"
    net = ResNet18(**config["network"]).to(device)
    net.load_state_dict(ck[key])
    return nn.Sequential(*list(net.children())[:-1]).to(device).eval()


def run_one(ckpt, config, device, X, Y, Xte, Yte, n_cls, args, seed):
    enc = load_encoder(ckpt, config, device, args.which)
    init = [p.detach().clone() for p in enc.parameters()]

    fp_before = frozen_probe(enc, X, Y, Xte, Yte, n_cls, device)

    if args.bn_mode == "always_eval":
        enc.eval()
    else:
        # orig / train：一开始都在 train 模式（BN 用 batch 统计量并更新 running stats），
        # 差别在于 orig 第一次评估后就永久停在 eval（原脚本就是这样，没切回去）
        enc.train()
    head = LogReg(512, n_cls).to(device)
    head.train()
    groups = [{"params": list(enc.parameters()), "lr": args.encoder_lr or args.lr},
              {"params": list(head.parameters()), "lr": args.lr}]
    opt = torch.optim.Adam(groups)
    crit = nn.CrossEntropyLoss()

    loader = DataLoader(Data(X, Y), batch_size=16, shuffle=True, drop_last=True)
    best_acc, last_acc, wb_step = 0.0, 0.0, 0
    wb = args.wb_run
    grad_norm_first = None
    for epoch in range(args.epochs):
        if args.bn_mode == "train" and epoch > 0:
            enc.train()
        for x, y in loader:
            x = torch.from_numpy(np.asarray(x)).float().to(device)
            y = torch.from_numpy(np.asarray(y)).long().to(device)
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
            feats = enc(x).view(x.size(0), -1)
            if args.normalize:
                feats = nn.functional.normalize(feats, dim=1)
            loss = crit(head(feats), y)
            opt.zero_grad()
            loss.backward()
            if grad_norm_first is None:
                grad_norm_first = float(torch.sqrt(sum(
                    (p.grad ** 2).sum() for p in enc.parameters() if p.grad is not None)).item())
            opt.step()
            wb_step += 1

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            enc.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in DataLoader(Data(Xte, Yte), batch_size=256, drop_last=True):
                    x = torch.from_numpy(np.asarray(x)).float().to(device).reshape(-1, 1, x.shape[1], x.shape[2])
                    y = torch.from_numpy(np.asarray(y)).long().to(device)
                    f = enc(x).view(x.size(0), -1)
                    if args.normalize:
                        f = nn.functional.normalize(f, dim=1)
                    correct += (head(f).argmax(1) == y).sum().item()
                    total += y.size(0)
            last_acc = 100.0 * correct / max(total, 1)
            best_acc = max(best_acc, last_acc)
            if wb is not None:
                wb.log({"ft/test_acc": last_acc, "ft/best_acc": best_acc,
                        "ft/epoch": epoch, "ft/N": args.n}, step=wb_step)
            if args.bn_mode == "train":
                enc.train()

    num = torch.sqrt(sum(((p.detach() - q) ** 2).sum() for p, q in zip(enc.parameters(), init))).item()
    den = torch.sqrt(sum((q ** 2).sum() for q in init)).item()
    enc.eval()          # 评估口径统一用 eval（BN 走 running stats）
    with torch.no_grad():
        fn = extract(enc, X, device)
    fp_after = frozen_probe(enc, X, Y, Xte, Yte, n_cls, device)
    del enc
    if device == "cuda":
        torch.cuda.empty_cache()
    return dict(best=best_acc, last=last_acc, drift=num / max(den, 1e-12),
                grad=grad_norm_first, fp_before=fp_before, fp_after=fp_after,
                feat_norm=float(np.linalg.norm(fn, axis=1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="D2")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4, help="分类头 lr（与原脚本一致 1e-4）")
    ap.add_argument("--encoder-lr", type=float, default=None,
                    help="编码器单独的学习率，默认与 --lr 相同（原脚本是同一个 1e-4）")
    ap.add_argument("--normalize", action="store_true",
                    help="喂给分类头前对特征做 L2 归一化（抹平不同方法之间的特征尺度差）")
    ap.add_argument("--eval-every", type=int, default=10,
                    help="每几轮评估一次，默认 10（与 fineTuning-AE*.py 一致）")
    ap.add_argument("--bn-mode", choices=["orig", "always_eval", "train"], default="orig",
                    help="orig=完全复现原脚本（第 0 轮评估后 BN 永久停在 eval，running stats "
                         "已被第一个 epoch 的 bs=16 目标域 batch 污染一点点，相当于一次免费域适配）；"
                         "always_eval=BN 从一开始就用预训练统计量；train=每轮都切回 train 更新统计量")
    ap.add_argument("--which", choices=["student", "teacher"], default="student",
                    help="加载 student 还是 teacher 权重做微调（DINO 官方评测用的是 teacher）")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default="swallow-diag-ft-dynamics")
    ap.add_argument("--models", nargs="+", default=[
        "BYOL-200=SaveModel/D1-Undefence/model_epoch_200.pth",
        "DINO-200=SaveModel/D1-Undefence-DINO/model_epoch_200.pth",
        "DINO-800=SaveModel/D1-Undefence-DINO/model_epoch_800.pth",
    ])
    args = ap.parse_args()

    config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  domain={args.domain}  N={args.n}  epochs={args.epochs}  "
          f"bn_mode={args.bn_mode}  which={args.which}  head_lr={args.lr}  "
          f"enc_lr={args.encoder_lr or args.lr}  normalize={args.normalize}\n")

    z = np.load(config["path"]["finetune_data"] + f"{args.domain}-Undefence-FineTune.npz")
    uv = np.unique(z['Y_tune_train'])
    mp = {v: i for i, v in enumerate(uv)}
    Xtr_all = z['X_tune_train']
    Ytr_all = np.array([mp[v] for v in z['Y_tune_train']]).astype(np.int64)
    Xte, Yte = z['X_tune_test'], np.array([mp[v] for v in z['Y_tune_test']]).astype(np.int64)
    n_cls = len(uv)

    args.wb_run = None
    if not args.no_wandb:
        try:
            import wandb
            args.wb_run = wandb.init(project=args.wandb_project,
                                     name=f"{args.domain}-N{args.n}-bn{args.bn_mode}-lr{args.lr:g}",
                                     config=vars(args).copy() | {"wb_run": None}, reinit=True)
        except Exception as e:
            print(f"[wandb] 跳过：{type(e).__name__}: {e}")

    print(f"  {'model':>10s} {'fp_before':>10s} {'ft_best':>8s} {'ft_last':>8s} "
          f"{'fp_after':>9s} {'||dW||/||W||':>13s} {'grad':>8s} {'|feat|':>8s}")
    print("  " + "-" * 84)
    for spec in args.models:
        tag, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"  {tag:>10s}  缺 {path}")
            continue
        rows = []
        for r in range(args.runs):
            idx = sample_traces(Xtr_all, Ytr_all, args.n, 1000 * args.n + r)
            rows.append(run_one(path, config, device, Xtr_all[idx], Ytr_all[idx],
                                Xte, Yte, n_cls, args, r))
        agg = {k: float(np.mean([x[k] for x in rows])) for k in rows[0]}
        print(f"  {tag:>10s} {agg['fp_before']:10.2f} {agg['best']:8.2f} {agg['last']:8.2f} "
              f"{agg['fp_after']:9.2f} {agg['drift']:13.4f} {agg['grad']:8.3f} {agg['feat_norm']:8.2f}")
        if args.wb_run is not None:
            args.wb_run.log({f"{tag}/{k}": v for k, v in agg.items()})

    if args.wb_run is not None:
        args.wb_run.finish()

    print("\n判读：")
    print("  fp_after 比 fp_before 掉很多 -> 微调把预训练表征毁掉了（考虑降 lr / 差异化 lr / 只训头）")
    print("  ||dW||/||W|| 明显更大          -> 同一个 lr 对这套权重更猛，是 lr 口径不匹配")
    print("  grad 明显更大                  -> 同上，梯度尺度不同")
    print("\nDONE")


if __name__ == "__main__":
    main()