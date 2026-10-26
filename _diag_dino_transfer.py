"""诊断：DINO 续训到 800 之后，表征的「域内变好 / 跨域变差」到底成不成立（只读，不训练）。

上一轮已经看到：
  - eff_rank 18 -> 61.6（确实训练开了）
  - teacher 熵 0.238 -> 0.08（几乎 one-hot），原型与 D1 标签的 NMI 0.638 -> 0.744
  - 但端到端微调 D2 79.70->77.86、D6 80.18->75.65、D7 52.72->48.91（更差了）

本脚本用「冻结特征 N-shot 线性探针」分开量 D1（域内）和 D2/D6/D7（跨域）：
  D1 探针涨、跨域掉  -> 表征在往 D1 特有的结构上过拟合，延长训练只会更糟
  一起掉             -> 是别的环节（比如加载 student 而不是 teacher）

跑法：
  .venv/bin/python _diag_dino_transfer.py
  .venv/bin/python _diag_dino_transfer.py --epochs 200 400 600 800 --which student teacher
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import yaml

from models.resnet_base_network import ResNet18


@torch.no_grad()
def extract(net, x, device, bs=512):
    outs = []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(np.asarray(x[i:i + bs])).float().to(device)
        xb = xb.reshape(xb.shape[0], 1, xb.shape[1], xb.shape[2])
        h = net.encoder(xb)
        outs.append(h.view(xb.size(0), -1).cpu())
    return torch.cat(outs).numpy()


def fit_linear(f_tr, y_tr, n_cls, device, epochs=300, lr=1e-3, wd=1e-4):
    torch.manual_seed(0)
    X = torch.from_numpy(f_tr).float().to(device)
    Y = torch.from_numpy(y_tr).long().to(device)
    lin = nn.Linear(X.shape[1], n_cls).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    lin.train()
    for _ in range(epochs):
        opt.zero_grad()
        crit(lin(X), Y).backward()
        opt.step()
    return lin.eval()


@torch.no_grad()
def accuracy(lin, f, y, device):
    pred = lin(torch.from_numpy(f).float().to(device)).argmax(1).cpu().numpy()
    return 100.0 * (pred == y).mean()


def shot_index(Ytr, N, run):
    """固定种子采 N-shot，保证所有模型/所有 epoch 拿到完全相同的样本。"""
    r = np.random.RandomState(1000 * N + run)
    idx = []
    for lab in np.unique(Ytr):
        cand = np.where(Ytr == lab)[0]
        idx.extend(r.choice(cand, min(N, len(cand)), replace=False))
    return np.array(idx)


def load_backbone(path, config, device, which):
    ck = torch.load(path, map_location=device)
    key = "target_network_state_dict" if which == "teacher" else "online_network_state_dict"
    if key not in ck:
        key = "online_network_state_dict"
    net = ResNet18(**config["network"]).to(device)
    net.load_state_dict(ck[key])
    return net.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="SaveModel/D1-Undefence-DINO")
    ap.add_argument("--byol", default="SaveModel/D1-Undefence/model_epoch_200.pth")
    ap.add_argument("--epochs", type=int, nargs="+", default=[200, 400, 600, 800])
    ap.add_argument("--which", nargs="+", default=["student"], choices=["student", "teacher"])
    ap.add_argument("--domains", default="D1,D2,D6,D7")
    ap.add_argument("--n-shots", type=int, nargs="+", default=[5, 20])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--probe-epochs", type=int, default=300)
    args = ap.parse_args()

    config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}\n")

    # ---- 数据：D1 用 even/odd 切 train/test，其余用数据集自带的 tune_train/test ----
    domains = {}
    for d in [s.strip() for s in args.domains.split(",") if s.strip()]:
        if d == "D1":
            z = np.load(config["path"]["pretrain_data"] + "D1-Undefence-PreTrain.npy",
                        allow_pickle=True).item()
            X, Y = z['dataset'], np.asarray(z['label'])
            uv = np.unique(Y)
            Y = np.searchsorted(uv, Y).astype(np.int64)
            rng = np.random.RandomState(0)
            perm = rng.permutation(len(X))
            te, tr = perm[:len(perm) // 5], perm[len(perm) // 5:]
        else:
            z = np.load(config["path"]["finetune_data"] + f"{d}-Undefence-FineTune.npz")
            uv = np.unique(z['Y_tune_train'])
            mp = {v: i for i, v in enumerate(uv)}
            X = z['X_tune_train']
            tr = np.arange(len(X))
            Y = np.array([mp[v] for v in z['Y_tune_train']]).astype(np.int64)
            te = np.arange(len(z['X_tune_test']))
            domains[d] = dict(
                Xtr=X[tr], Ytr=Y[tr], Xte=z['X_tune_test'],
                Yte=np.array([mp[v] for v in z['Y_tune_test']]).astype(np.int64),
                n_cls=len(uv), idx_map={(N, r): shot_index(Y[tr], N, r)
                                        for N in args.n_shots for r in range(args.runs)})
            print(f"{d}: train={domains[d]['Xtr'].shape} test={domains[d]['Xte'].shape} "
                  f"classes={domains[d]['n_cls']}")
            continue
        domains[d] = dict(
            Xtr=X[tr], Ytr=Y[tr], Xte=X[te], Yte=Y[te], n_cls=len(uv),
            idx_map={(N, r): shot_index(Y[tr], N, r)
                     for N in args.n_shots for r in range(args.runs)})
        print(f"{d}: train={domains[d]['Xtr'].shape} test={domains[d]['Xte'].shape} "
              f"classes={domains[d]['n_cls']}")

    jobs = []
    if os.path.exists(args.byol):
        jobs.append(("BYOL-200", args.byol))
    for ep in args.epochs:
        p = os.path.join(args.dir, f"model_epoch_{ep}.pth")
        if os.path.exists(p):
            jobs.append((f"DINO-{ep}", p))
        else:
            print(f"[跳过] 缺 {p}")

    results = {}
    for tag, path in jobs:
        for which in (["student"] if tag.startswith("BYOL") else args.which):
            name = f"{tag}" + ("" if which == "student" else f"({which})")
            net = load_backbone(path, config, device, which)
            row = {}
            for d, dom in domains.items():
                ftr = extract(net, dom["Xtr"], device)
                fte = extract(net, dom["Xte"], device)
                for N in args.n_shots:
                    accs = []
                    for r in range(args.runs):
                        idx = dom["idx_map"][(N, r)]
                        lin = fit_linear(ftr[idx], dom["Ytr"][idx], dom["n_cls"],
                                         device, args.probe_epochs)
                        accs.append(accuracy(lin, fte, dom["Yte"], device))
                    row[(d, N)] = float(np.mean(accs))
                print(f"  [{name}] {d} 完成")
            results[name] = row
            del net
            if device == "cuda":
                torch.cuda.empty_cache()

    print(f"\n{'=' * 78}\n冻结特征 N-shot 线性探针（student == 微调时实际加载的那个）\n{'=' * 78}")
    for d in domains:
        print(f"\n--- {d}{'（域内/D1）' if d == 'D1' else '（跨域）'} ---")
        print(f"  {'ckpt':>20s} " + " ".join(f"{'N=' + str(N):>8s}" for N in args.n_shots))
        print("  " + "-" * 60)
        for name, row in results.items():
            print(f"  {name:>20s} " + " ".join(f"{row[(d, N)]:8.2f}" for N in args.n_shots))

    print("\n判读：")
    print("  D1 涨、D2/D6/D7 掉 -> 表征往 D1 独有结构上过拟合，继续延长训练只会更糟")
    print("  D1/D2/D6/D7 一起掉 -> 不是续训的锅，得查加载 student vs teacher、或微调设置")
    print("\nDONE")


if __name__ == "__main__":
    main()