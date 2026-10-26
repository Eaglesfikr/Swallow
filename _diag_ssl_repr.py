"""诊断：对比 BYOL 与 SimSiam 预训练编码器的表征质量（只读，不训练）。

分别测：
  1. 塌陷程度：逐维标准差、有效秩、特征范数
  2. 源域(D1)线性可分性：冻结特征上训一个线性分类器
  3. 跨域可迁移性：D1 上训 -> D2/D3/D4/D5 上测（zero-shot，需标签对齐）
  4. 冻结特征 + N-shot 探针：完全复刻微调的采样协议，但冻结编码器
     —— 用来区分「表征本身不好」还是「全量微调的动力学把特征毁了」
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CKPTS = {
    "BYOL   ": "SaveModel/D1-Undefence/model_epoch_200.pth",
    "SimSiam": "SaveModel/D1-Undefence-SimSiam/model_epoch_200.pth",
}


def load_backbone(path, config, device):
    from models.resnet_base_network import ResNet18
    enc = ResNet18(**config['network'])
    lp = torch.load(path, map_location=device)
    enc.load_state_dict(lp['online_network_state_dict'])
    enc = nn.Sequential(*list(enc.children())[:-1])   # 与微调一致：去掉 projection
    return enc.to(device).eval()


@torch.no_grad()
def extract(enc, x, device, bs=256):
    outs = []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(np.asarray(x[i:i + bs])).float().to(device)
        xb = xb.reshape(xb.shape[0], 1, xb.shape[1], xb.shape[2])
        outs.append(enc(xb).view(xb.size(0), -1).cpu())
    return torch.cat(outs).numpy()


def collapse_stats(f):
    f = f.astype(np.float64)
    std = f.std(axis=0)
    fc = f - f.mean(axis=0, keepdims=True)
    cov = np.cov(fc, rowvar=False)
    lam = np.linalg.eigvalsh(cov)
    lam = np.clip(lam, 0, None)
    eff_rank = (lam.sum() ** 2) / (np.square(lam).sum() + 1e-12)
    return {
        "dim_std_mean": float(std.mean()),
        "dim_std_min": float(std.min()),
        "dim_std_frac_below_1e-2": float((std < 1e-2).mean()),
        "eff_rank": float(eff_rank),
        "eff_rank_ratio": float(eff_rank / f.shape[1]),
        "feat_norm_mean": float(np.linalg.norm(f, axis=1).mean()),
    }


def fit_linear(f_tr, y_tr, n_cls, device, epochs=300, lr=1e-3, wd=1e-4):
    """在冻结特征上训线性分类器（两个编码器用完全相同的协议）。"""
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
    lin.eval()
    return lin


@torch.no_grad()
def score(lin, f, y, device):
    pred = lin(torch.from_numpy(f).float().to(device)).argmax(1).cpu().numpy()
    return 100.0 * (pred == y).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d1-subsample", type=int, default=20000)
    ap.add_argument("--probe-epochs", type=int, default=300)
    args = ap.parse_args()

    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}\n")

    # ---- 数据 ----
    d1 = np.load(config["path"]["pretrain_data"] + "D1-Undefence-PreTrain.npy",
                 allow_pickle=True).item()
    x1, y1 = d1['dataset'], d1['label']
    rng = np.random.RandomState(0)
    sel = rng.choice(len(x1), min(args.d1_subsample, len(x1)), replace=False)
    x1, y1 = x1[sel], y1[sel]
    print(f"D1 源域: {x1.shape} classes={len(np.unique(y1))}")

    domains = {}
    for d in ["D2", "D3", "D4", "D5"]:
        z = np.load(config["path"]["finetune_data"] + f"{d}-Undefence-FineTune.npz")
        uv = np.unique(z['Y_tune_train'])
        mp = {v: i for i, v in enumerate(uv)}
        ytr = np.array([mp[v] for v in z['Y_tune_train']]).astype(np.int64)
        yte = np.array([mp[v] for v in z['Y_tune_test']]).astype(np.int64)
        domains[d] = dict(Xtr=z['X_tune_train'], Ytr=ytr, Xte=z['X_tune_test'], Yte=yte,
                          raw_tr=uv, raw_te=np.unique(z['Y_tune_test']))
        print(f"{d}: train={z['X_tune_train'].shape} test={z['X_tune_test'].shape} classes={len(uv)}")

    # 标签对齐检查（决定 zero-shot 是否可比）
    aligned = np.array_equal(np.unique(y1), domains["D2"]["raw_tr"])
    print(f"\nD1 与 D2 的类别 ID 集合是否一致（zero-shot 可比性）: {aligned}")

    y1_local = np.array([{v: i for i, v in enumerate(np.unique(y1))}[v] for v in y1]).astype(np.int64)
    n_cls = len(np.unique(y1))

    report = {}
    feats = {}
    for tag, path in CKPTS.items():
        if not os.path.exists(path):
            print(f"[skip] {path} 不存在")
            continue
        print(f"\n{'='*66}\n{tag}  {path}\n{'='*66}")
        enc = load_backbone(path, config, device)

        f1 = extract(enc, x1, device)
        feats[tag] = {"D1": f1}
        st = collapse_stats(f1)
        print("  [塌陷指标 @D1]")
        for k, v in st.items():
            print(f"    {k:26s} {v:.4f}")

        lin_src = fit_linear(f1, y1_local, n_cls, device, epochs=args.probe_epochs)
        src_acc = score(lin_src, f1, y1_local, device)
        print(f"  [源域 D1 线性可分] {src_acc:.2f}%  (训练集上，仅作上限参考)")

        for d in domains:
            fd = extract(enc, domains[d]["Xtr"], device)
            feats[tag][d] = fd
            feats[tag][d + "_te"] = extract(enc, domains[d]["Xte"], device)
            # 域间距离（相对域内散度）
            mu1, mud = f1.mean(0), fd.mean(0)
            within = 0.5 * (f1.std(0).mean() + fd.std(0).mean())
            shift = np.linalg.norm(mu1 - mud) / (np.linalg.norm(mu1) + 1e-12)
            print(f"  [D1->{d}] 域均值相对偏移 {shift:.4f}   "
                  f"组内/组间 {(np.linalg.norm(mu1-mud) / (within+1e-12)):.3f}")

        report[tag] = dict(stats=st, src_acc=src_acc)

    # ---- 冻结特征 + N-shot 探针（复刻微调采样协议）----
    print(f"\n{'='*66}\n冻结特征 + N-shot 探针（编码器不更新，只看表征）\n{'='*66}")
    N_shots, num_runs = [5, 10, 20], 5
    for tag in feats:
        print(f"\n{tag}")
        for d in domains:
            Xtr_all, Ytr_all = domains[d]["Xtr"], domains[d]["Ytr"]
            Yte = domains[d]["Yte"]
            ftr_all, fte = feats[tag][d], feats[tag][d + "_te"]
            line = []
            for N in N_shots:
                accs = []
                for run in range(num_runs):
                    # 采样索引由种子决定 —— 两个编码器拿到完全相同的 shot，排除采样噪声
                    rng = np.random.RandomState(1000 * N + run)
                    idx = []
                    for lab in np.unique(Ytr_all):
                        cand = np.where(Ytr_all == lab)[0]
                        idx.extend(rng.choice(cand, min(N, len(cand)), replace=False))
                    idx = np.array(idx)
                    lin = fit_linear(ftr_all[idx], Ytr_all[idx], len(np.unique(Ytr_all)),
                                     device, epochs=args.probe_epochs)
                    accs.append(score(lin, fte, Yte, device))
                line.append(f"N={N}: {np.mean(accs):6.2f}% (std {np.std(accs):4.2f})")
            print(f"  {d}  " + " | ".join(line))

    print("\nDONE")


if __name__ == "__main__":
    main()