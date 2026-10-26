"""诊断：为什么 SimSiam / DINO 的微调结果低于 BYOL（只读评测，不训练）。

DINO 和 SimSiam 都低于 BYOL，说明不是某个框架自己的毛病，得先分清差在哪一步：

  A. 预训练表征本身就更差   -> 冻结编码器 + N-shot 线性探针就该明显低于 BYOL
  B. 表征差不多，是端到端微调的动力学把特征弄坏了
                            -> 冻结探针接近，端到端差距大
  C. 预训练塌陷             -> backbone 逐维标准差极小 / 有效秩远低于 512

本脚本对三个编码器用完全相同的协议（同样的 shot 采样种子）测冻结特征上的 N-shot 准确率，
最后把「冻结探针」和「端到端微调」两套数字并排放出来看。

跑法：
  .venv/bin/python _diag_repr_compare.py
  .venv/bin/python _diag_repr_compare.py --probe-epochs 300 --runs 5
"""

import argparse
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CKPTS = {
    "BYOL": "SaveModel/D1-Undefence/model_epoch_200.pth",
    "SimSiam": "SaveModel/D1-Undefence-SimSiam/model_epoch_200.pth",
    "DINO": "SaveModel/D1-Undefence-DINO/model_epoch_200.pth",
}

# 端到端微调的结果文件（用于和冻结探针并排对比）
E2E_RESULTS = {
    "BYOL": "results/close/results-Scenario#1-AE-D2D6D7.txt",
    "DINO": "results/close/results-Scenario#1-AE-D2D6D7-DINO.txt",
}

DOMAINS = ["D2", "D6", "D7"]
N_SHOTS = [5, 10, 15, 20]


def load_backbone(path, config, device):
    """与微调脚本一致：加载 online_network_state_dict，然后丢掉 projection 头。"""
    from models.resnet_base_network import ResNet18
    enc = ResNet18(**config['network'])
    enc.load_state_dict(torch.load(path, map_location=device)['online_network_state_dict'])
    enc = nn.Sequential(*list(enc.children())[:-1])
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
    """塌陷指标：逐维标准差 + 有效秩（协方差特征值的熵指数）。"""
    f = f.astype(np.float64)
    std = f.std(axis=0)
    fc = f - f.mean(axis=0, keepdims=True)
    lam = np.clip(np.linalg.eigvalsh(np.cov(fc, rowvar=False)), 0, None)
    eff_rank = (lam.sum() ** 2) / (np.square(lam).sum() + 1e-12)
    return {
        "dim_std_mean": float(std.mean()),
        "dim_std_min": float(std.min()),
        "dim_std_below_1e-2": float((std < 1e-2).mean()),
        "eff_rank": float(eff_rank),
        "eff_rank_ratio": float(eff_rank / f.shape[1]),
    }


def fit_linear(f_tr, y_tr, n_cls, device, epochs, lr=1e-3, wd=1e-4):
    """在冻结特征上训线性分类器（所有编码器用同一协议）。"""
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
def score(lin, f, y, device):
    pred = lin(torch.from_numpy(f).float().to(device)).argmax(1).cpu().numpy()
    return 100.0 * (pred == y).mean()


def parse_e2e(path):
    """读端到端微调的结果文件 -> {(domain, N): avg}"""
    out = {}
    if not os.path.exists(path):
        return out
    text = open(path).read()
    for m in re.finditer(r"^(D1-\S+) - (\d+)\s*\n\s*avg -> ([\d.]+)", text, flags=re.M):
        full, n_str, avg = m.group(1), m.group(2), m.group(3)
        # 形如 D1-Undefence-DINO-D2-Undefence -> 目标域取 D2
        dm = re.search(r"-(D\d)-", full)
        if dm:
            out[(dm.group(1), int(n_str))] = float(avg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-epochs", type=int, default=300, help="线性探针的 epoch 数")
    ap.add_argument("--runs", type=int, default=5, help="每个 N 重复多少次（和微调一致）")
    ap.add_argument("--d1-subsample", type=int, default=20000, help="D1 取多少条算塌陷指标")
    args = ap.parse_args()

    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}\n")

    # ---- 数据 ----
    d1 = np.load(config["path"]["pretrain_data"] + "D1-Undefence-PreTrain.npy",
                 allow_pickle=True).item()
    x1 = d1['dataset']
    rng = np.random.RandomState(0)
    x1 = x1[rng.choice(len(x1), min(args.d1_subsample, len(x1)), replace=False)]

    domains = {}
    for d in DOMAINS:
        z = np.load(config["path"]["finetune_data"] + f"{d}-Undefence-FineTune.npz")
        uv = np.unique(z['Y_tune_train'])
        mp = {v: i for i, v in enumerate(uv)}
        domains[d] = dict(
            Xtr=z['X_tune_train'],
            Ytr=np.array([mp[v] for v in z['Y_tune_train']]).astype(np.int64),
            Xte=z['X_tune_test'],
            Yte=np.array([mp[v] for v in z['Y_tune_test']]).astype(np.int64),
            n_cls=len(uv),
        )
        print(f"{d}: train={z['X_tune_train'].shape} test={z['X_tune_test'].shape} classes={len(uv)}")

    # ---- 逐编码器：塌陷指标 + 冻结特征 N-shot 探针 ----
    frozen = {}
    for tag, path in CKPTS.items():
        if not os.path.exists(path):
            print(f"\n[skip] {path} 不存在")
            continue
        print(f"\n{'=' * 70}\n{tag}  {path}\n{'=' * 70}")
        enc = load_backbone(path, config, device)

        st = collapse_stats(extract(enc, x1, device))
        print("  [塌陷指标 @D1 backbone 512-d]")
        for k, v in st.items():
            print(f"    {k:22s} {v:.4f}")

        frozen[tag] = {}
        for d in domains:
            ftr = extract(enc, domains[d]["Xtr"], device)
            fte = extract(enc, domains[d]["Xte"], device)
            Ytr, Yte, n_cls = domains[d]["Ytr"], domains[d]["Yte"], domains[d]["n_cls"]
            for N in N_SHOTS:
                accs = []
                for run in range(args.runs):
                    # 采样种子固定 -> 三个编码器拿到完全相同的 shot，排除采样噪声
                    r = np.random.RandomState(1000 * N + run)
                    idx = []
                    for lab in np.unique(Ytr):
                        cand = np.where(Ytr == lab)[0]
                        idx.extend(r.choice(cand, min(N, len(cand)), replace=False))
                    idx = np.array(idx)
                    lin = fit_linear(ftr[idx], Ytr[idx], n_cls, device, args.probe_epochs)
                    accs.append(score(lin, fte, Yte, device))
                frozen[tag][(d, N)] = (float(np.mean(accs)), float(np.std(accs)))

    # ---- 并排对比：冻结探针 vs 端到端微调 ----
    e2e = {tag: parse_e2e(p) for tag, p in E2E_RESULTS.items()}

    print(f"\n{'=' * 70}\n冻结探针 vs 端到端微调（同一批目标域）\n{'=' * 70}")
    for d in DOMAINS:
        print(f"\n--- {d} ---")
        print(f"  {'method':10s} {'N=5':>16s} {'N=10':>16s} {'N=15':>16s} {'N=20':>16s}   (冻结 | 端到端)")
        for tag in CKPTS:
            if tag not in frozen:
                continue
            parts = []
            for N in N_SHOTS:
                fr = frozen[tag].get((d, N))
                e = e2e.get(tag, {}).get((d, N))
                fr_s = f"{fr[0]:5.2f}" if fr else "  -  "
                e_s = f"{e:5.2f}" if e is not None else "  -  "
                parts.append(f"{fr_s} | {e_s}".rjust(16))
            print(f"  {tag:10s} " + " ".join(parts))

    print("\n判读：若冻结探针三者也差这么多 -> 预训练表征问题（要动预训练超参/框架设置）")
    print("      若冻结探针接近、端到端拉开 -> 微调动力学问题（试 --freeze-encoder 口径）")
    print("\nDONE")


if __name__ == "__main__":
    main()