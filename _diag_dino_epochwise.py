"""诊断：DINO 预训练「逐 epoch」演化 —— 分清是欠训练还是容量(K)不够（只读评测，不训练）。

起因：DINO 微调比 BYOL 低约 2 分，且冻结特征就已经低 2 分（表征问题，不是微调问题）。
本脚本看两件事：

  1. 有效秩 / 塌陷指标随 epoch 怎么走
     -> 200 epoch 还在涨 = 欠训练（加 K 没用，得延长训练）
     -> 早就平了       = 容量问题（head/原型维度 K 是瓶颈）

  2. student vs teacher 分开量
     DINO 官方评测用的是 teacher（EMA），而下游 fineTuning 加载的是
     online_network_state_dict = student。如果 teacher 明显更强，
     那「加载 student」这件事本身就是一部分差距来源。

跑法：
  .venv/bin/python _diag_dino_epochwise.py
  .venv/bin/python _diag_dino_epochwise.py --domains D2,D7 --runs 3
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DINO_DIR = "SaveModel/D1-Undefence-DINO"
REFS = {
    "BYOL-200": "SaveModel/D1-Undefence/model_epoch_200.pth",
    "SimSiam-200": "SaveModel/D1-Undefence-SimSiam/model_epoch_200.pth",
}


def load_backbones(path, config, device):
    """返回 (student, teacher) 两个 backbone（都去掉 projection 头，与微调一致）。"""
    from models.resnet_base_network import ResNet18
    ck = torch.load(path, map_location=device)

    def backbone(sd):
        enc = ResNet18(**config['network'])
        enc.load_state_dict(sd)
        return nn.Sequential(*list(enc.children())[:-1]).to(device).eval()

    student = backbone(ck['online_network_state_dict'])
    teacher = backbone(ck['target_network_state_dict']) if 'target_network_state_dict' in ck else None
    return student, teacher


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
    lam = np.clip(np.linalg.eigvalsh(np.cov(fc, rowvar=False)), 0, None)
    return {
        "eff_rank": float((lam.sum() ** 2) / (np.square(lam).sum() + 1e-12)),
        "dim_std_mean": float(std.mean()),
        "dim_std_min": float(std.min()),
    }


def fit_linear(f_tr, y_tr, n_cls, device, epochs, lr=1e-3, wd=1e-4):
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


def shot_index(Ytr, N, run):
    """固定种子采 N-shot，保证所有模型/所有 epoch 拿到完全相同的样本。"""
    r = np.random.RandomState(1000 * N + run)
    idx = []
    for lab in np.unique(Ytr):
        cand = np.where(Ytr == lab)[0]
        idx.extend(r.choice(cand, min(N, len(cand)), replace=False))
    return np.array(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="D2,D6,D7")
    ap.add_argument("--n-shots", type=int, nargs="+", default=[5, 20])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--probe-epochs", type=int, default=300)
    ap.add_argument("--d1-subsample", type=int, default=10000)
    ap.add_argument("--probe-every", type=int, default=2,
                    help="每隔几个 checkpoint 做一次冻结探针（默认 2，省时间；1=每个都做）")
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
    for d in [s.strip() for s in args.domains.split(",") if s.strip()]:
        z = np.load(config["path"]["finetune_data"] + f"{d}-Undefence-FineTune.npz")
        uv = np.unique(z['Y_tune_train'])
        mp = {v: i for i, v in enumerate(uv)}
        Xtr = z['X_tune_train']
        Ytr = np.array([mp[v] for v in z['Y_tune_train']]).astype(np.int64)
        # 预先算好所有 (N, run) 的采样索引，保证各 epoch/各模型完全一致
        idx_map = {(N, r): shot_index(Ytr, N, r) for N in args.n_shots for r in range(args.runs)}
        domains[d] = dict(Xtr=Xtr, Ytr=Ytr, Xte=z['X_tune_test'],
                          Yte=np.array([mp[v] for v in z['Y_tune_test']]).astype(np.int64),
                          n_cls=len(uv), idx_map=idx_map)
        print(f"{d}: train={Xtr.shape} test={z['X_tune_test'].shape} classes={len(uv)}")

    def evaluate(ckpt_path, tag, do_probe):
        """量一个 checkpoint：塌陷指标 + （可选）冻结 N-shot 探针。"""
        student, teacher = load_backbones(ckpt_path, config, device)
        row = {}
        for name, enc in [("student", student), ("teacher", teacher)]:
            if enc is None:
                continue
            st = collapse_stats(extract(enc, x1, device))
            row[name] = dict(stats=st, probe={})
            if do_probe:
                for d, dom in domains.items():
                    ftr = extract(enc, dom["Xtr"], device)
                    fte = extract(enc, dom["Xte"], device)
                    for N in args.n_shots:
                        accs = []
                        for r in range(args.runs):
                            idx = dom["idx_map"][(N, r)]
                            lin = fit_linear(ftr[idx], dom["Ytr"][idx], dom["n_cls"],
                                             device, args.probe_epochs)
                            accs.append(score(lin, fte, dom["Yte"], device))
                        row[name]["probe"][(d, N)] = float(np.mean(accs))
            del enc
        del student, teacher
        torch.cuda.empty_cache() if device == 'cuda' else None
        return row

    # ---- 参照：BYOL / SimSiam 在 200 epoch（同协议）----
    ref_rows = {}
    for tag, path in REFS.items():
        if os.path.exists(path):
            print(f"[ref] {tag}")
            ref_rows[tag] = evaluate(path, tag, do_probe=True)

    # ---- DINO 逐 epoch ----
    ckpts = sorted(glob.glob(os.path.join(DINO_DIR, 'model_epoch_*.pth')),
                   key=lambda p: int(re.search(r'model_epoch_(\d+)', p).group(1)))
    if not ckpts:
        print(f"没找到 {DINO_DIR}/model_epoch_*.pth")
        return

    dino_rows = {}
    for i, p in enumerate(ckpts):
        ep = int(re.search(r'model_epoch_(\d+)', p).group(1))
        do_probe = (i % args.probe_every == 0) or (i == len(ckpts) - 1)
        print(f"[dino] epoch {ep}{' (+探针)' if do_probe else ''}")
        dino_rows[ep] = evaluate(p, f"DINO-{ep}", do_probe)

    # ---- 输出 1：塌陷指标随 epoch ----
    print(f"\n{'=' * 78}\n有效秩 / 塌陷指标随 epoch（D1 backbone 512-d）\n{'=' * 78}")
    print(f"  {'epoch':>6s} {'eff_rank(tea)':>14s} {'eff_rank(stu)':>14s} "
          f"{'std_min(tea)':>13s} {'std_min(stu)':>13s}")
    for tag, row in ref_rows.items():
        t = row.get("teacher") or row.get("student")
        s = row.get("student", t)
        print(f"  {tag:>10s} {t['stats']['eff_rank']:14.2f} {s['stats']['eff_rank']:14.2f} "
              f"{t['stats']['dim_std_min']:13.5f} {s['stats']['dim_std_min']:13.5f}")
    print("  " + "-" * 74)
    for ep, row in dino_rows.items():
        t = row.get("teacher", row.get("student"))
        s = row.get("student")
        print(f"  {ep:6d} {t['stats']['eff_rank']:14.2f} {s['stats']['eff_rank']:14.2f} "
              f"{t['stats']['dim_std_min']:13.5f} {s['stats']['dim_std_min']:13.5f}")

    # ---- 输出 2：冻结探针随 epoch（student vs teacher）----
    print(f"\n{'=' * 78}\n冻结特征 N-shot 探针（student vs teacher）\n{'=' * 78}")
    for d in domains:
        print(f"\n--- {d} ---")
        header = "  ".join(f"N={N}" for N in args.n_shots)
        print(f"  {'epoch':>6s} | {'student ' + header:>26s} | {'teacher ' + header:>26s}")
        for tag, row in ref_rows.items():
            s, t = row.get("student"), row.get("teacher", row.get("student"))
            ss = "  ".join(f"{s['probe'].get((d, N), float('nan')):5.2f}" for N in args.n_shots)
            ts = "  ".join(f"{t['probe'].get((d, N), float('nan')):5.2f}" for N in args.n_shots)
            print(f"  {tag:>10s} | {ss:>26s} | {ts:>26s}")
        print("  " + "-" * 74)
        for ep, row in dino_rows.items():
            s, t = row.get("student"), row.get("teacher", row.get("student"))
            ss = "  ".join(f"{s['probe'][(d, N)]:5.2f}" if (d, N) in s['probe'] else "    -"
                           for N in args.n_shots)
            ts = "  ".join(f"{t['probe'][(d, N)]:5.2f}" if (d, N) in t['probe'] else "    -"
                           for N in args.n_shots)
            print(f"  {ep:6d} | {ss:>26s} | {ts:>26s}")

    print("\n判读：")
    print("  eff_rank 到 200 epoch 还在涨     -> 欠训练，延长 epoch 比改 K 更有用")
    print("  eff_rank 早早平在 20 上下        -> 容量瓶颈，K（原型维度）是主要嫌疑")
    print("  teacher 明显强于 student         -> 下游加载 student 本身就吃亏")
    print("\nDONE")


if __name__ == "__main__":
    main()