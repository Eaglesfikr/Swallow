"""扫几个微调配置，看 DINO-800 的差距在哪一档被补回来（只读评测，不动训练脚本）。

起因（D2 N=5，复现微调循环后量的）：
  BYOL-200  冻结探针 82.54 -> 微调后特征 84.53，微调最好 85.28   （微调有增益）
  DINO-800  冻结探针 80.86 -> 微调后特征 78.89，微调最好 79.13   （微调没增益，还略降）
  DINO-800 特征范数 333.6（BYOL 183.8），编码器梯度范数 3.76（BYOL 1.22）
            —— 同一套 lr=1e-4 对两边的「实际步长」根本不是一个量级

所以怀疑是微调超参口径不匹配，而不是预训练权重不好。这里对每个配置跑 3 个 run 的完整
微调循环（与 fineTuning-AE*.py 同 bs=16 / 同 40 轮 / 同 10 轮评估一次），比较：
  ft_best   —— 与原脚本口径一致的「5 次评估里最好的一次」
  fp_after  —— 微调后编码器再冻结做线性探针，看预训练结构还剩多少

跑法：
  .venv/bin/python _sweep_ft.py                       # D2 N=5
  .venv/bin/python _sweep_ft.py --domain D7 --n 5
  .venv/bin/python _sweep_ft.py --models BYOL-200=SaveModel/D1-Undefence/model_epoch_200.pth
"""

import argparse
import os
from types import SimpleNamespace

import numpy as np
import torch
import yaml

import _diag_ft_dynamics as dyn

CONFIGS = [
    ("base", {}),
    ("normalize", {"normalize": True}),
    ("normalize_ep100", {"normalize": True, "epochs": 100}),
    ("normalize_teacher", {"normalize": True, "which": "teacher"}),
    ("teacher", {"which": "teacher"}),
    ("bn_train", {"bn_mode": "train"}),
    ("enc_lr3e-5", {"encoder_lr": 3e-5}),
    ("head1e-3_enc1e-5", {"lr": 1e-3, "encoder_lr": 1e-5}),
    ("ep100", {"epochs": 100}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="D2")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--models", nargs="+", default=[
        "BYOL-200=SaveModel/D1-Undefence/model_epoch_200.pth",
        "DINO-200=SaveModel/D1-Undefence-DINO/model_epoch_200.pth",
        "DINO-800=SaveModel/D1-Undefence-DINO/model_epoch_800.pth",
    ])
    ap.add_argument("--only", default=None, help="只跑一个模型的配置扫描")
    ap.add_argument("--only-config", default=None,
                    help="只跑名字完全等于该值的配置（省时间），见文件顶部 CONFIGS")
    ap.add_argument("--which", choices=["student", "teacher"], default="student")
    args = ap.parse_args()

    config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    z = np.load(config["path"]["finetune_data"] + f"{args.domain}-Undefence-FineTune.npz")
    uv = np.unique(z['Y_tune_train'])
    mp = {v: i for i, v in enumerate(uv)}
    Xtr_all = z['X_tune_train']
    Ytr_all = np.array([mp[v] for v in z['Y_tune_train']]).astype(np.int64)
    Xte, Yte = z['X_tune_test'], np.array([mp[v] for v in z['Y_tune_test']]).astype(np.int64)
    n_cls = len(uv)

    print(f"domain={args.domain} N={args.n} runs={args.runs}\n", flush=True)
    for spec in args.models:
        tag, path = spec.split("=", 1)
        if args.only and tag != args.only:
            continue
        if not os.path.exists(path):
            print(f"[跳过] {tag}: 缺 {path}", flush=True)
            continue
        print(f"{'=' * 78}\n{tag}  ({path})\n{'=' * 78}", flush=True)
        print(f"  {'配置':>18s} {'ft_best':>8s} {'ft_last':>8s} {'fp_before':>10s} "
              f"{'fp_after':>9s} {'||dW||/||W||':>13s} {'grad':>8s}", flush=True)
        print("  " + "-" * 82, flush=True)
        for ctag, over in CONFIGS:
            if args.only_config and args.only_config != ctag:
                continue
            kw = dict(domain=args.domain, n=args.n, runs=args.runs, epochs=40, lr=1e-4,
                      encoder_lr=None, normalize=False, eval_every=10, bn_mode="orig",
                      which=args.which, wb_run=None)
            kw.update(over)
            a = SimpleNamespace(**kw)
            rows = []
            for r in range(args.runs):
                idx = dyn.sample_traces(Xtr_all, Ytr_all, args.n, 1000 * args.n + r)
                rows.append(dyn.run_one(path, config, device, Xtr_all[idx], Ytr_all[idx],
                                        Xte, Yte, n_cls, a, r))
            agg = {k: float(np.mean([x[k] for x in rows])) for k in rows[0]}
            print(f"  {ctag:>18s} {agg['best']:8.2f} {agg['last']:8.2f} "
                  f"{agg['fp_before']:10.2f} {agg['fp_after']:9.2f} "
                  f"{agg['drift']:13.4f} {agg['grad']:8.3f}", flush=True)
        print(flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()