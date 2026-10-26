"""补充诊断：在 projection 空间（128 维）看塌陷 —— 也就是训练时 train/feature_std 监控的那个空间。"""
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

config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

d1 = np.load(config["path"]["pretrain_data"] + "D1-Undefence-PreTrain.npy", allow_pickle=True).item()
x1 = d1['dataset']
rng = np.random.RandomState(0)
x1 = x1[rng.choice(len(x1), min(10000, len(x1)), replace=False)]

from models.resnet_base_network import ResNet18


@torch.no_grad()
def extract(model, x, bs=256):
    outs = []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(np.asarray(x[i:i + bs])).float().to(device)
        xb = xb.reshape(xb.shape[0], 1, xb.shape[1], xb.shape[2])
        outs.append(model(xb).cpu())
    return torch.cat(outs).numpy()


def stats(f):
    f = f.astype(np.float64)
    std = f.std(axis=0)
    fc = f - f.mean(axis=0, keepdims=True)
    lam = np.clip(np.linalg.eigvalsh(np.cov(fc, rowvar=False)), 0, None)
    er = (lam.sum() ** 2) / (np.square(lam).sum() + 1e-12)
    return dict(feature_std=float(std.mean()),            # 训练时 wandb 里 log 的就是这个量
                dim_std_min=float(std.min()),
                eff_rank=float(er),
                eff_rank_ratio=float(er / f.shape[1]),    # 占维度数的比例
                top1_var_explained=float(lam[-1] / lam.sum()),
                top5_var_explained=float(lam[-5:].sum() / lam.sum()))


for tag, path in CKPTS.items():
    if not os.path.exists(path):
        continue
    m = ResNet18(**config['network']).to(device).eval()
    m.load_state_dict(torch.load(path, map_location=device)['online_network_state_dict'])
    z = extract(m, x1)
    s = stats(z)
    print(f"\n{tag}  projection 空间 ({z.shape[1]}-d)")
    for k, v in s.items():
        print(f"  {k:22s} {v:.4f}")
print("\nDONE")