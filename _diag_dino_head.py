"""诊断：DINO 的 teacher 头是不是退化成「硬聚类」了（只读，不训练）。

背景：续训到 800 epoch 后 eff_rank 从 18 涨到 61.6（确实开了），但下游 D2/D6 反而更差。
怀疑点是 head 的「过度锐化」：teacher 温度 0.04 是按 K=65536 设计的，我们 K=128，
于是 teacher 输出几乎 one-hot（wandb 里 teacher_max_prob≈0.989、entropy≈0.032），
自蒸馏退化成「128 类的硬聚类」，student 只需要学原型编号，backbone 就把与 D1 类别
无关的细节全扔了 —— 这正好解释「域内 eff_rank 涨、跨域反而掉」。

本脚本对若干 checkpoint 量（都在 D1 上，与训练时同一口径）：
  teacher 熵 / max_prob / 实际用到的原型数
  原型与 D1 标签的对齐度（每个原型取多数标签的纯度、以及 NMI）
  student 熵、center_norm、backbone 有效秩

跑法：
  .venv/bin/python _diag_dino_head.py
  .venv/bin/python _diag_dino_head.py --epochs 200 400 600 800 --t-teacher 0.04
  .venv/bin/python _diag_dino_head.py --dir SaveModel/D1-Undefence-DINO-K1024 --epochs 100

K 从 checkpoint 里推断（读 projection 末层的 out_features），所以 K=128 和 K=1024 的权重
都能量，不用改 config.yaml。
"""

import argparse
import copy
import glob
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from models.resnet_base_network import ResNet18


def eff_rank(f):
    f = f.astype(np.float64)
    fc = f - f.mean(axis=0, keepdims=True)
    lam = np.clip(np.linalg.eigvalsh(np.cov(fc, rowvar=False)), 0, None)
    return float((lam.sum() ** 2) / (np.square(lam).sum() + 1e-12))


@torch.no_grad()
def forward_all(net, x, device, bs=512):
    """返回 (logits head 输出, backbone 特征)，都是 numpy。"""
    logits, feats = [], []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(np.asarray(x[i:i + bs])).float().to(device)
        xb = xb.reshape(xb.shape[0], 1, xb.shape[1], xb.shape[2])
        enc = net.encoder(xb)
        flat = enc.view(xb.size(0), -1)          # encoder 输出是 (B,512,1,1)，先拉平
        logits.append(net.projection(flat).cpu())
        feats.append(flat.cpu())
    return torch.cat(logits).numpy(), torch.cat(feats).numpy()


def purity_nmi(pred, label, n_proto):
    """原型编号 vs D1 标签：多数标签纯度 + 归一化互信息。"""
    n_lab = label.max() + 1
    cnt = np.zeros((n_proto, n_lab), dtype=np.int64)
    np.add.at(cnt, (pred, label), 1)
    used = cnt.sum(axis=1) > 0
    purity = cnt.max(axis=1)[used].sum() / max(1, cnt.sum())
    # NMI
    pxy = cnt / cnt.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(pxy * np.log(pxy / (px @ py) + 1e-12))
        hx = -np.nansum(px * np.log(px + 1e-12))
        hy = -np.nansum(py * np.log(py + 1e-12))
    nmi = mi / max(1e-12, np.sqrt(hx * hy))
    return float(purity), float(nmi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="SaveModel/D1-Undefence-DINO")
    ap.add_argument("--epochs", type=int, nargs="+", default=None,
                    help="要量的 epoch；不给就自动挑（目录最大 epoch <=200 时取 100 的倍数，"
                         "否则从 200 起每 100 一个）")
    ap.add_argument("--refs", nargs="*", default=[
        "SaveModel/D1-Undefence/model_epoch_200.pth",
        "SaveModel/D1-Undefence-SimSiam/model_epoch_200.pth"])
    ap.add_argument("--t-teacher", type=float, default=None,
                    help="覆盖 teacher 温度，默认用 checkpoint 里的 t_t_end（存的是 0.07）")
    ap.add_argument("--t-student", type=float, default=0.1)
    ap.add_argument("--n", type=int, default=10000, help="D1 上量多少个样本")
    args = ap.parse_args()

    config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    d1 = np.load(config["path"]["pretrain_data"] + "D1-Undefence-PreTrain.npy",
                 allow_pickle=True).item()
    x1, y1 = d1['dataset'], d1['label']
    rng = np.random.RandomState(0)
    sel = rng.choice(len(x1), min(args.n, len(x1)), replace=False)
    x1, y1 = x1[sel], np.asarray(y1)[sel]
    uv = np.unique(y1)
    y1 = np.searchsorted(uv, y1)
    print(f"D1 评测样本 {x1.shape}，{len(uv)} 类\n")

    ckpts = sorted(glob.glob(os.path.join(args.dir, "model_epoch_*.pth")),
                   key=lambda p: int(re.search(r"model_epoch_(\d+)", p).group(1)))
    max_ep = int(re.search(r"model_epoch_(\d+)", ckpts[-1]).group(1))
    if args.epochs:
        wanted = args.epochs
    elif max_ep <= 200:
        # 短跑（比如 --epochs 100 的预筛）只有 20/40/.../100，别按 200 起步挑空
        wanted = list(range(100, max_ep + 1, 100)) or [max_ep]
    else:
        wanted = list(range(200, max_ep + 1, 100))
    # tag 从目录名推：D1-Undefence-DINO -> DINO，D1-Undefence-DINO-K1024 -> DINO-K1024
    tag_base = os.path.basename(os.path.normpath(args.dir)).replace("D1-Undefence", "").strip("-") or "BYOL"
    todo = [(f"{tag_base}-{ep}", os.path.join(args.dir, f"model_epoch_{ep}.pth")) for ep in wanted]
    todo = [(t, p) for t, p in todo if os.path.exists(p)]
    refs = [(os.path.basename(os.path.dirname(p)), p) for p in args.refs if os.path.exists(p)]

    hdr = (f"  {'ckpt':>18s} {'K':>5s} {'t_ent':>7s} {'t_maxp':>7s} {'used':>5s} {'pur':>6s} "
           f"{'nmi':>6s} {'s_ent':>7s} {'|C|':>7s} {'rank(t)':>8s}")
    print("=" * 98)
    print(f"teacher 头退化诊断（D1，dir={args.dir}）")
    print("=" * 98)
    print(hdr)
    print("  " + "-" * 94)

    for tag, path in refs + todo:
        ck = torch.load(path, map_location=device)
        sd = ck.get("target_network_state_dict") or ck.get("online_network_state_dict")
        # K（原型数）从 checkpoint 推断，不跟 config.yaml 走，否则加载 K=1024 的权重会形状不匹配
        w_out = sd.get("projection.net.3.weight")
        k_out = int(w_out.shape[0]) if w_out is not None else config["network"]["projection_head"]["projection_size"]
        net_cfg = copy.deepcopy(config["network"])
        net_cfg["projection_head"]["projection_size"] = k_out
        net = ResNet18(**net_cfg).to(device).eval()
        net.load_state_dict(sd)
        logits, feats = forward_all(net, x1, device)

        if "center" not in ck:
            # BYOL / SimSiam 的 head 不是原型头（没有 center、也不做 sharpening），
            # 它们的熵/纯度列没有可比性，只报有效秩
            print(f"  {tag:>18s} {k_out:5d} {'n/a':>7s} {'n/a':>7s} {'-':>5s} {'-':>6s} "
                  f"{'-':>6s} {'-':>7s} {'-':>7s} {eff_rank(feats):8.2f}")
            del net, ck
            if device == "cuda":
                torch.cuda.empty_cache()
            continue

        center = ck["center"].cpu().numpy()
        t_t = args.t_teacher if args.t_teacher is not None else 0.07
        z = torch.from_numpy((logits - center) / t_t).float()
        pt = F.softmax(z, dim=-1).numpy()
        ps = F.softmax(torch.from_numpy(logits / args.t_student).float(), dim=-1).numpy()

        t_ent = float(-(pt * np.log(pt + 1e-8)).sum(1).mean())
        s_ent = float(-(ps * np.log(ps + 1e-8)).sum(1).mean())
        t_maxp = float(pt.max(1).mean())
        share = pt.mean(0)                      # 每个原型平均拿到的概率质量
        used = int((share > 1.0 / (2 * len(share))).sum())   # 拿到「均匀份额一半」以上的原型数
        pred = pt.argmax(1)
        pur, nmi = purity_nmi(pred, y1, logits.shape[1])

        print(f"  {tag:>18s} {k_out:5d} {t_ent:7.3f} {t_maxp:7.3f} {used:5d} {pur:6.3f} "
              f"{nmi:6.3f} {s_ent:7.3f} {np.linalg.norm(center):7.3f} {eff_rank(feats):8.2f}")
        del net, ck
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n判读：")
    print("  t_ent 接近 0 且 t_maxp 接近 1  -> teacher 几乎 one-hot，自蒸馏退化成硬聚类")
    print("  pur/nmi 很高（接近 D1 的 100 类）-> 原型学的就是 D1 的类别，跨域不迁移")
    print("  used 远小于 K                  -> 大量原型没被用上，head 容量其实在浪费")
    print("\nDONE")


if __name__ == "__main__":
    main()