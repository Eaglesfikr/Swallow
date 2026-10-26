"""列出所有本地 wandb run 的 summary，按微调任务汇总，用于对比 BYOL / DINO 的微调过程。"""
import glob
import json
import os

import yaml

ROOT = "wandb"
rows = []
for d in sorted(glob.glob(os.path.join(ROOT, "run-*"))):
    cfg, sm = {}, {}
    p = os.path.join(d, "files", "config.yaml")
    if os.path.exists(p):
        try:
            cfg = yaml.load(open(p), Loader=yaml.FullLoader) or {}
        except Exception:
            pass
    p = os.path.join(d, "files", "wandb-summary.json")
    if os.path.exists(p):
        try:
            sm = json.load(open(p))
        except Exception:
            pass

    def g(dct, k, default=None):
        v = dct.get(k, default)
        return v.get("value") if isinstance(v, dict) and "value" in v else v

    rows.append(dict(
        dir=os.path.basename(d),
        model=g(cfg, "model_name"),
        ckpt=str(g(cfg, "checkpoint") or ""),
        nshot=g(sm, "N_shot"),
        bs=g(cfg, "batch_size"),
        lr=g(cfg, "lr"),
        train_acc=g(sm, "train/accuracy"),
        train_loss=g(sm, "train/loss"),
        test_acc=g(sm, "test/accuracy"),
        best=sm.get("test/best_accuracy"),
        ep=sm.get("epoch"),
    ))


def fmt(v, spec, dash="-"):
    return format(v, spec) if isinstance(v, (int, float)) else dash


ft = [r for r in rows if r["model"]]
ft.sort(key=lambda r: (str(r["model"]), -(r["nshot"] or 0), r["ckpt"]))
print(f"共 {len(ft)} 个微调 run\n")
print(f"  {'model':>24s} {'ckpt(末8)':>13s} {'N':>4s} {'bs':>4s} {'lr':>8s} "
      f"{'tr_acc':>7s} {'tr_loss':>9s} {'te_acc':>7s} {'best':>7s} {'ep':>4s}  run")
for r in ft:
    print(f"  {str(r['model']):>24s} {r['ckpt'][-13:]:>13s} {str(r['nshot']):>4s} "
          f"{str(r['bs']):>4s} {fmt(r['lr'], '.0e'):>8s} {fmt(r['train_acc'], '.1f'):>7s} "
          f"{fmt(r['train_loss'], '.5f'):>9s} {fmt(r['test_acc'], '.2f'):>7s} "
          f"{fmt(r['best'], '.2f'):>7s} {str(r['ep']):>4s}  {r['dir']}")