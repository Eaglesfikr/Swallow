import importlib
print("=== 关键包版本 ===")
for m in ["wandb", "google.protobuf", "tensorboard", "torch", "numpy", "yaml"]:
    try:
        mod = importlib.import_module(m)
        print(f"  OK   {m:20s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  FAIL {m:20s} {type(e).__name__}: {e}")

print("\n=== torch.utils.tensorboard (trainer.py 依赖) ===")
try:
    from torch.utils.tensorboard import SummaryWriter
    print("  OK  SummaryWriter 可导入")
except Exception as e:
    print(f"  FAIL {type(e).__name__}: {e}")

print("\n=== 实际建一个 SummaryWriter 看能不能写 ===")
try:
    import tempfile, os
    d = tempfile.mkdtemp(prefix="tb_check_")
    w = SummaryWriter(log_dir=d)
    w.add_scalar("x", 1.0, 0)
    w.close()
    print("  OK  写入成功:", os.listdir(d))
except Exception as e:
    print(f"  FAIL {type(e).__name__}: {e}")