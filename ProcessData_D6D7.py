# -*- coding: utf-8 -*-
"""
ProcessData_D6D7.py —— 处理 D6 / D7 Trace 数据集（Swallow 复现用）

与原 ProcessData.py 的差异
--------------------------
1. 一次特征提取，同时产出预训练集（*.npy）和微调集（*.npz），
   用 --mode 控制（both / pretrain / finetune）。原脚本要跑两遍。
2. 解析改用纯 Python 逐行读取，不依赖 pandas。
   原因：.venv 里没有装 pandas；而且 D7 是 CRLF 行尾、D6 有第三列，
   原脚本 str.slice(0, -1) + astype("float") 的写法没必要再保留。
3. 站点数 / 实例数沿用 config.yaml 的 monitored_site_num / monitored_inst_num
   （100 × 100），与原脚本口径完全一致。
4. 输出 dtype 对齐已有数据集：特征 / 标签 int32，load_time / slot_duration float64。

数据格式
--------
D6-Undefence : `时间 \t 方向 \t 0`   第 3 列恒为 0（全库 7306 万行中仅 4 行为 3），CIF 不使用
D7-Undefence : `时间 \t 方向`        CRLF 行尾
D1–D5        : `时间 \t 方向`
方向列取值 ±1：+1 记为上行、-1 记为下行，与 CIF.fun 中的约定一致。

用法
----
    python3 ProcessData_D6D7.py                    # 同时产出 PreTrain + FineTune
    python3 ProcessData_D6D7.py --mode finetune    # 只产出微调集
    python3 ProcessData_D6D7.py --overwrite        # 覆盖已存在的同名输出
    python3 ProcessData_D6D7.py --max-files 200    # 只处理前 200 个文件（快速自检）
    python3 ProcessData_D6D7.py --datasets D6-Undefence
"""

import argparse
import os
import sys
from multiprocessing import Pool
from os.path import abspath, dirname, isabs, join

import numpy as np
import yaml
from tqdm import tqdm

PROJECT_ROOT = dirname(abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from CIF import fun  # noqa: E402  自定义特征提取函数

# ---------------------------------------------------------------- 配置读取

with open(join(PROJECT_ROOT, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

WEBSITE_NUM = config['dataset_info']['monitored_site_num']   # 100，站点数
TRAIN_NUM = config['dataset_info']['monitored_inst_num']     # 100，每站点实例数
MAX_FINETUNE_NUM = 50                                        # 微调集：每站点前 50 条作训练

# 默认处理的数据集（Trace 目录名）
DEFAULT_DATASETS = ["D6-Undefence", "D7-Undefence"]


def _resolve(path):
    """把 config 里的相对路径按项目根目录解析，避免受当前工作目录影响。"""
    return path if isabs(path) else join(PROJECT_ROOT, path)


TRACE_PATH = _resolve(config["path"]["trace"])
FINE_TUNE_DATA_PATH = _resolve(config["path"]["finetune_data"])
PRETRAIN_DATA_PATH = _resolve(config["path"]["pretrain_data"])


# ---------------------------------------------------------------- 解析与特征提取

def parse_trace(file_path):
    """
    读取一个原始 trace 文件，返回 (相对时间, 方向序列)。

    兼容 2 列（D1–D5、D7）和 3 列（D6）两种格式；行尾 LF / CRLF 均可。
    等价于原脚本的：
        seq = pd.Series(tcp_dump).str.slice(0, -1).str.split("\\t", expand=True).astype("float")
        times = seq[:, 0] - seq[0, 0];  length_seq = seq[:, 1].astype("int")
    """
    times = []
    directions = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            # float() 会自行吃掉行尾残留的 \r，无需额外 strip
            times.append(float(parts[0]))
            directions.append(int(float(parts[1])))

    if len(times) < 2:
        return None, None

    times = np.asarray(times, dtype=np.float64)
    times -= times[0]
    directions = np.asarray(directions, dtype=np.int64)
    return times, directions


def process_single_file(file_name, folder_path):
    """提取单个 raw trace 文件的 CIF 特征。"""
    if '-' not in file_name:
        return None

    label, num = (int(x) for x in file_name.split('-'))
    if label >= WEBSITE_NUM or num >= TRAIN_NUM:
        return None

    times, directions = parse_trace(join(folder_path, file_name))
    if times is None:
        return None

    feature, load_time, slot_duration = fun(times, directions)

    return (feature, label, num, load_time, slot_duration)


def process_file_wrapper(args):
    """multiprocessing 的参数解包。"""
    file_name, folder_path = args
    return process_single_file(file_name, folder_path)


def parallel_process(file_list, folder_path, n_jobs=15):
    """多进程并行提取特征，保持与 file_list 相同的顺序。"""
    args_list = [(file_name, folder_path) for file_name in file_list]

    pool = Pool(n_jobs)
    try:
        results = list(tqdm(
            pool.imap(process_file_wrapper, args_list),
            total=len(file_list),
            desc="Processing files"
        ))
    finally:
        pool.close()
        pool.join()
    return results


# ---------------------------------------------------------------- 数据集处理

def _build_file_list(folder_path, max_files=None):
    """按 (站点, 实例) 顺序枚举存在的 trace 文件，与原脚本的枚举顺序一致。"""
    file_list = []
    for i in tqdm(range(WEBSITE_NUM), desc="Building file list"):
        for j in range(TRAIN_NUM):
            file_name = f"{i}-{j}"
            if os.path.exists(join(folder_path, file_name)):
                file_list.append(file_name)
    if max_files is not None:
        file_list = file_list[:max_files]
    return file_list


def _save_pretrain(name, features, labels, load_times, slot_durations, overwrite):
    save_path = join(PRETRAIN_DATA_PATH, f"{name}-PreTrain.npy")
    if os.path.exists(save_path) and not overwrite:
        print(f"  [跳过] 已存在（加 --overwrite 可覆盖）: {save_path}")
        return None
    os.makedirs(PRETRAIN_DATA_PATH, exist_ok=True)
    np.save(save_path, {
        'dataset': features,
        'label': labels,
        'load_time': load_times,
        'slot_duration': slot_durations
    })
    return save_path


def _save_finetune(name, features, labels, nums, overwrite):
    save_path = join(FINE_TUNE_DATA_PATH, f"{name}-FineTune.npz")
    if os.path.exists(save_path) and not overwrite:
        print(f"  [跳过] 已存在（加 --overwrite 可覆盖）: {save_path}")
        return None

    # 每站点前 MAX_FINETUNE_NUM 条作训练，其余作测试
    train_mask = nums < MAX_FINETUNE_NUM
    x_train = features[train_mask]
    y_train = labels[train_mask]
    x_test = features[~train_mask]
    y_test = labels[~train_mask]

    print(f"  训练样本: {len(x_train)}    测试样本: {len(x_test)}")

    os.makedirs(FINE_TUNE_DATA_PATH, exist_ok=True)
    np.savez(save_path,
             X_tune_train=x_train,
             Y_tune_train=y_train,
             X_tune_test=x_test,
             Y_tune_test=y_test)
    return save_path


def process_dataset(folder_path, name, mode="both", overwrite=False,
                    n_jobs=15, max_files=None):
    """处理单个数据集目录，按 mode 产出预训练集和 / 或微调集。"""
    print(f"\n{'=' * 60}")
    print(f"数据集     : {name}")
    print(f"输入路径   : {folder_path}")
    print(f"口径       : {WEBSITE_NUM} 站点 × {TRAIN_NUM} 实例")
    print(f"输出模式   : {mode}")

    if not os.path.isdir(folder_path):
        print(f"  [错误] 目录不存在，跳过：{folder_path}")
        print('=' * 60)
        return

    file_list = _build_file_list(folder_path, max_files=max_files)
    print(f"待处理文件 : {len(file_list)}")

    if not file_list:
        print("  [错误] 没有找到可处理的文件，跳过")
        print('=' * 60)
        return

    results = parallel_process(file_list, folder_path, n_jobs=n_jobs)

    features, labels, nums = [], [], []
    load_times, slot_durations = [], []

    skipped = 0
    for result in results:
        if result is None:
            skipped += 1
            continue
        feature, label, num, load_time, slot_duration = result
        features.append(feature)
        labels.append(label)
        nums.append(num)
        load_times.append(load_time)
        slot_durations.append(slot_duration)

    if not features:
        print("  [错误] 所有文件都解析失败，跳过")
        print('=' * 60)
        return

    # dtype 对齐已有数据集：特征 / 标签 int32，时间 float64
    features = np.asarray(features, dtype=np.int32)
    if features.ndim < 3:
        features = features[:, np.newaxis, :]
    labels = np.asarray(labels, dtype=np.int32)
    nums = np.asarray(nums, dtype=np.int64)
    load_times = np.asarray(load_times, dtype=np.float64)
    slot_durations = np.asarray(slot_durations, dtype=np.float64)

    print("\n  统计:")
    print(f"    特征      : {features.shape} {features.dtype}")
    print(f"    标签      : {labels.shape} {labels.dtype}，"
          f"取值 {labels.min()}..{labels.max()}（{len(np.unique(labels))} 个站点）")
    print(f"    每站点实例: {len(nums) // max(len(np.unique(labels)), 1)}")
    if skipped:
        print(f"    解析失败/被过滤: {skipped} 个")

    saved = []
    if mode in ("both", "pretrain"):
        p = _save_pretrain(name, features, labels, load_times, slot_durations, overwrite)
        if p:
            saved.append(p)
    if mode in ("both", "finetune"):
        p = _save_finetune(name, features, labels, nums, overwrite)
        if p:
            saved.append(p)

    print("\n  输出:")
    if saved:
        for p in saved:
            print(f"    -> {p}")
    else:
        print("    （无新文件写出）")
    print('=' * 60)


# ---------------------------------------------------------------- 入口

def main():
    parser = argparse.ArgumentParser(
        description="处理 D6 / D7 Trace 数据集，产出与 D1–D5 一致的 PreTrain / FineTune 文件")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        help=f"Trace 目录名，默认为 {' '.join(DEFAULT_DATASETS)}")
    parser.add_argument("--mode", choices=["both", "pretrain", "finetune"], default="both",
                        help="产出哪种数据集，默认 both")
    parser.add_argument("--jobs", type=int, default=15, help="并行进程数，默认 15")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已存在的输出文件（默认跳过并提示）")
    parser.add_argument("--max-files", type=int, default=None,
                        help="每个数据集最多处理多少个文件，仅用于快速自检")
    args = parser.parse_args()

    print("配置:")
    print(f"  Trace 目录   : {TRACE_PATH}")
    print(f"  PreTrain 输出: {PRETRAIN_DATA_PATH}")
    print(f"  FineTune 输出: {FINE_TUNE_DATA_PATH}")
    print(f"  切分         : 每站点前 {MAX_FINETUNE_NUM} 条训练 / 其余测试")
    print(f"  并行进程     : {args.jobs}")

    for name in args.datasets:
        process_dataset(
            folder_path=join(TRACE_PATH, name),
            name=name,
            mode=args.mode,
            overwrite=args.overwrite,
            n_jobs=args.jobs,
            max_files=args.max_files,
        )

    print("\n全部完成。")


if __name__ == '__main__':
    main()