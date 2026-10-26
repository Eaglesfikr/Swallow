#!/bin/bash
# bn_mode 修正为 orig（=完全复现原脚本：第 0 轮 BN 用目标域 batch 统计量并更新 running stats，
# 之后永久停在 eval）后，重新量 base / normalize 在 D2、D7 上的对照。
cd /mnt/d/workspace/Swallow
P=.venv/bin/python
BYOL=SaveModel/D1-Undefence/model_epoch_200.pth
D800=SaveModel/D1-Undefence-DINO/model_epoch_800.pth

run () {  # $1=domain $2=model_tag $3=model_path $4=config
  echo "########## $1  $2  cfg=$4"
  $P -u _sweep_ft.py --domain "$1" --n 5 --runs 3 \
      --models "$2=$3" --only "$2" --only-config "$4" 2>&1 | grep -v -i warn
}

for D in D2 D7; do
  run $D BYOL-200 $BYOL base
  run $D BYOL-200 $BYOL normalize
  run $D DINO-800 $D800 base
  run $D DINO-800 $D800 normalize
done
run D2 DINO-800 $D800 normalize_teacher
echo "ALL DONE"