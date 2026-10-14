# 这是我的复现实验

## 过程
```

```

## 观察Swallow数据集
好像确实是低频的更有用

## 添加`ProcessData_D6D7.py`预处理文件
会产出 4 个文件，其中 D6-Undefence-FineTune.npz 因已存在(原版单独的swallow-dataset就有)会被跳过（反正内容和我们生成的完全一样）：
```
SaveData/PreTrain/D6-Undefence-PreTrain.npy    (10000, 2, 1000)
SaveData/PreTrain/D7-Undefence-PreTrain.npy    (10000, 2, 1000)
SaveData/FineTune/D6-Undefence-FineTune.npz    [跳过，已存在]
SaveData/FineTune/D7-Undefence-FineTune.npz    (5000 / 5000)
```