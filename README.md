# Recall System

本项目包含 MovieLens 数据转换，以及 UserCF、ItemCF、User2Emb、DSSM 等召回模型示例。

## 数据准备

先将已解压的 MovieLens 数据集转换为统一格式：

```bash
python convert_dataset.py -o convert_dataset
```

转换后默认输出目录为：

```text
convert_dataset/
```

其中包含：

- `ratings.csv`
- `movies.csv`
- `users.csv`
- `interactions.csv`
- `user_sequences.csv`
- `metadata.json`

## DSSM 模型运行选项

DSSM 默认读取根目录下的 `convert_dataset/`。如果需要指定其他转换后目录，可以设置环境变量：

```bash
export DSSM_DATA_DIR=/path/to/convert_dataset
```

Windows CMD：

```cmd
set DSSM_DATA_DIR=D:\path\to\convert_dataset
```

### 训练

进入 `dssm` 目录：

```bash
cd dssm
```

默认 pointwise 训练：

```bash
python train.py
```

Windows CMD 常用可选项：

```cmd
python train.py ^
  --mode pointwise ^
  --batch_size 256 ^
  --epochs 5 ^
  --lr 0.001 ^
  --embed_dim 32 ^
  --neg_ratio 3 ^
  --device cpu
```

Linux/macOS 常用可选项：

```bash
python train.py \
  --mode pointwise \
  --batch_size 256 \
  --epochs 5 \
  --lr 0.001 \
  --embed_dim 32 \
  --neg_ratio 3 \
  --device cpu
```

训练参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--mode` | `pointwise` | 训练模式，可选 `pointwise` 或 `pairwise` |
| `--batch_size` | `256` | 每个 batch 的样本数 |
| `--epochs` | `5` | 训练轮数 |
| `--lr` | `0.001` | 学习率 |
| `--embed_dim` | `32` | 用户、物品、类型等 embedding 维度 |
| `--neg_ratio` | `3` | pointwise 模式下每个正样本对应的负样本数量 |
| `--margin` | `0.2` | pairwise 模式下 MarginRankingLoss 的 margin |
| `--device` | 自动选择 | 默认有 CUDA 用 `cuda`，否则用 `cpu` |

pairwise 训练示例：

```bash
python train.py --mode pairwise --epochs 5 --batch_size 256 --margin 0.2
```

训练完成后会在 `dssm/` 下保存模型：

```text
dssm_pointwise.pth
dssm_pairwise.pth
```

### 评估

评估默认读取：

```text
dssm_pointwise.pth
```

运行：

```bash
python evaluate.py
```

评估可选项：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model_path` | `dssm_pointwise.pth` | 要加载的模型权重路径 |
| `--embed_dim` | `32` | 必须与训练时的 `--embed_dim` 保持一致 |
| `--device` | 自动选择 | 默认有 CUDA 用 `cuda`，否则用 `cpu` |

指定 pairwise 模型评估：

```bash
python evaluate.py --model_path dssm_pairwise.pth --embed_dim 32 --device cpu
```
