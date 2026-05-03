# 数据准备与运行说明

本项目包含 MovieLens 数据转换，以及 UserCF、ItemCF、User2Emb、DSSM 等召回模型示例。

## 下载数据集并解压

windows
```bash
mkdir dataset

curl -L -o dataset/ml-latest-small.zip https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
curl -L -o dataset/ml-100k.zip https://files.grouplens.org/datasets/movielens/ml-100k.zip
curl -L -o dataset/ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip
curl -L -o dataset/ml-10m.zip https://files.grouplens.org/datasets/movielens/ml-10m.zip
curl -L -o dataset/ml-20m.zip https://files.grouplens.org/datasets/movielens/ml-20m.zip
curl -L -o dataset/ml-25m.zip https://files.grouplens.org/datasets/movielens/ml-25m.zip
```

linux
```bash
mkdir -p dataset && cd dataset

for name in ml-latest-small ml-100k ml-1m ml-10m ml-20m ml-25m
do
  curl -L -o ${name}.zip https://files.grouplens.org/datasets/movielens/${name}.zip \
  && unzip -q ${name}.zip \
  && rm ${name}.zip
done
```

## 处理任一数据集

```bash
python convert_dataset.py -d ml-1m
python convert_dataset.py
```

默认会输出统一格式数据到：

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

## 运行 ItemCF（基于内容）

```bash
python itemcf/item_content_based.py
python itemcf/evaluate.py
```

## 运行 UserCF（基于用户）

```bash
python usercf/user_cf.py
python usercf/evaluate.py
```

## 运行 DSSM

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
  --device cuda

python evaluate.py --model_path dssm_pointwise.pth --embed_dim 32 --device cuda

python train.py ^
  --mode pairwise ^
  --batch_size 256 ^
  --epochs 5 ^
  --lr 0.001 ^
  --embed_dim 32 ^
  --neg_ratio 3 ^
  --device cuda

python evaluate.py --model_path dssm_pairwise.pth --embed_dim 32 --device cuda
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
  --device cuda

python evaluate.py --model_path dssm_pointwise.pth --embed_dim 32 --device cuda

python train.py \
  --mode pairwise \
  --batch_size 4096 \
  --epochs 30 \
  --lr 0.001 \
  --embed_dim 64 \
  --margin 0.2 \
  --device cuda

python evaluate.py --model_path dssm_pairwise.pth --embed_dim 64 --device cuda
```

基于网格搜索最优参数训练 100 轮：

```bash
python train.py \
  --mode pointwise \
  --batch_size 512 \
  --epochs 100 \
  --lr 0.0001 \
  --embed_dim 128 \
  --neg_ratio 3 \
  --margin 0.2

python train.py \
  --mode pairwise \
  --batch_size 256 \
  --epochs 100 \
  --lr 0.001 \
  --embed_dim 128 \
  --neg_ratio 3 \
  --margin 0.2
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
python evaluate.py --model_path dssm_pairwise.pth --embed_dim 32 --device cuda
```

### 轮数测试

如果需要观察不同训练轮数对 DSSM 效果的影响，可以使用 `sweep_epochs.py` 依次训练并评估多个轮数。该脚本会对每个轮数完成一次训练和评估，并汇总 `Recall@K`、`NDCG@K`、`MRR@K`、`Coverage@K` 四类指标。

示例：

```bash
python sweep_epochs.py \
  --mode pairwise \
  --start 10 \
  --end 100 \
  --step 10 \
  --batch_size 32768 \
  --lr 0.001 \
  --embed_dim 64 \
  --margin 0.2 \
  --device cuda
```

参数含义：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--mode` | `pairwise` | 训练模式，可选 `pointwise` 或 `pairwise` |
| `--start` | `10` | 起始训练轮数 |
| `--end` | `100` | 结束训练轮数 |
| `--step` | `10` | 每次增加的训练轮数 |
| `--batch_size` | `32768` | 每个 batch 的样本数 |
| `--lr` | `0.001` | 学习率 |
| `--embed_dim` | `64` | embedding 维度，评估时会保持一致 |
| `--neg_ratio` | `3` | pointwise 模式下每个正样本对应的负样本数量 |
| `--margin` | `0.2` | pairwise 模式下的排序损失间隔 |
| `--device` | 自动选择 | 默认有 CUDA 用 `cuda`，否则用 `cpu` |
| `--results_root` | `dssm/epoch_sweep_results` | 轮数测试结果保存目录 |
| `--no_keep_models` | 关闭 | 不保存每个轮数训练出的模型副本 |

运行完成后，每次轮数测试会在结果目录下生成一个带时间戳的子目录，主要包含：

```text
epoch_metrics.csv
epoch_metrics.json
logs/
eval_json/
models/
```

其中，`epoch_metrics.csv` 是最常用的汇总文件，记录每个训练轮数对应的 `Recall@50/100/200`、`NDCG@50/100/200`、`MRR@50/100/200` 和 `Coverage@50/100/200`。

生成轮数测试曲线：

```bash
python plot_epoch_sweep.py
```

默认会读取最新一次轮数测试结果，并将图片保存到：

```text
dssm/output/epoch_sweep_plots/
```

如果需要指定某一次轮数测试结果：

```bash
python plot_epoch_sweep.py --result_dir epoch_sweep_results/20260426_205928
```

### 特征消融实验

如果需要验证不同特征对 DSSM 双塔模型的影响，可以运行特征消融实验。该实验会依次测试 6 组特征组合，并输出每组实验的 `Recall@50` 和 `NDCG@50`。

运行：

```bash
python feature_ablation.py \
  --mode pairwise \
  --epochs 30 \
  --batch_size 4096 \
  --lr 0.001 \
  --embed_dim 64 \
  --margin 0.2 \
  --device cuda
```

如果使用 pointwise 训练方式：

```bash
python feature_ablation.py \
  --mode pointwise \
  --epochs 30 \
  --batch_size 256 \
  --lr 0.001 \
  --embed_dim 32 \
  --neg_ratio 3 \
  --device cuda
```

实验结果会保存到：

```text
dssm/output/feature_ablation_时间戳/
```

其中 `feature_ablation_results.csv` 记录每组消融实验的特征组合、`Recall@50` 和 `NDCG@50`，`feature_ablation_results.json` 记录完整实验配置和最优结果。

### 结构消融实验

如果需要验证 DSSM 双塔模型中 MLP 层数对推荐效果的影响，可以运行结构消融实验。该实验保持输入特征不变，只改变用户塔和物品塔中的 MLP 结构，并输出每组实验的 `Recall@50` 和 `NDCG@50`。

实验组包括：

| 实验组 | MLP 结构 | 目的 |
| --- | --- | --- |
| 1 层 MLP | `64` | 验证浅层结构效果 |
| 2 层 MLP | `128, 64` | 验证中等深度效果 |
| 3 层 MLP | `256, 128, 64` | 当前完整结构 |
| 4 层 MLP | `512, 256, 128, 64` | 验证加深网络是否有效 |

运行：

```bash
python structure_ablation.py \
  --mode pairwise \
  --epochs 30 \
  --batch_size 4096 \
  --lr 0.001 \
  --embed_dim 64 \
  --margin 0.2 \
  --device cuda
```

如果使用 pointwise 训练方式：

```bash
python structure_ablation.py \
  --mode pointwise \
  --epochs 30 \
  --batch_size 256 \
  --lr 0.001 \
  --embed_dim 32 \
  --neg_ratio 3 \
  --device cuda
```

实验结果会保存到：

```text
dssm/output/structure_ablation_时间戳/
```

其中 `structure_ablation_results.csv` 记录每组结构消融实验的 MLP 结构、`Recall@50` 和 `NDCG@50`，`structure_ablation_results.json` 记录完整实验配置和最优结果。
