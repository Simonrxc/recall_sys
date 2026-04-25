# 数据准备与运行说明

## 下载数据集并解压
```bash
mkdir dataset

curl -L -o dataset/ml-latest-small.zip https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
curl -L -o dataset/ml-100k.zip https://files.grouplens.org/datasets/movielens/ml-100k.zip
curl -L -o dataset/ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip
curl -L -o dataset/ml-10m.zip https://files.grouplens.org/datasets/movielens/ml-10m.zip
curl -L -o dataset/ml-20m.zip https://files.grouplens.org/datasets/movielens/ml-20m.zip
curl -L -o dataset/ml-25m.zip https://files.grouplens.org/datasets/movielens/ml-25m.zip
```
## 处理任一数据集
```bash
python convert_dataset.py -d ml-1m
python convert_dataset.py
```

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

## 运行 dssm
```bash
python dssm/train.py
python dssm/evaluate.py
```