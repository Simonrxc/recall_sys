#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert extracted MovieLens datasets into one stable format.

The script scans the dataset root, lets you choose one dataset folder, clears
the output directory, and writes normalized CSV files for UserCF, ItemCF,
user2emb, and DSSM experiments.
"""

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path


DEFAULT_DATASET_ROOT = "dataset"
DEFAULT_OUTPUT_DIR = "converted_dataset"
POSITIVE_RATING_THRESHOLD = 3.0

RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
MOVIE_COLUMNS = ["movie_id", "title", "genres"]
USER_COLUMNS = ["user_id", "gender", "age", "occupation", "zip_code"]
INTERACTION_COLUMNS = ["user_id", "movie_id", "rating", "timestamp", "label"]
SEQUENCE_COLUMNS = ["user_id", "movie_id_sequence"]

ML100K_GENRES = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def list_dataset_dirs(dataset_root):
    """Return extracted dataset directories under dataset_root."""
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"数据集根目录不存在: {root}")

    dirs = sorted([path for path in root.iterdir() if path.is_dir()], key=lambda path: path.name.lower())
    if not dirs:
        raise FileNotFoundError(f"{root} 下没有找到已解压的数据集文件夹")
    return dirs


def choose_dataset(dataset_dirs):
    """Ask the user to choose a dataset directory."""
    print("请选择要转换的 MovieLens 数据集:")
    for idx, path in enumerate(dataset_dirs, start=1):
        print(f"  {idx}. {path.name}")

    while True:
        choice = input("输入序号: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(dataset_dirs):
            return dataset_dirs[int(choice) - 1]
        print("输入无效，请重新输入。")


def resolve_dataset_dir(dataset_root, dataset_name):
    """Resolve a dataset by folder name or path."""
    candidate = Path(dataset_name)
    if candidate.exists() and candidate.is_dir():
        return candidate

    candidate = Path(dataset_root) / dataset_name
    if candidate.exists() and candidate.is_dir():
        return candidate

    available = ", ".join(path.name for path in list_dataset_dirs(dataset_root))
    raise FileNotFoundError(f"找不到数据集 {dataset_name}。可选数据集: {available}")


def read_delimited_file(path, delimiter, encoding="utf-8", skip_header=False):
    """Read a delimited text file as rows."""
    with open(path, "r", encoding=encoding, newline="") as file_obj:
        reader = csv.reader(file_obj, delimiter=delimiter)
        if skip_header:
            next(reader, None)
        return [row for row in reader if row]


def read_double_colon_file(path, encoding="latin-1"):
    """Read MovieLens .dat files whose separator is ::."""
    rows = []
    with open(path, "r", encoding=encoding) as file_obj:
        for line in file_obj:
            line = line.rstrip("\n")
            if line:
                rows.append(line.split("::"))
    return rows


def read_ratings(dataset_dir):
    """Read ratings from ratings.csv, ratings.dat, or u.data."""
    ratings_csv = dataset_dir / "ratings.csv"
    ratings_dat = dataset_dir / "ratings.dat"
    ratings_100k = dataset_dir / "u.data"

    if ratings_csv.exists():
        with open(ratings_csv, "r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            rows = [
                {
                    "user_id": int(row["userId"]),
                    "movie_id": int(row["movieId"]),
                    "rating": float(row["rating"]),
                    "timestamp": int(row["timestamp"]),
                }
                for row in reader
            ]
    elif ratings_dat.exists():
        rows = [
            {
                "user_id": int(row[0]),
                "movie_id": int(row[1]),
                "rating": float(row[2]),
                "timestamp": int(row[3]),
            }
            for row in read_double_colon_file(ratings_dat)
        ]
    elif ratings_100k.exists():
        rows = [
            {
                "user_id": int(row[0]),
                "movie_id": int(row[1]),
                "rating": float(row[2]),
                "timestamp": int(row[3]),
            }
            for row in read_delimited_file(ratings_100k, delimiter="\t", encoding="latin-1")
        ]
    else:
        raise FileNotFoundError(f"{dataset_dir} 中未找到 ratings.csv、ratings.dat 或 u.data")

    return sorted(rows, key=lambda row: (row["user_id"], row["timestamp"], row["movie_id"]))


def read_movies(dataset_dir):
    """Read movie metadata from movies.csv, movies.dat, or u.item."""
    movies_csv = dataset_dir / "movies.csv"
    movies_dat = dataset_dir / "movies.dat"
    movies_100k = dataset_dir / "u.item"

    if movies_csv.exists():
        with open(movies_csv, "r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            rows = [
                {
                    "movie_id": int(row["movieId"]),
                    "title": row.get("title", "") or "",
                    "genres": row.get("genres", "") or "(no genres listed)",
                }
                for row in reader
            ]
    elif movies_dat.exists():
        rows = [
            {
                "movie_id": int(row[0]),
                "title": row[1],
                "genres": row[2] if len(row) > 2 and row[2] else "(no genres listed)",
            }
            for row in read_double_colon_file(movies_dat)
        ]
    elif movies_100k.exists():
        rows = []
        for row in read_delimited_file(movies_100k, delimiter="|", encoding="latin-1"):
            genre_flags = row[5 : 5 + len(ML100K_GENRES)]
            genres = [genre for genre, enabled in zip(ML100K_GENRES, genre_flags) if int(enabled) == 1]
            rows.append(
                {
                    "movie_id": int(row[0]),
                    "title": row[1],
                    "genres": "|".join(genres) if genres else "unknown",
                }
            )
    else:
        raise FileNotFoundError(f"{dataset_dir} 中未找到 movies.csv、movies.dat 或 u.item")

    return sorted(rows, key=lambda row: row["movie_id"])


def read_users(dataset_dir, ratings):
    """Read user metadata when available, otherwise create placeholder rows."""
    users_dat = dataset_dir / "users.dat"
    users_100k = dataset_dir / "u.user"

    if users_dat.exists():
        rows = [
            {
                "user_id": int(row[0]),
                "gender": row[1] or "Unknown",
                "age": int(row[2]) if row[2] else 0,
                "occupation": row[3] or "Unknown",
                "zip_code": row[4] if len(row) > 4 else "",
            }
            for row in read_double_colon_file(users_dat)
        ]
    elif users_100k.exists():
        rows = [
            {
                "user_id": int(row[0]),
                "gender": row[2] or "Unknown",
                "age": int(row[1]) if row[1] else 0,
                "occupation": row[3] or "Unknown",
                "zip_code": row[4] if len(row) > 4 else "",
            }
            for row in read_delimited_file(users_100k, delimiter="|", encoding="latin-1")
        ]
    else:
        rows = [
            {
                "user_id": user_id,
                "gender": "Unknown",
                "age": 0,
                "occupation": "Unknown",
                "zip_code": "",
            }
            for user_id in sorted({row["user_id"] for row in ratings})
        ]

    return sorted(rows, key=lambda row: row["user_id"])


def build_interactions(ratings, positive_threshold):
    """Create implicit interaction labels from explicit ratings."""
    interactions = []
    for row in ratings:
        interactions.append(
            {
                "user_id": row["user_id"],
                "movie_id": row["movie_id"],
                "rating": row["rating"],
                "timestamp": row["timestamp"],
                "label": 1 if row["rating"] >= positive_threshold else 0,
            }
        )
    return interactions


def build_user_sequences(interactions):
    """Create per-user positive item sequences ordered by timestamp."""
    grouped = defaultdict(list)
    for row in interactions:
        if row["label"] == 1:
            grouped[row["user_id"]].append((row["timestamp"], row["movie_id"]))

    sequences = []
    for user_id in sorted(grouped):
        movie_ids = [str(movie_id) for _, movie_id in sorted(grouped[user_id])]
        sequences.append({"user_id": user_id, "movie_id_sequence": "|".join(movie_ids)})
    return sequences


def clear_output_dir(output_dir, dataset_root, dataset_dir):
    """Remove previous converted files and recreate the output directory."""
    output = Path(output_dir).resolve()
    root = Path(dataset_root).resolve()
    source = Path(dataset_dir).resolve()

    if output in {root, source}:
        raise ValueError("输出目录不能等于 dataset 根目录或当前源数据集目录，避免误删原始数据。")

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    return output


def write_csv(path, fieldnames, rows):
    """Write dictionaries to CSV with stable column order."""
    with open(path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def convert_dataset(dataset_root, dataset_dir, output_dir, positive_threshold):
    ratings = read_ratings(dataset_dir)
    movies = read_movies(dataset_dir)
    users = read_users(dataset_dir, ratings)
    interactions = build_interactions(ratings, positive_threshold)
    sequences = build_user_sequences(interactions)

    output = clear_output_dir(output_dir, dataset_root, dataset_dir)
    write_csv(output / "ratings.csv", RATING_COLUMNS, ratings)
    write_csv(output / "movies.csv", MOVIE_COLUMNS, movies)
    write_csv(output / "users.csv", USER_COLUMNS, users)
    write_csv(output / "interactions.csv", INTERACTION_COLUMNS, interactions)
    write_csv(output / "user_sequences.csv", SEQUENCE_COLUMNS, sequences)

    metadata = {
        "source_dataset": str(dataset_dir),
        "positive_rating_threshold": positive_threshold,
        "num_ratings": len(ratings),
        "num_positive_interactions": sum(row["label"] for row in interactions),
        "num_users": len(users),
        "num_movies": len(movies),
        "files": {
            "ratings": "ratings.csv",
            "movies": "movies.csv",
            "users": "users.csv",
            "interactions": "interactions.csv",
            "user_sequences": "user_sequences.csv",
        },
        "columns": {
            "ratings": RATING_COLUMNS,
            "movies": MOVIE_COLUMNS,
            "users": USER_COLUMNS,
            "interactions": INTERACTION_COLUMNS,
            "user_sequences": SEQUENCE_COLUMNS,
        },
    }
    with open(output / "metadata.json", "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, ensure_ascii=False, indent=2)

    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="把已解压的 MovieLens 数据集转换成统一 CSV 格式。")
    parser.add_argument(
        "-r",
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help=f"保存多个 MovieLens 文件夹的根目录，默认: {DEFAULT_DATASET_ROOT}",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        help="要转换的数据集文件夹名或路径；不传则进入交互式选择。",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"统一格式输出目录，默认: {DEFAULT_OUTPUT_DIR}。每次运行会自动清空该目录。",
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=POSITIVE_RATING_THRESHOLD,
        help=f"生成 interactions.csv 的正反馈阈值，默认: {POSITIVE_RATING_THRESHOLD}",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)

    if args.dataset:
        dataset_dir = resolve_dataset_dir(dataset_root, args.dataset)
    else:
        dataset_dir = choose_dataset(list_dataset_dirs(dataset_root))

    metadata = convert_dataset(dataset_root, dataset_dir, args.output_dir, args.positive_threshold)

    print("\n转换完成:")
    print(f"  源数据集: {metadata['source_dataset']}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  用户数: {metadata['num_users']:,}")
    print(f"  电影数: {metadata['num_movies']:,}")
    print(f"  评分数: {metadata['num_ratings']:,}")
    print(f"  正反馈数: {metadata['num_positive_interactions']:,}")
    print("  输出文件: ratings.csv, movies.csv, users.csv, interactions.csv, user_sequences.csv, metadata.json")


if __name__ == "__main__":
    main()
