import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "epoch_sweep_results"
METRIC_NAMES = ["recall", "hr", "ndcg"]
KS = [50, 100, 200]
METRIC_DISPLAY_NAMES = {
    "recall": "Recall",
    "hr": "HR",
    "ndcg": "NDCG",
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train and evaluate DSSM with epoch values from start to end."
    )
    parser.add_argument("--mode", type=str, default="pairwise", choices=["pointwise", "pairwise"])
    parser.add_argument("--start", type=int, default=10)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--neg_ratio", type=int, default=3)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--results_root", type=str, default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument(
        "--no_keep_models",
        action="store_true",
        help="Do not copy each trained model into the sweep result directory.",
    )
    return parser


def metric_columns():
    return [f"{METRIC_DISPLAY_NAMES[name]}@{k}" for k in KS for name in METRIC_NAMES]


def run_command(command, log_path):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    output_lines = []

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(command) + "\n\n")
        process = subprocess.Popen(
            command,
            cwd=SCRIPT_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            output_lines.append(line)

        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"Command failed with exit code {return_code}: {' '.join(command)}"
            )

    return "".join(output_lines)


def parse_metrics_from_stdout(output):
    pattern = re.compile(r"(Recall|HR|NDCG)@(\d+):\s*([0-9.]+)")
    metrics = {}
    for metric_name, k, value in pattern.findall(output):
        metrics[f"{metric_name}@{k}"] = float(value)
    return metrics


def latest_eval_json(before_paths):
    output_dir = SCRIPT_DIR / "output"
    after_paths = set(output_dir.glob("experiment_dssm_*.json"))
    new_paths = list(after_paths - before_paths)
    candidates = new_paths or list(after_paths)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def flatten_metrics(metrics_by_k):
    row = {}
    for k in KS:
        values = metrics_by_k.get(str(k)) or metrics_by_k.get(k) or {}
        row[f"Recall@{k}"] = values.get("recall")
        row[f"HR@{k}"] = values.get("hr")
        row[f"NDCG@{k}"] = values.get("ndcg")
    return row


def append_csv(csv_path, row):
    fieldnames = ["epoch"] + metric_columns()
    file_exists = csv_path.exists()
    with open(csv_path, "a", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def write_summary_json(summary_path, config, records):
    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "records": records,
    }
    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def display_path(path):
    try:
        return str(path.relative_to(SCRIPT_DIR))
    except ValueError:
        return str(path)


def copy_model(args, epoch, model_dir):
    model_path = SCRIPT_DIR / f"dssm_{args.mode}.pth"
    if not model_path.exists():
        return None

    target_path = model_dir / f"dssm_{args.mode}_epochs_{epoch:03d}.pth"
    shutil.copy2(model_path, target_path)
    return display_path(target_path)


def main():
    args = build_parser().parse_args()
    if args.start <= 0 or args.end < args.start or args.step <= 0:
        raise ValueError("--start/--end/--step must define a valid positive range.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(args.results_root).resolve() / run_id
    log_dir = result_dir / "logs"
    model_dir = result_dir / "models"
    eval_json_dir = result_dir / "eval_json"
    for path in [log_dir, eval_json_dir]:
        path.mkdir(parents=True, exist_ok=True)
    if not args.no_keep_models:
        model_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = result_dir / "epoch_metrics.csv"
    summary_json = result_dir / "epoch_metrics.json"
    config = vars(args).copy()
    config["run_id"] = run_id
    config["result_dir"] = str(result_dir)
    records = []

    print(f"Epoch sweep results will be saved to: {result_dir}")
    for epoch in range(args.start, args.end + 1, args.step):
        print("=" * 80)
        print(f"Training and evaluating epochs={epoch}")
        print("=" * 80)

        train_command = [
            sys.executable,
            "train.py",
            "--mode",
            args.mode,
            "--batch_size",
            str(args.batch_size),
            "--epochs",
            str(epoch),
            "--lr",
            str(args.lr),
            "--embed_dim",
            str(args.embed_dim),
            "--neg_ratio",
            str(args.neg_ratio),
            "--margin",
            str(args.margin),
            "--device",
            args.device,
        ]
        run_command(train_command, log_dir / f"train_epochs_{epoch:03d}.log")

        model_path = SCRIPT_DIR / f"dssm_{args.mode}.pth"
        before_eval_json = set((SCRIPT_DIR / "output").glob("experiment_dssm_*.json"))
        eval_command = [
            sys.executable,
            "evaluate.py",
            "--model_path",
            str(model_path),
            "--embed_dim",
            str(args.embed_dim),
            "--device",
            args.device,
        ]
        eval_output = run_command(eval_command, log_dir / f"eval_epochs_{epoch:03d}.log")

        eval_json_path = latest_eval_json(before_eval_json)
        record = {"epoch": epoch}
        if eval_json_path:
            with open(eval_json_path, "r", encoding="utf-8") as file_obj:
                eval_payload = json.load(file_obj)
            record.update(flatten_metrics(eval_payload["metrics"]["by_k"]))
            copied_eval_json = eval_json_dir / f"eval_epochs_{epoch:03d}.json"
            shutil.copy2(eval_json_path, copied_eval_json)
            record["eval_json"] = display_path(copied_eval_json)
        else:
            record.update(parse_metrics_from_stdout(eval_output))

        if not args.no_keep_models:
            saved_model = copy_model(args, epoch, model_dir)
            if saved_model:
                record["model_path"] = saved_model

        records.append(record)
        append_csv(summary_csv, record)
        write_summary_json(summary_json, config, records)
        print(f"Saved epoch={epoch} metrics to {summary_csv}")

    print("=" * 80)
    print(f"Done. Summary CSV: {summary_csv}")
    print(f"Done. Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
