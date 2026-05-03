import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
DEFAULT_EMBED_DIMS = [32, 64, 128]
DEFAULT_LRS = [1e-3, 5e-4, 1e-4]
DEFAULT_BATCH_SIZES = [256, 512]
DEFAULT_MODES = ["pointwise", "pairwise"]
OBJECTIVE_NAME = "Recall@50"


def build_parser():
    parser = argparse.ArgumentParser(description="Grid search DSSM hyperparameters.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    return parser


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


def parse_recall_at_50(output):
    match = re.search(r"Recall@50:\s*([0-9.]+)", output)
    if not match:
        raise ValueError("Could not parse Recall@50 from evaluation output.")
    return float(match.group(1))


def write_csv(csv_path, records):
    fieldnames = [
        "mode",
        "epochs",
        "embed_dim",
        "lr",
        "batch_size",
        "objective",
        "model_path",
        "train_log",
        "eval_log",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_json(json_path, payload):
    with open(json_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def main():
    args = build_parser().parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).resolve()
    result_dir = output_dir / f"grid_search_{run_id}"
    log_dir = result_dir / "logs"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    records = []
    best_record = None

    print(f"Grid search results will be saved to: {result_dir}")
    for mode in DEFAULT_MODES:
        for embed_dim in DEFAULT_EMBED_DIMS:
            for lr in DEFAULT_LRS:
                for batch_size in DEFAULT_BATCH_SIZES:
                    run_name = (
                        f"{mode}_d{embed_dim}_lr{lr:g}_bs{batch_size}"
                    ).replace(".", "p")
                    print("=" * 80)
                    print(
                        "Running "
                        f"mode={mode}, epochs={args.epochs}, "
                        f"embed_dim={embed_dim}, lr={lr}, batch_size={batch_size}"
                    )
                    print("=" * 80)

                    train_command = [
                        sys.executable,
                        "train.py",
                        "--mode",
                        mode,
                        "--batch_size",
                        str(batch_size),
                        "--epochs",
                        str(args.epochs),
                        "--lr",
                        str(lr),
                        "--embed_dim",
                        str(embed_dim),
                        "--device",
                        args.device,
                    ]
                    train_log = log_dir / f"train_{run_name}.log"
                    run_command(train_command, train_log)

                    model_path = SCRIPT_DIR / f"dssm_{mode}.pth"
                    eval_command = [
                        sys.executable,
                        "evaluate.py",
                        "--model_path",
                        str(model_path),
                        "--embed_dim",
                        str(embed_dim),
                        "--device",
                        args.device,
                    ]
                    eval_log = log_dir / f"eval_{run_name}.log"
                    eval_output = run_command(eval_command, eval_log)
                    objective = parse_recall_at_50(eval_output)

                    record = {
                        "mode": mode,
                        "epochs": args.epochs,
                        "embed_dim": embed_dim,
                        "lr": lr,
                        "batch_size": batch_size,
                        "objective": objective,
                        "model_path": str(model_path),
                        "train_log": str(train_log),
                        "eval_log": str(eval_log),
                    }
                    records.append(record)

                    if best_record is None or objective > best_record["objective"]:
                        best_record = record
                        print(f"New best {OBJECTIVE_NAME}: {objective:.6f}")

                    write_csv(result_dir / "grid_search_results.csv", records)
                    write_json(
                        result_dir / "grid_search_summary.json",
                        {
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "objective_name": OBJECTIVE_NAME,
                            "search_space": {
                                "mode": DEFAULT_MODES,
                                "epochs": args.epochs,
                                "embed_dim": DEFAULT_EMBED_DIMS,
                                "lr": DEFAULT_LRS,
                                "batch_size": DEFAULT_BATCH_SIZES,
                            },
                            "best": best_record,
                            "records": records,
                        },
                    )

    print("=" * 80)
    print("Grid search finished.")
    print(f"Best {OBJECTIVE_NAME}: {best_record['objective']:.6f}")
    print(
        "Best params: "
        f"mode={best_record['mode']}, "
        f"epochs={best_record['epochs']}, "
        f"embed_dim={best_record['embed_dim']}, "
        f"lr={best_record['lr']}, "
        f"batch_size={best_record['batch_size']}"
    )
    print(f"Summary: {result_dir / 'grid_search_summary.json'}")
    print(f"CSV: {result_dir / 'grid_search_results.csv'}")


if __name__ == "__main__":
    main()
