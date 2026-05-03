import argparse
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "epoch_sweep_results"
DEFAULT_PLOTS_ROOT = SCRIPT_DIR / "output" / "epoch_sweep_plots"
METRIC_GROUPS = {
    "recall": ["Recall@50", "Recall@100", "Recall@200"],
    "ndcg": ["NDCG@50", "NDCG@100", "NDCG@200"],
    "mrr": ["MRR@50", "MRR@100", "MRR@200"],
    "coverage": ["Coverage@50", "Coverage@100", "Coverage@200"],
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot DSSM epoch sweep metrics from epoch_metrics.csv."
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help="Directory created by sweep_epochs.py. Defaults to the latest run.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for generated images. Defaults to dssm/output/epoch_sweep_plots/<run_id>.",
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser


def latest_result_dir():
    if not DEFAULT_RESULTS_ROOT.exists():
        raise FileNotFoundError(
            f"No sweep result directory found under {DEFAULT_RESULTS_ROOT}."
        )

    candidates = [
        path
        for path in DEFAULT_RESULTS_ROOT.iterdir()
        if path.is_dir() and (path / "epoch_metrics.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No epoch_metrics.csv found under {DEFAULT_RESULTS_ROOT}."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_metrics(csv_path):
    with open(csv_path, "r", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        records = []
        for row in reader:
            record = {"epoch": int(row["epoch"])}
            for columns in METRIC_GROUPS.values():
                for column in columns:
                    record[column] = float(row[column]) if row.get(column) else None
            records.append(record)

    if not records:
        raise ValueError(f"No records found in {csv_path}.")
    return sorted(records, key=lambda item: item["epoch"])


def plot_lines(records, columns, title, output_path, dpi):
    import matplotlib.pyplot as plt

    epochs = [record["epoch"] for record in records]
    plt.figure(figsize=(10, 6))
    for column in columns:
        values = [record[column] for record in records]
        plt.plot(epochs, values, marker="o", linewidth=2, label=column)

    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.xticks(epochs)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_metric_groups(records, metric_groups, title, output_path, dpi):
    import matplotlib.pyplot as plt

    epochs = [record["epoch"] for record in records]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for axis, (metric_name, columns) in zip(axes, metric_groups.items()):
        for column in columns:
            values = [record[column] for record in records]
            axis.plot(epochs, values, marker="o", linewidth=2, label=column)

        axis.set_title(metric_name.upper())
        axis.set_xlabel("Epochs")
        axis.set_ylabel("Metric Value")
        axis.set_xticks(epochs)
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend()

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main():
    args = build_parser().parse_args()
    result_dir = Path(args.result_dir).resolve() if args.result_dir else latest_result_dir()
    csv_path = result_dir / "epoch_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {csv_path}")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else DEFAULT_PLOTS_ROOT / result_dir.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_metrics(csv_path)
    plot_metric_groups(
        records,
        METRIC_GROUPS,
        "DSSM Epoch Sweep - All Metrics",
        output_dir / "all_metrics.png",
        args.dpi,
    )
    for metric_name, columns in METRIC_GROUPS.items():
        plot_lines(
            records,
            columns,
            f"DSSM Epoch Sweep - {metric_name.upper()}",
            output_dir / f"{metric_name}.png",
            args.dpi,
        )

    print(f"Read metrics from: {csv_path}")
    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
