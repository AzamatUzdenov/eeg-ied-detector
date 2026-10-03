"""Compare IED training modes with one dataset/split and validation-only selection."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

import train


MODES = tuple(train.MODES)
DEFAULT_MODELS = MODES[:3]
METRICS = ("auroc", "average_precision", "sensitivity", "specificity", "precision",
           "f1", "balanced_accuracy")
CI_METRICS = ("auroc", "average_precision", "sensitivity", "specificity", "f1")


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value, name, minimum=0.0, maximum=1.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside the expected range")
    return value


def aggregate_test(result):
    """Whitelist aggregate fields, excluding any unexpected metadata or identifiers."""
    test = result["test"]
    metrics = {name: finite_number(test["metrics"][name], name) for name in METRICS}
    confusion = {name: test["metrics"]["confusion"][name] for name in ("tn", "fp", "fn", "tp")}
    counts = {name: test["counts"][name] for name in
              ("patients", "windows", "positive_windows", "negative_windows")}
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0
           for v in (*confusion.values(), *counts.values())):
        raise ValueError("invalid aggregate test counts")
    if counts["windows"] != sum(confusion.values()) or counts["patients"] < 1:
        raise ValueError("aggregate test counts are inconsistent")
    metrics["confusion"] = confusion
    bootstrap = test["bootstrap"]
    intervals = {}
    for name in CI_METRICS:
        interval = bootstrap["intervals"][name]
        if interval is not None:
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError("invalid bootstrap confidence interval")
            interval = [finite_number(value, f"{name} CI") for value in interval]
            if interval[0] > interval[1]:
                raise ValueError("reversed bootstrap confidence interval")
        intervals[name] = interval
    replicate_counts = {name: bootstrap[name] for name in
                        ("requested_replicates", "valid_replicates", "skipped_single_class_replicates")}
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in replicate_counts.values()):
        raise ValueError("invalid bootstrap replicate counts")
    if replicate_counts["requested_replicates"] != (
        replicate_counts["valid_replicates"] + replicate_counts["skipped_single_class_replicates"]
    ):
        raise ValueError("inconsistent bootstrap replicate counts")
    return {"counts": counts, "metrics": metrics, "bootstrap": {
        "method": "patient-cluster percentile bootstrap", "confidence_level": 0.95,
        "threshold_fixed": True, **replicate_counts, "intervals": intervals,
    }}


def load_history(path, epochs):
    history = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(history, list) or len(history) != epochs:
        raise ValueError("training history does not match requested epochs")
    for index, item in enumerate(history, start=1):
        if item["epoch"] != index:
            raise ValueError("training history epochs are not consecutive")
        finite_number(item["loss"], "training loss", maximum=float("inf"))
        finite_number(item["validation_ap"], "history validation AP")
    return history


def plot_curves(histories, data_kind, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    try:
        for mode, history in histories.items():
            epochs = [item["epoch"] for item in history]
            axes[0].plot(epochs, [item["loss"] for item in history], marker=".", label=mode)
            axes[1].plot(epochs, [item["validation_ap"] for item in history], marker=".", label=mode)
        axes[0].set(xlabel="Epoch", ylabel="Training loss")
        axes[1].set(xlabel="Epoch", ylabel="Validation average precision", ylim=(0, 1.03))
        axes[1].legend(fontsize=8)
        for axis in axes:
            axis.grid(alpha=0.2)
        title = "Synthetic demonstration" if data_kind == "synthetic" else f"Data provenance: {data_kind}"
        figure.suptitle(title + "; model selection uses validation only")
        figure.tight_layout()
        figure.savefig(output, dpi=160)
    finally:
        plt.close(figure)


def compare(dataset, output, models=None, epochs=20, batch_size=16,
            learning_rate=3e-4, seed=42, target_sensitivity=0.8, bootstrap=500,
            pretraining_overlap="unknown"):
    output = Path(output)
    if os.path.lexists(output):
        raise FileExistsError("output directory already exists")
    modes = list(DEFAULT_MODELS if models is None else models)
    if not modes or len(modes) != len(set(modes)) or any(mode not in MODES for mode in modes):
        raise ValueError("models must be a nonempty list of distinct supported modes")
    if not isinstance(epochs, int) or epochs < 1 or not isinstance(batch_size, int) or batch_size < 2:
        raise ValueError("epochs must be >=1 and batch size >=2")
    if not isinstance(seed, int) or seed < 0 or not isinstance(bootstrap, int) or bootstrap < 1:
        raise ValueError("seed must be >=0 and bootstrap >=1")
    if not math.isfinite(learning_rate) or not 0 < learning_rate < 1:
        raise ValueError("learning rate must be in (0,1)")
    if not math.isfinite(target_sensitivity) or not 0 < target_sensitivity <= 1:
        raise ValueError("target sensitivity must be in (0,1]")
    if pretraining_overlap not in {"unknown", "present", "excluded_by_user"}:
        raise ValueError("invalid pretraining overlap status")
    dataset_hash = file_digest(dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    reserved = False
    try:
        rows, histories = [], {}
        common_split, data_kind = None, None
        for mode in modes:
            run = stage / mode
            metadata = train.fit(
                dataset=dataset, output=run, mode=mode, epochs=epochs,
                batch_size=batch_size, learning_rate=learning_rate, seed=seed,
                target_sensitivity=target_sensitivity, bootstrap=bootstrap,
                pretraining_overlap=pretraining_overlap,
            )
            if metadata["mode"] != mode or metadata["dataset_sha256"] != dataset_hash:
                raise ValueError("training mode or dataset checksum changed during comparison")
            split_hash = metadata["split_sha256"]
            if not isinstance(split_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", split_hash):
                raise ValueError("invalid split checksum")
            if common_split is not None and common_split != split_hash:
                raise ValueError("models used different patient splits")
            common_split = split_hash
            kind = metadata["data_kind"]
            if kind not in {"clinical", "synthetic", "unspecified"} or (data_kind is not None and kind != data_kind):
                raise ValueError("inconsistent data provenance")
            data_kind = kind
            validation_ap = finite_number(metadata["validation_ap"], "validation AP")
            threshold = finite_number(metadata["threshold"], "threshold")
            if not isinstance(metadata["best_epoch"], int) or not 1 <= metadata["best_epoch"] <= epochs:
                raise ValueError("invalid selected epoch")
            history = load_history(run / "history.json", epochs)
            if not math.isclose(validation_ap, max(item["validation_ap"] for item in history), abs_tol=1e-12):
                raise ValueError("selected validation AP disagrees with training history")
            histories[mode] = history
            evaluation = json.loads((run / "evaluation/metrics.json").read_text(encoding="utf-8"))
            if evaluation["data_kind"] != data_kind or not math.isclose(evaluation["threshold"], threshold, abs_tol=1e-12):
                raise ValueError("training and evaluation metadata disagree")
            rows.append({
                "mode": mode, "validation_ap": validation_ap,
                "best_epoch": metadata["best_epoch"], "threshold": threshold,
                "pretraining_overlap": metadata["pretraining_overlap"],
                "test": aggregate_test(evaluation),
            })
        if file_digest(dataset) != dataset_hash:
            raise ValueError("dataset changed during comparison")
        ranked = sorted(rows, key=lambda row: row["validation_ap"], reverse=True)
        selected = ranked[0]["mode"]
        limitations = [
            "Window-level comparison; it does not establish clinical validity or localize IED events.",
            "Modes use their own preprocessing; this compares complete pipelines, not architecture alone.",
            "Shared hyperparameters define a fixed-budget experiment, not optimal tuning of each architecture.",
            "Selection and epoch tuning reuse validation patients; test metrics never select a model.",
            "Test bootstrap intervals exclude epoch/model/threshold selection uncertainty.",
            "Pretraining patient overlap is user supplied and has not been independently verified.",
            "Local model folders contain identifiers and weights; publish only reviewed aggregate results.",
        ]
        if data_kind == "synthetic":
            limitations.insert(0, "Synthetic demonstration only; these metrics are not clinical performance.")
        if data_kind == "unspecified":
            limitations.insert(0, "Data provenance unspecified; clinical interpretation is unsupported.")
        summary = {
            "schema_version": 1, "data_kind": data_kind,
            "dataset_sha256": dataset_hash, "split_sha256": common_split,
            "selected_model": selected,
            "selection": "maximum validation_ap; input model order breaks ties; test is not used",
            "validation_ranking": [row["mode"] for row in ranked],
            "parameters": {"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                           "seed": seed, "target_sensitivity": target_sensitivity, "bootstrap": bootstrap},
            "models": rows, "limitations": limitations,
        }
        (stage / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        with (stage / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            columns = ["data_kind", "model", "validation_ap", "best_epoch", "threshold"] + [f"test_{name}" for name in METRICS]
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({"data_kind": data_kind, "model": row["mode"],
                                 "validation_ap": row["validation_ap"], "best_epoch": row["best_epoch"],
                                 "threshold": row["threshold"],
                                 **{f"test_{name}": row["test"]["metrics"][name] for name in METRICS}})
        plot_curves(histories, data_kind, stage / "learning_curves.png")
        output.mkdir()
        reserved = True
        os.rename(stage, output)
        reserved = False
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        if reserved:
            output.rmdir()
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--models", nargs="+", choices=MODES, default=list(DEFAULT_MODELS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-sensitivity", type=float, default=0.8)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--pretraining-overlap", choices=["unknown", "present", "excluded_by_user"], default="unknown")
    args = parser.parse_args(argv)
    try:
        result = compare(args.dataset, args.output, args.models, args.epochs, args.batch_size,
                         args.lr, args.seed, args.target_sensitivity, args.bootstrap, args.pretraining_overlap)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Saved {args.output}; selected {result['selected_model']} using validation AP only")


if __name__ == "__main__":
    main()
