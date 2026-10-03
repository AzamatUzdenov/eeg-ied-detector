"""Evaluate binary EEG window scores on patient-disjoint validation/test sets."""

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class PredictionData:
    subjects: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


def validate_data(data):
    subjects = np.asarray(data.subjects)
    labels = np.asarray(data.labels)
    scores = np.asarray(data.scores)
    if any(a.ndim != 1 for a in (subjects, labels, scores)):
        raise ValueError("subject, label and score must be one-dimensional")
    if not len(labels) or len(subjects) != len(labels) or len(scores) != len(labels):
        raise ValueError("subject, label and score must have equal nonzero lengths")
    if any(not isinstance(s, str) or not s.strip() for s in subjects):
        raise ValueError("subject must be a nonempty string")
    subjects = np.array([str(s).strip() for s in subjects])
    if labels.dtype.kind not in "biuf" or not np.isin(labels, [0, 1]).all():
        raise ValueError("label must be 0 or 1")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("each split must contain both classes")
    if scores.dtype.kind not in "iuf" or not np.isfinite(scores).all():
        raise ValueError("score must be finite")
    if (scores < 0).any() or (scores > 1).any():
        raise ValueError("score must be in [0, 1]")
    return PredictionData(subjects, labels.astype(np.int8), scores.astype(float))


def read_predictions(path):
    subjects, labels, scores = [], [], []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["subject", "label", "score"]:
            raise ValueError("CSV header must be exactly subject,label,score")
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"malformed CSV row {row_number}")
            subject = row["subject"].strip()
            label = row["label"].strip()
            if not subject or label not in {"0", "1"}:
                raise ValueError(f"invalid subject or binary label at row {row_number}")
            try:
                score = float(row["score"])
            except ValueError as exc:
                raise ValueError(f"invalid score at row {row_number}") from exc
            subjects.append(subject)
            labels.append(int(label))
            scores.append(score)
    return validate_data(PredictionData(np.array(subjects), np.array(labels), np.array(scores)))


def select_threshold(validation, target_sensitivity=0.8):
    """Highest threshold meeting sensitivity on validation; positive means score >= t."""
    validation = validate_data(validation)
    if not np.isfinite(target_sensitivity) or not 0 < target_sensitivity <= 1:
        raise ValueError("target sensitivity must be in (0, 1]")
    _, sensitivity, thresholds = roc_curve(
        validation.labels, validation.scores, drop_intermediate=False
    )
    eligible = (sensitivity >= target_sensitivity) & np.isfinite(thresholds)
    return float(np.max(thresholds[eligible]))


def point_metrics(labels, scores, threshold):
    tn, fp, fn, tp = confusion_matrix(labels, scores >= threshold, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn))
    specificity = float(tn / (tn + fp))
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    f1 = float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def cluster_bootstrap(data, threshold, repetitions=500, seed=42):
    """Resample complete patients, retaining all their windows and multiplicities."""
    data = validate_data(data)
    if not isinstance(repetitions, (int, np.integer)) or repetitions < 1:
        raise ValueError("bootstrap repetitions must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    patients, inverse = np.unique(data.subjects, return_inverse=True)
    groups = [np.flatnonzero(inverse == index) for index in range(len(patients))]
    names = ("auroc", "average_precision", "sensitivity", "specificity", "f1")
    samples = {name: [] for name in names}
    rng = np.random.default_rng(seed)
    skipped = 0
    for _ in range(repetitions):
        selected = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in selected])
        if len(np.unique(data.labels[indices])) != 2:
            skipped += 1
            continue
        metrics = point_metrics(data.labels[indices], data.scores[indices], threshold)
        for name in names:
            samples[name].append(metrics[name])
    intervals = {
        name: np.quantile(values, [0.025, 0.975]).tolist() if values else None
        for name, values in samples.items()
    }
    return {
        "method": "patient-cluster percentile bootstrap",
        "confidence_level": 0.95,
        "requested_replicates": int(repetitions),
        "valid_replicates": int(repetitions - skipped),
        "skipped_single_class_replicates": int(skipped),
        "threshold_fixed": True,
        "seed": int(seed),
        "intervals": intervals,
    }


def describe(data):
    return {
        "patients": int(len(np.unique(data.subjects))),
        "windows": int(len(data.labels)),
        "positive_windows": int(data.labels.sum()),
        "negative_windows": int((data.labels == 0).sum()),
    }


def evaluate(validation, test, target_sensitivity=0.8, bootstrap=500, seed=42,
             data_kind="unspecified"):
    validation, test = validate_data(validation), validate_data(test)
    if np.intersect1d(validation.subjects, test.subjects).size:
        raise ValueError("validation and test subjects overlap")
    if data_kind not in {"clinical", "synthetic", "unspecified"}:
        raise ValueError("data kind must be clinical, synthetic or unspecified")
    threshold = select_threshold(validation, target_sensitivity)
    warnings = [
        "Window-level metrics; bootstrap resamples patients, not independent windows.",
        "Bootstrap intervals condition on a fixed threshold and exclude its selection uncertainty.",
        "Scores must come from a model trained without validation/test patients; CSVs cannot verify this.",
        "This report does not establish diagnostic or clinical validity.",
    ]
    if data_kind == "synthetic":
        warnings.insert(0, "Synthetic demonstration only; metrics are not clinical performance.")
    elif data_kind == "unspecified":
        warnings.insert(0, "Data provenance unspecified; clinical interpretation is unsupported.")
    if len(np.unique(test.subjects)) < 2:
        warnings.append("Only one test patient; bootstrap intervals cannot estimate between-patient variability.")
    return {
        "schema_version": 1,
        "data_kind": data_kind,
        "positive_class": "label 1: annotated IED window",
        "prediction_rule": "score >= threshold",
        "threshold": threshold,
        "threshold_selection": {
            "split": "validation",
            "target_sensitivity": float(target_sensitivity),
            "rule": "highest threshold meeting target sensitivity (maximal specificity)",
        },
        "validation": {"counts": describe(validation),
                       "metrics": point_metrics(validation.labels, validation.scores, threshold)},
        "test": {"counts": describe(test),
                 "metrics": point_metrics(test.labels, test.scores, threshold),
                 "bootstrap": cluster_bootstrap(test, threshold, bootstrap, seed)},
        "limitations": warnings,
    }


def write_report(result, test, output):
    """Publish an aggregate-only report directory; refuse any existing destination."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    if os.path.lexists(output):
        raise FileExistsError("output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    reserved = False
    try:
        (staging / "metrics.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        title = "Synthetic demonstration" if result["data_kind"] == "synthetic" else "Held-out test windows"
        fpr, tpr, _ = roc_curve(test.labels, test.scores)
        precision, recall, _ = precision_recall_curve(test.labels, test.scores)
        metrics = result["test"]["metrics"]
        figure, axis = plt.subplots(figsize=(5, 4))
        axis.plot(fpr, tpr, label=f"AUROC {metrics['auroc']:.3f}")
        axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        axis.scatter(1 - metrics["specificity"], metrics["sensitivity"], label="Validation threshold")
        axis.set(xlabel="False positive rate", ylabel="Sensitivity", title=title,
                 xlim=(0, 1), ylim=(0, 1))
        axis.legend(loc="lower right")
        figure.tight_layout()
        figure.savefig(staging / "roc.png", dpi=160)
        plt.close(figure)
        figure, axis = plt.subplots(figsize=(5, 4))
        axis.step(recall, precision, where="post", label=f"Average precision {metrics['average_precision']:.3f}")
        axis.axhline(test.labels.mean(), linestyle="--", color="gray", linewidth=1, label="Prevalence")
        axis.set(xlabel="Recall (sensitivity)", ylabel="Precision", title=title,
                 xlim=(0, 1), ylim=(0, 1.02))
        axis.legend(loc="lower left")
        figure.tight_layout()
        figure.savefig(staging / "precision_recall.png", dpi=160)
        plt.close(figure)
        matrix = confusion_matrix(test.labels, test.scores >= result["threshold"], labels=[0, 1])
        figure, axis = plt.subplots(figsize=(4, 4))
        axis.imshow(matrix, cmap="Blues")
        for (row, col), value in np.ndenumerate(matrix):
            color = "white" if value > matrix.max() / 2 else "black"
            axis.text(col, row, str(value), ha="center", va="center", color=color)
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Non-IED", "IED"],
                 yticklabels=["Non-IED", "IED"], xlabel="Predicted", ylabel="Annotated", title=title)
        figure.tight_layout()
        figure.savefig(staging / "confusion.png", dpi=160)
        plt.close(figure)
        boot = result["test"]["bootstrap"]
        lines = [title, f"Data kind: {result['data_kind']}",
                 f"Threshold: {result['threshold']:.8g} (selected on validation only)",
                 f"Test: {result['test']['counts']['patients']} patients, {len(test.labels)} windows",
                 f"Bootstrap: {boot['valid_replicates']}/{boot['requested_replicates']} valid draws"]
        for name in ("auroc", "average_precision", "sensitivity", "specificity", "precision", "f1", "balanced_accuracy"):
            interval = boot["intervals"].get(name)
            suffix = f"; 95% CI [{interval[0]:.4f}, {interval[1]:.4f}]" if interval else ""
            lines.append(f"{name}: {metrics[name]:.4f}{suffix}")
        lines.extend(["", *result["limitations"]])
        (staging / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        # Reserve without overwriting, then atomically replace our own empty reservation.
        output.mkdir()
        reserved = True
        os.rename(staging, output)
        reserved = False
    finally:
        plt.close("all")
        shutil.rmtree(staging, ignore_errors=True)
        if reserved:
            output.rmdir()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-sensitivity", type=float, default=0.8)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-kind", choices=["clinical", "synthetic", "unspecified"], default="unspecified")
    args = parser.parse_args(argv)
    try:
        if os.path.lexists(args.output):
            raise FileExistsError("output directory already exists")
        validation, test = read_predictions(args.validation), read_predictions(args.test)
        result = evaluate(validation, test, args.target_sensitivity, args.bootstrap, args.seed, args.data_kind)
        write_report(result, test, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Saved aggregate metrics and plots to {args.output}; threshold {result['threshold']:.6g}")


if __name__ == "__main__":
    main()
