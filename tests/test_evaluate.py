import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import evaluate as evaluation


def data(prefix, labels=(1, 1, 1, 1, 0, 0), scores=(.9, .8, .7, .1, .6, .2)):
    return evaluation.PredictionData(
        np.array([f"{prefix}-{i // 2}" for i in range(len(labels))]),
        np.array(labels), np.array(scores),
    )


def test_patient_overlap_rejected_including_whitespace():
    val, test = data("shared"), data("shared")
    with pytest.raises(ValueError, match="subjects overlap"):
        evaluation.evaluate(val, test, bootstrap=10)
    whitespace = evaluation.PredictionData(
        np.array([f" {s} " for s in test.subjects]), test.labels, test.scores
    )
    with pytest.raises(ValueError, match="subjects overlap"):
        evaluation.evaluate(val, whitespace, bootstrap=10)


def test_threshold_uses_validation_only_and_handles_ties():
    val, test = data("v"), data("t")
    first = evaluation.evaluate(val, test, .75, bootstrap=20)
    changed_test = evaluation.PredictionData(test.subjects, 1 - test.labels, 1 - test.scores)
    second = evaluation.evaluate(val, changed_test, .75, bootstrap=20)
    assert first["threshold"] == second["threshold"] == .7
    assert first["validation"]["metrics"]["sensitivity"] == .75
    assert first["validation"]["metrics"]["specificity"] == 1
    assert first["test"]["metrics"] != second["test"]["metrics"]
    tied = data("tie", labels=(1, 1, 0, 0), scores=(.8, .5, .5, .1))
    assert evaluation.select_threshold(tied, 1) == .5
    for target in (0, -.1, 1.1, np.nan, np.inf):
        with pytest.raises(ValueError, match="target sensitivity"):
            evaluation.select_threshold(val, target)


def test_cluster_bootstrap_keeps_patient_windows_and_is_deterministic():
    test = evaluation.PredictionData(
        np.array(["patient-pos"] * 10 + ["patient-neg"] * 3),
        np.array([1] * 10 + [0] * 3),
        np.array([.8] * 10 + [.1] * 3),
    )
    first = evaluation.cluster_bootstrap(test, .5, repetitions=100, seed=42)
    second = evaluation.cluster_bootstrap(test, .5, repetitions=100, seed=42)
    assert first == second
    draws = np.random.default_rng(42).integers(0, 2, size=(100, 2))
    expected_valid = int(np.sum(draws[:, 0] != draws[:, 1]))
    assert first["valid_replicates"] == expected_valid
    assert first["skipped_single_class_replicates"] == 100 - expected_valid
    assert all(interval == [1, 1] for interval in first["intervals"].values())
    with pytest.raises(ValueError, match="positive integer"):
        evaluation.cluster_bootstrap(test, .5, 0)


@pytest.mark.parametrize("score", [np.nan, np.inf, -.001, 1.001])
def test_invalid_scores_are_rejected(score):
    valid = data("valid")
    invalid = evaluation.PredictionData(valid.subjects, valid.labels, np.full(len(valid.labels), score))
    with pytest.raises(ValueError, match="score"):
        evaluation.validate_data(invalid)


@pytest.mark.parametrize("contents", [
    "subject,label,score\na,0,.2\nb,1,\n",
    "subject,label,score\na,0,.2\n,1,.8\n",
    "subject,label,score\na,0,.2\nb,2,.8\n",
    "subject,label,score\na,0,.2\nb,1,.8,extra\n",
    "subject,label,score\na,0,.2\nb,0,.8\n",
    "subject,label,score\n",
    "label,subject,score\n0,a,.2\n1,b,.8\n",
])
def test_invalid_csv_is_rejected(tmp_path, contents):
    path = tmp_path / "invalid.csv"
    path.write_text(contents)
    with pytest.raises(ValueError):
        evaluation.read_predictions(path)


def save_csv(path, dataset):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject", "label", "score"])
        writer.writerows(zip(dataset.subjects, dataset.labels, dataset.scores))


def test_cli_report_is_aggregate_only_and_refuses_overwrite(tmp_path):
    val_path, test_path, output = tmp_path / "v.csv", tmp_path / "t.csv", tmp_path / "report"
    save_csv(val_path, data("PRIVATE_VALIDATION_ID"))
    save_csv(test_path, data("PRIVATE_TEST_ID"))
    command = [sys.executable, str(REPO / "evaluate.py"), "--validation", str(val_path),
               "--test", str(test_path), "--output", str(output), "--bootstrap", "25",
               "--target-sensitivity", ".75", "--data-kind", "synthetic"]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert {p.name for p in output.iterdir()} == {
        "metrics.json", "roc.png", "precision_recall.png", "confusion.png", "report.txt"
    }
    result = json.loads((output / "metrics.json").read_text())
    assert result["data_kind"] == "synthetic"
    assert result["threshold"] == .7
    assert result["test"]["counts"]["patients"] == 3
    assert "Synthetic demonstration" in (output / "report.txt").read_text()
    for path in (output / "metrics.json", output / "report.txt"):
        assert "PRIVATE_" not in path.read_text()
    original = {p.name: p.read_bytes() for p in output.iterdir()}
    rerun = subprocess.run(command, capture_output=True, text=True)
    assert rerun.returncode == 1 and "already exists" in rerun.stderr
    assert original == {p.name: p.read_bytes() for p in output.iterdir()}


def test_plot_failure_does_not_publish_partial_report(tmp_path, monkeypatch):
    import matplotlib.figure
    output = tmp_path / "should-not-exist"
    val, test = data("v"), data("t")
    result = evaluation.evaluate(val, test, .75, bootstrap=10, data_kind="synthetic")
    original_savefig = matplotlib.figure.Figure.savefig
    calls = 0

    def fail_second_plot(figure, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated disk failure")
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", fail_second_plot)
    with pytest.raises(OSError, match="simulated disk failure"):
        evaluation.write_report(result, test, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".should-not-exist.*"))


def test_publication_failure_removes_empty_destination(tmp_path, monkeypatch):
    output = tmp_path / "failed-publication"
    val, test = data("v"), data("t")
    result = evaluation.evaluate(val, test, .75, bootstrap=10, data_kind="synthetic")

    def fail_rename(*args):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(evaluation.os, "rename", fail_rename)
    with pytest.raises(OSError, match="simulated publication failure"):
        evaluation.write_report(result, test, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".failed-publication.*"))
