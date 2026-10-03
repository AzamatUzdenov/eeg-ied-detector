import csv
import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import compare as comparison


@pytest.fixture
def dataset(tmp_path):
    # train.fit is mocked: this file tests checksum orchestration, not NPZ parsing.
    path = tmp_path / "synthetic.npz"
    path.write_bytes(b"synthetic dataset placeholder")
    return path


def install_mock_fit(monkeypatch, dataset, ap=None, test_auc=None, change=None):
    calls = []
    source_hash = comparison.file_digest(dataset)
    ap = ap or {mode: .9 - index * .1 for index, mode in enumerate(comparison.MODES)}
    test_auc = test_auc or {mode: .5 for mode in comparison.MODES}

    def fit(**kwargs):
        calls.append(kwargs)
        mode, run = kwargs["mode"], Path(kwargs["output"])
        run.mkdir()
        (run / "evaluation").mkdir()
        # Local run files deliberately contain identifiers; aggregates must not copy them.
        (run / "test.csv").write_text("subject,label,score\nPRIVATE_PATIENT_ID,1,.8\n")
        (run / "model.safetensors").write_bytes(b"mock checkpoint")
        metrics = {name: .5 for name in comparison.METRICS}
        metrics["auroc"] = test_auc[mode]
        metrics["confusion"] = {"tn": 3, "fp": 1, "fn": 1, "tp": 3}
        result = {"data_kind": "synthetic", "threshold": .5, "test": {
            "counts": {"patients": 2, "windows": 8, "positive_windows": 4,
                       "negative_windows": 4, "private_ids": ["PRIVATE_PATIENT_ID"]},
            "metrics": {**metrics, "private_ids": ["PRIVATE_PATIENT_ID"]},
            "bootstrap": {"requested_replicates": kwargs["bootstrap"],
                          "valid_replicates": kwargs["bootstrap"],
                          "skipped_single_class_replicates": 0,
                          "intervals": {name: [.1, .9] for name in comparison.CI_METRICS}},
        }}
        metadata = {"mode": mode, "dataset_sha256": source_hash,
                    "split_sha256": "a" * 64, "data_kind": "synthetic",
                    "validation_ap": ap[mode], "threshold": .5,
                    "best_epoch": kwargs["epochs"], "private_ids": ["PRIVATE_PATIENT_ID"],
                    "pretraining_overlap": "not_applicable" if mode == "resnet_scratch" else "unknown"}
        history = [{"epoch": epoch, "loss": 1 / epoch,
                    "validation_ap": ap[mode] * epoch / kwargs["epochs"]}
                   for epoch in range(1, kwargs["epochs"] + 1)]
        (run / "history.json").write_text(json.dumps(history))
        (run / "evaluation/metrics.json").write_text(json.dumps(result))
        if change:
            change(mode, metadata, run)
        return metadata

    monkeypatch.setattr(comparison.train, "fit", fit)
    return calls


def test_selection_uses_validation_even_when_other_test_score_is_better(tmp_path, dataset, monkeypatch):
    calls = install_mock_fit(monkeypatch, dataset,
                            test_auc={"resnet_scratch": .1, "resnet_finetune": .99,
                                      "labram_linear": .95, "labram_finetune": .9})
    output = tmp_path / "comparison"
    summary = comparison.compare(dataset, output, epochs=2, bootstrap=10)
    assert summary["selected_model"] == "resnet_scratch"
    assert summary["validation_ranking"] == list(comparison.DEFAULT_MODELS)
    assert [call["mode"] for call in calls] == list(comparison.DEFAULT_MODELS)
    for call in calls:
        assert call["dataset"] == dataset
        assert call["epochs"] == 2 and call["bootstrap"] == 10
        assert call["seed"] == 42 and call["batch_size"] == 16
        assert call["learning_rate"] == 3e-4 and call["target_sensitivity"] == .8
        assert call["pretraining_overlap"] == "unknown"
    assert summary == json.loads((output / "summary.json").read_text())
    assert summary["dataset_sha256"] == comparison.file_digest(dataset)
    assert summary["split_sha256"] == "a" * 64
    assert (output / "learning_curves.png").stat().st_size > 1000
    for filename in ("summary.json", "comparison.csv"):
        assert "PRIVATE_PATIENT_ID" not in (output / filename).read_text()
    with (output / "comparison.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3 and all(row["data_kind"] == "synthetic" for row in rows)
    assert rows[1]["test_auroc"] == "0.99"


def test_tied_validation_ap_keeps_requested_order(tmp_path, dataset, monkeypatch):
    install_mock_fit(monkeypatch, dataset, ap={mode: .7 for mode in comparison.MODES})
    requested = ["labram_finetune", "resnet_scratch"]
    summary = comparison.compare(dataset, tmp_path / "tie", models=requested, epochs=2, bootstrap=2)
    assert summary["selected_model"] == requested[0]
    assert summary["validation_ranking"] == requested


@pytest.mark.parametrize("which", ["split", "dataset"])
def test_mismatched_split_or_dataset_is_rejected_and_removed(tmp_path, dataset, monkeypatch, which):
    def change(mode, metadata, run):
        if mode == "resnet_finetune":
            metadata[f"{which}_sha256"] = "b" * 64

    install_mock_fit(monkeypatch, dataset, change=change)
    output = tmp_path / "bad-checksum"
    with pytest.raises(ValueError, match="different patient splits|dataset checksum"):
        comparison.compare(dataset, output, epochs=2, bootstrap=2)
    assert not output.exists()
    assert not list(tmp_path.glob(".bad-checksum.*"))


def test_actual_dataset_mutation_is_detected(tmp_path, dataset, monkeypatch):
    def change(mode, metadata, run):
        dataset.write_bytes(b"changed during model fitting")

    install_mock_fit(monkeypatch, dataset, change=change)
    output = tmp_path / "mutated"
    with pytest.raises(ValueError, match="dataset changed"):
        comparison.compare(dataset, output, models=["resnet_scratch"], epochs=2, bootstrap=2)
    assert not output.exists()
    assert not list(tmp_path.glob(".mutated.*"))


def test_later_fit_failure_removes_all_completed_and_partial_runs(tmp_path, dataset, monkeypatch):
    def change(mode, metadata, run):
        if mode == "resnet_finetune":
            (run / "partial-checkpoint").write_bytes(b"partial")
            raise RuntimeError("simulated fit failure")

    calls = install_mock_fit(monkeypatch, dataset, change=change)
    output = tmp_path / "failed-fit"
    with pytest.raises(RuntimeError, match="simulated fit failure"):
        comparison.compare(dataset, output, epochs=2, bootstrap=2)
    assert len(calls) == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".failed-fit.*"))


def test_existing_output_is_preserved_without_training(tmp_path, dataset, monkeypatch):
    calls = install_mock_fit(monkeypatch, dataset)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "sentinel").write_bytes(b"keep me")
    with pytest.raises(FileExistsError, match="already exists"):
        comparison.compare(dataset, output, epochs=2, bootstrap=2)
    assert calls == []
    assert (output / "sentinel").read_bytes() == b"keep me"


@pytest.mark.parametrize("arguments", [
    {"models": []}, {"models": ["resnet_scratch", "resnet_scratch"]},
    {"models": ["unknown"]}, {"epochs": 0}, {"batch_size": 1},
    {"learning_rate": float("nan")}, {"seed": -1}, {"bootstrap": 0},
    {"target_sensitivity": 0},
])
def test_invalid_arguments_do_not_start_training(tmp_path, dataset, monkeypatch, arguments):
    calls = install_mock_fit(monkeypatch, dataset)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        comparison.compare(dataset, output, **arguments)
    assert calls == [] and not output.exists()


def test_cli_defaults_match_training_contract(tmp_path, dataset, monkeypatch, capsys):
    captured = []

    def compare(*args):
        captured.append(args)
        return {"selected_model": "resnet_scratch"}

    monkeypatch.setattr(comparison, "compare", compare)
    output = tmp_path / "cli"
    comparison.main([str(dataset), "--output", str(output)])
    assert captured == [(dataset, output, list(comparison.DEFAULT_MODELS), 20, 16,
                         3e-4, 42, .8, 500, "unknown")]
    assert "using validation AP only" in capsys.readouterr().out
