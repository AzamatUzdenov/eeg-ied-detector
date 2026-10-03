"""Regression checks for malformed annotations, publication races and provenance."""

import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from demo import synthetic_dataset
import prepare
import train


class TinyClassifier(nn.Module):
    """A cheap substitute for architecture weights; exercise the real fit/report code."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(19, 2)

    def forward(self, x):
        return self.head(x.std(dim=-1))


@pytest.fixture
def tiny_training(tmp_path, monkeypatch):
    dataset = tmp_path / "synthetic.npz"
    synthetic_dataset(dataset, subjects=6, per_class=1)
    monkeypatch.setattr(train, "build_network", lambda mode: TinyClassifier())
    return dataset


def fit_tiny(dataset, output, **kwargs):
    return train.fit(dataset=dataset, output=output, mode="resnet_finetune",
                     epochs=1, batch_size=4, bootstrap=3, **kwargs)


@pytest.mark.parametrize("row", [
    "record.edf,synthetic-patient,0\n",
    "record.edf,synthetic-patient,0,1,extra-field\n",
])
def test_malformed_annotation_row_fails_before_eeg_read(tmp_path, monkeypatch, row):
    (tmp_path / "record.edf").touch()
    labels = tmp_path / "annotations.csv"
    labels.write_text("recording,subject,start_seconds,label\n" + row)
    output = tmp_path / "prepared.npz"
    calls = []

    def forbidden_read(*args, **kwargs):
        calls.append(args)
        raise AssertionError("Malformed annotations must not reach EEG reading")

    monkeypatch.setattr(prepare.mne.io, "read_raw_edf", forbidden_read)
    with pytest.raises(ValueError, match="Malformed CSV row 2"):
        prepare.prepare(labels, output, "synthetic")
    assert calls == [] and not output.exists()
    assert not list(tmp_path.glob("*.npz"))


def test_new_empty_output_is_preserved_even_with_a_stale_exists_check(tiny_training, tmp_path, monkeypatch):
    output = tmp_path / "foreign-output"
    original_save = train.save_file
    original_exists = Path.exists
    foreign_inode = []

    def create_foreign_output_at_final_checkpoint(*args, **kwargs):
        original_save(*args, **kwargs)
        output.mkdir()
        foreign_inode.append(output.stat().st_ino)

    # Simulate a stale check-then-rename result. Exclusive mkdir must consult the
    # filesystem itself and preserve the newly appeared directory.
    monkeypatch.setattr(Path, "exists", lambda path: False if path == output else original_exists(path))
    monkeypatch.setattr(train, "save_file", create_foreign_output_at_final_checkpoint)
    with pytest.raises(FileExistsError):
        fit_tiny(tiny_training, output)
    assert original_exists(output) and output.is_dir()
    assert output.stat().st_ino == foreign_inode[0]
    assert list(output.iterdir()) == []
    assert not list(tmp_path.glob(".foreign-output.*"))


def test_dataset_change_during_loading_stops_before_training(tiny_training, tmp_path, monkeypatch):
    output = tmp_path / "changed-load"
    original_load = train.load_windows
    built = []

    def load_then_change(path):
        windows = original_load(path)
        path.write_bytes(path.read_bytes() + b"changed during loading")
        return windows

    def forbidden_network(mode):
        built.append(mode)
        raise AssertionError("Changed data must fail before model construction")

    monkeypatch.setattr(train, "load_windows", load_then_change)
    monkeypatch.setattr(train, "build_network", forbidden_network)
    with pytest.raises(ValueError, match="Dataset changed while loading"):
        fit_tiny(tiny_training, output)
    assert built == [] and not output.exists()
    assert not list(tmp_path.glob(".changed-load.*"))


def test_dataset_change_during_training_does_not_publish(tiny_training, tmp_path, monkeypatch):
    output = tmp_path / "changed-training"
    original_scores = train.scores
    changed = []

    def score_then_change(model, x, batch_size):
        result = original_scores(model, x, batch_size)
        if not changed:
            tiny_training.write_bytes(tiny_training.read_bytes() + b"changed after first training epoch")
            changed.append(True)
        return result

    monkeypatch.setattr(train, "scores", score_then_change)
    with pytest.raises(ValueError, match="Dataset changed during training"):
        fit_tiny(tiny_training, output)
    assert changed and not output.exists()
    assert not list(tmp_path.glob(".changed-training.*"))


@pytest.mark.parametrize("overlap", ["unknown", "present", "excluded_by_user"])
def test_pretrained_overlap_disclosure_survives_aggregate_export(tiny_training, tmp_path, overlap):
    output = tmp_path / "overlap-report"
    metadata = fit_tiny(tiny_training, output, pretraining_overlap=overlap)
    metrics = json.loads((output / "evaluation/metrics.json").read_text())
    report = (output / "evaluation/report.txt").read_text()
    assert metadata["pretraining_overlap"] == metrics["pretraining_overlap"] == overlap
    assert f"Pretraining overlap: {overlap}" in report
    assert "not independently verified" in report
    assert "Patient separation is guaranteed only for the current fine-tuning split" in report
    assert metrics["data_kind"] == "synthetic"
