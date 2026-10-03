import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import labram_preprocess, load_windows, patient_split
from demo import synthetic_dataset
from predict import preprocess
from train import fit, load_trained, scores, transform


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "synthetic.npz"
    synthetic_dataset(path, subjects=6, per_class=2)
    return path


def test_patient_split_is_disjoint_and_repeatable(dataset):
    windows = load_windows(dataset)
    a, b = patient_split(windows), patient_split(windows)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
        assert set(windows.y[a[name]]) == {0, 1}
    groups = [set(windows.subjects[a[name]]) for name in a]
    assert all(not left & right for i, left in enumerate(groups) for right in groups[i + 1:])
    assert sum(map(len, a.values())) == len(windows.x)


def test_dataset_rejects_wrong_units_or_nonfinite(dataset, tmp_path):
    with np.load(dataset) as source:
        arrays = {name: source[name] for name in source.files}
    arrays["units"] = "uV"
    invalid = tmp_path / "invalid.npz"
    np.savez(invalid, **arrays)
    with pytest.raises(ValueError, match="units"):
        load_windows(invalid)
    arrays["units"] = "V"
    arrays["X"][0, 0, 0] = np.nan
    np.savez(invalid, **arrays)
    with pytest.raises(ValueError, match="non-finite"):
        load_windows(invalid)


def test_labram_shape_and_voltage_conversion(dataset):
    windows = load_windows(dataset)
    a = labram_preprocess(windows.x[:2], windows.sfreq)
    b = labram_preprocess(windows.x[:2] * 2, windows.sfreq)
    assert a.shape == (2, 19, 800)
    np.testing.assert_allclose(b, a * 2, rtol=1e-5, atol=1e-6)


def test_training_checkpoint_roundtrip(dataset, tmp_path):
    output = tmp_path / "trained"
    metadata = fit(dataset, output, "resnet_scratch", epochs=1, batch_size=8, bootstrap=20)
    network, config = load_trained(output)
    windows = load_windows(dataset)
    x = preprocess(windows.x[:2], windows.sfreq)
    first = scores(network, x, 2)
    network2, _ = load_trained(output)
    np.testing.assert_array_equal(first, scores(network2, x, 2))
    assert config["data_kind"] == "synthetic" and config["pretraining_overlap"] == "not_applicable"
    assert 0 <= config["threshold"] <= 1
    assert (output / "evaluation/metrics.json").exists()
    with pytest.raises(ValueError, match="already exists"):
        fit(dataset, output, "resnet_scratch", epochs=1, bootstrap=10)


@pytest.mark.integration
def test_real_labram_weights_and_frozen_encoder_roundtrip(dataset, tmp_path):
    from train import build_network
    torch.set_num_threads(2)
    model = build_network("labram_linear")
    model.train()
    assert not model.encoder.training
    assert all(not p.requires_grad for p in model.encoder.parameters())
    windows = load_windows(dataset)
    x = torch.from_numpy(transform(windows.x[:2], windows.sfreq, "labram_linear"))
    logits = model(x)
    assert logits.shape == (2, 2) and torch.isfinite(logits).all()
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    assert model.head.weight.grad is not None
    assert all(p.grad is None for p in model.encoder.parameters())
    metadata = fit(dataset, tmp_path / "labram", "labram_linear", epochs=1, batch_size=8, bootstrap=10)
    loaded, _ = load_trained(tmp_path / "labram")
    assert loaded(x).shape == (2, 2)
    validation = patient_split(windows)["validation"]
    expected = list(csv.DictReader((tmp_path / "labram/validation.csv").open()))
    restored = scores(loaded, transform(windows.x[validation], windows.sfreq, "labram_linear"), 8)
    np.testing.assert_array_equal(restored, np.array([float(row["score"]) for row in expected], dtype=np.float32))
    assert metadata["pretraining_overlap"] == "unknown"


def test_prepare_uses_raw_volts_and_rejects_duplicate_aliases(dataset, tmp_path, monkeypatch):
    import mne
    from prepare import prepare
    from predict import CHANNELS
    windows = load_windows(dataset)
    raw = mne.io.RawArray(windows.x[0], mne.create_info(CHANNELS, 250, "eeg"), verbose="ERROR")
    source = tmp_path / "record.edf"
    source.touch()
    monkeypatch.setattr(mne.io, "read_raw_edf", lambda *a, **k: raw.copy())
    labels = tmp_path / "labels.csv"
    labels.write_text("recording,subject,start_seconds,label\nrecord.edf,person,0,1\n")
    output = tmp_path / "prepared.npz"
    assert prepare(labels, output, "synthetic")["accepted"] == 1
    with np.load(output) as data:
        np.testing.assert_array_equal(data["X"][0], windows.x[0])
        assert data["units"].item() == "V"
    labels.write_text("recording,subject,start_seconds,label\nrecord.edf,person,0,1\n./record.edf,person,0.0001,0\n")
    with pytest.raises(ValueError, match="Duplicate"):
        prepare(labels, tmp_path / "duplicate.npz")
    assert not (tmp_path / "duplicate.npz").exists()
