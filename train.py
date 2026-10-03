"""Train an IED classifier with patient-separated model/threshold selection."""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data import labram_preprocess, load_windows, patient_split, split_digest
from evaluate import evaluate, read_predictions, write_report
from model import EEGResNetAttention
from predict import CHANNELS, CONFIG, load_model, preprocess

LABRAM_REPO = "braindecode/labram-pretrained"
LABRAM_REVISION = "0563b6c626e7b40d9a36653b763715db94d945d7"
MODES = ["resnet_scratch", "resnet_finetune", "labram_linear", "labram_finetune"]


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LaBraMClassifier(nn.Module):
    def __init__(self, encoder, frozen=False):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(200, 2)
        self.frozen = frozen
        if frozen:
            self.encoder.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, x):
        features = self.encoder(x, ch_names=CHANNELS)
        if features.ndim != 2 or features.shape[1] != 200:
            raise ValueError("Unexpected LaBraM embedding shape.")
        return self.head(features)


def build_network(mode, pretrained=True, encoder_config=None):
    if mode not in MODES:
        raise ValueError("Unknown model mode.")
    if mode.startswith("resnet_"):
        return load_model() if mode == "resnet_finetune" and pretrained else EEGResNetAttention(
            num_channels=19, num_classes=2, input_length=1000)
    from braindecode.models import Labram
    from huggingface_hub import hf_hub_download
    if encoder_config is None:
        config_path = hf_hub_download(LABRAM_REPO, "config.json", revision=LABRAM_REVISION)
        encoder_config = json.loads(Path(config_path).read_text())
    encoder_config = dict(encoder_config)
    encoder_config.pop("braindecode_version", None)
    encoder = Labram(**encoder_config)
    if pretrained:
        weights = hf_hub_download(LABRAM_REPO, "model.safetensors", revision=LABRAM_REVISION)
        encoder.load_state_dict(load_file(weights, device="cpu"), strict=True)
        # Load the complete original state first, then explicitly adapt 15s -> 4s.
        # Keep the learned embeddings for the first four one-second patches.
        encoder.temporal_embedding = nn.Parameter(encoder.temporal_embedding[:, :5].detach().clone())
        encoder.patch_embed[0].n_times = 800
        encoder.patch_embed[0].n_patchs = 4
        encoder.n_path = 4
        encoder_config["n_times"] = 800
    network = LaBraMClassifier(encoder, frozen=mode == "labram_linear")
    network.encoder_config = encoder_config
    return network


def load_trained(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    weights = directory / "model.safetensors"
    if file_digest(weights) != config["checkpoint_sha256"]:
        raise ValueError("Trained checkpoint checksum mismatch.")
    network = build_network(config["mode"], pretrained=False, encoder_config=config.get("encoder_config"))
    network.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    return network.eval(), config


def transform(x, sfreq, mode):
    function = labram_preprocess if mode.startswith("labram_") else preprocess
    return np.concatenate([function(x[i:i + 64], sfreq) for i in range(0, len(x), 64)])


@torch.inference_mode()
def scores(model, x, batch_size):
    model.eval()
    return np.concatenate([model(torch.from_numpy(x[i:i + batch_size])).softmax(1)[:, 1].numpy()
                           for i in range(0, len(x), batch_size)])


def write_scores(path, subjects, y, score):
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject", "label", "score"])
        writer.writerows(zip(subjects, y.tolist(), score.tolist()))


def fit(dataset, output, mode="resnet_finetune", epochs=20, batch_size=16,
        learning_rate=3e-4, seed=42, target_sensitivity=0.8, bootstrap=500,
        pretraining_overlap="unknown"):
    output = Path(output)
    if os.path.lexists(output):
        raise ValueError("Output already exists.")
    if mode not in MODES or epochs < 1 or batch_size < 2 or not 0 < learning_rate < 1:
        raise ValueError("Invalid training arguments (batch_size must be >=2).")
    if not 0 < target_sensitivity <= 1 or bootstrap < 1:
        raise ValueError("Sensitivity target must be in (0,1], bootstrap >=1.")
    if pretraining_overlap not in {"unknown", "present", "excluded_by_user"}:
        raise ValueError("Invalid pretraining overlap status.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.use_deterministic_algorithms(True)
    dataset_hash = file_digest(dataset)
    windows = load_windows(dataset)
    if file_digest(dataset) != dataset_hash:
        raise ValueError("Dataset changed while loading.")
    split = patient_split(windows, seed)
    x = transform(windows.x, windows.sfreq, mode)
    model = build_network(mode)
    training = TensorDataset(torch.from_numpy(x[split["train"]]), torch.from_numpy(windows.y[split["train"]]))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(training, batch_size=batch_size, shuffle=True, generator=generator)
    positive = int(windows.y[split["train"]].sum())
    negative = len(training) - positive
    weights = torch.tensor([len(training) / (2 * negative), len(training) / (2 * positive)])
    loss_function = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    best_ap, best_state, best_epoch, history = -1.0, None, 0, []
    validation_index = split["validation"]
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features), labels)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_scores = scores(model, x[validation_index], batch_size)
        validation_ap = float(average_precision_score(windows.y[validation_index], validation_scores))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation_ap": validation_ap})
        print(f"{mode}: epoch {epoch}/{epochs}, validation AP {validation_ap:.4f}", flush=True)
        if validation_ap > best_ap:
            best_ap, best_epoch = validation_ap, epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    model.load_state_dict(best_state, strict=True)
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    reserved = False
    try:
        for name in ["validation", "test"]:
            index = split[name]
            write_scores(stage / f"{name}.csv", windows.subjects[index], windows.y[index],
                         scores(model, x[index], batch_size))
        evaluation = evaluate(read_predictions(stage / "validation.csv"), read_predictions(stage / "test.csv"),
                              target_sensitivity=target_sensitivity, bootstrap=bootstrap,
                              seed=seed, data_kind=windows.kind)
        overlap = "not_applicable" if mode == "resnet_scratch" else pretraining_overlap
        evaluation["pretraining_overlap"] = overlap
        if mode != "resnet_scratch":
            evaluation["limitations"].append(
                f"Pretraining overlap: {overlap} (user supplied, not independently verified). "
                "Patient separation is guaranteed only for the current fine-tuning split.")
        # Patient IDs and predictions stay in the local run, never in aggregate reports.
        write_report(evaluation, read_predictions(stage / "test.csv"), stage / "evaluation")
        save_file(best_state, str(stage / "model.safetensors"))
        checkpoint_hash = file_digest(stage / "model.safetensors")
        metadata = {"mode": mode, "channels": CHANNELS, "source_sfreq": windows.sfreq,
                    "window_seconds": 4, "threshold": evaluation["threshold"],
                    "data_kind": windows.kind, "best_epoch": best_epoch, "validation_ap": best_ap,
                    "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                    "seed": seed, "split_sha256": split_digest(split),
                    "dataset_sha256": dataset_hash,
                    "checkpoint_sha256": checkpoint_hash, "selection": "validation average precision only",
                    "task": "binary four-second window IED classification, not event localization",
                    "pretraining_overlap": overlap,
                    "overlap_status_is_user_supplied": mode != "resnet_scratch",
                    "clinical_validation": "not established by this pipeline",
                    "preprocessing": ("adaptation: Butterworth4 0.1-75Hz, IIR notch50 Q30, 200Hz, V*10000"
                                      if mode.startswith("labram_") else
                                      "source: Butterworth4 1-45Hz, common average, 250Hz, global zscore clip8"),
                    "encoder_config": getattr(model, "encoder_config", None),
                    "labram_transfer": ("strict full checkpoint load, then retain first 4 temporal patches; explicit channel names"
                                        if mode.startswith("labram_") else None),
                    "pretrained_source": None if mode == "resnet_scratch" else (
                        {"repo": LABRAM_REPO, "revision": LABRAM_REVISION} if mode.startswith("labram_") else
                        {"repo": CONFIG["source"], "revision": CONFIG["revision"], "dataset": "vEpiSet"}),
                    "split_counts": {name: {"windows": len(index), "subjects": len(np.unique(windows.subjects[index])),
                                            "positive": int(windows.y[index].sum())} for name, index in split.items()}}
        (stage / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (stage / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if file_digest(dataset) != dataset_hash:
            raise ValueError("Dataset changed during training; refusing to publish mismatched metadata.")
        output.mkdir(exist_ok=False)
        reserved = True
        os.rename(stage, output)
        reserved = False
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if reserved:
            output.rmdir()
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", choices=MODES, default="resnet_finetune")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-sensitivity", type=float, default=0.8)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--pretraining-overlap", choices=["unknown", "present", "excluded_by_user"], default="unknown")
    args = parser.parse_args()
    try:
        result = fit(args.dataset, args.output, args.model, args.epochs, args.batch_size,
                     args.lr, args.seed, args.target_sensitivity, args.bootstrap, args.pretraining_overlap)
        print(f"Saved {args.output}; threshold={result['threshold']:.6f}")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
