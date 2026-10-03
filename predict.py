"""Score four-second scalp EEG windows using released SenuaLab weights."""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

import mne
import numpy as np
import torch
from safetensors.torch import load_file
from scipy.signal import butter, resample_poly, sosfiltfilt

from model import EEGResNetAttention

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text())
CHANNELS = CONFIG["channels"]
ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}


def channel_indices(names):
    found, references = {}, set()
    for i, name in enumerate(names):
        match = re.fullmatch(
            r"(?:EEG\s*)?([A-Z][A-Z0-9]*)(?:-(REF|LE|AR|AVG|CAR|A1|A2|M1|M2))?",
            name.strip().upper(),
        )
        if not match:
            continue
        channel = ALIASES.get(match[1], match[1]).casefold()
        if channel not in {c.casefold() for c in CHANNELS}:
            continue
        if channel in found:
            raise ValueError(f"Duplicate EEG channel: {name}")
        found[channel] = i
        references.add(match[2] or "unlabelled")
    missing = [c for c in CHANNELS if c.casefold() not in found]
    if missing:
        raise ValueError(f"Missing referential channels: {', '.join(missing)}. Bipolar input is unsupported.")
    if len(references) != 1:
        raise ValueError("EEG channels must share a common reference; mixed reference labels found.")
    return [found[c.casefold()] for c in CHANNELS]


def preprocess(batch, sfreq, return_valid=False):
    # Same per-window operations and float32 arithmetic as robust_train.py.
    x = np.asarray(batch, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (19, 4 * sfreq):
        raise ValueError("Expected full four-second, 19-channel windows.")
    if not np.isfinite(x).all():
        raise ValueError("EEG contains NaN or infinity.")
    sos = butter(4, [1.0, 45.0], btype="bandpass", fs=sfreq, output="sos")
    x = sosfiltfilt(sos, x, axis=-1).astype(np.float32)
    x -= x.mean(axis=1, keepdims=True)
    if sfreq != 250:
        common = math.gcd(250, sfreq)
        x = resample_poly(x, 250 // common, sfreq // common, axis=-1).astype(np.float32)
    mean = x.mean(axis=(1, 2), keepdims=True)
    std = x.std(axis=(1, 2), keepdims=True)
    processed = np.clip((x - mean) / np.maximum(std, 1e-7), -8, 8).astype(np.float32)
    if return_valid:
        return processed, std[:, 0, 0] > 1e-7
    return processed


def load_model():
    weights = ROOT / "model.safetensors"
    if hashlib.sha256(weights.read_bytes()).hexdigest() != CONFIG["sha256"]:
        raise ValueError("Model checksum mismatch.")
    model = EEGResNetAttention(num_channels=19, num_classes=2, input_length=1000)
    model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    return model.eval()


@torch.inference_mode()
def predict(raw, model, stride=0.5, batch_size=16):
    source_rate = float(raw.info["sfreq"])
    if not math.isfinite(source_rate) or source_rate <= 90 or not source_rate.is_integer():
        raise ValueError("An integer sampling rate above 90 Hz is required.")
    if not math.isfinite(stride) or stride <= 0 or stride > 4:
        raise ValueError("Stride must be greater than 0 and at most 4 seconds.")
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    sfreq = int(source_rate)
    picks = channel_indices(raw.ch_names)
    if set(raw.info["bads"]) & {raw.ch_names[i] for i in picks}:
        raise ValueError("A required EEG channel is marked bad.")
    window, step = 4 * sfreq, max(1, round(stride * sfreq))
    if raw.n_times < window:
        raise ValueError("Recording must contain at least four seconds.")
    starts = list(range(0, raw.n_times - window + 1, step))
    if starts[-1] != raw.n_times - window:
        starts.append(raw.n_times - window)  # Cover the tail with a full window.
    for begin in range(0, len(starts), batch_size):
        current = starts[begin:begin + batch_size]
        batches, valid, rows = [], [], []
        for start in current:
            data = raw.get_data(picks=picks, start=start, stop=start + window, reject_by_annotation="NaN")
            status = "ok"
            if not np.isfinite(data).all():
                status = "skipped_bad_or_nonfinite"
            elif np.any(np.ptp(data, axis=1) == 0):
                status = "skipped_flat_channel"
            rows.append([start / sfreq, (start + window) / sfreq, "", "", status])
            if status == "ok":
                valid.append(len(rows) - 1)
                batches.append(data)
        if batches:
            processed, usable = preprocess(np.stack(batches), sfreq, return_valid=True)
            for index, keep in zip(valid, usable):
                if not keep:
                    rows[index][4] = "skipped_low_variance"
            if usable.any():
                scores = model(torch.from_numpy(processed[usable])).softmax(1)[:, 1].numpy()
                if not np.isfinite(scores).all():
                    raise ValueError("Model returned non-finite scores.")
                for index, score in zip(np.array(valid)[usable], scores):
                    rows[index][2:4] = [float(score), int(score >= CONFIG["threshold"])]
        yield from rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Referential EDF, BDF or BrainVision .vhdr")
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--stride", type=float, default=0.5, help="Window step in seconds")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    readers = {".edf": mne.io.read_raw_edf, ".bdf": mne.io.read_raw_bdf, ".vhdr": mne.io.read_raw_brainvision}
    try:
        reader = readers.get(args.input.suffix.lower())
        if reader is None:
            raise ValueError("Supported formats: .edf, .bdf, .vhdr")
        if args.output.resolve() == args.input.resolve():
            raise ValueError("Output must differ from input.")
        torch.set_num_threads(min(4, torch.get_num_threads()))
        raw = reader(str(args.input), preload=False, verbose="ERROR")
        temporary = None
        try:
            rows = predict(raw, load_model(), args.stride, args.batch_size)
            first = next(rows)  # Validate before creating output.
            with tempfile.NamedTemporaryFile(mode="w", newline="", dir=args.output.parent,
                                             prefix=f".{args.output.name}.", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                writer = csv.writer(handle)
                writer.writerow(["start_seconds", "end_seconds", "ied_score", "candidate_ied", "status"])
                writer.writerow(first)
                scored = int(first[4] == "ok")
                candidates = int(first[3] == 1)
                skipped = int(first[4] != "ok")
                for row in rows:
                    writer.writerow(row)
                    scored += row[4] == "ok"
                    candidates += row[3] == 1
                    skipped += row[4] != "ok"
            os.link(temporary, args.output)  # Atomic publication; never overwrite an existing result.
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raw.close()
        print(f"{args.output}: {scored} scored, {candidates} candidate windows, {skipped} skipped.")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
