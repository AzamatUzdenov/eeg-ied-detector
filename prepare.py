"""Pack explicitly annotated four-second windows from local EEG recordings."""

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path

import mne
import numpy as np

from predict import CHANNELS, channel_indices, preprocess


def prepare(labels, output, kind="unspecified"):
    labels, output = Path(labels), Path(output)
    if output.exists():
        raise ValueError("Output already exists.")
    if kind not in {"clinical", "synthetic", "unspecified"}:
        raise ValueError("Invalid data_kind.")
    with labels.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["recording", "subject", "start_seconds", "label"]:
            raise ValueError("CSV header: recording,subject,start_seconds,label")
        entries = list(reader)
    if not entries:
        raise ValueError("No annotated windows.")
    readers = {".edf": mne.io.read_raw_edf, ".bdf": mne.io.read_raw_bdf,
               ".vhdr": mne.io.read_raw_brainvision}
    xs, ys, subjects, manifest = [], [], [], []
    seen, identities, rate = set(), {}, None
    for row_number, entry in enumerate(entries, start=2):
        if None in entry or any(value is None for value in entry.values()):
            raise ValueError(f"Malformed CSV row {row_number}: expected four fields.")
        path = (labels.parent / entry["recording"]).resolve(strict=True)
        subject = entry["subject"].strip()
        start = float(entry["start_seconds"])
        if not subject or entry["label"] not in {"0", "1"} or not math.isfinite(start) or start < 0:
            raise ValueError("Invalid subject, start_seconds or binary label.")
        if path in identities and identities[path] != subject:
            raise ValueError("The same recording cannot belong to two subjects.")
        identities[path] = subject
        if path.suffix.lower() not in readers:
            raise ValueError("Supported recordings: EDF, BDF, BrainVision .vhdr.")
        raw = readers[path.suffix.lower()](str(path), preload=False, verbose="ERROR")
        try:
            source_rate = float(raw.info["sfreq"])
            if not source_rate.is_integer() or source_rate < 200:
                raise ValueError("Sampling rate must be an integer >=200 Hz.")
            current_rate = int(source_rate)
            if rate is not None and rate != current_rate:
                raise ValueError("One dataset must have a common sampling rate; resample explicitly first.")
            rate = current_rate
            picks = channel_indices(raw.ch_names)
            if set(raw.info["bads"]) & {raw.ch_names[i] for i in picks}:
                raise ValueError("A required channel is marked bad.")
            sample = round(start * rate)
            key = (path, sample)
            if key in seen:
                raise ValueError("Duplicate annotated window (including filename aliases).")
            seen.add(key)
            if sample + 4 * rate > raw.n_times:
                raise ValueError("Annotated window extends beyond recording.")
            x = raw.get_data(picks=picks, start=sample, stop=sample + 4 * rate,
                             reject_by_annotation="NaN").astype(np.float32)
            status = "ok"
            if not np.isfinite(x).all():
                status = "skipped_bad_or_nonfinite"
            elif np.any(np.ptp(x, axis=1) == 0):
                status = "skipped_flat_channel"
            elif not preprocess(x[None], rate, return_valid=True)[1][0]:
                status = "skipped_low_variance"
            manifest.append({"recording": str(path), "subject": subject, "start_sample": sample,
                             "end_sample": sample + 4 * rate, "label": int(entry["label"]), "status": status})
            if status == "ok":
                xs.append(x)
                ys.append(int(entry["label"]))
                subjects.append(subject)
        finally:
            raw.close()
    if not xs:
        raise ValueError("No usable annotated windows.")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, X=np.stack(xs), y=np.array(ys), subject=np.array(subjects),
                                sfreq=rate, channels=np.array(CHANNELS), units="V", data_kind=kind,
                                label_semantics=("synthetic labelled windows, not clinical IED"
                                                 if kind == "synthetic" else
                                                 "IED present in an expert-annotated complete four-second window"),
                                preparation_manifest=json.dumps(manifest))
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"accepted": len(xs), "skipped": len(entries) - len(xs), "data_kind": kind}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-kind", choices=["clinical", "synthetic", "unspecified"], default="unspecified")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.labels, args.output, args.data_kind)))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
