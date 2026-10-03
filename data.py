"""Raw-window contract and deterministic patient-separated splits."""

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, iirnotch, resample_poly, sosfiltfilt, filtfilt

from predict import CHANNELS, preprocess


@dataclass
class Windows:
    x: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    sfreq: int
    kind: str


def load_windows(path):
    with np.load(path, allow_pickle=False) as archive:
        required = {"X", "y", "subject", "sfreq", "channels", "units", "data_kind"}
        if not required.issubset(archive.files):
            raise ValueError(f"Dataset needs: {', '.join(sorted(required))}")
        x = np.asarray(archive["X"], dtype=np.float32)
        y = np.asarray(archive["y"])
        subjects = np.asarray(archive["subject"])
        rate = float(archive["sfreq"].item())
        if not math.isfinite(rate) or not rate.is_integer() or rate < 200:
            raise ValueError("Dataset sampling rate must be an integer >=200 Hz.")
        rate = int(rate)
        if archive["channels"].tolist() != CHANNELS or archive["units"].item() != "V":
            raise ValueError("Expected canonical 19-channel order and units='V'.")
        kind = str(archive["data_kind"].item())
    if x.ndim != 3 or x.shape[1:] != (19, 4 * rate) or len(x) == 0:
        raise ValueError("X must have shape (windows,19,4*sfreq).")
    if y.shape != (len(x),) or not np.isin(y, [0, 1]).all():
        raise ValueError("y must contain one binary label per window.")
    if subjects.shape != (len(x),) or subjects.dtype.kind not in "US":
        raise ValueError("subject must contain one string ID per window.")
    subjects = np.char.strip(subjects.astype(str))
    if np.any(subjects == "") or kind not in {"clinical", "synthetic", "unspecified"}:
        raise ValueError("Empty subject or invalid data_kind.")
    if not np.isfinite(x).all() or np.any(np.ptp(x, axis=-1) == 0):
        raise ValueError("Dataset contains non-finite or flat-channel windows.")
    # All modes compare exactly the same population of usable windows.
    for begin in range(0, len(x), 64):
        _, usable = preprocess(x[begin:begin + 64], rate, return_valid=True)
        if not usable.all():
            raise ValueError("Dataset contains unusable low-variance windows.")
    return Windows(x, y.astype(np.int64), subjects, rate, kind)


def patient_split(windows, seed=42):
    patients = np.unique(windows.subjects)
    if len(patients) < 5:
        raise ValueError("At least five subjects are required for train/validation/test.")
    count = max(1, round(len(patients) * 0.2))
    rng = np.random.default_rng(seed)
    for _ in range(500):
        ordered = rng.permutation(patients)
        groups = {"test": ordered[:count], "validation": ordered[count:2 * count],
                  "train": ordered[2 * count:]}
        indices = {name: np.flatnonzero(np.isin(windows.subjects, group))
                   for name, group in groups.items()}
        if all(len(np.unique(windows.y[index])) == 2 for index in indices.values()):
            return indices
    raise ValueError("Cannot make patient-separated splits with both labels in each split.")


def split_digest(indices):
    digest = hashlib.sha256()
    for name in sorted(indices):
        digest.update(name.encode())
        digest.update(indices[name].astype("<i8").tobytes())
    return digest.hexdigest()


def labram_preprocess(x, sfreq):
    """LaBraM adaptation: raw volts -> 200 Hz in units of 100 microvolts."""
    x = np.asarray(x, dtype=np.float32)
    sos = butter(4, [0.1, 75], fs=sfreq, btype="bandpass", output="sos")
    x = sosfiltfilt(sos, x, axis=-1)
    b, a = iirnotch(50, 30, fs=sfreq)
    x = filtfilt(b, a, x, axis=-1)
    common = math.gcd(200, sfreq)
    x = resample_poly(x, 200 // common, sfreq // common, axis=-1)
    x = (x * 1e4).astype(np.float32)
    if x.shape[1:] != (19, 800) or not np.isfinite(x).all():
        raise ValueError("Invalid LaBraM input after processing.")
    return x
