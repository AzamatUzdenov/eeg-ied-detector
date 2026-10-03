"""Generate labelled synthetic waveforms and exercise the training pipeline."""

import argparse
from pathlib import Path

import numpy as np

from predict import CHANNELS


def synthetic_dataset(path, seed=42, subjects=10, per_class=6, sfreq=250):
    if Path(path).exists():
        raise ValueError("Synthetic dataset already exists.")
    rng = np.random.default_rng(seed)
    time = np.arange(4 * sfreq) / sfreq
    x, y, groups = [], [], []
    for subject in range(subjects):
        frequency = rng.uniform(7, 12)
        for label in [0, 1]:
            for _ in range(per_class):
                wave = np.stack([15e-6 * np.sin(2 * np.pi * (frequency + i * 0.07) * time + rng.uniform(0, 6.28))
                                 + rng.normal(0, 4e-6, len(time)) for i in range(19)])
                if label:
                    centre = rng.uniform(1, 3)
                    pulse = 90e-6 * np.exp(-((time - centre) / 0.018) ** 2)
                    pulse -= 30e-6 * np.exp(-((time - centre - 0.12) / 0.07) ** 2)
                    for channel, scale in [(12, 1), (14, 0.8), (4, 0.5)]:
                        wave[channel] += scale * pulse
                x.append(wave.astype(np.float32))
                y.append(label)
                groups.append(f"synthetic-{subject:02d}")
    np.savez_compressed(path, X=np.stack(x), y=np.array(y), subject=np.array(groups),
                        sfreq=sfreq, channels=np.array(CHANNELS), units="V", data_kind="synthetic",
                        label_semantics="generated pulse class; not real expert-labelled IED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    parser.add_argument("--model", choices=["resnet_scratch", "resnet_finetune", "labram_linear", "labram_finetune"],
                        default="resnet_scratch")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists.")
    from train import fit
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset = args.output.parent / f"{args.output.name}-synthetic.npz"
    try:
        synthetic_dataset(dataset, args.seed)
        fit(dataset, args.output, args.model, args.epochs, batch_size=12, seed=args.seed, bootstrap=100)
        print("SYNTHETIC technical demonstration only. No clinical accuracy claim.")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
