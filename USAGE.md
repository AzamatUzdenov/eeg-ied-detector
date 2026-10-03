# Usage

```bash
pip install -r requirements-train.txt
python prepare.py labels.csv --output dataset.npz --data-kind clinical
python train.py dataset.npz --output runs/trained --epochs 20
python predict.py recording.edf --checkpoint runs/trained --output predictions.csv
```

`labels.csv` header: `recording,subject,start_seconds,label`. Paths are relative
to the CSV; time is measured from the start of the recording. Each expert label
is 0 or 1 for a complete four-second window: 1 means IED present, 0 means IED absent.
One recording belongs to one subject.

Data preparation selects and orders the 19 channels listed in `config.json`.
Input requires a common reference and one integer sampling rate >=200 Hz. T7/T8/P7/P8 aliases
are accepted; bipolar or mixed-reference input is unsupported. MNE reads samples
in volts. Bad, non-finite, flat and low-variance windows are skipped.

The NPZ contains `X`, `y`, `subject`, `sfreq`, `channels`, `units` and `data_kind`.
`X` has shape `(windows,19,4*sfreq)` and `units` must be `V`.
The complete dataset and processed windows are loaded into memory.
At least five subjects are required, with both labels in every split.

Patient-separated train/validation/test splits use approximately 60/20/20 of
subjects and a fixed seed. Checkpoints are selected by validation average
precision. The threshold maximizes specificity at validation sensitivity >=0.8
(`--target-sensitivity`). Test data is evaluated after selection.

Runs save weights, configuration, history, local prediction CSVs, aggregate
metrics and ROC/PR/confusion plots. Confidence intervals resample whole patients;
they condition on the selected threshold and omit selection uncertainty.
Intervals from small patient samples are unreliable.

Patient separation applies to the current training split. Prior pretraining
overlap requires a separate audit; `--pretraining-overlap` only records the user's
assessment. Pretrained sources and transfer settings are recorded in run metadata.
Metrics describe windows and do not locate individual events or measure false events per hour.

Local CSVs and preparation manifests contain subject IDs and recording paths.
`runs/`, datasets and recordings are excluded from Git. Review aggregate outputs
before publishing. Existing outputs are never overwritten.

```bash
python demo.py --output runs/demo --epochs 2
python -m pytest -q
```

The demo and tests use synthetic signals. They verify execution, not clinical
performance. Additional options are listed with `--help`; optional components
download pinned weights on first use.
