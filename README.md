# EEG IED Detector

This public research toolkit accompanies an EEG analysis project completed at Hadassah Medical Center in spring 2026.

Detect candidate interictal epileptiform discharge (IED) windows in EEG recordings. A released checkpoint is included (~20 MB).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python predict.py recording.edf --output predictions.csv
```

Python 3.11+. EDF, BDF or BrainVision `.vhdr` input; 19 channels with a common reference. Four-second windows, 0.5-second step. Output: window scores and quality status.

[Usage](USAGE.md) covers data preparation, training and evaluation. [Example outputs](examples/synthetic) use synthetic signals.

Research use only. Outputs require expert review; the public toolkit has not undergone independent clinical validation.

Code: MIT. Included weights: CC BY 4.0. Sources and attribution: [NOTICE](NOTICE).
