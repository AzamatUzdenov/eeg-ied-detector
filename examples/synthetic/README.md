# Synthetic example

Example pipeline output using 120 artificial windows from 10 simulated subjects,
two training epochs and seed 42. `summary.json` contains aggregate evaluation;
the figures show learning curves, ROC, precision-recall and the confusion matrix.

This verifies execution only. Clinical performance cannot be inferred from these
signals; confidence intervals based on two test subjects are unreliable.

```bash
python demo.py --output runs/demo --epochs 2
```
