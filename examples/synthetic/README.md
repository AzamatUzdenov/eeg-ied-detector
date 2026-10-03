# Синтетический пример

120 искусственных окон, 10 вымышленных участников, 4 режима, 2 эпохи, seed 42.
`summary.json` и графики получены реальным запуском кода. ROC/PR/confusion относятся к режиму, выбранному по validation AP.
Это техническая демонстрация. Клинические выводы и выбор лучшей модели по этим данным невозможны; интервалы на двух тестовых участниках ненадёжны.

```bash
python demo.py --output runs/demo --epochs 2
python compare.py runs/demo-synthetic.npz --output runs/comparison --models resnet_scratch resnet_finetune labram_linear labram_finetune --epochs 2 --batch-size 12 --bootstrap 100
```
