# EEG IED detector

Готовая SenuaLab ResNet-Attention для поиска окон ЭЭГ с возможными эпилептиформными разрядами. Исходные веса включены (~20 МБ).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python predict.py recording.edf --output predictions.csv
```

Python 3.11+. Вход: EDF/BDF или BrainVision `.vhdr`, 19 каналов из `config.json`, общий референс (биполярный монтаж не поддерживается). Допустимы T7/T8/P7/P8. Окна 4 с, шаг 0.5 с; CSV содержит оценку и статус. Плохие/нечисловые/плоские окна пропускаются. Это кандидаты для просмотра, точное время разряда не определяется.

Дополнительно: подготовка экспертной разметки, обучение ResNet и LaBraM, сравнение с разделением по пациентам, выбор порога по validation, метрики с интервалами и графиками. [Команды и формат данных](USAGE.md), [проверенный синтетический пример](examples/synthetic).

```bash
pip install -r requirements-train.txt
python demo.py --model labram_linear --epochs 2
python -m pytest -q -m "not integration"
```

Исследовательская модель, независимая клиническая проверка не проведена. Нужен просмотр специалистом. Результаты авторов на vEpiSet (13 участников): AUROC 0.900, F1 0.667, чувствительность 0.608, специфичность 0.973. Диагноз и приступы не определяет.

Источник: [SenuaLab](https://github.com/SenuaLab/EEG-IED-Detection), [веса](https://huggingface.co/SenuaLab/EEG-IED-Detection). Код MIT, веса CC BY 4.0, атрибуция в `NOTICE`.
