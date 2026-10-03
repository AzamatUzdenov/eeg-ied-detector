# EEG IED detector

Готовая SenuaLab ResNet-Attention для поиска окон ЭЭГ с возможными эпилептиформными разрядами. Исходные веса включены (~20 МБ).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python predict.py recording.edf --output predictions.csv
```

Python 3.10+. Вход: EDF/BDF или BrainVision `.vhdr`, 19 каналов из `config.json`, общий референс (биполярный монтаж не поддерживается). Допустимы названия T7/T8/P7/P8. Обработка: 1-45 Гц, усреднённый референс, 250 Гц, нормализация окна. Окна 4 с, шаг 0.5 с. CSV: время окна, оценка, `candidate_ied`, статус. Не оцениваются помеченные плохие участки, NaN, плоские каналы и остаточное стандартное отклонение ≤0.1 мкВ. Это окна для просмотра, точное время разряда не определяется.

Исследовательская модель, независимая клиническая проверка не проведена. Нужен просмотр специалистом. Результаты авторов на vEpiSet (13 участников): AUROC 0.900, F1 0.667, чувствительность 0.608, специфичность 0.973. Диагноз и приступы не определяет.

Источник: [SenuaLab](https://github.com/SenuaLab/EEG-IED-Detection), [веса](https://huggingface.co/SenuaLab/EEG-IED-Detection). Код MIT, веса CC BY 4.0, атрибуция в `NOTICE`.
