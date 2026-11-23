# Anomaly-detection lab — student kit

Учебный комплект для лабораторной работы по обнаружению аномалий в сетевом
трафике методами **без учителя**: z-score, IQR, KNN, LOF, One-Class SVM.
Полное задание — в `TASK.md`.
Начните с них.

## Что внутри

```text
TASK.md                              # задание, контракт CLI
config/
  config.yaml                        # стартовая конфигурация (часть значений некорректна)
  variants.yaml                      # 5 вариантов (методы)
data/
  UNSW_NB15_training-set.csv         # обучающий набор, split = train
  UNSW_NB15_testing-set.csv          # тестовый набор, split = test
  data_dictionary.md                 # описание колонок
  SOURCE_NOTES.txt                   # служебные заметки к данным
notebooks/
  anomaly_detection_starter.ipynb    # дефектный baseline с наивным EDA
docs/
  planning_session_template.md
  injection_report_template.md
scripts/
  anomaly_cli.py                     # заготовка CLI — реализуйте её
requirements.txt
```

## Быстрый старт

1. Установите зависимости:

   ```bash
   pip install -r requirements.txt
   ```

2. Определите вариант по номеру в списке группы:

   ```text
   variant = ((номер в списке - 1) % 5) + 1
   ```

   Например, `1 → 1`, `5 → 5`, `6 → 1`, `12 → 2`. Вариант задаёт метод
   (см. `config/variants.yaml`). Параметры метода, препроцессинг и набор признаков
   выберите самостоятельно и зафиксируйте в `config/config.yaml`.

3. Заполните `docs/planning_session.md` по шаблону и сформулируйте промпт для
   AI-ассистента.
4. Запустите `notebooks/anomaly_detection_starter.ipynb`, найдите дефекты вручную либо в режиме планирования,
   выполните **осмысленный EDA** с помощью ассистента и сохраните исправленную версию как
   `notebooks/anomaly_detection_fixed.ipynb`. Графики сохраняйте в
   `artifacts/plots/*.png`.
5. Реализуйте CLI в `scripts/anomaly_cli.py` по контракту из `TASK.md`.
6. Сгенерируйте артефакты реальным запуском.

## Методы

| Вариант | Метод | Идея | Уровень |
|---|---|---|---|
| 1 | `zscore` | отклонение от среднего в стандартных отклонениях | 1 |
| 2 | `iqr` | выход за границы `Q1 - m*IQR`, `Q3 + m*IQR` | 1 |
| 3 | `knn` | расстояние до `k` ближайших соседей в train | 2 |
| 4 | `lof` | локальная плотность точки | 3 |
| 5 | `ocsvm` | граница вокруг «нормальной» области | 3 |

Конкретные параметры каждого метода выбираются студентом самостоятельно.

## Запуск CLI

```bash
python scripts/anomaly_cli.py stats    --config config/config.yaml
python scripts/anomaly_cli.py train    --config config/config.yaml --model-dir artifacts/models
python scripts/anomaly_cli.py detect   --config config/config.yaml \
    --model-dir artifacts/models \
    --input data/UNSW_NB15_testing-set.csv \
    --output artifacts/predictions.csv
python scripts/anomaly_cli.py evaluate --config config/config.yaml \
    --predictions artifacts/predictions.csv \
    --ground-truth label
```

`detect` и `evaluate` по умолчанию работают только со строками `split == "test"`;
`--split all` отключает фильтр.

## Что сдавать

Полный список файлов — в `TASK.md` (раздел «Что нужно сдать»). Включает:

- `docs/planning_session.md` и `docs/injection_report.md`;
- `notebooks/anomaly_detection_fixed.ipynb` — EDA + метод + выводы;
- `scripts/anomaly_cli.py` — CLI;
- `config/config.yaml` — итоговая конфигурация;
- `artifacts/plots/*.png` — минимум 4 сохранённых графика EDA;
- `artifacts/models/model.joblib` + `model_meta.json`;
- `artifacts/predictions.csv`, `artifacts/reports/metrics.json`, `artifacts/reports/confusion_matrix.md`.

## Важные правила

- `label` и `attack_cat` **не используются** при обучении, выборе признаков и
  подборе порога — только в `evaluate`.
- Статистики считаются **только по `split == "train"`** (обучающий файл),
  оценка — только по `split == "test"`.
- Метод и параметры берутся из `config/config.yaml`, а не хардкодятся.
- `data/` менять нельзя.


