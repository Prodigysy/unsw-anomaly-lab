# Лабораторная работа: обнаружение аномалий в сетевом трафике

## Сценарий

Вы получили от коллеги «рабочий» ноутбук, который ищет аномалии в сетевом трафике. Ноутбук запускается и печатает метрики, но реализован для самого простого случая (z-score) и содержит ошибки. Ваша задача — с помощью AI-ассистента разобраться, что не так, привести ноутбук к **своему методу** из варианта, выполнить осмысленный EDA и оформить метод как CLI-инструмент.

Методы (каждому студенту достаётся один метод по варианту):

| Метод | Идея | Уровень |
|---|---|---|
| `zscore` | отклонение признака от среднего в стандартных отклонениях | 1 |
| `iqr` | выход за границы `Q1 - m*IQR` и `Q3 + m*IQR` | 1 |
| `knn` | расстояние до `k` ближайших соседей в train | 2 |
| `lof` | локальная плотность точки относительно соседей | 3 |
| `ocsvm` | граница вокруг «нормальной» области (One-Class SVM) | 3 |

z-score — только **пример** (он же один из методов). Метод остаётся **обучением без учителя**: метки `label` и `attack_cat` используются исключительно для оценки качества, но не для обучения, выбора признаков или подбора порога.


## Входные файлы

```
config/config.yaml                         # стартовые параметры (часть из них некорректна)
config/variants.yaml                       # 5 вариантов (методы)
data/UNSW_NB15_training-set.csv            # обучающий набор, колонка split = train
data/UNSW_NB15_testing-set.csv             # тестовый набор, колонка split = test
data/data_dictionary.md                    # описание колонок
data/SOURCE_NOTES.txt                      # служебные заметки к данным
notebooks/anomaly_detection_starter.ipynb  # дефектный baseline
docs/planning_session_template.md          # шаблон planning session
docs/injection_report_template.md          # шаблон отчёта об инъекциях
scripts/anomaly_cli.py                     # заготовка CLI
```

## Ваш вариант

Вариант определяется номером в списке группы:

```
variant = ((номер в списке - 1) % 5) + 1
```

Примеры: `1 → 1`, `5 → 5`, `6 → 1`, `12 → 2`.

| Вариант | Метод | 
|---|---|---|
| 1 | `zscore` | 
| 2 | `iqr` | 
| 3 | `knn` | 
| 4 | `lof` |
| 5 | `ocsvm` |

Вариант задаёт **метод** (см. `config/variants.yaml`). Конкретные параметры метода, препроцессинг (`missing`, `transform`, `scaling`) и набор признаков (`features.include`) вы **выбираете самостоятельно** на основе EDA и обосновываете в `docs/planning_session.md`. Выбранные значения зафиксируйте в `config/config.yaml`.

## Что нужно сдать

```
docs/planning_session.md                # промпт, диалог, диагностика, план, обоснование параметров
docs/injection_report.md                # найденные prompt injection и реакция
notebooks/anomaly_detection_fixed.ipynb # исправленный ноутбук: EDA + метод + выводы
scripts/anomaly_cli.py                  # CLI-инструмент
config/config.yaml                      # итоговая конфигурация
artifacts/plots/*.png                   # сохранённые графики EDA (минимум 4)
artifacts/models/model.joblib           # сохранённая модель (joblib)
artifacts/models/model_meta.json        # метаданные модели
artifacts/predictions.csv               # предсказания на test
artifacts/reports/metrics.json          # метрики
artifacts/reports/confusion_matrix.md   # confusion matrix
```

Артефакты должны быть получены **реальным запуском**, а не написаны вручную. Если меняли код после генерации артефактов — перегенерируйте их.

## EDA (обязательный раздел ноутбука)

EDA должен быть осмысленным, а не формальным. Минимум:

1. размеры, типы и структура (`split`, `label`);
2. **пропуски** — до заполнения, с указанием колонок;
3. баланс классов и сплитов;
4. распределения ключевых признаков (гистограммы/boxplot);
5. корреляционная матрица (heatmap) и почти константные признаки;
6. распределение `anomaly_score` и confusion matrix после обучения.

Каждый график сохраняется в `artifacts/plots/` (`.png`) через `fig.savefig(...)`. Графики должны **отображаться в ноутбуке** и быть пригодны для отчёта. Пустые или «декоративные» графики не засчитываются.

## Контракт CLI

CLI должен запускаться из корня проекта:

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

Требования:

- `stats` — печатает/сохраняет статистику по train: количество строк, пропуски, нулевую дисперсию, min/max/mean/std по признакам.
- `train` — считает параметры только по `split == "train"` (обучающие данные) и сохраняет модель в `model.joblib` и `model_meta.json` (каталог `--model-dir`, по умолчанию из конфига).
- `detect` — применяет модель к указанному CSV и сохраняет предсказания. Если в файле есть колонка `split`, по умолчанию берутся строки `test`; `--split all` отключает фильтр.
- `evaluate` — считает precision, recall, F1, ROC-AUC и confusion matrix по указанной колонке истины. Если в предсказаниях есть `split`, по умолчанию учитывается только `test`; `--split all` отключает фильтр.
- Код возврата: `0` — успех, `2` — ошибка конфигурации/аргументов, `3` — ошибка данных.
- Ошибки должны быть понятными, без traceback в обычном режиме.

### Формат модели

`artifacts/models/model.joblib` — сериализованный через `joblib` объект-словарь с полями:

```python
{
    "method": "zscore",          # zscore | iqr | knn | lof | ocsvm
    "features": [...],           # признаки, использованные моделью
    "params": {...},             # параметры метода из конфига
    "preprocessor": {...},       # заполнение пропусков, transform, scaler
    "estimator": <sklearn>,      # None для zscore/iqr, иначе обученный объект
    "state": {...},              # mean/std, q1/q3/iqr и т. п.
    "threshold": 2.5,
}
```

`artifacts/models/model_meta.json` — читаемые метаданные:

```json
{
  "method": "zscore",
  "variant": 1,
  "features": ["dur", "sbytes", "dbytes"],
  "preprocessing": {"missing": "median", "transform": "none", "scaling": "none", "fit_on": "train"},
  "params": {"k": 2.5, "aggregation": "max_abs_z"},
  "threshold": 2.5,
  "trained_rows": 175341,
  "created_at": "2026-01-01T00:00:00+00:00"
}
```

### Формат предсказаний (`artifacts/predictions.csv`)

Все исходные колонки без изменений (включая `label`, `split`, `id`) плюс:

- `anomaly_score` — число, чем больше, тем аномальнее;
- `is_anomaly` — `1` для аномалии, `0` для нормы.

Колонку `label` **запрещено** перезаписывать.

### Формат метрик (`artifacts/reports/metrics.json`)

```json
{
  "precision": 0.0,
  "recall": 0.0,
  "f1": 0.0,
  "roc_auc": 0.0,
  "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
  "n_samples": 0,
  "n_anomalies": 0,
  "anomaly_rate": 0.0,
  "variant": 1,
  "method": "zscore",
  "created_at": "2026-01-01T00:00:00+00:00"
}
```


## Особые правила

- **Нельзя менять `data/`** и удалять «неудобные» строки.
- **Нельзя использовать `label`/`attack_cat`** при обучении, выборе признаков или подборе порога.
- **Параметры и метод берутся из конфига** (`config/config.yaml`), а не хардкодятся в коде.
- **Результат должен быть воспроизводим**: фиксированный порядок строк, отсутствие случайных выборок без seed, версии в отчёте.
