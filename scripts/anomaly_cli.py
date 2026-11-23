#!/usr/bin/env python3
"""CLI-инструмент обнаружения аномалий в сетевом трафике (вариант 1, метод zscore).

Подкоманды по контракту TASK.md:

    python scripts/anomaly_cli.py stats    --config config/config.yaml
    python scripts/anomaly_cli.py train    --config config/config.yaml --model-dir artifacts/models
    python scripts/anomaly_cli.py detect   --config config/config.yaml \
        --model-dir artifacts/models --input data/UNSW_NB15_testing-set.csv \
        --output artifacts/predictions.csv
    python scripts/anomaly_cli.py evaluate --config config/config.yaml \
        --predictions artifacts/predictions.csv --ground-truth label

Коды возврата: 0 — успех, 2 — ошибка конфигурации/аргументов, 3 — ошибка данных.
Метод и параметры читаются из конфига; статистики считаются только по
split == "train"; label/attack_cat в обучении не используются.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_DATA = 3

EPS_STD = 1e-12  # замена нулевого std (политика zero_variance: epsilon)


class ConfigError(Exception):
    """Некорректный конфиг или аргументы (код возврата 2)."""


class DataError(Exception):
    """Некорректные/отсутствующие данные (код возврата 3)."""


# --------------------------------------------------------------------------
# Конфиг
# --------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    if not config_path.is_file():
        raise ConfigError(f"файл конфигурации не найден: {config_path}")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"не удалось разобрать YAML: {exc}") from exc
    if not isinstance(config, dict):
        raise ConfigError("конфигурация должна быть YAML-отображением")

    for section in ("variant", "data", "features", "method", "preprocessing", "paths"):
        if section not in config:
            raise ConfigError(f"в конфиге отсутствует секция/ключ: {section}")

    variant = config["variant"]
    method = config["method"].get("name")
    expected = ((int(variant) - 1) % 5) + 1
    if expected != variant or method != "zscore":
        raise ConfigError(
            "этот CLI реализует вариант 1 (метод zscore); "
            f"в конфиге variant={variant}, method={method!r}"
        )
    return config


def resolve_path(value: str, base: Path = ROOT) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base / p


# --------------------------------------------------------------------------
# Данные
# --------------------------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise DataError(f"файл данных не найден: {path}")
    try:
        # utf-8-sig: у CSV есть BOM в первой колонке; low_memory=False —
        # устойчивый вывод типов для колонок со смешанными значениями.
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise DataError(f"не удалось прочитать CSV {path}: {exc}") from exc


def require_columns(df: pd.DataFrame, columns: list[str], where: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataError(f"в {where} отсутствуют колонки: {', '.join(missing)}")


def filter_split(df: pd.DataFrame, split_column: str, split_value: str, split_arg: str) -> pd.DataFrame:
    """--split all отключает фильтр; иначе берём строки split == test."""
    if split_arg == "all":
        return df
    if split_column not in df.columns:
        raise DataError(f"колонка '{split_column}' не найдена, используйте --split all")
    out = df[df[split_column] == split_value]
    if out.empty:
        raise DataError(f"нет строк со значением '{split_value}' в колонке '{split_column}'")
    return out


def feature_columns(config: dict, df: pd.DataFrame) -> list[str]:
    exclude = set(config["data"]["exclude_columns"])
    include = config["features"].get("include") or []
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    if include:
        unknown = [c for c in include if c not in df.columns]
        if unknown:
            raise ConfigError(f"features.include содержит неизвестные колонки: {', '.join(unknown)}")
        return [c for c in include if c in numeric] or list(include)
    return numeric


# --------------------------------------------------------------------------
# Препроцессинг и метод zscore
# --------------------------------------------------------------------------

def _signed_log1p(X: pd.DataFrame) -> pd.DataFrame:
    return np.sign(X) * np.log1p(np.abs(X))


def apply_transform(X: pd.DataFrame, transform: str) -> pd.DataFrame:
    if transform == "none":
        return X
    if transform == "signed_log1p":
        return _signed_log1p(X)
    raise ConfigError(f"неизвестный transform: {transform!r}")


def impute(X: pd.DataFrame, fill_values: pd.Series | None) -> pd.DataFrame:
    if fill_values is None:
        return X
    return X.fillna(fill_values)


def fit_preprocessor(config: dict, X: pd.DataFrame) -> dict:
    pre = config["preprocessing"]
    missing = pre.get("missing", "median")
    transform = pre.get("transform", "none")
    scaling = pre.get("scaling", "none")

    fill_values = None
    if missing == "median":
        fill_values = X.median(numeric_only=True)
    elif missing != "zero":
        raise ConfigError(f"неизвестный missing: {missing!r}")

    Xt = apply_transform(impute(X, fill_values), transform)

    if scaling not in ("none", "standard", "robust"):
        raise ConfigError(f"неизвестный scaling: {scaling!r}")
    if scaling == "standard":
        scaler = {"kind": "standard", "mean": Xt.mean(), "std": Xt.std(ddof=0).replace(0, 1.0)}
    elif scaling == "robust":
        iqr = Xt.quantile(0.75) - Xt.quantile(0.25)
        scaler = {"kind": "robust", "median": Xt.median(), "iqr": iqr.replace(0, 1.0)}
    else:
        scaler = {"kind": "none"}

    return {"missing": missing, "transform": transform, "scaling": scaling,
            "fill_values": fill_values, "scaler": scaler, "fit_on": pre.get("fit_on", "train")}


def transform_features(X: pd.DataFrame, preprocessor: dict) -> pd.DataFrame:
    X = impute(X, preprocessor["fill_values"])
    X = apply_transform(X, preprocessor["transform"])
    scaler = preprocessor["scaler"]
    if scaler["kind"] == "standard":
        X = (X - scaler["mean"]) / scaler["std"]
    elif scaler["kind"] == "robust":
        X = (X - scaler["median"]) / scaler["iqr"]
    return X


def fit_zscore_state(X: pd.DataFrame, ddof: int, zero_variance: str) -> dict:
    if zero_variance not in ("epsilon", "drop", "zero"):
        raise ConfigError(f"неизвестная политика zero_variance: {zero_variance!r}")
    mu = X.mean()
    sd = X.std(ddof=ddof)
    if zero_variance == "epsilon":
        sd = sd.replace(0.0, EPS_STD)
    elif zero_variance == "zero":
        # z для константной колонки считается нулевым (деление даёт NaN -> 0)
        sd = sd.replace(0.0, np.inf)
    return {"mean": mu, "std": sd, "ddof": ddof, "zero_variance": zero_variance}


def zscore_scores(X: pd.DataFrame, state: dict, aggregation: str) -> pd.Series:
    if aggregation not in ("max_abs_z", "count", "euclidean"):
        raise ConfigError(f"неизвестная агрегация: {aggregation!r}")
    z = (X - state["mean"]) / state["std"].replace(0.0, EPS_STD)
    z = z.fillna(0.0)  # политика zero_variance: zero -> NaN заменяем нулём
    zabs = z.abs()
    if aggregation == "max_abs_z":
        return zabs.max(axis=1)
    if aggregation == "count":
        return (zabs > 3.0).sum(axis=1).astype(float)
    return np.sqrt((z ** 2).sum(axis=1))


# --------------------------------------------------------------------------
# Подкоманды
# --------------------------------------------------------------------------

def cmd_stats(config: dict, args: argparse.Namespace) -> int:
    train_path = resolve_path(config["data"]["train_path"])
    df = load_csv(train_path)
    split_col = config["data"]["split_column"]
    require_columns(df, [split_col, config["data"]["label_column"]], train_path.name)
    train_df = filter_split(df, split_col, config["data"]["train_split"], "train")

    feats = feature_columns(config, df)
    print(f"Файл: {train_path.name}; строк (split == train): {len(train_df)}")
    print(f"Числовых признаков: {len(feats)}")

    X = train_df[feats].astype(float)
    missing = X.isna().sum()
    print("\nПропуски по признакам (только ненулевые):")
    print(missing[missing > 0].to_string() if (missing > 0).any() else "  нет")

    sd = X.std(ddof=0)
    zero_var = sd[sd == 0].index.tolist()
    print(f"\nПризнаки с нулевой дисперсией: {zero_var if zero_var else 'нет'}")

    desc = X.agg(["min", "max", "mean", "std"]).T.round(4)
    print("\nmin/max/mean/std по признакам:")
    print(desc.to_string())

    if args.output:
        out = resolve_path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "n_rows": int(len(train_df)),
            "n_features": len(feats),
            "features": feats,
            "missing": {k: int(v) for k, v in missing[missing > 0].items()},
            "zero_variance_features": zero_var,
            "describe": {c: {s: float(desc.loc[c, s]) for s in desc.columns} for c in desc.index},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nСтатистика сохранена: {out}")
    return EXIT_OK


def cmd_train(config: dict, args: argparse.Namespace) -> int:
    train_path = resolve_path(config["data"]["train_path"])
    df = load_csv(train_path)
    split_col = config["data"]["split_column"]
    require_columns(df, [split_col, config["data"]["label_column"]], train_path.name)
    train_df = filter_split(df, split_col, config["data"]["train_split"], "train")

    feats = feature_columns(config, df)
    if len(feats) < int(config["method"]["params"].get("min_features", 1)):
        raise DataError(f"слишком мало признаков для обучения: {len(feats)}")

    X = train_df[feats].astype(float)
    preprocessor = fit_preprocessor(config, X)
    Xt = transform_features(X, preprocessor)
    state = fit_zscore_state(Xt, ddof=int(config["method"]["params"]["ddof"]),
                             zero_variance=config["method"]["params"]["zero_variance"])
    threshold = float(config["method"]["params"]["k"])

    model = {
        "method": config["method"]["name"],
        "features": feats,
        "params": config["method"]["params"],
        "preprocessor": preprocessor,
        "estimator": None,  # zscore не требует обучаемого sklearn-объекта
        "state": state,
        "threshold": threshold,
    }

    model_dir = resolve_path(args.model_dir or config["paths"]["models_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")

    meta = {
        "method": model["method"],
        "variant": config["variant"],
        "features": feats,
        "preprocessing": {"missing": preprocessor["missing"], "transform": preprocessor["transform"],
                          "scaling": preprocessor["scaling"], "fit_on": preprocessor["fit_on"]},
        "params": model["params"],
        "threshold": threshold,
        "trained_rows": int(len(train_df)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (model_dir / "model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Обучено на {len(train_df)} строках, признаков: {len(feats)}, порог k={threshold}")
    print(f"Модель сохранена: {model_dir / 'model.joblib'}")
    print(f"Метаданные: {model_dir / 'model_meta.json'}")
    return EXIT_OK


def load_model(model_dir: Path) -> dict:
    path = model_dir / "model.joblib"
    if not path.is_file():
        raise DataError(f"модель не найдена: {path}. Сначала выполните train.")
    model = joblib.load(path)
    if not isinstance(model, dict) or "state" not in model:
        raise DataError(f"файл {path} не является моделью этого CLI")
    return model


def cmd_detect(config: dict, args: argparse.Namespace) -> int:
    model = load_model(resolve_path(args.model_dir or config["paths"]["models_dir"]))
    input_path = resolve_path(args.input)
    df = load_csv(input_path)
    require_columns(df, model["features"], input_path.name)

    work = filter_split(df, config["data"]["split_column"], config["data"]["test_split"], args.split)
    X = transform_features(work[model["features"]].astype(float), model["preprocessor"])
    score = zscore_scores(X, model["state"], model["params"]["aggregation"])

    out = df.loc[work.index].copy()  # все исходные колонки без изменений
    out["anomaly_score"] = score.round(6)
    out["is_anomaly"] = (score > float(model["threshold"])).astype(int)

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Строк оценено: {len(out)} (split={args.split})")
    print(f"Аномалий (is_anomaly=1): {int(out['is_anomaly'].sum())} "
          f"({out['is_anomaly'].mean():.2%}) при пороге {model['threshold']}")
    print(f"Предсказания сохранены: {output_path}")
    return EXIT_OK


def cmd_evaluate(config: dict, args: argparse.Namespace) -> int:
    pred_path = resolve_path(args.predictions)
    df = load_csv(pred_path)
    truth_col = args.ground_truth
    require_columns(df, [truth_col, "anomaly_score", "is_anomaly"], pred_path.name)

    work = filter_split(df, config["data"]["split_column"], config["data"]["test_split"], args.split)
    y_true = work[truth_col].astype(int)
    y_pred = work["is_anomaly"].astype(int)
    scores = work["anomaly_score"].astype(float)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    # ROC-AUC по ранговой формуле Манна-Уитни (эквивалент sklearn.roc_auc_score):
    # AUC = (сумма средних рангов позитивов - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    ranks = scores.rank(method="average")
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise DataError("ROC-AUC не определён: в выборке один класс по ground truth")
    rank_sum_pos = float(ranks[y_true == 1].sum())
    roc_auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    metrics = {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "roc_auc": round(roc_auc, 6),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "n_samples": int(len(work)),
        "n_anomalies": int(y_pred.sum()),
        "anomaly_rate": round(float(y_pred.mean()), 6),
        "variant": config["variant"],
        "method": config["method"]["name"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    reports_dir = resolve_path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    cm_md = (
        "# Confusion matrix (метод zscore, тестовый набор)\n\n"
        "| | Предсказано: норма (0) | Предсказано: аномалия (1) |\n"
        "|---|---|---|\n"
        f"| **Истина: норма (0)** | TN = {tn} | FP = {fp} |\n"
        f"| **Истина: атака (1)** | FN = {fn} | TP = {tp} |\n\n"
        f"- precision = {precision:.4f}, recall = {recall:.4f}, F1 = {f1:.4f}, ROC-AUC = {roc_auc:.4f}\n"
        f"- строк оценено: {len(work)}, доля флагов is_anomaly: {y_pred.mean():.2%}\n"
    )
    (reports_dir / "confusion_matrix.md").write_text(cm_md, encoding="utf-8")

    print(f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} roc_auc={roc_auc:.4f}")
    print(f"TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"Метрики сохранены: {reports_dir / 'metrics.json'}")
    print(f"Confusion matrix: {reports_dir / 'confusion_matrix.md'}")
    return EXIT_OK


# --------------------------------------------------------------------------
# Точка входа
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anomaly_cli.py", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p, with_model_dir=False):
        p.add_argument("--config", required=True, help="путь к config/config.yaml")
        if with_model_dir:
            p.add_argument("--model-dir", default=None, help="каталог модели")
        p.add_argument("--split", default="test", choices=["test", "all"],
                       help="фильтр по split (по умолчанию test; all отключает фильтр)")

    p = sub.add_parser("stats", help="статистика по train")
    add_common(p)
    p.add_argument("--output", default=None, help="куда сохранить stats.json")

    p = sub.add_parser("train", help="обучить модель на train")
    add_common(p, with_model_dir=True)

    p = sub.add_parser("detect", help="применить модель к CSV")
    add_common(p, with_model_dir=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)

    p = sub.add_parser("evaluate", help="метрики по предсказаниям")
    add_common(p)
    p.add_argument("--predictions", required=True)
    p.add_argument("--ground-truth", default="label")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(resolve_path(args.config))
        if args.command == "stats":
            return cmd_stats(config, args)
        if args.command == "train":
            return cmd_train(config, args)
        if args.command == "detect":
            return cmd_detect(config, args)
        if args.command == "evaluate":
            return cmd_evaluate(config, args)
        parser.error(f"неизвестная команда: {args.command}")
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except DataError as exc:
        print(f"Ошибка данных: {exc}", file=sys.stderr)
        return EXIT_DATA
    except KeyboardInterrupt:
        print("Прервано пользователем", file=sys.stderr)
        return 130
    except Exception as exc:  # непредвиденное — без traceback в обычном режиме
        print(f"Непредвиденная ошибка: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_DATA


if __name__ == "__main__":
    raise SystemExit(main())
