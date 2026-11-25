"""Entrenamiento, validación cruzada agrupada, evaluación y gráficos del módulo ML."""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ROOT, get_logger, load_settings
from src.data.synthetic import validate_tabular_frame
from src.ml.baseline import SeverityTrafficRuleBaseline
from src.ml.priority import CLASSES, FEATURES

logger = get_logger(__name__)


def _eval_split(estimator, X_split: pd.DataFrame, y_split: pd.Series) -> dict:
    preds = estimator.predict(X_split)
    actual_idx = y_split.map(CLASSES.index).to_numpy()
    pred_idx = np.array([CLASSES.index(p) for p in preds])
    return {
        "f1_macro": round(
            float(f1_score(y_split, preds, labels=CLASSES, average="macro", zero_division=0)),
            4,
        ),
        "mae_ordinal": round(float(mean_absolute_error(actual_idx, pred_idx)), 4),
        "exactitud": round(float(accuracy_score(y_split, preds)), 4),
        "matriz": confusion_matrix(y_split, preds, labels=CLASSES).tolist(),
    }


def _subgroup_analysis(estimator, X_test: pd.DataFrame, y_test: pd.Series) -> list[dict]:
    preds = pd.Series(estimator.predict(X_test), index=X_test.index)
    rows: list[dict] = []

    for col, values in [
        ("tipo_via", sorted(X_test["tipo_via"].unique())),
        ("severidad_inspeccionada", sorted(X_test["severidad_inspeccionada"].unique())),
    ]:
        for val in values:
            mask = X_test[col] == val
            if not mask.any():
                continue
            yt = y_test[mask]
            yp = preds[mask]
            actual_idx = yt.map(CLASSES.index).to_numpy()
            pred_idx = np.array([CLASSES.index(p) for p in yp])
            rows.append(
                {
                    "subgrupo": f"{col}={val}",
                    "n": int(mask.sum()),
                    "f1_macro": round(
                        float(f1_score(yt, yp, labels=CLASSES, average="macro", zero_division=0)),
                        4,
                    ),
                    "mae_ordinal": round(float(mean_absolute_error(actual_idx, pred_idx)), 4),
                    "exactitud": round(float(accuracy_score(yt, yp)), 4),
                }
            )
    return rows


def _save_figures(scores: dict, best_name: str, subgroups: list[dict]) -> None:
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Comparación Train / Validación / Test
    names = list(scores.keys())
    f1_train = [scores[n]["f1_train"] for n in names]
    f1_val = [scores[n]["f1_validacion"] for n in names]
    f1_test = [scores[n]["f1_test"] for n in names]

    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w, f1_train, width=w, label="Train", color="#8ab0ab")
    ax.bar(x, f1_val, width=w, label="Validación CV", color="#006e65")
    ax.bar(x + w, f1_test, width=w, label="Test intacto", color="#122c2b")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Comparación de modelos tabulares (Train / Validación / Test)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(results_dir / "fig_ml_comparacion.png", dpi=150)
    plt.close(fig)

    # 2. Matriz de confusión del mejor modelo
    cm = np.array(scores[best_name]["matriz"])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="YlGn")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels([c.capitalize() for c in CLASSES])
    ax.set_yticklabels([c.capitalize() for c in CLASSES])
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión en test ({best_name})")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "#122c2b", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(results_dir / "fig_ml_matriz_confusion.png", dpi=150)
    plt.close(fig)

    # 3. Análisis por subgrupo en test
    sub_df = pd.DataFrame(subgroups)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.barh(sub_df["subgrupo"], sub_df["mae_ordinal"], color="#006e65")
    ax.set_xlabel("MAE ordinal en test (menor es mejor)")
    ax.set_title(f"Error ordinal por subgrupo en test ({best_name})")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(results_dir / "fig_ml_subgrupos.png", dpi=150)
    plt.close(fig)


def train_priority_models(frame: pd.DataFrame, demo: bool = True, group: str = "id_tramo") -> dict:
    """Entrena baselines y candidatos con partición por grupos y cero fugas de datos."""
    settings = load_settings().get("ml", {})
    seed = int(settings.get("seed", 42))
    test_size = float(settings.get("test_size", 0.25))
    cv_splits = int(settings.get("cv_splits", 4))

    cleaned = validate_tabular_frame(frame, group)
    x, y, groups = cleaned[FEATURES], cleaned["prioridad_validada"], cleaned[group]
    train_rows, test_rows = next(
        GroupShuffleSplit(test_size=test_size, random_state=seed).split(x, y, groups)
    )
    if any(set(y.iloc[rows]) != set(CLASSES) for rows in (train_rows, test_rows)):
        raise ValueError("Cada partición debe contener las tres clases.")

    x_train, y_train, g_train = x.iloc[train_rows], y.iloc[train_rows], groups.iloc[train_rows]
    x_test, y_test = x.iloc[test_rows], y.iloc[test_rows]

    numeric = [f for f in FEATURES if f != "tipo_via"]
    preprocess = ColumnTransformer(
        [
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            (
                "category",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                ["tipo_via"],
            ),
        ]
    )
    cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    specs = {
        "regla_baseline": (SeverityTrafficRuleBaseline(), {}, False),
        "dummy_frecuente": (make_pipeline(preprocess, DummyClassifier(strategy="most_frequent")), {}, False),
        "regresion_logistica": (
            make_pipeline(preprocess, LogisticRegression(max_iter=1000)),
            {"logisticregression__C": [0.2, 1.0, 5.0]},
            True,
        ),
        "random_forest": (
            make_pipeline(preprocess, RandomForestClassifier(n_estimators=100, random_state=seed)),
            {"randomforestclassifier__max_depth": [4, 8, None]},
            True,
        ),
        "gradient_boosting": (
            make_pipeline(preprocess, GradientBoostingClassifier(random_state=seed)),
            {
                "gradientboostingclassifier__max_depth": [2, 3],
                "gradientboostingclassifier__learning_rate": [0.05, 0.1],
                "gradientboostingclassifier__n_estimators": [60, 120],
            },
            True,
        ),
    }

    models, scores = {}, {}
    for name, (estimator, grid, selectable) in specs.items():
        search = GridSearchCV(
            estimator,
            grid,
            scoring="f1_macro",
            cv=cv,
            error_score="raise",
        )
        search.fit(x_train, y_train, groups=g_train)
        best_est = search.best_estimator_
        tr_metrics = _eval_split(best_est, x_train, y_train)
        te_metrics = _eval_split(best_est, x_test, y_test)
        best_idx = int(search.best_index_)
        cv_std = float(search.cv_results_["std_test_score"][best_idx])

        models[name] = best_est
        scores[name] = {
            "seleccionable": selectable,
            "f1_train": tr_metrics["f1_macro"],
            "mae_train": tr_metrics["mae_ordinal"],
            "f1_validacion": round(float(search.best_score_), 4),
            "f1_validacion_std": round(cv_std, 4),
            "parametros": search.best_params_,
            "f1_test": te_metrics["f1_macro"],
            "mae_ordinal": te_metrics["mae_ordinal"],
            "exactitud_test": te_metrics["exactitud"],
            "matriz": te_metrics["matriz"],
        }

    candidates = [k for k, v in scores.items() if v["seleccionable"]]
    best_name = max(candidates, key=lambda k: scores[k]["f1_validacion"])

    suffix = "demo" if demo else "validado"
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    joblib.dump(models[best_name], ROOT / "models" / f"{suffix}.joblib")

    subgroups = _subgroup_analysis(models[best_name], x_test, y_test)
    _save_figures(scores, best_name, subgroups)

    comparison_rows = [
        {
            "modelo": k,
            "tipo": "candidato" if v["seleccionable"] else "baseline",
            "f1_train": v["f1_train"],
            "f1_validacion": v["f1_validacion"],
            "f1_validacion_std": v["f1_validacion_std"],
            "f1_test": v["f1_test"],
            "mae_train": v["mae_train"],
            "mae_ordinal_test": v["mae_ordinal"],
            "exactitud_test": v["exactitud_test"],
        }
        for k, v in scores.items()
    ]
    pd.DataFrame(comparison_rows).to_csv(
        ROOT / "results" / f"ml_tabla_comparativa{'' if demo else '_validado'}.csv",
        index=False,
    )
    pd.DataFrame(subgroups).to_csv(
        ROOT / "results" / f"ml_subgrupos{'' if demo else '_validado'}.csv",
        index=False,
    )

    cleaned.assign(particion=np.where(cleaned.index.isin(test_rows), "test", "train"))[
        [group, "particion"]
    ].drop_duplicates().to_csv(ROOT / "results" / f"split_{suffix}.csv", index=False)

    report = {
        "origen": "sintético" if demo else "archivo aportado",
        "modelo": best_name,
        "clases": CLASSES,
        "train": int(len(train_rows)),
        "test": int(len(test_rows)),
        "grupos_train": int(g_train.nunique()),
        "grupos_test": int(groups.iloc[test_rows].nunique()),
        "resultados": scores,
        "subgrupos_test": subgroups,
    }
    (ROOT / "results" / f"ml_{suffix}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Entrenamiento ML completado (%s): ganador=%s, f1_val=%.4f, f1_test=%.4f",
        suffix,
        best_name,
        scores[best_name]["f1_validacion"],
        scores[best_name]["f1_test"],
    )
    return report
