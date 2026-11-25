"""Pruebas del módulo ML tabular: coherencia intra-tramo, sin fugas, rangos e incertidumbre."""
import pandas as pd
import pytest

from src.data.synthetic import generate_synthetic_tramos
from src.ml.baseline import SeverityTrafficRuleBaseline
from src.ml.priority import calcular_prioridad


def test_synthetic_intra_tramo_consistency():
    df = generate_synthetic_tramos()
    assert len(df) == 360
    assert df["id_tramo"].nunique() == 120
    # Cada tramo mantiene un único tipo_via
    per_tramo_road = df.groupby("id_tramo")["tipo_via"].nunique()
    assert (per_tramo_road == 1).all()
    # La antigüedad progresa de forma monotónica no decreciente dentro de cada tramo
    for _, grp in df.groupby("id_tramo"):
        assert grp["antiguedad_anios"].is_monotonic_increasing


def test_baseline_rule_learns_thresholds_without_leakage():
    df = generate_synthetic_tramos()
    base = SeverityTrafficRuleBaseline().fit(df.drop(columns=["prioridad_validada"]), df["prioridad_validada"])
    preds = base.predict(df)
    probs = base.predict_proba(df)
    assert len(preds) == len(df)
    assert probs.shape == (len(df), 3)
    assert set(preds).issubset({"baja", "media", "alta"})


def test_calcular_prioridad_ranges_and_monotonic_traffic():
    base_vals = dict(
        trafico_diario=6500,
        antiguedad_anios=14,
        reparaciones_24m=3,
        tipo_via="colectora",
        precipitacion_mm_30d=25,
        severidad_inspeccionada=2,
    )
    low = calcular_prioridad({**base_vals, "trafico_diario": 500}, True)
    mid = calcular_prioridad(base_vals, True)
    high = calcular_prioridad({**base_vals, "trafico_diario": 9800}, True)

    assert low["puntaje"] < mid["puntaje"] < high["puntaje"]
    assert mid["nivel_incertidumbre"] in {"baja", "media", "alta"}
    assert mid["factores"]

    with pytest.raises(ValueError):
        calcular_prioridad({**base_vals, "trafico_diario": -10}, True)
    with pytest.raises(ValueError):
        calcular_prioridad({**base_vals, "tipo_via": "autopista"}, True)
    with pytest.raises(ValueError):
        calcular_prioridad(base_vals, False)
