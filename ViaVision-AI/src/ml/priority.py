"""Inferencia de prioridad vial, incertidumbre y factores explicativos."""
from functools import lru_cache
import math

import joblib
import pandas as pd

from src.config import ROOT, get_logger, load_settings

logger = get_logger(__name__)

_ML_CFG = load_settings().get("ml", {})
FEATURES = _ML_CFG.get(
    "features",
    [
        "trafico_diario",
        "antiguedad_anios",
        "reparaciones_24m",
        "tipo_via",
        "precipitacion_mm_30d",
        "severidad_inspeccionada",
    ],
)
CLASSES = _ML_CFG.get("classes", ["baja", "media", "alta"])
RANGES = {
    k: tuple(v)
    for k, v in _ML_CFG.get(
        "ranges",
        {
            "trafico_diario": [0, 100000],
            "antiguedad_anios": [0, 100],
            "reparaciones_24m": [0, 100],
            "precipitacion_mm_30d": [0, 5000],
            "severidad_inspeccionada": [0, 3],
        },
    ).items()
}
VIA = _ML_CFG.get("road_types", ["local", "colectora", "arterial"])


@lru_cache(maxsize=8)
def _load_cached(path_str: str, mtime_ns: int):
    return joblib.load(path_str)


def load_model(path):
    """Carga un modelo joblib invalidando caché si cambia el archivo en disco."""
    p = ROOT / path if not hasattr(path, "stat") else path
    return _load_cached(str(p), p.stat().st_mtime_ns)


def _explain_factors(datos: dict) -> list[str]:
    factores: list[str] = []
    if float(datos["severidad_inspeccionada"]) >= 2:
        factores.append(f"Severidad inspeccionada nivel {int(datos['severidad_inspeccionada'])}/3")
    if float(datos["trafico_diario"]) >= 6000:
        factores.append(f"Flujo vehicular elevado ({int(datos['trafico_diario'])} veh/día)")
    elif float(datos["trafico_diario"]) <= 1500:
        factores.append(f"Flujo vehicular moderado-bajo ({int(datos['trafico_diario'])} veh/día)")
    if float(datos["antiguedad_anios"]) >= 12:
        factores.append(f"Pavimento antiguo ({int(datos['antiguedad_anios'])} años)")
    if float(datos["reparaciones_24m"]) >= 3:
        factores.append(f"Historial frecuente de parches ({int(datos['reparaciones_24m'])} en 24 meses)")
    if float(datos["precipitacion_mm_30d"]) >= 60:
        factores.append(f"Precipitación acumulada relevante ({float(datos['precipitacion_mm_30d']):.1f} mm/30d)")
    if datos["tipo_via"] == "arterial":
        factores.append("Jerarquía arterial de alta exposición")
    if not factores:
        factores.append(f"Vía {datos['tipo_via']} sin factores extremos aislados")
    return factores


def _uncertainty_metrics(probs: dict[str, float]) -> tuple[float, float, str]:
    ordered = sorted(probs.values(), reverse=True)
    margin = (ordered[0] - ordered[1]) if len(ordered) >= 2 else 1.0
    entropy = -sum(p * math.log(p) for p in ordered if p > 0.0)
    norm_entropy = entropy / math.log(max(len(CLASSES), 2))
    margin_thr = float(_ML_CFG.get("uncertainty_margin_threshold", 0.15))
    if margin < margin_thr:
        level = "alta"
    elif margin < 0.35:
        level = "media"
    else:
        level = "baja"
    return round(float(margin), 4), round(float(norm_entropy), 4), level


def calcular_prioridad(datos: dict, modo_demo: bool = True) -> dict:
    """Valida variables tabulares y predice prioridad, puntaje, incertidumbre y factores."""
    if any(k not in datos for k in FEATURES):
        raise ValueError("Faltan variables del tramo.")
    for name, (low, high) in RANGES.items():
        val = float(datos[name])
        if math.isnan(val) or not low <= val <= high:
            raise ValueError(f"{name} debe estar entre {low} y {high}.")
    if datos["tipo_via"] not in VIA:
        raise ValueError("Tipo de vía desconocido.")

    path = ROOT / "models" / ("demo.joblib" if modo_demo else "validado.joblib")
    if not path.is_file():
        logger.warning("Modelo no encontrado en %s (modo_demo=%s)", path, modo_demo)
        raise ValueError("Instala primero un modelo entrenado con registros validados.")

    model = load_model(path)
    frame = pd.DataFrame([datos], columns=FEATURES)
    probs = dict(zip(model.classes_, model.predict_proba(frame)[0]))
    probs_clean = {str(k): round(float(v), 4) for k, v in probs.items()}
    expected = sum(i * probs.get(c, 0.0) for i, c in enumerate(CLASSES)) * 50.0
    margin, norm_entropy, unc_level = _uncertainty_metrics(probs_clean)
    clase = str(model.predict(frame)[0])

    result = {
        "clase": clase,
        "puntaje": round(float(expected), 1),
        "probabilidades": probs_clean,
        "margen_probabilidad": margin,
        "entropia_normalizada": norm_entropy,
        "nivel_incertidumbre": unc_level,
        "factores": _explain_factors(datos),
        "origen": "entrenamiento sintético" if modo_demo else "modelo con registros validados",
        "variables": dict(datos),
    }
    logger.info(
        "Prioridad calculada: clase=%s puntaje=%.1f incertidumbre=%s demo=%s",
        clase,
        result["puntaje"],
        unc_level,
        modo_demo,
    )
    return result
