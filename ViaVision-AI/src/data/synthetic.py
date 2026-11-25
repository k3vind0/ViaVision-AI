"""Generación sintética coherente por tramo y validación de datos tabulares."""
import numpy as np
import pandas as pd

from src.config import get_logger, load_settings

logger = get_logger(__name__)

_TRAFFIC_BY_ROAD = {
    "local": (300, 3200),
    "colectora": (2500, 7200),
    "arterial": (5500, 10200),
}


def generate_synthetic_tramos(
    n_tramos: int = 120,
    obs_per_tramo: int = 3,
    seed: int | None = None,
) -> pd.DataFrame:
    """Genera observaciones sintéticas agrupadas manteniendo coherencia física por tramo.

    Cada id_tramo conserva constante su tipo_via y una progresión consistente de
    antigüedad y rango de tráfico vehicular entre inspecciones.
    """
    settings = load_settings().get("ml", {})
    classes = settings.get("classes", ["baja", "media", "alta"])
    seed = settings.get("seed", 42) if seed is None else seed
    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    road_choices = ["local", "colectora", "arterial"]
    road_probs = [0.40, 0.35, 0.25]

    for idx in range(n_tramos):
        tramo_id = f"SIM-{idx:03d}"
        road_type = str(rng.choice(road_choices, p=road_probs))
        low_t, high_t = _TRAFFIC_BY_ROAD[road_type]
        base_traffic = float(rng.integers(low_t, high_t))
        base_age = int(rng.integers(1, 31))

        for obs_idx in range(obs_per_tramo):
            traffic = int(np.clip(round(base_traffic * rng.uniform(0.88, 1.12)), 100, 10500))
            age = int(np.clip(base_age + obs_idx, 0, 100))
            repairs = int(rng.integers(0, 8))
            rain = int(rng.integers(0, 160))
            severity = int(rng.integers(0, 4))

            road_bonus = {"local": 0.0, "colectora": 0.35, "arterial": 0.65}[road_type]
            score = (
                traffic / 2800.0
                + age / 18.0
                + repairs / 5.5
                + rain / 180.0
                + severity * 1.10
                + road_bonus
                + float(rng.normal(0.0, 0.55))
            )
            cls_idx = int(np.digitize(score, [4.35, 6.65]))
            rows.append(
                {
                    "id_tramo": tramo_id,
                    "trafico_diario": traffic,
                    "antiguedad_anios": age,
                    "reparaciones_24m": repairs,
                    "tipo_via": road_type,
                    "precipitacion_mm_30d": rain,
                    "severidad_inspeccionada": severity,
                    "prioridad_validada": classes[cls_idx],
                }
            )

    frame = pd.DataFrame(rows)
    logger.info(
        "Dataset sintético generado: %d filas, %d tramos, distribución=%s",
        len(frame),
        n_tramos,
        frame["prioridad_validada"].value_counts().to_dict(),
    )
    return frame


def validate_tabular_frame(frame: pd.DataFrame, group_col: str = "id_tramo") -> pd.DataFrame:
    """Valida columnas requeridas, nulos en grupo/objetivo y presencia de clases."""
    settings = load_settings().get("ml", {})
    features = settings.get(
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
    classes = settings.get("classes", ["baja", "media", "alta"])
    target = "prioridad_validada"

    required = set(features) | {group_col, target}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {sorted(missing)}")

    cleaned = frame.drop_duplicates().reset_index(drop=True)
    if cleaned[[group_col, target]].isna().any().any():
        raise ValueError("Existen valores nulos en la columna de agrupación o en prioridad_validada.")
    if set(cleaned[target]) != set(classes):
        raise ValueError("Se requieren las tres etiquetas: baja, media y alta.")
    return cleaned
