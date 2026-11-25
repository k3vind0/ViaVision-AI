"""Configuración centralizada y logging estructurado para VíaVision AI."""
from functools import lru_cache
import logging
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=4)
def load_settings(config_path: str | None = None) -> dict:
    """Carga la configuración YAML del proyecto desde configs/settings.yaml."""
    rel = config_path or os.getenv("VIAVISION_CONFIG", "configs/settings.yaml")
    path = Path(rel)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def get_logger(name: str) -> logging.Logger:
    """Devuelve un logger configurado con salida a archivo y consola."""
    settings = load_settings()
    log_cfg = settings.get("logging", {})
    level_name = os.getenv("VIAVISION_LOG_LEVEL") or log_cfg.get("level", "INFO")
    level = getattr(logging, str(level_name).upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    log_file = ROOT / log_cfg.get("file", "logs/viavision.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        pass

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger
