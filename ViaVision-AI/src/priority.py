"""Compatibilidad hacia atrás: re-exporta src.ml.priority."""
from src.ml.priority import CLASSES, FEATURES, RANGES, ROOT, VIA, calcular_prioridad, load_model

__all__ = ["CLASSES", "FEATURES", "RANGES", "ROOT", "VIA", "calcular_prioridad", "load_model"]
