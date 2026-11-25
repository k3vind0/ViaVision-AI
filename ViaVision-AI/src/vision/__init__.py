"""Módulo de visión computacional (detección YOLO ONNX y evaluación de cajas)."""
from src.vision.detector import LABELS, detectar_danos, draw_boxes, draw_comparison_boxes, iou, read_image, session
from src.vision.evaluator import evaluate_manifest

__all__ = [
    "LABELS",
    "detectar_danos",
    "draw_boxes",
    "draw_comparison_boxes",
    "evaluate_manifest",
    "iou",
    "read_image",
    "session",
]
