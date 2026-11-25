"""Detección de deterioros viales mediante YOLO11n exportado a ONNX."""
from functools import lru_cache
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.config import ROOT, get_logger, load_settings

logger = get_logger(__name__)

LABELS = ["Grieta longitudinal", "Grieta transversal", "Fisuración por fatiga", "Bache"]


def _vision_cfg() -> dict:
    return load_settings().get("vision", {})


def read_image(raw: bytes) -> Image.Image:
    """Abre, valida dimensiones y normaliza orientación EXIF a RGB."""
    image = Image.open(BytesIO(raw))
    if max(image.size) > 8000 or min(image.size) < 32:
        raise ValueError("Carga una imagen entre 32 y 8000 píxeles por lado.")
    return ImageOps.exif_transpose(image).convert("RGB")


@lru_cache(maxsize=2)
def session():
    """Carga y cachea la sesión de inferencia ONNX en CPU."""
    import onnxruntime as ort

    rel = _vision_cfg().get("model_path", "models/road_damage.onnx")
    model_path = ROOT / rel
    logger.info("Cargando detector ONNX desde %s", model_path)
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def iou(a: list[int] | list[float], b: list[int] | list[float]) -> float:
    """Calcula Intersection over Union entre dos cajas [x1, y1, x2, y2]."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return float(w * h / union) if union > 0 else 0.0


def _visual_uncertainty(selected: list[dict]) -> str:
    thr = float(_vision_cfg().get("uncertainty_confidence_threshold", 0.45))
    if not selected:
        return "alta"
    max_conf = max(item["confianza"] for item in selected)
    min_conf = min(item["confianza"] for item in selected)
    if max_conf < thr:
        return "alta"
    if min_conf < thr:
        return "media"
    return "baja"


def detectar_danos(raw: bytes, confidence: float | None = None) -> dict:
    """Ejecuta YOLO11n ONNX con letterbox 640x640 y NMS por clase."""
    cfg = _vision_cfg()
    conf_thr = float(cfg.get("confidence_threshold", 0.25) if confidence is None else confidence)
    iou_thr = float(cfg.get("iou_threshold", 0.50))
    max_det = int(cfg.get("max_detections", 50))
    input_size = int(cfg.get("input_size", 640))
    excluded = set(cfg.get("excluded_classes", ["Tapa de registro"]))

    image = read_image(raw)
    model_path = ROOT / cfg.get("model_path", "models/road_damage.onnx")
    if not model_path.is_file():
        logger.error("Detector ONNX no encontrado en %s", model_path)
        return {
            "estado": "no evaluable: detector ONNX no instalado",
            "danos": [],
            "umbral_confianza": conf_thr,
            "incertidumbre_visual": "alta",
        }

    model = session()
    scale = min(input_size / image.width, input_size / image.height)
    w, h = round(image.width * scale), round(image.height * scale)
    px, py = (input_size - w) // 2, (input_size - h) // 2

    canvas = Image.new("RGB", (input_size, input_size), (114, 114, 114))
    canvas.paste(image.resize((w, h), Image.Resampling.BILINEAR), (px, py))
    pixels = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0

    output = model.run(None, {model.get_inputs()[0].name: pixels})[0][0]
    if output.ndim != 2 or output.shape[0] not in (7, 8):
        raise ValueError("El modelo debe exportar YOLO ONNX sin NMS de tres o cuatro clases.")

    labels = ["Bache", "Grieta", "Tapa de registro"] if output.shape[0] == 7 else LABELS
    candidates = []
    for x, y, bw, bh, *scores in output.T:
        cls = int(np.argmax(scores))
        score = float(scores[cls])
        if score < conf_thr or labels[cls] in excluded:
            continue
        box = [
            (x - bw / 2 - px) / scale,
            (y - bh / 2 - py) / scale,
            (x + bw / 2 - px) / scale,
            (y + bh / 2 - py) / scale,
        ]
        box = [int(np.clip(v, 0, image.size[i % 2] - 1)) for i, v in enumerate(box)]
        if box[2] > box[0] and box[3] > box[1]:
            candidates.append({"clase": labels[cls], "caja": box, "confianza": score})

    selected = []
    for item in sorted(candidates, key=lambda d: d["confianza"], reverse=True)[:1000]:
        if all(old["clase"] != item["clase"] or iou(old["caja"], item["caja"]) < iou_thr for old in selected):
            selected.append(item)
        if len(selected) == max_det:
            break

    estado = "evaluada" if selected else "sin hallazgos sobre el umbral"
    unc = _visual_uncertainty(selected)
    logger.info("Visión ejecutada: estado=%s, hallazgos=%d, incertidumbre=%s", estado, len(selected), unc)
    return {
        "estado": estado,
        "danos": selected,
        "umbral_confianza": conf_thr,
        "incertidumbre_visual": unc,
    }


def draw_boxes(raw: bytes, detections: list[dict]) -> Image.Image:
    """Dibuja cajas de deterioro y confianza sobre la imagen de entrada."""
    image = read_image(raw)
    draw = ImageDraw.Draw(image)
    for item in detections:
        x, y, _, _ = item["caja"]
        draw.rectangle(item["caja"], outline="#00ffd1", width=max(2, image.width // 200))
        draw.text(
            (x, max(y - 13, 0)),
            f"{item['clase']} {item['confianza']:.0%}",
            fill="#00ffd1",
            stroke_width=2,
            stroke_fill="black",
        )
    return image


def draw_comparison_boxes(raw: bytes, gt_boxes: list[dict], pred_boxes: list[dict]) -> Image.Image:
    """Dibuja cajas Ground Truth (verde) frente a predicciones YOLO (rojo/cian)."""
    image = read_image(raw)
    draw = ImageDraw.Draw(image)
    w = max(2, image.width // 180)
    for item in gt_boxes:
        x, y, _, _ = item["caja"]
        draw.rectangle(item["caja"], outline="#22c55e", width=w)
        draw.text((x, max(y - 13, 0)), f"GT: {item['clase']}", fill="#22c55e", stroke_width=2, stroke_fill="black")
    for item in pred_boxes:
        x, _, _, y2 = item["caja"]
        color = "#00ffd1" if item["clase"] == "Bache" else "#ff4d4f"
        draw.rectangle(item["caja"], outline=color, width=w)
        draw.text(
            (x, min(y2 + 2, max(0, image.height - 14))),
            f"Pred: {item['clase']} {item['confianza']:.0%}",
            fill=color,
            stroke_width=2,
            stroke_fill="black",
        )
    return image
