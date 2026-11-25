"""Evaluación cuantitativa por calle/recorrido y análisis de errores visuales."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.config import get_logger
from src.vision.detector import detectar_danos, draw_comparison_boxes, iou, read_image

logger = get_logger(__name__)


def _contrast_baseline(raw: bytes) -> list[dict]:
    """Referencia heurística simple (sin entrenamiento) basada en contraste oscuro local."""
    img = read_image(raw).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    threshold = max(20.0, float(np.mean(arr) - 2.0 * np.std(arr)))
    mask = arr < threshold
    if mask.sum() < 80:
        return []
    ys, xs = np.where(mask)
    box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    if box[2] <= box[0] or box[3] <= box[1]:
        return []
    return [{"clase": "Bache", "caja": box, "confianza": 0.30}]


def _score_predictions(frames: list[dict], preds_by_frame: list[list[dict]], conf_eval: float) -> dict:
    actual, predictions = {}, {}
    for i, frame in enumerate(frames):
        for box in frame["cajas"]:
            actual.setdefault(box["clase"], {}).setdefault(i, []).append(box["caja"])
        for item in preds_by_frame[i]:
            predictions.setdefault(item["clase"], []).append((i, item))

    aps, recall, precision = {}, {}, {}
    false_alarms = {f["recorrido"]: 0 for f in frames}
    for label in sorted(set(actual) | set(predictions)):
        gt = actual.get(label, {})
        count = sum(map(len, gt.values()))
        used, flags, selected = set(), [], []
        for i, item in sorted(
            predictions.get(label, []),
            key=lambda entry: entry[1]["confianza"],
            reverse=True,
        ):
            overlaps = [
                (iou(item["caja"], b), j)
                for j, b in enumerate(gt.get(i, []))
                if (i, j) not in used
            ]
            match = max(overlaps, default=(0.0, -1))
            positive = match[0] >= 0.5
            if positive:
                used.add((i, match[1]))
            flags.append(int(positive))
            if item["confianza"] >= conf_eval:
                selected.append(int(positive))
                false_alarms[frames[i]["recorrido"]] += int(not positive)
        if count:
            tps = np.cumsum(flags)
            recalls = tps / count
            prec = tps / np.arange(1, len(flags) + 1)
            aps[label] = round(
                float(np.mean([max(prec[recalls >= r], default=0.0) for r in np.linspace(0, 1, 101)])),
                4,
            )
            recall[label] = round(sum(selected) / count, 4)
            precision[label] = round((sum(selected) / len(selected)) if selected else 0.0, 4)

    return {
        "mAP50": round(float(np.mean(list(aps.values()))), 4) if aps else None,
        "ap50": aps,
        "recall": recall,
        "precision": precision,
        "falsas_alarmas_por_recorrido": false_alarms,
    }


def _save_case_study_figure(frames: list[dict], manifest_dir: Path, figure_path: Path) -> None:
    fig, axes = plt.subplots(1, len(frames), figsize=(5.5 * len(frames), 4.2))
    if len(frames) == 1:
        axes = [axes]
    for ax, frame in zip(axes, frames):
        raw = (manifest_dir / frame["imagen"]).read_bytes()
        pred = detectar_danos(raw, confidence=0.25)["danos"]
        annotated = draw_comparison_boxes(raw, frame["cajas"], pred)
        ax.imshow(annotated)
        ax.set_title(f"{frame['imagen']}\n(calle: {frame['calle']} | {len(pred)} pred / {len(frame['cajas'])} GT)", fontsize=9)
        ax.axis("off")
    fig.suptitle("Evaluación visual: Ground Truth (verde) vs YOLO11n (cian=Bache, rojo=Grieta)", fontsize=10)
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def evaluate_manifest(
    manifest_path: Path,
    output_json: Path | None = None,
    figure_path: Path | None = None,
) -> dict:
    """Evalúa detecciones contra cajas verificadas de un manifiesto JSON."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        frames = payload.get("muestras", [])
        scope_note = payload.get("alcance", "evaluación por manifiesto")
    else:
        frames = payload
        scope_note = "evaluación por manifiesto"

    if not frames or any(not f.get("calle") or not f.get("recorrido") for f in frames):
        raise ValueError("Cada foto debe tener calle y recorrido identificados.")
    if len({f["imagen"] for f in frames}) != len(frames):
        raise ValueError("Hay entradas de imagen repetidas.")

    preds_yolo_low, preds_yolo_25, preds_yolo_50, preds_baseline = [], [], [], []
    image_details = []

    for frame in frames:
        raw = (manifest_path.parent / frame["imagen"]).read_bytes()
        res_low = detectar_danos(raw, confidence=0.01)
        if res_low["estado"].startswith("no evaluable"):
            raise ValueError(res_low["estado"])
        all_preds = res_low["danos"]
        p25 = [p for p in all_preds if p["confianza"] >= 0.25]
        p50 = [p for p in all_preds if p["confianza"] >= 0.50]
        p_base = _contrast_baseline(raw)

        preds_yolo_low.append(all_preds)
        preds_yolo_25.append(p25)
        preds_yolo_50.append(p50)
        preds_baseline.append(p_base)

        gt_boxes = frame["cajas"]
        best_ious = []
        for g in gt_boxes:
            same_cls = [iou(g["caja"], p["caja"]) for p in p25 if p["clase"] == g["clase"]]
            any_cls = [iou(g["caja"], p["caja"]) for p in p25]
            best_ious.append(
                {
                    "gt_clase": g["clase"],
                    "gt_caja": g["caja"],
                    "mejor_iou_misma_clase": round(max(same_cls, default=0.0), 4),
                    "mejor_iou_cualquier_clase": round(max(any_cls, default=0.0), 4),
                }
            )
        image_details.append(
            {
                "imagen": frame["imagen"],
                "calle": frame["calle"],
                "recorrido": frame["recorrido"],
                "gt_count": len(gt_boxes),
                "predicciones_025": p25,
                "iou_por_gt": best_ious,
                "nota_caso": frame.get("nota", ""),
            }
        )

    eval_025 = _score_predictions(frames, preds_yolo_low, conf_eval=0.25)
    eval_050 = _score_predictions(frames, preds_yolo_low, conf_eval=0.50)
    eval_base = _score_predictions(frames, preds_baseline, conf_eval=0.25)

    if figure_path is not None:
        _save_case_study_figure(frames, manifest_path.parent, figure_path)

    report = {
        "alcance": scope_note,
        "imagenes": len(frames),
        "mAP50": eval_025["mAP50"],
        "ap50": eval_025["ap50"],
        "recall_umbral_025": eval_025["recall"],
        "precision_umbral_025": eval_025["precision"],
        "falsas_alarmas_por_recorrido": eval_025["falsas_alarmas_por_recorrido"],
        "comparativa_modelos": {
            "yolo11n_umbral_025": eval_025,
            "yolo11n_umbral_050": eval_050,
            "baseline_contraste_local": eval_base,
        },
        "detalle_por_imagen": image_details,
    }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Evaluación visual finalizada: imagenes=%d mAP50=%s", len(frames), report["mAP50"])
    return report
