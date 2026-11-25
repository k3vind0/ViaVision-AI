"""Pruebas del módulo de visión computacional y evaluación por manifiesto."""
from src.config import ROOT
from src.vision.detector import detectar_danos
from src.vision.evaluator import evaluate_manifest


def test_yolo_demo_and_fhwa_error_case():
    demo_raw = (ROOT / "data_sample" / "tramo-ejemplo.jpg").read_bytes()
    res_demo = detectar_danos(demo_raw)
    assert res_demo["estado"] == "evaluada"
    assert len(res_demo["danos"]) == 2
    assert all(d["clase"] == "Bache" for d in res_demo["danos"])

    fhwa_raw = (ROOT / "data_sample" / "fhwa-bache.jpg").read_bytes()
    res_fhwa = detectar_danos(fhwa_raw)
    assert res_fhwa["estado"] == "evaluada"
    assert res_fhwa["danos"][0]["clase"] == "Grieta"
    assert res_fhwa["incertidumbre_visual"] in {"media", "alta"}


def test_evaluate_vision_manifest_without_writing_repo():
    manifest = ROOT / "data_sample" / "vision_demo_manifest.json"
    report = evaluate_manifest(manifest)
    assert report["imagenes"] == 2
    assert "Bache" in report["ap50"]
    assert "comparativa_modelos" in report
    assert len(report["detalle_por_imagen"]) == 2
