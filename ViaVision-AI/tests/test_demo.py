"""Pruebas integradas del grafo LangGraph sin sobrescribir artefactos en results/."""
from scripts.evaluate_demo import run_demo_suite


def test_piloto_local_sin_mutar_resultados():
    report = run_demo_suite(save_outputs=False)
    assert all(report["pruebas"].values())
    assert report["traza"] == [
        "verificar_metadatos",
        "calcular_prioridad",
        "detectar_danos",
        "recuperar_normativa",
        "evaluar_incertidumbre",
        "revision_humana",
        "registrar_inspeccion_preliminar",
    ]
