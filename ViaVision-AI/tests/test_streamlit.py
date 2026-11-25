"""Pruebas automatizadas de la interfaz Streamlit con AppTest aislado."""
import json
import re

from streamlit.testing.v1 import AppTest

from src.config import ROOT


def _extract_score(delta_text: str) -> float:
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)/100", delta_text or "")
    assert m, f"No se encontró puntaje en {delta_text!r}"
    return float(m.group(1))


def test_demo_streamlit(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
    assert not app.exception

    score_default = _extract_score(app.metric[0].delta)
    app.slider[0].set_value(500).run()
    score_low = _extract_score(app.metric[0].delta)
    assert score_low < score_default

    app.slider[0].set_value(9800).run()
    score_high = _extract_score(app.metric[0].delta)
    assert score_high > score_default

    q_input = next(w for w in app.text_input if w.label == "Pregunta sobre la inspección")
    q_input.set_value("¿Cuándo aplica el parchado superficial en pavimento flexible?").run()

    analyze_btn = next(b for b in app.button if b.label == "Analizar tramo")
    analyze_btn.click().run()
    assert not app.exception
    assert any("MTC-410" in x.value for x in app.markdown)
    assert not app.get("download_button")

    # Probar chat multi-turno con pregunta oficial de prioridad
    if app.chat_input:
        app.chat_input[0].set_value("¿En qué sección se sustenta la prioridad?").run()
        assert not app.exception
        assert any("MTC-PRI" in x.value for x in app.markdown)

    approve_btn = next(b for b in app.button if "Aprobar" in b.label)
    approve_btn.click().run()
    assert not app.exception
    downloads = app.get("download_button")
    assert len(downloads) == 1
    assert downloads[0].label == "Descargar expediente"
    exported = app.session_state["result"]["record"]
    assert exported["tramo"] == "PILOTO-01"
    assert exported["ubicacion"]["provincia"] == "Trujillo"
    assert "informe_preliminar" in exported and exported["informe_preliminar"]
