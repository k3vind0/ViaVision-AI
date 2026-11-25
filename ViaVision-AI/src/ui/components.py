"""Componentes reutilizables de la interfaz Streamlit para VíaVision AI."""
import json

import pandas as pd
import streamlit as st

from src.config import ROOT
from src.rag.assistant import answer
from src.rag.retriever import recuperar_normativa


def render_sidebar() -> tuple[bool, str, str, str, dict, bool]:
    """Renderiza la barra lateral con modo, identificación y ubicación en Trujillo."""
    with st.sidebar:
        st.header("Tramo vial")
        mode = st.radio("Modo", ["Demostración", "Piloto con datos validados"])
        demo = mode == "Demostración"
        tramo = st.text_input("Identificador", "PILOTO-01", max_chars=50)
        st.caption("Provincia de Trujillo · La Libertad")
        district = st.text_input(
            "Distrito",
            "Trujillo" if demo else "",
            max_chars=80,
            key=f"district_{demo}",
        ).strip()
        road = st.text_input(
            "Vía o sector",
            "Tramo simulado" if demo else "",
            max_chars=120,
            key=f"road_{demo}",
        ).strip()

        coordinates = None
        if st.checkbox("Añadir coordenadas de inspección"):
            lat = st.number_input(
                "Latitud",
                min_value=-90.0,
                max_value=90.0,
                value=-8.1117 if demo else None,
                format="%.6f",
                key=f"lat_{demo}",
            )
            lon = st.number_input(
                "Longitud",
                min_value=-180.0,
                max_value=180.0,
                value=-79.0289 if demo else None,
                format="%.6f",
                key=f"lon_{demo}",
            )
            if lat is not None and lon is not None:
                coordinates = {"latitud": lat, "longitud": lon}
            else:
                st.caption("Completa ambas coordenadas para incluirlas.")

        model_ready = demo or (ROOT / "models" / "validado.joblib").is_file()
        if not model_ready:
            st.warning(
                "No se encontró models/validado.joblib. Ejecuta "
                "`python -m scripts.train_ml --csv data_private/tramos.csv` antes de usar modo piloto."
            )

        location = {
            "departamento": "La Libertad",
            "provincia": "Trujillo",
            "distrito": district,
            "via_o_sector": road,
            "coordenadas": coordinates,
            "origen": "simulado" if demo else "declarado por el usuario",
        }
    return demo, tramo, district, road, location, model_ready


def render_priority_details(preview: dict) -> None:
    """Muestra probabilidades de clase, incertidumbre y factores explicativos."""
    probs = preview.get("probabilidades", {})
    cols = st.columns(3)
    for col, cls in zip(cols, ("baja", "media", "alta")):
        p = float(probs.get(cls, 0.0))
        with col:
            st.progress(min(max(p, 0.0), 1.0), text=f"{cls.capitalize()}: {p:.1%}")
    st.caption(
        f"Incertidumbre ML: **{preview.get('nivel_incertidumbre', 'media')}** "
        f"(margen={preview.get('margen_probabilidad', 0):.2f}, "
        f"entropía={preview.get('entropia_normalizada', 0):.2f})"
    )
    factores = preview.get("factores", [])
    if factores:
        st.caption("Factores principales: " + " · ".join(factores))


def render_evidence_and_trace(result: dict) -> None:
    """Muestra las fichas recuperadas y la traza con latencias por nodo."""
    for item in result.get("evidence", []):
        meta = f"{item.get('documento', '')} · {item.get('seccion', '')} · sim={item.get('similaridad', 0):.3f}"
        with st.expander(f"{item['id']} · {item['fuente']} ({meta})"):
            st.write(item["texto"])
            st.link_button("Consultar fuente original", item["url"])

    latencias = result.get("latencias_ms", {})
    total_ms = round(sum(latencias.values()), 2) if latencias else 0.0
    with st.expander(f"Traza del agente ({total_ms} ms)"):
        for name in result.get("trace", []):
            ms = latencias.get(name)
            st.write(f"- `{name}`" + (f" ({ms} ms)" if ms is not None else ""))


def render_chat_panel(result: dict, location: dict) -> None:
    """Panel de chatbot contextual multi-turno sobre el tramo analizado."""
    st.markdown("**Asistente contextual del tramo**")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": result["response"], "mode": result["mode"]}
        ]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("mode"):
                st.caption(msg["mode"])

    follow_up = st.chat_input("Consulta adicional sobre este tramo o normativa (ej. ¿En qué sección se sustenta la prioridad?)")
    if follow_up and follow_up.strip():
        clean_q = follow_up.strip()[:500]
        st.session_state.chat_history.append({"role": "user", "content": clean_q})
        hits = recuperar_normativa(
            clean_q,
            detected_damages=result.get("vision", {}).get("danos", []),
        )
        resp, mode = answer(
            clean_q,
            hits,
            vision=result.get("vision"),
            priority=result.get("priority"),
            location=location,
        )
        st.session_state.chat_history.append({"role": "assistant", "content": resp, "mode": mode})
        st.rerun()


def render_approval_and_download(result: dict, on_approve) -> None:
    """Control humano explícito antes de permitir la descarga del expediente."""
    vision = result.get("vision", {})
    ready = (
        bool(result.get("priority"))
        and not result.get("errors")
        and not vision.get("estado", "").startswith("no evaluable")
    )
    if ready and not result.get("record"):
        if st.button("Aprobar y preparar expediente preliminar"):
            on_approve()
            st.rerun()
    if result.get("record"):
        st.success("Revisión aprobada. Expediente preliminar listo para descarga.")
        st.download_button(
            "Descargar expediente",
            json.dumps(result["record"], ensure_ascii=False, indent=2),
            file_name="expediente-viavision.json",
            mime="application/json",
        )
