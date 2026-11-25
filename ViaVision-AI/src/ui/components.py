"""Componentes reutilizables y ejecutivos de la interfaz Streamlit para VíaVision AI."""
from html import escape
import json

import streamlit as st

from src.config import ROOT
from src.rag.assistant import answer
from src.rag.retriever import recuperar_normativa
from src.ui.styles import badge_class


def render_hero() -> None:
    """Renderiza la cabecera institucional UNT Grupo 11."""
    st.markdown(
        """
        <div class="vv-hero">
            <div class="vv-hero-kicker">Universidad Nacional de Trujillo · Ingeniería de Sistemas · Grupo 11 · Provincia de Trujillo</div>
            <h1 class="vv-hero-title">VíaVision AI</h1>
            <p class="vv-hero-sub">
                Plataforma integrada de inspección vial: detección de deterioros (YOLO11n),
                priorización tabular explicable (ML), evidencia normativa verificable (RAG) y control humano (LangGraph).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[bool, str, str, str, dict, bool]:
    """Renderiza la barra lateral con modo, identificación y ubicación en Trujillo."""
    with st.sidebar:
        st.header("Tramo vial")
        mode = st.radio("Modo", ["Demostración", "Piloto con datos validados"])
        demo = mode == "Demostración"
        tramo = st.text_input("Identificador", "PILOTO-01", max_chars=50)
        st.caption("Provincia de Trujillo · La Libertad · Grupo 11")
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

    unc = preview.get("nivel_incertidumbre", "media")
    st.markdown(
        f'<span class="{badge_class(unc)}">Incertidumbre ML: {escape(unc.upper())}</span> '
        f'<span class="vv-kpi-foot">Margen={preview.get("margen_probabilidad", 0):.2f} · '
        f'Entropía={preview.get("entropia_normalizada", 0):.2f}</span>',
        unsafe_allow_html=True,
    )
    factores = preview.get("factores", [])
    if factores:
        chips = "".join(f'<span class="vv-chip">{escape(f)}</span>' for f in factores)
        st.markdown(f'<div class="vv-chips">{chips}</div>', unsafe_allow_html=True)


def render_kpi_strip(result: dict) -> None:
    """Renderiza una franja ejecutiva de 4 tarjetas KPI para el tramo analizado."""
    prio = result.get("priority") or {}
    vis = result.get("vision") or {}
    unc = result.get("incertidumbre") or {}
    latencias = result.get("latencias_ms") or {}

    clase = str(prio.get("clase", "N/D")).capitalize()
    puntaje = prio.get("puntaje", "N/D")
    danos = vis.get("danos", [])
    total_ms = round(sum(latencias.values()), 1) if latencias else 0.0
    unc_lvl = str(unc.get("nivel", "media"))
    vis_unc = str(vis.get("incertidumbre_visual", "media"))
    n_hits = len(result.get("evidence", []))

    st.markdown(
        f"""
        <div class="vv-kpi-grid">
            <div class="vv-kpi-card">
                <div class="vv-kpi-label">Prioridad del tramo</div>
                <div class="vv-kpi-value">{escape(clase)} · {escape(str(puntaje))}/100</div>
                <span class="{badge_class(clase)}">{escape(prio.get("origen", "ML"))}</span>
            </div>
            <div class="vv-kpi-card">
                <div class="vv-kpi-label">Hallazgos YOLO11n</div>
                <div class="vv-kpi-value">{len(danos)} deterioro(s)</div>
                <span class="{badge_class(vis_unc)}">Incert. visual: {escape(vis_unc)}</span>
            </div>
            <div class="vv-kpi-card">
                <div class="vv-kpi-label">Incertidumbre conjunta</div>
                <div class="vv-kpi-value">{escape(unc_lvl.capitalize())}</div>
                <span class="vv-badge vv-badge-info">{n_hits} fuente(s) RAG</span>
            </div>
            <div class="vv-kpi-card">
                <div class="vv-kpi-label">Latencia LangGraph</div>
                <div class="vv-kpi-value">{total_ms} ms</div>
                <span class="vv-kpi-foot">{len(result.get("trace", []))} nodos ejecutados</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_timeline(result: dict) -> None:
    """Renderiza una línea de tiempo visual con los nodos ejecutados y su latencia."""
    latencias = result.get("latencias_ms", {})
    trace = result.get("trace", [])
    items_html = []
    for idx, name in enumerate(trace, 1):
        ms = latencias.get(name)
        ms_txt = f"{ms} ms" if ms is not None else "—"
        items_html.append(
            f'<div class="vv-step">'
            f'  <div class="vv-step-left">'
            f'    <span class="vv-step-idx">{idx}</span>'
            f'    <span class="vv-step-name">{escape(str(name))}</span>'
            f'  </div>'
            f'  <span class="vv-step-ms">{escape(ms_txt)}</span>'
            f'</div>'
        )
    st.markdown(f'<div class="vv-timeline">{"".join(items_html)}</div>', unsafe_allow_html=True)


def render_evidence_and_trace(result: dict) -> None:
    """Muestra la respuesta principal, las fichas normativas y la traza visual LangGraph."""
    st.markdown("**Respuesta fundamentada del sistema**")
    st.markdown(result["response"])
    st.caption(f"Modo de respuesta: {result['mode']}")

    evidence = result.get("evidence", [])
    if evidence:
        st.markdown("**Fuentes normativas recuperadas**")
        for item in evidence:
            meta = f"{item.get('documento', '')} · {item.get('seccion', '')} · sim={item.get('similaridad', 0):.3f}"
            with st.expander(f"{item['id']} · {item['fuente']} ({meta})"):
                st.write(item["texto"])
                st.link_button("Consultar fuente original", item["url"])

    total_ms = round(sum((result.get("latencias_ms") or {}).values()), 2)
    with st.expander(f"Traza del agente ({total_ms} ms)", expanded=True):
        render_timeline(result)


def render_chat_panel(result: dict, location: dict) -> None:
    """Panel de chatbot contextual multi-turno sobre el tramo analizado (fuera de tabs)."""
    st.subheader("Asistente contextual multi-turno")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": result["response"], "mode": result["mode"]}
        ]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("mode"):
                st.caption(msg["mode"])

    follow_up = st.chat_input(
        "Consulta adicional sobre este tramo o normativa (ej. ¿En qué sección se sustenta la prioridad?)"
    )
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
    """Panel destacado de revisión humana obligatorio antes de exportar expediente."""
    vision = result.get("vision", {})
    ready = (
        bool(result.get("priority"))
        and not result.get("errors")
        and not vision.get("estado", "").startswith("no evaluable")
    )
    if ready and not result.get("record"):
        st.markdown(
            """
            <div class="vv-review-pending">
                <div class="vv-review-title">Control Humano Obligatorio (LangGraph interrupt)</div>
                <p class="vv-review-desc">
                    El expediente preliminar permanece bloqueado hasta confirmar la revisión técnica del inspector.
                    Este sistema no emite órdenes de obra automáticas.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Aprobar y preparar expediente preliminar"):
            on_approve()
            st.rerun()

    if result.get("record"):
        st.markdown(
            """
            <div class="vv-review-approved">
                <div class="vv-review-title">Revisión aprobada · Expediente preliminar validado (Pydantic)</div>
                <p class="vv-review-desc">
                    Flujo LangGraph reanudado correctamente. Puedes descargar el expediente técnico en formato JSON.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            "Descargar expediente",
            json.dumps(result["record"], ensure_ascii=False, indent=2),
            file_name="expediente-viavision.json",
            mime="application/json",
        )
