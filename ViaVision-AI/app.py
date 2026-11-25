"""Aplicación principal Streamlit de VíaVision AI (Grupo 11 · UNT)."""
import hashlib
from uuid import uuid4

from dotenv import load_dotenv
from langgraph.types import Command
import streamlit as st

from src.agents.flow import graph
from src.config import ROOT, get_logger
from src.ml.priority import VIA, calcular_prioridad
from src.ui import (
    inject_styles,
    render_approval_and_download,
    render_chat_panel,
    render_evidence_and_trace,
    render_hero,
    render_kpi_strip,
    render_priority_details,
    render_sidebar,
)
from src.vision.detector import draw_boxes

load_dotenv()
logger = get_logger(__name__)

st.set_page_config(page_title="VíaVision AI · Grupo 11", page_icon="🛣️", layout="wide")
inject_styles()
render_hero()

demo, tramo, district, road, location, model_ready = render_sidebar()

if demo:
    st.info(
        "Modo Demostración activo. El tramo y la prioridad provienen de entrenamiento sintético; "
        "la imagen incluida procede de un conjunto externo a Trujillo."
    )

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Imagen de la vía")
    upload = st.file_uploader("Carga JPG, PNG o WebP", type=["jpg", "jpeg", "png", "webp"])
    example = ROOT / "data_sample" / "tramo-ejemplo.jpg"
    use_sample = st.checkbox(
        "Usar foto de demostración",
        value=demo,
        disabled=not demo,
        key=f"use_sample_{demo}",
    )
    raw = (
        upload.getvalue()
        if upload
        else (example.read_bytes() if demo and use_sample and example.exists() else None)
    )
    if raw:
        try:
            st.image(draw_boxes(raw, []), caption="Imagen de entrada", width="stretch")
        except (OSError, ValueError):
            st.error("El archivo no contiene una imagen compatible.")
            raw = None

with right:
    st.subheader("Datos para estimar prioridad")
    traffic = st.slider("Tráfico diario (vehículos)", 0, 100000, 6500, step=100)
    age = st.number_input("Antigüedad del pavimento (años)", 0, 100, 14)
    repairs = st.slider("Reparaciones en los últimos 24 meses", 0, 100, 3)
    road_type = st.selectbox("Tipo de vía", VIA, index=1)
    rain = st.number_input("Precipitación en 30 días (mm)", 0.0, 5000.0, 25.0)
    severity = st.selectbox(
        "Severidad declarada por inspección",
        [0, 1, 2, 3],
        index=2,
        format_func=lambda x: ["0 · ninguna", "1 · baja", "2 · media", "3 · alta"][x],
    )
    values = dict(
        trafico_diario=traffic,
        antiguedad_anios=age,
        reparaciones_24m=repairs,
        tipo_via=road_type,
        precipitacion_mm_30d=rain,
        severidad_inspeccionada=severity,
    )
    try:
        preview = calcular_prioridad(values, demo)
        st.metric("Prioridad estimada", preview["clase"].capitalize(), f"{preview['puntaje']}/100")
        render_priority_details(preview)
    except ValueError as exc:
        st.warning(str(exc))

st.subheader("Consulta técnica y ejecución integrada")
question = st.text_input(
    "Pregunta sobre la inspección",
    "¿Qué debo medir en un bache?",
    max_chars=500,
)
signature = (
    hashlib.sha256(raw).hexdigest() if raw else None,
    tuple(values.items()),
    demo,
    tramo,
    question,
    location,
)


def workflow():
    if "workflow" not in st.session_state:
        st.session_state.workflow = graph()
    return st.session_state.workflow


def run(approved: bool) -> None:
    if approved:
        payload = Command(resume=True)
    else:
        st.session_state.thread_config = {"configurable": {"thread_id": str(uuid4())}}
        st.session_state.pop("chat_history", None)
        payload = {
            "image": raw,
            "values": values,
            "demo": demo,
            "tramo": tramo.strip(),
            "question": question,
            "location": location,
        }
    with st.spinner("Procesando imagen, prioridad y fuentes…"):
        st.session_state.result = workflow().invoke(payload, st.session_state.thread_config)
        st.session_state.signature = signature
        logger.info("Flujo invocado desde Streamlit (approved=%s, tramo=%s)", approved, tramo)


if st.button(
    "Analizar tramo",
    type="primary",
    disabled=not raw or not tramo.strip() or not district or not road or not question.strip() or not model_ready,
):
    run(False)

if st.session_state.get("signature") == signature:
    result = st.session_state.result
    for error in result.get("errors", []):
        st.error(error)
    for warning in result.get("warnings", []):
        st.warning(warning)

    render_kpi_strip(result)

    tab_vis, tab_ml, tab_rag = st.tabs(
        [
            "Escenario 1 · Visión (YOLO11n)",
            "Escenario 2 · Prioridad y Factores (ML)",
            "Escenario 3 · Evidencia RAG y Traza LangGraph",
        ]
    )

    with tab_vis:
        vision = result["vision"]
        if vision["estado"].startswith("no evaluable"):
            st.warning(vision["estado"])
        else:
            st.image(
                draw_boxes(raw, vision["danos"]),
                caption=f"Estado: {vision['estado']} · Incertidumbre visual: {vision.get('incertidumbre_visual', 'media')}",
                width="stretch",
            )
            if vision["danos"]:
                st.dataframe(vision["danos"], hide_index=True)
            st.caption(
                "Nota técnica: YOLO11n ubica deterioros superficiales en píxeles; "
                "la profundidad física debe verificarse en campo."
            )

    with tab_ml:
        prio = result.get("priority") or {}
        if prio:
            st.markdown(
                f"**Prioridad calculada:** `{prio['clase'].capitalize()}` (**{prio['puntaje']}/100**) · "
                f"Origen: `{prio.get('origen', 'N/D')}`"
            )
            render_priority_details(prio)
            st.json(prio.get("variables", {}), expanded=False)

    with tab_rag:
        render_evidence_and_trace(result)

    render_approval_and_download(result, lambda: run(True))
    render_chat_panel(result, location)
