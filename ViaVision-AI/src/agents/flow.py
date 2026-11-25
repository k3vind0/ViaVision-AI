"""Grafo LangGraph con estado tipado, flujo entre módulos, incertidumbre y control humano."""
from datetime import datetime, timezone
import operator
import time
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.config import get_logger
from src.ml.priority import calcular_prioridad
from src.rag.assistant import answer
from src.rag.retriever import recuperar_normativa
from src.schemas import ExpedientePreliminar
from src.vision.detector import detectar_danos

logger = get_logger(__name__)


class State(TypedDict, total=False):
    image: bytes
    values: dict
    demo: bool
    tramo: str
    question: str
    approved: bool
    metadata_ok: bool
    location: dict
    vision: dict
    priority: dict
    incertidumbre: dict
    evidence: list
    response: str
    mode: str
    record: dict
    errors: Annotated[list, operator.add]
    warnings: Annotated[list, operator.add]
    trace: Annotated[list, operator.add]
    latencias_ms: Annotated[dict, operator.or_]


def verificar_metadatos(s: State) -> dict:
    """Valida metadatos mínimos del tramo y ubicación antes de ejecutar herramientas."""
    t0 = time.perf_counter()
    tramo = str(s.get("tramo", "")).strip()
    location = s.get("location") or {}
    errors: list[str] = []
    warnings: list[str] = []

    if not tramo:
        errors.append("El identificador del tramo es obligatorio.")
    if not s.get("demo") and (not location.get("distrito") or not location.get("via_o_sector")):
        errors.append("En modo piloto se requiere distrito y vía o sector.")
    if not location.get("coordenadas"):
        warnings.append("Inspección sin coordenadas geográficas; se recomienda registrar latitud/longitud.")

    ms = round((time.perf_counter() - t0) * 1000.0, 2)
    logger.info("Nodo verificar_metadatos: ok=%s tramo=%s (%.2f ms)", not errors, tramo, ms)
    return {
        "metadata_ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "trace": ["verificar_metadatos"],
        "latencias_ms": {"verificar_metadatos": ms},
    }


def vision_node(s: State) -> dict:
    t0 = time.perf_counter()
    try:
        res = detectar_danos(s["image"])
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "vision": res,
            "trace": ["detectar_danos"],
            "latencias_ms": {"detectar_danos": ms},
        }
    except (ValueError, OSError, RuntimeError) as exc:
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        logger.warning("Error en detectar_danos: %s", exc)
        return {
            "vision": {"estado": "no evaluable", "danos": [], "incertidumbre_visual": "alta"},
            "trace": ["detectar_danos"],
            "errors": ["No se pudo evaluar la imagen o cargar el detector."],
            "latencias_ms": {"detectar_danos": ms},
        }


def priority_node(s: State) -> dict:
    t0 = time.perf_counter()
    try:
        res = calcular_prioridad(s["values"], s.get("demo", True))
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "priority": res,
            "trace": ["calcular_prioridad"],
            "latencias_ms": {"calcular_prioridad": ms},
        }
    except ValueError as exc:
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        logger.warning("Error en calcular_prioridad: %s", exc)
        return {
            "priority": {},
            "trace": ["calcular_prioridad"],
            "errors": [str(exc)],
            "latencias_ms": {"calcular_prioridad": ms},
        }


def evidence_node(s: State) -> dict:
    t0 = time.perf_counter()
    question = str(s.get("question", "")).strip()[:500]
    vision = s.get("vision") or {}
    priority = s.get("priority") or {}
    location = s.get("location") or {}

    hits = recuperar_normativa(question, detected_damages=vision.get("danos", []))
    response, mode = answer(
        question,
        hits,
        vision=vision,
        priority=priority,
        location=location,
    )
    ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "evidence": hits,
        "response": response,
        "mode": mode,
        "trace": ["recuperar_normativa"],
        "latencias_ms": {"recuperar_normativa": ms},
    }


def evaluar_incertidumbre(s: State) -> dict:
    """Evalúa incertidumbre conjunta ML + Visión para alertar antes de la revisión."""
    t0 = time.perf_counter()
    motivos: list[str] = []
    warnings: list[str] = []

    ml_unc = (s.get("priority") or {}).get("nivel_incertidumbre", "alta")
    vis_unc = (s.get("vision") or {}).get("incertidumbre_visual", "alta")
    if ml_unc == "alta":
        motivos.append(
            f"Margen de probabilidad estrecho en prioridad ({s['priority'].get('margen_probabilidad')})."
        )
    if vis_unc == "alta":
        motivos.append("Confianza visual baja o ausencia de hallazgos sobre el umbral.")
    if not s.get("evidence"):
        motivos.append("Sin respaldo normativo recuperado para la consulta actual.")

    if "alta" in (ml_unc, vis_unc) or not s.get("evidence"):
        nivel = "alta"
    elif "media" in (ml_unc, vis_unc):
        nivel = "media"
    else:
        nivel = "baja"

    if nivel == "alta":
        warnings.append("Incertidumbre conjunta alta: requiere verificación técnica obligatoria en campo.")

    ms = round((time.perf_counter() - t0) * 1000.0, 2)
    logger.info("Incertidumbre conjunta=%s motivos=%s", nivel, motivos)
    return {
        "incertidumbre": {
            "nivel": nivel,
            "ml": ml_unc,
            "vision": vis_unc,
            "motivos": motivos,
            "requiere_verificacion_campo": nivel != "baja",
        },
        "warnings": warnings,
        "trace": ["evaluar_incertidumbre"],
        "latencias_ms": {"evaluar_incertidumbre": ms},
    }


def revision_node(s: State) -> dict:
    t0 = time.perf_counter()
    decision = interrupt(
        {
            "mensaje": "Revisar el expediente preliminar.",
            "tramo": s.get("tramo"),
            "incertidumbre": s.get("incertidumbre"),
        }
    )
    ms = round((time.perf_counter() - t0) * 1000.0, 2)
    approved = decision is True
    logger.info("Revisión humana completada: approved=%s", approved)
    return {
        "approved": approved,
        "trace": ["revision_humana"],
        "latencias_ms": {"revision_humana": ms},
    }


def registrar_inspeccion_preliminar(s: State) -> dict:
    t0 = time.perf_counter()
    latencias = dict(s.get("latencias_ms") or {})
    traza = list(s.get("trace") or []) + ["registrar_inspeccion_preliminar"]
    ms = round((time.perf_counter() - t0) * 1000.0, 2)
    latencias["registrar_inspeccion_preliminar"] = ms

    raw_record = {
        "tramo": s["tramo"],
        "ubicacion": s.get("location") or {},
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "alcance": "demostración sintética" if s.get("demo", True) else "piloto",
        "danos_detectados": (s.get("vision") or {}).get("danos", []),
        "vision": s["vision"],
        "puntaje_prioridad": (s.get("priority") or {}).get("puntaje", 0.0),
        "prioridad": s["priority"],
        "factores": (s.get("priority") or {}).get("factores", []),
        "evidencia_normativa": s.get("evidence", []),
        "evidencia": s.get("evidence", []),
        "nivel_incertidumbre": (s.get("incertidumbre") or {}).get("nivel", "media"),
        "informe_preliminar": s.get("response", ""),
        "modo_respuesta": s.get("mode", ""),
        "revision_humana": "aprobado para exportar expediente preliminar",
        "traza": traza,
        "latencias_ms": latencias,
        "nota": "Documento de inspección preliminar. No sustituye ensayos estructurales.",
    }
    validated = ExpedientePreliminar.model_validate(raw_record).model_dump()
    logger.info("Expediente preliminar validado (Pydantic) para tramo=%s", s["tramo"])
    return {
        "record": validated,
        "trace": ["registrar_inspeccion_preliminar"],
        "latencias_ms": {"registrar_inspeccion_preliminar": ms},
    }


def _can_review(s: State) -> str:
    if (
        s.get("metadata_ok")
        and s.get("priority")
        and not s.get("errors")
        and not (s.get("vision") or {}).get("estado", "").startswith("no evaluable")
    ):
        return "revision"
    return END


def graph():
    builder = StateGraph(State)
    nodes = [
        ("verificar_metadatos", verificar_metadatos),
        ("prioridad", priority_node),
        ("vision", vision_node),
        ("evidencia", evidence_node),
        ("incertidumbre", evaluar_incertidumbre),
        ("revision", revision_node),
        ("registro", registrar_inspeccion_preliminar),
    ]
    for name, fn in nodes:
        builder.add_node(name, fn)

    builder.add_edge(START, "verificar_metadatos")
    builder.add_edge("verificar_metadatos", "prioridad")
    builder.add_edge("prioridad", "vision")
    builder.add_edge("vision", "evidencia")
    builder.add_edge("evidencia", "incertidumbre")
    builder.add_conditional_edges("incertidumbre", _can_review, {"revision": "revision", END: END})
    builder.add_conditional_edges("revision", lambda s: "registro" if s.get("approved") else END, {"registro": "registro", END: END})
    builder.add_edge("registro", END)
    return builder.compile(checkpointer=InMemorySaver())
