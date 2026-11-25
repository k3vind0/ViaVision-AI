import operator
from datetime import datetime, timezone
from typing import Annotated, TypedDict

from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt
from src.evidence import answer, recuperar_normativa
from src.priority import calcular_prioridad
from src.vision import detectar_danos


class State(TypedDict, total=False):
    image: bytes
    values: dict
    demo: bool
    tramo: str
    question: str
    approved: bool
    location: dict
    vision: dict
    priority: dict
    evidence: list
    response: str
    mode: str
    record: dict
    errors: Annotated[list, operator.add]
    trace: Annotated[list, operator.add]


def vision_node(s):
    try:
        return {'vision': detectar_danos(s['image']), 'trace': ['detectar_danos']}
    except (ValueError, OSError, RuntimeError):
        return {'vision': {'estado':'no evaluable', 'danos':[]}, 'trace':['detectar_danos'],
                'errors':['No se pudo evaluar la imagen o cargar el detector.']}


def priority_node(s):
    try:
        return {'priority': calcular_prioridad(s['values'], s['demo']),
                'trace': ['calcular_prioridad']}
    except ValueError as e:
        return {'priority': {}, 'trace': ['calcular_prioridad'], 'errors': [str(e)]}


def evidence_node(s):
    question = s['question'].strip()[:500]
    hits = recuperar_normativa(question)
    response, mode = answer(question, hits)
    return {'evidence': hits, 'response': response, 'mode': mode, 'trace': ['recuperar_normativa']}


def registrar_inspeccion_preliminar(s):
    return {'record': {'tramo': s['tramo'], 'ubicacion': s.get('location'),
                       'fecha_utc': datetime.now(timezone.utc).isoformat(),
                       'alcance': 'demostración sintética' if s['demo'] else 'piloto',
                       'vision': s['vision'], 'prioridad': s['priority'], 'evidencia': s['evidence'],
                       'revision_humana': 'aprobado para exportar expediente preliminar',
                       'nota': 'Documento de inspección preliminar.'},
            'trace': ['registrar_inspeccion_preliminar']}


def revision_node(s):
    return {'approved': interrupt('Revisar el expediente preliminar.') is True,
            'trace': ['revision_humana']}


def graph():
    builder = StateGraph(State)
    for name, node in [('vision', vision_node), ('prioridad', priority_node),
                       ('evidencia', evidence_node), ('revision', revision_node),
                       ('registro', registrar_inspeccion_preliminar)]:
        builder.add_node(name, node)
    builder.add_edge(START, 'vision')
    builder.add_edge('vision', 'prioridad')
    builder.add_edge('prioridad', 'evidencia')
    builder.add_conditional_edges('evidencia',
        lambda s: 'revision' if s.get('priority')
        and not s.get('errors') and not s['vision']['estado'].startswith('no evaluable') else END,
        {'revision': 'revision', END: END})
    builder.add_conditional_edges('revision', lambda s: 'registro' if s['approved'] else END,
                                  {'registro': 'registro', END: END})
    builder.add_edge('registro', END)
    return builder.compile(checkpointer=InMemorySaver())
