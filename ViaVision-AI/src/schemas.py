"""Esquemas Pydantic v2 para validación de entradas, salidas estructuradas y expediente."""
from typing import Any, Optional

from pydantic import BaseModel, Field


class Coordenadas(BaseModel):
    latitud: float = Field(..., ge=-90.0, le=90.0)
    longitud: float = Field(..., ge=-180.0, le=180.0)


class UbicacionTramo(BaseModel):
    departamento: str = "La Libertad"
    provincia: str = "Trujillo"
    distrito: str = ""
    via_o_sector: str = ""
    coordenadas: Optional[Coordenadas] = None
    origen: str = "simulado"


class HallazgoVisual(BaseModel):
    clase: str
    caja: list[int] = Field(..., min_length=4, max_length=4)
    confianza: float = Field(..., ge=0.0, le=1.0)


class ResultadoVision(BaseModel):
    estado: str
    danos: list[HallazgoVisual] = Field(default_factory=list)
    umbral_confianza: float = 0.25
    incertidumbre_visual: str = "media"


class ResultadoPrioridad(BaseModel):
    clase: str
    puntaje: float = Field(..., ge=0.0, le=100.0)
    probabilidades: dict[str, float]
    margen_probabilidad: float = Field(..., ge=0.0, le=1.0)
    entropia_normalizada: float = Field(..., ge=0.0, le=1.0)
    nivel_incertidumbre: str
    factores: list[str] = Field(default_factory=list)
    origen: str
    variables: dict[str, Any]


class FragmentoNormativo(BaseModel):
    id: str
    documento: str = ""
    seccion: str = ""
    pagina: str = ""
    tema: str = ""
    fuente: str
    url: str
    texto: str
    similaridad: float = Field(..., ge=0.0)


class ExplicacionEstructurada(BaseModel):
    resumen: str
    interpretacion_resultados: str
    evidencia_citada: list[str] = Field(default_factory=list)
    preguntas_aclaracion: list[str] = Field(default_factory=list)
    advertencia: str = (
        "La prioridad calculada no sustituye ensayos estructurales ni decisión de un "
        "ingeniero vial; no inferir profundidad de grietas sólo desde RGB."
    )


class ExpedientePreliminar(BaseModel):
    tramo: str
    ubicacion: UbicacionTramo
    fecha_utc: str
    alcance: str
    danos_detectados: list[HallazgoVisual]
    vision: ResultadoVision
    puntaje_prioridad: float
    prioridad: ResultadoPrioridad
    factores: list[str]
    evidencia_normativa: list[FragmentoNormativo]
    evidencia: list[FragmentoNormativo]
    nivel_incertidumbre: str
    informe_preliminar: str
    modo_respuesta: str
    revision_humana: str
    traza: list[str]
    latencias_ms: dict[str, float]
    nota: str
