"""Carga, limpieza y segmentación (chunking) con metadatos del corpus normativo."""
import json
import re

from src.config import ROOT, get_logger, load_settings

logger = get_logger(__name__)


def clean_text(text: str) -> str:
    """Normaliza espacios en blanco manteniendo puntuación y citas intactas."""
    return re.sub(r"\s+", " ", text).strip()


def chunk_document(
    text: str,
    doc_id: str,
    metadata: dict,
    max_chars: int = 450,
    overlap: int = 60,
) -> list[dict]:
    """Segmenta un documento en fragmentos trazables con identificador y página."""
    cleaned = clean_text(text)
    if len(cleaned) <= max_chars:
        return [{**metadata, "id": doc_id, "chunk_id": f"{doc_id}#c0", "texto": cleaned}]

    chunks = []
    start = 0
    idx = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + max_chars)
        piece = cleaned[start:end].strip()
        chunks.append(
            {
                **metadata,
                "id": doc_id,
                "chunk_id": f"{doc_id}#c{idx}",
                "texto": piece,
            }
        )
        if end >= len(cleaned):
            break
        start = max(start + 1, end - overlap)
        idx += 1
    return chunks


def load_corpus() -> list[dict]:
    """Carga el corpus JSON curado y aplica segmentación con trazabilidad documental."""
    cfg = load_settings().get("rag", {})
    corpus_path = ROOT / cfg.get("corpus_path", "data_sample/corpus.json")
    raw_records = json.loads(corpus_path.read_text(encoding="utf-8"))
    chunks: list[dict] = []
    for rec in raw_records:
        meta = {k: v for k, v in rec.items() if k not in {"id", "texto"}}
        chunks.extend(chunk_document(rec["texto"], rec["id"], meta))
    logger.info("Corpus cargado: %d fichas -> %d fragmentos", len(raw_records), len(chunks))
    return chunks
