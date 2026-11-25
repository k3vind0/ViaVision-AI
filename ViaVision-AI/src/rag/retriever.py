"""Recuperación híbrida (TF-IDF + embeddings densos SVD) con guardia anti-inyección."""
from functools import lru_cache
import re
import unicodedata

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.config import get_logger, load_settings
from src.rag.corpus import load_corpus

logger = get_logger(__name__)

_SPANISH_STOPWORDS = [
    "el", "la", "los", "las", "de", "del", "se", "que", "para", "en", "por",
    "un", "una", "unos", "unas", "con", "su", "sus", "al", "lo", "como", "es",
    "son", "cuando", "cual", "cuales", "sobre", "este", "esta", "y", "o", "a",
]

_DOMAIN_PATTERN = re.compile(
    r"\b("
    r"bache\w*|grieta\w*|fisur\w*|pavimento\w*|calzada\w*|parchad\w*|sellad\w*|"
    r"rodadura|asfalto|ahuellamiento\w*|prioridad\w*|prioriz\w*|inspeccion\w*|"
    r"conservacion\w*|mantenimiento|tramo\w*|via\w*|mtc|fhwa|deterioro\w*|"
    r"severidad\w*|profundidad\w*|fatiga|cocodrilo|borde\w*|bloque\w*|expediente\w*|"
    r"estructural\w*|rgb|orden\w*\s+de\s+obra|repar\w*"
    r")\b",
    re.I,
)

_INJECTION_PATTERN = re.compile(
    r"\b("
    r"contrasen\w*|password\w*|credencial\w*|api[_\s-]?key|secret\w*|token\w*|"
    r"ignora\w*|olvida\w*|descarta\w*|omite\w*|anula\w*|reemplaza\w*|revela\w*|"
    r"system\s+prompt|jailbreak|bypass|override|ignore\s+previous|forget\s+your|"
    r"salario\w*|sueldo\w*"
    r")\b",
    re.I,
)


def _strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


@lru_cache(maxsize=1)
def corpus_index():
    """Construye índice léxico TF-IDF e índice vectorial denso L2-normalizado."""
    cfg = load_settings().get("rag", {})
    records = load_corpus()
    texts = [
        f"{r.get('id', '')} {r.get('seccion', '')} {r.get('tema', '')} {r.get('texto', '')}"
        for r in records
    ]

    word_vec = TfidfVectorizer(
        strip_accents="unicode",
        ngram_range=(1, 2),
        stop_words=_SPANISH_STOPWORDS,
    )
    tfidf_matrix = word_vec.fit_transform(texts)

    char_vec = TfidfVectorizer(
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5),
    )
    char_matrix = char_vec.fit_transform(texts)

    req_dim = int(cfg.get("embedding_dim", 12))
    n_comp = max(2, min(req_dim, len(records) - 1, char_matrix.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    dense_raw = svd.fit_transform(char_matrix)
    dense_index = normalize(dense_raw, norm="l2")

    return records, word_vec, tfidf_matrix, char_vec, svd, dense_index


def recuperar_normativa(
    question: str,
    top_k: int | None = None,
    detected_damages: list[dict] | None = None,
) -> list[dict]:
    """Recupera fichas normativas con filtrado de dominio, índice híbrido y reranking visual."""
    cfg = load_settings().get("rag", {})
    k = int(cfg.get("top_k", 3) if top_k is None else top_k)
    min_sim = float(cfg.get("min_similarity", 0.08))
    w_tfidf = float(cfg.get("tfidf_weight", 0.45))
    w_dense = float(cfg.get("dense_weight", 0.55))
    damage_boost = float(cfg.get("damage_boost", 0.08))

    normalized_q = _strip_accents(question or "")
    if _INJECTION_PATTERN.search(normalized_q):
        logger.warning("Consulta bloqueada por guardia anti-inyección/seguridad: %r", question[:80])
        return []
    if not _DOMAIN_PATTERN.search(normalized_q):
        logger.info("Consulta fuera de dominio vial (abstención): %r", question[:80])
        return []

    records, word_vec, tfidf_matrix, char_vec, svd, dense_index = corpus_index()
    lex_scores = (tfidf_matrix @ word_vec.transform([question]).T).toarray().ravel()
    q_dense = normalize(svd.transform(char_vec.transform([question])), norm="l2")[0]
    dense_scores = np.clip(dense_index @ q_dense, 0.0, 1.0)

    hybrid = w_tfidf * lex_scores + w_dense * dense_scores

    # Reranking ligero cuando hay daños detectados por visión y coinciden con el tema de la ficha
    if detected_damages:
        damage_terms = {
            _strip_accents(str(d.get("clase", "")).lower())
            for d in detected_damages
            if d.get("clase")
        }
        for idx, rec in enumerate(records):
            haystack = _strip_accents((rec.get("tema", "") + " " + rec.get("texto", "")).lower())
            if any(term in haystack for term in damage_terms) and lex_scores[idx] > 0.05:
                hybrid[idx] += damage_boost

    # Deduplicar por id de ficha conservando el mejor fragmento
    best_by_id: dict[str, dict] = {}
    for idx in np.argsort(hybrid)[::-1]:
        score = float(hybrid[idx])
        if score < min_sim:
            continue
        rec = records[idx]
        rec_id = rec["id"]
        if rec_id not in best_by_id:
            best_by_id[rec_id] = {**rec, "similaridad": round(score, 3)}
        if len(best_by_id) >= k:
            break

    hits = list(best_by_id.values())
    logger.info(
        "RAG recuperó %d fichas para %r -> %s",
        len(hits),
        question[:60],
        [(h["id"], h["similaridad"]) for h in hits],
    )
    return hits
