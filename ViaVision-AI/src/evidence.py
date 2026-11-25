"""Compatibilidad hacia atrás: re-exporta src.rag."""
from src.rag.assistant import answer, answer_without_rag
from src.rag.retriever import corpus_index, recuperar_normativa

__all__ = ["answer", "answer_without_rag", "corpus_index", "recuperar_normativa"]
