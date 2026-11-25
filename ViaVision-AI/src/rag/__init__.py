"""Módulo RAG híbrido (TF-IDF + índice vectorial denso) y asistente LangChain."""
from src.rag.assistant import answer, answer_without_rag
from src.rag.corpus import chunk_document, load_corpus
from src.rag.retriever import corpus_index, recuperar_normativa

__all__ = [
    "answer",
    "answer_without_rag",
    "chunk_document",
    "corpus_index",
    "load_corpus",
    "recuperar_normativa",
]
