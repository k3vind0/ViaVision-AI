# Bitácora del Grupo 03

- **Repositorio oficial:** https://github.com/k3vind0/ViaVision-AI
- **Versión base heredada:** `v0.2.0-trujillo` -> **Actualización Fase 1:** refactorización modular, ML completo, RAG híbrido (24 consultas), LangGraph integrado, pruebas `pytest` y capturas reales.

## Integrantes y responsabilidades

| Integrante | Rol asignado | Aportes principales verificados en Fase 1 |
|---|---|---|
| **Kevin Alexis Alva Neciosup** | Líder Técnico / Pipeline ML y Visión | Reestructuración `src/data/` y `src/ml/`, generador coherente por tramo, `SeverityTrafficRuleBaseline`, `GradientBoostingClassifier`, análisis por subgrupos, manifiesto visual `data_sample/vision_demo_manifest.json` y análisis del caso `fhwa-bache.jpg`. |
| **Aguilar Meza Yolver Yonel** | Arquitectura RAG y Agente LangGraph | Ampliación de `data_sample/corpus.json` a 13 fichas trazables, chunking e índice híbrido (TF-IDF + embeddings SVD) en `src/rag/`, guardia anti-inyección, esquemas Pydantic (`src/schemas.py`), nodos `verificar_metadatos` y `evaluar_incertidumbre` en `src/agents/flow.py`, y evaluación de 24 consultas. |
| **Chacon Garcia Anthony Alexander** | Interfaz Streamlit, Pruebas y Documentación | Modularización en `src/ui/components.py` y `app.py` (barras de probabilidad, chat multi-turno, control de modo piloto), suite completa `tests/test_*.py` (`9 passed`), ejecución de `notebooks/Experimentacion.ipynb`, script `scripts/capture_screenshots.py` y sincronización de `docs/`. |

## Registro de hitos

| Fecha | Responsable(s) | Actividad realizada | Evidencia enlazada | Estado |
|---|---|---|---|---|
| 2026-11-23 | Grupo 03 | Prototipo base reproducible e incorporación de Trujillo | `Repositorio.bundle` (`v0.1.0-feria-local`, `v0.2.0-trujillo`) | Completado |
| 2026-11-24 | Kevin A. Alva / Yolver Y. Aguilar / Anthony A. Chacon | Fase 1: Arquitectura modular `src/*`, ML (5 modelos + subgrupos), RAG híbrido (13 fichas, 24 consultas), LangGraph integrado con Pydantic y `logging` | `configs/settings.yaml`, `results/ml_demo.json`, `results/vision_demo.json`, `results/demo.json`, `logs/viavision.log` | Completado |
| 2026-11-24 | Anthony A. Chacon / Kevin A. Alva | Ejecución de `notebooks/Experimentacion.ipynb`, suite `pytest` (9/9) y capturas reales Streamlit | `notebooks/Experimentacion.ipynb`, `tests/`, `docs/capturas/*.png` | Completado |
