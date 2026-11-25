# Bitácora del Grupo 11

- **Repositorio oficial:** https://github.com/k3vind0/ViaVision-AI
- **Versión base heredada:** `v0.2.0-trujillo` -> **Actualización Fase 1–2:** arquitectura modular `src/*`, ML completo, RAG híbrido (24 consultas), LangGraph integrado con Pydantic, interfaz ejecutiva Streamlit, pruebas `pytest` y entregables académicos completos.

## Integrantes y responsabilidades

| Integrante | Rol oficial | Aportes y autoría verificada |
|---|---|---|
| **Kevin Alexis Alva Neciosup** | Ingeniería de Datos, ML y Visión Computacional | Reestructuración `src/data/` y `src/ml/`, generador coherente por tramo, `SeverityTrafficRuleBaseline`, `GradientBoostingClassifier`, análisis por subgrupos, manifiesto `data_sample/vision_demo_manifest.json`, análisis `fhwa-bache.jpg` y capítulos metodológicos/resultados ML-Visión. |
| **Aguilar Meza Yolver Yonel** | LLM, LangChain y Pipeline RAG | Ampliación de `data_sample/corpus.json` a 13 fichas trazables, chunking e índice híbrido (TF-IDF + embeddings SVD) en `src/rag/`, guardia anti-inyección, esquemas Pydantic (`src/schemas.py`), evaluación de 24 consultas y documentación RAG/LLM/Referencias. |
| **Chacon Garcia Anthony Alexander** | Orquestación LangGraph, UI Streamlit y QA | Grafo LangGraph de 7 nodos (`src/agents/flow.py`), rediseño ejecutivo de `app.py` y `src/ui/`, suite `tests/test_*.py` (`9 passed`), cuaderno `notebooks/Experimentacion.ipynb`, script `scripts/capture_screenshots.py`, manuales, póster y sustentación. |

## Registro de hitos

| Fecha | Responsable(s) | Actividad realizada | Evidencia enlazada | Estado |
|---|---|---|---|---|
| 2026-11-23 | Grupo 11 | Prototipo base reproducible e incorporación de Trujillo | `Repositorio.bundle` (`v0.1.0-feria-local`, `v0.2.0-trujillo`) | Completado |
| 2026-11-24 | Kevin A. Alva / Yolver Y. Aguilar / Anthony A. Chacon | Fase 1: Arquitectura modular `src/*`, ML (5 modelos + subgrupos), RAG híbrido (13 fichas, 24 consultas), LangGraph integrado con Pydantic y `logging` | `configs/settings.yaml`, `results/ml_demo.json`, `results/vision_demo.json`, `results/demo.json`, `logs/viavision.log` | Completado |
| 2026-11-24 | Anthony A. Chacon / Kevin A. Alva | Ejecución de `notebooks/Experimentacion.ipynb`, rediseño UI/UX Streamlit, `pytest` (9/9) y capturas reales | `notebooks/Experimentacion.ipynb`, `tests/`, `docs/capturas/*.png` | Completado |
| 2026-11-24 | Grupo 11 | Fase 2: Informe final Word, Manuales Word, Póster PDF con QR y Sustentación PPTX con plantilla oficial | `docs/Informe_ViaVision_AI.docx`, `docs/Manuales_ViaVision_AI.docx`, `docs/Poster_ViaVision_AI.pdf`, `docs/Sustentacion_ViaVision_AI.pptx` | En cierre |
