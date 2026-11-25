# Estado de entrega frente a la guía

| Criterio | Evidencia disponible (Fase 1 verificada) | Pendiente (Fase 2 / trabajo de campo futuro) |
|---|---|---|
| Estructura, config y logging | `src/{data,ml,vision,rag,agents,ui}/`, `configs/settings.yaml`, `src/schemas.py` (Pydantic v2), `logs/viavision.log`, `requirements.txt` compatible Python 3.12–3.14 | Despliegue en servidor municipal si se requiere |
| Machine Learning | Generador coherente por tramo, `regla_baseline` + `dummy_frecuente` + 3 candidatos (`LogisticRegression`, `RandomForest`, `GradientBoosting`), métricas train/val/test, subgrupos y 3 figuras PNG | Sustituir fichas sintéticas por registros municipales con `prioridad_validada` real |
| Visión | YOLO11n ONNX, `evaluate_vision.py`, manifiesto de muestra `vision_demo_manifest.json`, análisis cuantitativo/visual de `fhwa-bache.jpg` (`fig_vision_fhwa_analisis.png`) | Reunir y anotar dataset propio por calle/recorrido en Trujillo y reentrenar/ajustar detector |
| RAG y LLM | 13 fichas curadas (FHWA/MTC), chunking, índice híbrido TF-IDF + SVD denso, evaluación de 24 consultas (Sin-RAG vs Con-RAG, 100 % Hit@3 y bloqueo adversarial), Pydantic `with_structured_output` | Pruebas de fidelidad con clave OpenAI activa en producción |
| LangGraph y Streamlit | Grafo de 7 nodos (`verificar_metadatos` → `prioridad` → `vision` → `evidencia` → `incertidumbre` → `revision` → `registro`), chatbot multi-turno, barras de probabilidad, 9/9 `pytest`, 7 capturas reales en `docs/capturas/` | — |
| Material académico (Fase 2) | `notebooks/Experimentacion.ipynb` ejecutado con salidas; `README.md`, `Bitacora.md`, `Diccionario.md`, `Fichas-modelos.md`, `Fuentes.md` actualizados | Actualizar `Informe_ViaVision_AI.docx`, `Manuales_ViaVision_AI.docx`, `Poster_ViaVision_AI.pdf`, `Sustentacion_ViaVision_AI.pptx` y video continuo en vivo |
