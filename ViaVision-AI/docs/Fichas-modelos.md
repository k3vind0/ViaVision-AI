# Fichas de los modelos (Model Cards)

## 1. Clasificador de prioridad vial (`models/demo.joblib`)

- **Modelo seleccionado:** Regresión logística multiclase (`LogisticRegression`, `C=5.0`), `scikit-learn 1.8.0`, seleccionado por mayor **macro-F1 de validación** entre los candidatos entrenados (`regresion_logistica`, `random_forest`, `gradient_boosting`) y contrastado frente a `regla_baseline` (regla severidad + tráfico sin fuga) y `dummy_frecuente`.
- **Datos y partición:** 360 observaciones sintéticas coherentes de 120 tramos (`seed=42`). Split disjunto por `id_tramo`: 270 filas (90 tramos) en entrenamiento con `StratifiedGroupKFold(n_splits=4)` y 90 filas (30 tramos) en prueba intacta.
- **Métricas observadas (`results/ml_demo.json` / `results/ml_tabla_comparativa.csv`):**
  - `regresion_logistica` (ganador): F1-train `0.9065`, F1-val `0.8733 ± 0.0566`, F1-test `0.8710`, MAE ordinal test `0.1222`, exactitud test `0.8778`.
  - `random_forest`: F1-train `1.0000`, F1-val `0.7927`, F1-test `0.8477`, MAE ordinal `0.1444`.
  - `gradient_boosting`: F1-train `0.9926`, F1-val `0.8100`, F1-test `0.7790`, MAE ordinal `0.2111`.
  - `regla_baseline` (severidad + tráfico): F1-val `0.6609`, F1-test `0.6033`, MAE ordinal `0.4111`.
  - `dummy_frecuente`: F1-val `0.2028`, F1-test `0.2381`, MAE ordinal `0.4444`.
- **Salidas y riesgo:** Tres probabilidades (`baja`, `media`, `alta`), puntaje `50 × (P(media) + 2·P(alta))`, margen, entropía normalizada, nivel de incertidumbre y factores explicativos. El modelo sintético refleja las regularidades del generador; el modo piloto requiere datos reales validados.

## 2. Detector visual externo (`models/road_damage.onnx`)

- **Arquitectura y versión:** YOLO11n exportado a ONNX (CPU, entrada 640×640, opset 12), procedente del repositorio de Rishabh Aryan (`commit 31406345`). No se realizó reentrenamiento propio en este paquete.
- **Clases e inferencia:** Clases originales `Pothole`, `Crack`, `Manhole`; la aplicación mapea a `Bache` y `Grieta` y excluye `Tapa de registro` (`Manhole`). Umbral operativo `0.25`, NMS por clase con IoU `0.50`, máximo 50 detecciones.
- **Verificación de protocolo (`results/vision_demo.json` sobre 2 muestras):**
  - En `tramo-ejemplo.jpg`: 2/2 baches detectados (`61.8%` y `48.1%`, origen compartido con el repositorio del modelo).
  - En `fhwa-bache.jpg`: fallo documentado — bache real (`[102, 60, 178, 86]`) clasificado como `Grieta` (`29.3%`, caja `[52, 8, 302, 189]`, IoU misma clase `0.0`).
  - Métrica conjunta de protocolo en las 2 muestras: `mAP@50 = 0.8653`, `recall(Bache) = 0.6667`, `precision(Bache) = 1.0`, 1 falsa alarma en `fhwa-control` (vs. `0.0` del baseline heurístico de contraste).
- **Limitaciones:** Sin un dataset de campo anotado e independiente por calle y recorrido, estos valores solo verifican el protocolo de evaluación y muestran la caída de generalización fuera de dominio.
