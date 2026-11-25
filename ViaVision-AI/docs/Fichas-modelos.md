# Fichas de los modelos

## Regresión logística multiclase

- Versión: scikit-learn 1.8.0. Artefacto: models/demo.joblib.
- Entradas: seis variables descritas en Diccionario.md. Salida: tres probabilidades y un índice ordinal de 0 a 100.
- Datos: generador reproducible, 360 filas sintéticas agrupadas por tramo. Ajuste C=0.2 seleccionado por macro-F1 de validación. Prueba: 90 filas y 30 tramos.
- Métricas: macro-F1 0.766 y MAE ordinal 0.222. La matriz y las alternativas están en results/ml_demo.json. De los 21 casos altos, 9 pasaron a la categoría media.
- Uso: demostración académica del pipeline. Para piloto se necesita entrenar con etiquetas respaldadas por inspección.
- Riesgo: el modelo reproduce las regularidades del generador. El índice es una agregación de probabilidades, no una probabilidad de daño.

## YOLO11n externo

- Versión: ONNX, CPU, entrada 640×640, fijo por hash del commit descrito en Fuentes.md.
- Entrenamiento: realizado por Rishabh Aryan. Consulte su receta pública, licencia y datos. No hay reentrenamiento propio en este paquete.
- Entradas: fotos de superficie vial. Salidas: cajas y confianza, con umbral 0.25 e IoU 0.5 para supresión por clase.
- Uso: orientar revisión de baches y grietas visibles. La clase tapa de registro se excluye.
- Limitaciones: la foto de demo puede compartir origen con entrenamiento. La foto FHWA examinada produjo una predicción errónea como grieta. La evaluación de campo requiere calles y recorridos independientes.
