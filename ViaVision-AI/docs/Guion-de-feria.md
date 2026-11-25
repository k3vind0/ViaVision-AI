# Guion de feria

**0:00–0:30 | Problema.** La inspección vial combina fotos, datos de servicio y criterios técnicos dispersos. VíaVision AI reúne esas piezas en un expediente preliminar para revisión.

**0:30–1:10 | Visión.** Analizar una foto, señalar las cajas y leer la confianza. Explicar que YOLO ubica daños visibles y que la medición de profundidad requiere inspección en campo.

**1:10–1:50 | Prioridad.** Variar tráfico de 500 a 9800 vehículos diarios. Indicar que la regresión logística se eligió por macro-F1 de validación. Comparar con el baseline y recordar que se usaron fichas sintéticas.

**1:50–2:35 | Criterios.** Consultar cuándo aplica el parchado superficial en pavimento flexible. Abrir MTC §410.1. Explicar que TF-IDF recupera fichas por similitud y que LangChain puede redactar con ellas cuando hay una clave configurada.

**2:35–3:15 | Trazas y revisión.** Abrir la traza. Mostrar que antes de aprobar no se puede descargar. Aprobar, descargar y abrir el JSON para ubicar los daños, la prioridad y las fuentes.

**3:15–3:30 | Límites.** Para trasladar el piloto a un municipio, reunir etiquetas de prioridad, evaluar visión con recorridos nuevos y probar la fidelidad del generador con un corpus ampliado.

## Preguntas para practicar

1. ¿Cómo se verificó que los tramos de entrenamiento y prueba fueran disjuntos?
2. ¿Por qué usar macro-F1 y además MAE ordinal?
3. ¿Qué significa la confianza de una caja y cómo se diferencia del puntaje de prioridad?
4. ¿Cómo se distingue una coincidencia documental de una afirmación bien sustentada?
5. ¿Qué estado conserva LangGraph al solicitar la revisión humana?
6. ¿Qué evidencia faltaría para validar el sistema en una zona real?

El MP4 incluido dura aproximadamente 3:15 y resume la evidencia de la ejecución local con placas de texto y la imagen detectada. Para mostrar la interacción tal como se ve en la feria, usar el guion de arriba con Streamlit en vivo.

Al abrir la app, muestra el ámbito «Provincia de Trujillo · La Libertad». Completa el distrito y la vía o sector en la barra lateral. Al abrir el expediente, ubica el bloque 'ubicacion' y explica que las coordenadas de demostración son opcionales y que la foto de ejemplo es externa.
