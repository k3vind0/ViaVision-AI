# Diccionario de variables

| Campo | Unidad o categorías | Uso |
|---|---|---|
| id_tramo | Texto | Identificador de agrupación, excluido de las entradas del predictor |
| trafico_diario | Vehículos por día | Entrada |
| antiguedad_anios | Años desde instalación o renovación | Entrada |
| reparaciones_24m | Conteo en los 24 meses anteriores | Entrada |
| tipo_via | local, colectora, arterial | Entrada |
| precipitacion_mm_30d | Milímetros acumulados en 30 días | Entrada |
| severidad_inspeccionada | Entero de 0 a 3 registrado por inspección | Entrada |
| prioridad_validada | baja, media, alta | Etiqueta de entrenamiento. En demo se deriva de una regla sintética con ruido. En piloto requiere evidencia de inspección. |

Consultar licencia institucional antes de distribuir registros reales. Los faltantes de las entradas se imputan dentro de cada partición de entrenamiento. El formulario interactivo exige valores válidos.
