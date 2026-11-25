# Diccionario de variables

## 1. Variables tabulares de prioridad vial (`data_sample/tramos_sinteticos.csv`)

| Campo | Tipo / Unidad o categorías | Uso y validación |
|---|---|---|
| `id_tramo` | Texto (`SIM-000` .. `SIM-119`) | Identificador de agrupación para `GroupShuffleSplit` y `StratifiedGroupKFold`. Excluido como predictor. Cada tramo mantiene constante su `tipo_via` y una progresión coherente de antigüedad. |
| `trafico_diario` | Entero (vehículos/día, `0` a `100000`) | Entrada numérica. Rango acorde al tipo de vía en datos sintéticos. |
| `antiguedad_anios` | Entero (años, `0` a `100`) | Entrada numérica desde instalación o renovación. |
| `reparaciones_24m` | Entero (conteo en 24 meses, `0` a `100`) | Entrada numérica de intervenciones recientes. |
| `tipo_via` | Categórica (`local`, `colectora`, `arterial`) | Entrada categórica codificada con `OneHotEncoder(handle_unknown="ignore")` dentro de cada partición. |
| `precipitacion_mm_30d` | Numérico (mm en 30 días, `0.0` a `5000.0`) | Entrada numérica climática. |
| `severidad_inspeccionada` | Entero ordinal (`0` a `3`) | Entrada declarada por inspección (`0 · ninguna`, `1 · baja`, `2 · media`, `3 · alta`). |
| `prioridad_validada` | Categórica ordinal (`baja`, `media`, `alta`) | Variable objetivo. En demo proviene de regla sintética con ruido; en piloto requiere etiqueta validada por inspección. |

## 2. Campos del corpus documental (`data_sample/corpus.json`)

| Campo | Descripción |
|---|---|
| `id` | Identificador único verificable (ej. `FHWA-B8`, `MTC-410`, `MTC-PRI`). |
| `documento` | Norma o manual de origen (`FHWA-HRT-13-092` o `MTC MCV-2014/2016`). |
| `seccion` / `pagina` | Ubicación exacta de capítulo, sección o página dentro del documento. |
| `tema` | Palabras clave de dominio para indexación híbrida. |
| `fuente` / `url` | Cita corta y enlace oficial público verificable. |
| `texto` | Paráfrasis curada utilizada como evidencia en `ExpedientePreliminar`. |

Los valores faltantes en entradas tabulares reales se imputan dentro del `ColumnTransformer` exclusivamente sobre la partición de entrenamiento (mediana para numéricas, moda para `tipo_via`).
