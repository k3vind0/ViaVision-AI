from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['trafico_diario', 'antiguedad_anios', 'reparaciones_24m', 'tipo_via',
            'precipitacion_mm_30d', 'severidad_inspeccionada']
CLASSES = ['baja', 'media', 'alta']
RANGES = {'trafico_diario': (0, 100000), 'antiguedad_anios': (0, 100),
          'reparaciones_24m': (0, 100), 'precipitacion_mm_30d': (0, 5000),
          'severidad_inspeccionada': (0, 3)}
VIA = ['local', 'colectora', 'arterial']


@lru_cache
def load_model(path):
    return joblib.load(path)


def calcular_prioridad(datos, modo_demo=True):
    if any(k not in datos for k in FEATURES):
        raise ValueError('Faltan variables del tramo.')
    for name, (low, high) in RANGES.items():
        if not low <= float(datos[name]) <= high:
            raise ValueError(f'{name} debe estar entre {low} y {high}.')
    if datos['tipo_via'] not in VIA:
        raise ValueError('Tipo de vía desconocido.')
    path = ROOT / 'models' / ('demo.joblib' if modo_demo else 'validado.joblib')
    if not path.is_file():
        raise ValueError('Instala primero un modelo entrenado con registros validados.')
    model = load_model(path)
    frame = pd.DataFrame([datos], columns=FEATURES)
    probs = dict(zip(model.classes_, model.predict_proba(frame)[0]))
    expected = sum(i * probs.get(c, 0.0) for i, c in enumerate(CLASSES)) * 50
    return {'clase': str(model.predict(frame)[0]), 'puntaje': round(float(expected), 1),
            'probabilidades': {str(k): round(float(v), 4) for k, v in probs.items()},
            'origen': 'entrenamiento sintético' if modo_demo else 'modelo con registros validados',
            'variables': datos}
