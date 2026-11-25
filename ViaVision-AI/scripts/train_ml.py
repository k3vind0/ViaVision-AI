"""Comparación con grupos disjuntos y ajuste exclusivamente en entrenamiento."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.priority import CLASSES, FEATURES, ROOT


def synthetic():
    rng = np.random.default_rng(42)
    count = 360
    traffic = rng.integers(100, 10500, count)
    age = rng.integers(0, 35, count)
    repairs = rng.integers(0, 8, count)
    severity = rng.integers(0, 4, count)
    rain = rng.integers(0, 160, count)
    score = traffic / 3100 + age / 20 + repairs / 5 + severity * 1.05 + rng.normal(0, .7, count)
    labels = np.asarray(CLASSES)[np.digitize(score, [4.2, 6.4])]
    return pd.DataFrame(dict(id_tramo=[f'SIM-{i//3:03}' for i in range(count)],
                             trafico_diario=traffic, antiguedad_anios=age, reparaciones_24m=repairs,
                             tipo_via=rng.choice(['local', 'colectora', 'arterial'], count),
                             precipitacion_mm_30d=rain, severidad_inspeccionada=severity,
                             prioridad_validada=labels))


def train(frame, demo, group):
    target = 'prioridad_validada'
    frame = frame.drop_duplicates().reset_index(drop=True)
    if frame[[group, target]].isna().any().any() or set(frame[target]) != set(CLASSES):
        raise ValueError('Se requieren grupos completos y las tres etiquetas baja, media y alta.')
    x, y, groups = frame[FEATURES], frame[target], frame[group]
    train_rows, test_rows = next(GroupShuffleSplit(test_size=.25, random_state=42).split(x, y, groups))
    if any(set(y.iloc[rows]) != set(CLASSES) for rows in [train_rows, test_rows]):
        raise ValueError('Cada partición debe contener las tres clases.')
    numeric = [f for f in FEATURES if f != 'tipo_via']
    preprocess = ColumnTransformer([
        ('numeric', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric),
        ('category', make_pipeline(SimpleImputer(strategy='most_frequent'),
                                   OneHotEncoder(handle_unknown='ignore')), ['tipo_via'])])
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    candidates = {
        'baseline': (DummyClassifier(strategy='most_frequent'), {}),
        'logistica': (LogisticRegression(max_iter=800), {'logisticregression__C': [.2, 1, 5]}),
        'bosque': (RandomForestClassifier(n_estimators=80, random_state=42),
                   {'randomforestclassifier__max_depth': [4, 8, None]})}
    models, scores = {}, {}
    x_train, y_train = x.iloc[train_rows], y.iloc[train_rows]
    for name, (estimator, grid) in candidates.items():
        search = GridSearchCV(make_pipeline(preprocess, estimator), grid, scoring='f1_macro',
                              cv=cv, error_score='raise')
        search.fit(x_train, y_train, groups=groups.iloc[train_rows])
        predictions = search.predict(x.iloc[test_rows])
        actual = y.iloc[test_rows]
        models[name] = search.best_estimator_
        scores[name] = dict(f1_validacion=search.best_score_, parametros=search.best_params_,
                            f1_test=f1_score(actual, predictions, labels=CLASSES, average='macro',
                                             zero_division=0),
                            mae_ordinal=mean_absolute_error(actual.map(CLASSES.index),
                                                            [CLASSES.index(p) for p in predictions]),
                            matriz=confusion_matrix(actual, predictions, labels=CLASSES).tolist())
    best = max(models, key=lambda k: scores[k]['f1_validacion'])
    suffix = 'demo' if demo else 'validado'
    joblib.dump(models[best], ROOT / 'models' / f'{suffix}.joblib')
    report = dict(origen='sintético' if demo else 'archivo aportado', modelo=best, clases=CLASSES,
                  train=len(train_rows), test=len(test_rows), resultados=scores)
    (ROOT / 'results' / f'ml_{suffix}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    frame.assign(particion=np.where(frame.index.isin(test_rows), 'test', 'train')
                 )[[group, 'particion']].drop_duplicates().to_csv(
                     ROOT / 'results' / f'split_{suffix}.csv', index=False)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--grupo', default='id_tramo')
    args = parser.parse_args()
    frame = pd.read_csv(args.csv) if args.csv else synthetic()
    if not args.csv:
        frame.to_csv(ROOT / 'data_sample' / 'tramos_sinteticos.csv', index=False)
    print(json.dumps(train(frame, args.csv is None, args.grupo), indent=2))
