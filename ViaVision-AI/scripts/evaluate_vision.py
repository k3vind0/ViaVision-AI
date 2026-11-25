"""Evalúa detecciones contra cajas verificadas de calles y recorridos nuevos."""
import argparse
import json
from pathlib import Path

import numpy as np
from src.vision import detectar_danos, iou


def evaluate(manifest_path):
    frames = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not frames or any(not f.get('calle') or not f.get('recorrido') for f in frames):
        raise ValueError('Cada foto debe tener calle y recorrido identificados.')
    if len({f['imagen'] for f in frames}) != len(frames):
        raise ValueError('Hay entradas de imagen repetidas.')
    actual, predictions = {}, {}
    for i, frame in enumerate(frames):
        for box in frame['cajas']:
            actual.setdefault(box['clase'], {}).setdefault(i, []).append(box['caja'])
        raw = (manifest_path.parent / frame['imagen']).read_bytes()
        result = detectar_danos(raw, confidence=.01)
        if result['estado'].startswith('no evaluable'):
            raise ValueError(result['estado'])
        for item in result['danos']:
            predictions.setdefault(item['clase'], []).append((i, item))
    aps, recall = {}, {}
    false_alarms = {f['recorrido']: 0 for f in frames}
    for label in sorted(set(actual) | set(predictions)):
        gt = actual.get(label, {})
        count = sum(map(len, gt.values()))
        used, flags, selected = set(), [], []
        for i, item in sorted(predictions.get(label, []),
                              key=lambda entry: entry[1]['confianza'], reverse=True):
            overlaps = [(iou(item['caja'], b), j) for j,b in enumerate(gt.get(i, []))
                        if (i,j) not in used]
            match = max(overlaps, default=(0.0,-1))
            positive = match[0] >= .5
            if positive:
                used.add((i,match[1]))
            flags.append(int(positive))
            if item['confianza'] >= .25:
                selected.append(int(positive))
                false_alarms[frames[i]['recorrido']] += int(not positive)
        if count:
            tps = np.cumsum(flags)
            recalls = tps / count
            precision = tps / np.arange(1,len(flags)+1)
            aps[label] = float(np.mean([max(precision[recalls >= r], default=0.)
                                        for r in np.linspace(0,1,101)]))
            recall[label] = sum(selected)/count
    return dict(mAP50=float(np.mean(list(aps.values()))) if aps else None, ap50=aps,
                recall_umbral_025=recall, falsas_alarmas_por_recorrido=false_alarms,
                imagenes=len(frames))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    print(json.dumps(evaluate(p.parse_args().manifest), ensure_ascii=False, indent=2))
