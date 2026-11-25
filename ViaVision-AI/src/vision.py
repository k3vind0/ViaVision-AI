from functools import lru_cache
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from src.priority import ROOT

LABELS = ['Grieta longitudinal', 'Grieta transversal', 'Fisuración por fatiga', 'Bache']


def read_image(raw):
    image = Image.open(BytesIO(raw))
    if max(image.size) > 8000 or min(image.size) < 32:
        raise ValueError('Carga una imagen entre 32 y 8000 píxeles por lado.')
    return ImageOps.exif_transpose(image).convert('RGB')


@lru_cache
def session():
    import onnxruntime as ort
    return ort.InferenceSession(str(ROOT / 'models' / 'road_damage.onnx'),
                                providers=['CPUExecutionProvider'])


def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - w*h
    return w*h/union if union > 0 else 0.0


def detectar_danos(raw, confidence=.25):
    image = read_image(raw)
    if not (ROOT / 'models' / 'road_damage.onnx').is_file():
        return {'estado': 'no evaluable: detector ONNX no instalado', 'danos': []}
    model = session()
    scale = min(640/image.width, 640/image.height)
    w, h = round(image.width*scale), round(image.height*scale)
    px, py = (640-w)//2, (640-h)//2
    canvas = Image.new('RGB', (640, 640), (114, 114, 114))
    canvas.paste(image.resize((w, h), Image.Resampling.BILINEAR), (px, py))
    pixels = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    output = model.run(None, {model.get_inputs()[0].name: pixels})[0][0]
    if output.ndim != 2 or output.shape[0] not in (7, 8):
        raise ValueError('El modelo debe exportar YOLO ONNX sin NMS de tres o cuatro clases.')
    labels = ['Bache', 'Grieta', 'Tapa de registro'] if output.shape[0] == 7 else LABELS
    candidates = []
    for x, y, bw, bh, *scores in output.T:
        cls = int(np.argmax(scores))
        score = float(scores[cls])
        if score < confidence or labels[cls] == 'Tapa de registro':
            continue
        box = [(x-bw/2-px)/scale, (y-bh/2-py)/scale,
               (x+bw/2-px)/scale, (y+bh/2-py)/scale]
        box = [int(np.clip(v, 0, image.size[i%2]-1)) for i, v in enumerate(box)]
        if box[2] > box[0] and box[3] > box[1]:
            candidates.append({'clase': labels[cls], 'caja': box, 'confianza': score})
    selected = []
    for item in sorted(candidates, key=lambda d: d['confianza'], reverse=True)[:1000]:
        if all(old['clase'] != item['clase'] or iou(old['caja'], item['caja']) < .5 for old in selected):
            selected.append(item)
        if len(selected) == 50:
            break
    return {'estado': 'evaluada' if selected else 'sin hallazgos sobre el umbral', 'danos': selected}


def draw_boxes(raw, detections):
    image = read_image(raw)
    draw = ImageDraw.Draw(image)
    for item in detections:
        x, y, _, _ = item['caja']
        draw.rectangle(item['caja'], outline='#00ffd1', width=max(2, image.width//200))
        draw.text((x, max(y-13, 0)), f"{item['clase']} {item['confianza']:.0%}",
                  fill='#00ffd1', stroke_width=2, stroke_fill='black')
    return image
