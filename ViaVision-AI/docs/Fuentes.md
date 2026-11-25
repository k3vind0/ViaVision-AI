# Fuentes y procedencia

| Artefacto | Procedencia y licencia |
|---|---|
| `models/road_damage.onnx` | [Rishabh Aryan, commit 31406345](https://github.com/Rishabh-bgp/machine-learning-part-3/tree/31406345c784f2e66216f42a221627da2e26e7c8). YOLO11n (clases originales `Pothole`, `Crack`, `Manhole`; la app excluye `Manhole`). El repositorio de origen declara MIT (`docs/LICENSE-MODEL-MIT.txt`) y los metadatos ONNX declaran AGPL-3.0 de Ultralytics (`docs/LICENSE-AGPL-3.0.txt`). |
| `data_sample/tramo-ejemplo.jpg` | Mismo repositorio (`docs/samples/sample4.jpg`, 640×360 px). Uso demostrativo; dataset atribuido a Lorenzo Arcioni bajo licencia MIT. |
| `data_sample/fhwa-bache.jpg` | FHWA-HRT-13-092, capítulo 1, figura 28 (303×193 px). https://www.fhwa.dot.gov/publications/research/infrastructure/pavements/ltpp/13092/images/image030.jpg |
| `data_sample/vision_demo_manifest.json` | Manifiesto de verificación de protocolo anotado sobre las 2 imágenes incluidas en `data_sample/`. |
| `data_sample/tramos_sinteticos.csv` | Generado por `src/data/synthetic.py` (`seed=42`, 360 filas, 120 tramos coherentes). |
| `data_sample/corpus.json` | 13 fichas con paráfrasis curadas y sección/página trazables de FHWA-HRT-13-092 (cap. 1) y Manual de Carreteras del MTC — Conservación Vial (RD 08-2014-MTC/14 y 05-2016-MTC/14), con URL oficial. |
| `RDD2022` | Dataset externo de referencia para futura validación de campo: https://doi.org/10.6084/m9.figshare.21431547.v1 |
