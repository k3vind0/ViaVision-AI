# Fuentes y procedencia

| Artefacto | Procedencia |
|---|---|
| road_damage.onnx | [Rishabh Aryan, commit 31406345](https://github.com/Rishabh-bgp/machine-learning-part-3/tree/31406345c784f2e66216f42a221627da2e26e7c8). YOLO11n, clases originales Pothole, Crack, Manhole. La app descarta Manhole. El proyecto de origen declara MIT y el ONNX lleva la declaración AGPL-3.0 de Ultralytics. Se adjuntan avisos y enlaces al código y receta originales. |
| tramo-ejemplo.jpg | Mismo proyecto, docs/samples/sample4.jpg. Uso de demostración. El repositorio atribuye los datos a Lorenzo Arcioni y declara MIT para ese dataset. |
| fhwa-bache.jpg | FHWA-HRT-13-092, capítulo 1, figura 28. https://www.fhwa.dot.gov/publications/research/infrastructure/pavements/ltpp/13092/images/image030.jpg |
| tramos_sinteticos.csv | scripts/train_ml.py. Semilla 42. 360 filas de 120 grupos sintéticos. |
| corpus.json | Paráfrasis curadas de FHWA-HRT-13-092, cap. 1, y del Manual de Carreteras del MTC, secciones 401 y 410.1. Cada ficha incluye la URL oficial. |
| RDD2022 | Dataset externo de referencia para futuras comparaciones. https://doi.org/10.6084/m9.figshare.21431547.v1 |
