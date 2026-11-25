import hashlib
import json
from uuid import uuid4
from langgraph.types import Command

import streamlit as st
from dotenv import load_dotenv
from src.flow import graph
from src.priority import ROOT, VIA, calcular_prioridad
from src.vision import draw_boxes

load_dotenv()
st.set_page_config(page_title='VíaVision AI', page_icon='🛣️', layout='wide')
st.markdown('**GRUPO 03 · SISTEMAS INTELIGENTES · UNT**')
st.title('VíaVision AI')
st.caption('Detecta daños visibles, estima la prioridad y prepara el expediente de inspección.')

with st.sidebar:
    st.header('Tramo vial')
    mode = st.radio('Modo', ['Demostración', 'Piloto con datos validados'])
    demo = mode == 'Demostración'
    tramo = st.text_input('Identificador', 'PILOTO-01', max_chars=50)
    st.caption('Provincia de Trujillo · La Libertad')
    district = st.text_input('Distrito', 'Trujillo' if demo else '', max_chars=80, key=f'district_{demo}').strip()
    road = st.text_input('Vía o sector', 'Tramo simulado' if demo else '', max_chars=120, key=f'road_{demo}').strip()
    coordinates = None
    if st.checkbox('Añadir coordenadas de inspección'):
        lat = st.number_input('Latitud', min_value=-90.0, max_value=90.0,
                              value=-8.1117 if demo else None, format='%.6f', key=f'lat_{demo}')
        lon = st.number_input('Longitud', min_value=-180.0, max_value=180.0,
                              value=-79.0289 if demo else None, format='%.6f', key=f'lon_{demo}')
        if lat is not None and lon is not None:
            coordinates = {'latitud': lat, 'longitud': lon}
        else:
            st.caption('Completa ambas coordenadas para incluirlas.')
    location = {'departamento': 'La Libertad', 'provincia': 'Trujillo',
                'distrito': district, 'via_o_sector': road, 'coordenadas': coordinates,
                'origen': 'simulado' if demo else 'declarado por el usuario'}

if demo:
    st.info('Demostración. El tramo y la prioridad son simulados. La foto incluida procede de un conjunto externo a Trujillo.')

left, right = st.columns(2, gap='large')
with left:
    st.subheader('Imagen de la vía')
    upload = st.file_uploader('Carga JPG, PNG o WebP', type=['jpg', 'jpeg', 'png', 'webp'])
    example = ROOT / 'data_sample' / 'tramo-ejemplo.jpg'
    use_sample = st.checkbox('Usar foto de demostración', value=True, disabled=not demo)
    raw = upload.getvalue() if upload else (
        example.read_bytes() if demo and use_sample and example.exists() else None)
    if raw:
        try:
            st.image(draw_boxes(raw, []), caption='Imagen de entrada', width='stretch')
        except (OSError, ValueError):
            st.error('El archivo no contiene una imagen compatible.')
            raw = None

with right:
    st.subheader('Datos para estimar prioridad')
    traffic = st.slider('Tráfico diario (vehículos)', 0, 100000, 6500, step=100)
    age = st.number_input('Antigüedad del pavimento (años)', 0, 100, 14)
    repairs = st.slider('Reparaciones en los últimos 24 meses', 0, 100, 3)
    road_type = st.selectbox('Tipo de vía', VIA, index=1)
    rain = st.number_input('Precipitación en 30 días (mm)', 0.0, 5000.0, 25.0)
    severity = st.selectbox('Severidad declarada por inspección', [0, 1, 2, 3], index=2,
                            format_func=lambda x: ['0 · ninguna', '1 · baja', '2 · media', '3 · alta'][x])
    values = dict(trafico_diario=traffic, antiguedad_anios=age, reparaciones_24m=repairs,
                  tipo_via=road_type, precipitacion_mm_30d=rain, severidad_inspeccionada=severity)
    try:
        preview = calcular_prioridad(values, demo)
        st.metric('Prioridad estimada', preview['clase'].capitalize(), f"{preview['puntaje']}/100")
    except ValueError as exc:
        st.warning(str(exc))

st.subheader('Consulta técnica y expediente')
question = st.text_input('Pregunta sobre la inspección', '¿Qué debo medir en un bache?', max_chars=500)
signature = (hashlib.sha256(raw).hexdigest() if raw else None, tuple(values.items()),
             demo, tramo, question, location)

def workflow():
    if 'workflow' not in st.session_state:
        st.session_state.workflow = graph()
    return st.session_state.workflow


def run(approved):
    if approved:
        payload = Command(resume=True)
    else:
        st.session_state.thread_config = {'configurable': {'thread_id': str(uuid4())}}
        payload = {'image': raw, 'values': values, 'demo': demo, 'tramo': tramo.strip(),
                   'question': question, 'location': location}
    with st.spinner('Procesando imagen, prioridad y fuentes…'):
        st.session_state.result = workflow().invoke(payload, st.session_state.thread_config)
        st.session_state.signature = signature

if st.button('Analizar tramo', type='primary',
             disabled=not raw or not tramo.strip() or not district or not road or not question.strip()):
    run(False)

if st.session_state.get('signature') == signature:
    result = st.session_state.result
    for error in result.get('errors', []):
        st.error(error)
    vision = result['vision']
    if vision['estado'].startswith('no evaluable'):
        st.warning(vision['estado'])
    else:
        st.image(draw_boxes(raw, vision['danos']), caption=vision['estado'], width='stretch')
        if vision['danos']:
            st.dataframe(vision['danos'], hide_index=True)
    st.write(result['response'])
    st.caption(result['mode'])
    for item in result['evidence']:
        with st.expander(f"{item['id']} · {item['fuente']}"):
            st.write(item['texto'])
            st.link_button('Consultar fuente original', item['url'])
    with st.expander('Traza del agente'):
        for name in result['trace']:
            st.write(name)
    ready = (bool(result['priority']) and not result.get('errors')
             and not vision['estado'].startswith('no evaluable'))
    if ready and not result.get('record'):
        if st.button('Aprobar y preparar expediente preliminar'):
            run(True)
            st.rerun()
    if result.get('record'):
        st.download_button('Descargar expediente', json.dumps(result['record'], ensure_ascii=False, indent=2),
                           file_name='expediente-viavision.json', mime='application/json')
