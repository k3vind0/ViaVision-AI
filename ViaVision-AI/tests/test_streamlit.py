from streamlit.testing.v1 import AppTest


def test_demo_streamlit():
    app = AppTest.from_file('app.py', default_timeout=30).run()
    assert not app.exception
    assert app.metric[0].value == 'Media'
    app.slider[0].set_value(500).run()
    assert '16.0/100' in app.metric[0].delta
    next(widget for widget in app.text_input if widget.label == 'Pregunta sobre la inspección').set_value('¿Cuándo aplica el parchado superficial en pavimento flexible?').run()
    app.button[0].click().run()
    assert not app.exception
    assert any('MTC-410' in x.value for x in app.markdown)
    assert not app.get('download_button')
    app.button[1].click().run()
    assert not app.exception
    assert len(app.get('download_button')) == 1
