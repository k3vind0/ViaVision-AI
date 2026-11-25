"""Captura automática de pantallas reales de la app Streamlit rediseñada."""
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
CAP_DIR = ROOT / "docs" / "capturas"
URL = os.getenv("VIAVISION_URL", "http://localhost:8501")

BROWSER_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]


def _pick_browser() -> str | None:
    for path in BROWSER_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def _wait_idle(page, extra_ms: int = 1000) -> None:
    page.wait_for_selector("text=Prioridad estimada", timeout=30000)
    page.wait_for_selector("button:has-text('Analizar tramo')", timeout=30000)
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"] button')",
        timeout=30000,
    )
    page.wait_for_timeout(extra_ms)


def _scroll_to(page, selector: str) -> None:
    page.evaluate(
        f"""() => {{
            const el = document.querySelector({selector!r});
            if (el) el.scrollIntoView({{block: 'start', behavior: 'instant'}});
        }}"""
    )
    page.wait_for_timeout(450)


def main() -> None:
    CAP_DIR.mkdir(parents=True, exist_ok=True)
    exe = _pick_browser()
    with sync_playwright() as p:
        launch_kwargs = {"headless": True}
        if exe:
            launch_kwargs["executable_path"] = exe
        browser = p.chromium.launch(**launch_kwargs)
        context = browser.new_context(viewport={"width": 1440, "height": 1350}, device_scale_factor=1.35)
        page = context.new_page()

        # 1. Vista inicial (cabecera ejecutiva, sidebar, formulario y KPIs previos)
        page.goto(URL, wait_until="domcontentloaded")
        _wait_idle(page, 1400)
        _scroll_to(page, ".vv-hero")
        page.screenshot(path=str(CAP_DIR / "01-interfaz-inicial.png"))

        # 2. Escenario 1: Analizar tramo (KPIs + imagen anotada YOLO + tabla)
        page.get_by_role("button", name="Analizar tramo").click()
        page.wait_for_selector("text=Control Humano Obligatorio", timeout=45000)
        _wait_idle(page, 1000)
        _scroll_to(page, ".vv-kpi-grid")
        page.screenshot(path=str(CAP_DIR / "02-escenario1-vision.png"))

        # 3. Escenario 2: Variar tráfico a 9800 veh/día, re-analizar y abrir pestaña Escenario 2
        slider = page.locator("[data-testid='stSlider'] [role='slider']").first
        slider.focus()
        for _ in range(33):
            slider.press("ArrowRight")
        _wait_idle(page, 800)
        page.get_by_role("button", name="Analizar tramo").click()
        page.wait_for_selector("text=Alta · 99.1/100", timeout=45000)
        _wait_idle(page, 700)
        page.get_by_role("tab", name="Escenario 2 · Prioridad y Factores (ML)").click()
        _wait_idle(page, 600)
        _scroll_to(page, ".vv-kpi-grid")
        page.screenshot(path=str(CAP_DIR / "03-escenario2-prioridad.png"))

        # 4. Escenario 3: Consultar MTC §410.1, abrir pestaña Escenario 3 y mostrar evidencia + timeline + revisión
        q_box = page.get_by_label("Pregunta sobre la inspección")
        q_box.fill("¿Cuándo aplica el parchado superficial en pavimento flexible?")
        q_box.press("Enter")
        _wait_idle(page, 600)
        page.get_by_role("button", name="Analizar tramo").click()
        page.wait_for_selector("text=Control Humano Obligatorio", timeout=45000)
        _wait_idle(page, 700)
        page.get_by_role("tab", name="Escenario 3 · Evidencia RAG y Traza LangGraph").click()
        _wait_idle(page, 500)
        mtc_exp = page.locator("summary", has_text="MTC-410").first
        if mtc_exp.count():
            mtc_exp.click()
        page.wait_for_timeout(500)
        _scroll_to(page, ".vv-kpi-grid")
        page.screenshot(path=str(CAP_DIR / "04-escenario3-rag-traza.png"))

        # 5. Aprobación humana y botón Descargar expediente
        page.get_by_role("button", name="Aprobar y preparar expediente preliminar").click()
        page.wait_for_selector("text=Revisión aprobada", timeout=30000)
        _wait_idle(page, 700)
        page.get_by_role("tab", name="Escenario 3 · Evidencia RAG y Traza LangGraph").click()
        _wait_idle(page, 500)
        _scroll_to(page, ".vv-timeline")
        page.screenshot(path=str(CAP_DIR / "05-aprobado-expediente.png"))

        # 6. Caso negativo / abstención verificable
        q_box = page.get_by_label("Pregunta sobre la inspección")
        q_box.fill("¿Cuál es la receta del ceviche?")
        q_box.press("Enter")
        _wait_idle(page, 600)
        page.get_by_role("button", name="Analizar tramo").click()
        page.wait_for_selector("text=Control Humano Obligatorio", timeout=45000)
        _wait_idle(page, 700)
        page.get_by_role("tab", name="Escenario 3 · Evidencia RAG y Traza LangGraph").click()
        _wait_idle(page, 600)
        _scroll_to(page, ".vv-kpi-grid")
        page.screenshot(path=str(CAP_DIR / "07-caso-negativo-abstencion.png"))

        browser.close()
    print("Capturas regeneradas con encuadre optimizado en", CAP_DIR)


if __name__ == "__main__":
    main()
