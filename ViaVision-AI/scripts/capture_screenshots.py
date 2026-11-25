"""Captura automática de pantallas reales de la app Streamlit con Playwright."""
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


def _wait_idle(page, extra_ms: int = 1200) -> None:
    page.wait_for_selector("text=Prioridad estimada", timeout=30000)
    page.wait_for_selector("button:has-text('Analizar tramo')", timeout=30000)
    # Esperar a que desaparezca el indicador de ejecución en curso (Running...)
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"] button')",
        timeout=30000,
    )
    page.wait_for_timeout(extra_ms)


def main() -> None:
    CAP_DIR.mkdir(parents=True, exist_ok=True)
    exe = _pick_browser()
    with sync_playwright() as p:
        launch_kwargs = {"headless": True}
        if exe:
            launch_kwargs["executable_path"] = exe
        browser = p.chromium.launch(**launch_kwargs)
        context = browser.new_context(viewport={"width": 1440, "height": 1100}, device_scale_factor=1.5)
        page = context.new_page()

        # 1. Vista inicial completa
        page.goto(URL, wait_until="domcontentloaded")
        _wait_idle(page, 1500)
        page.screenshot(path=str(CAP_DIR / "01-interfaz-inicial.png"), full_page=True)

        # 2. Escenario 1: Analizar tramo con foto de demostración
        page.get_by_role("button", name="Analizar tramo").click()
        page.wait_for_selector("text=Traza del agente", timeout=45000)
        _wait_idle(page, 1200)
        page.screenshot(path=str(CAP_DIR / "02-escenario1-vision.png"), full_page=True)

        # 3. Escenario 2: Cambiar tráfico a 9800 veh/día y mostrar métrica + barras + incertidumbre
        slider = page.locator("[data-testid='stSlider'] [role='slider']").first
        slider.focus()
        for _ in range(33):
            slider.press("ArrowRight")
        _wait_idle(page, 1200)
        page.screenshot(path=str(CAP_DIR / "03-escenario2-prioridad.png"), full_page=True)

        # 4. Escenario 3: Consultar criterio MTC §410.1, expandir fuente y traza
        q_box = page.get_by_label("Pregunta sobre la inspección")
        q_box.fill("¿Cuándo aplica el parchado superficial en pavimento flexible?")
        q_box.press("Enter")
        _wait_idle(page, 800)
        page.get_by_role("button", name="Analizar tramo").click()
        page.wait_for_selector("summary:has-text('MTC-410')", timeout=45000)
        _wait_idle(page, 1000)
        page.locator("summary", has_text="MTC-410").first.click()
        page.locator("summary", has_text="Traza del agente").first.click()
        page.wait_for_timeout(700)
        page.screenshot(path=str(CAP_DIR / "04-escenario3-rag-traza.png"), full_page=True)

        # 5. Aprobación humana y botón Descargar expediente
        page.get_by_role("button", name="Aprobar y preparar expediente preliminar").click()
        page.wait_for_selector("text=Revisión aprobada. Expediente preliminar listo para descarga.", timeout=30000)
        _wait_idle(page, 1000)
        page.screenshot(path=str(CAP_DIR / "05-aprobado-expediente.png"), full_page=True)

        # 6. Caso negativo / abstención verificable
        q_box = page.get_by_label("Pregunta sobre la inspección")
        q_box.fill("¿Cuál es la receta del ceviche?")
        q_box.press("Enter")
        _wait_idle(page, 600)
        page.get_by_role("button", name="Analizar tramo").click()
        page.wait_for_selector("text=abstención", timeout=45000)
        _wait_idle(page, 1000)
        page.screenshot(path=str(CAP_DIR / "07-caso-negativo-abstencion.png"), full_page=True)

        browser.close()
    print("Capturas generadas correctamente en", CAP_DIR)


if __name__ == "__main__":
    main()
