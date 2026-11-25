"""Estilos CSS modulares y paleta ejecutiva sincronizada con la presentación oficial."""
import streamlit as st

PALETTE = {
    "obsidian": "#0D1321",
    "cobalt": "#1C4587",
    "electric": "#2070C0",
    "sky": "#5FA0E0",
    "amber": "#C85103",
    "canvas": "#F7F9FC",
    "surface": "#FFFFFF",
    "surface_alt": "#EEF3F9",
    "border": "#D8E2EE",
    "text": "#0F172A",
    "muted": "#475569",
}

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Montserrat', 'Segoe UI', system-ui, -apple-system, sans-serif;
}

.stApp {
    background-color: #F7F9FC;
    color: #0F172A;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #EEF3F9 0%, #F7F9FC 100%);
    border-right: 1px solid #D8E2EE;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 1260px;
}

/* Cabecera institucional */
.vv-hero {
    background: linear-gradient(125deg, #0D1321 0%, #1C4587 62%, #2070C0 100%);
    color: #FFFFFF;
    padding: 1.5rem 1.85rem;
    border-radius: 14px;
    border-bottom: 4px solid #C85103;
    box-shadow: 0 8px 24px rgba(13, 19, 33, 0.14);
    margin-bottom: 1.25rem;
}

.vv-hero-kicker {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #93C5FD;
    margin-bottom: 0.35rem;
}

.vv-hero-title {
    font-size: 2.05rem;
    font-weight: 800;
    line-height: 1.15;
    margin: 0 0 0.35rem 0;
    color: #FFFFFF;
}

.vv-hero-sub {
    font-size: 0.96rem;
    color: #E2E8F0;
    margin: 0;
    max-width: 78ch;
}

/* Tarjetas KPI */
.vv-kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.9rem;
    margin: 0.85rem 0 1.1rem 0;
}

@media (max-width: 900px) {
    .vv-kpi-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

.vv-kpi-card {
    background: #FFFFFF;
    border: 1px solid #D8E2EE;
    border-radius: 12px;
    padding: 0.95rem 1.05rem;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}

.vv-kpi-label {
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.35rem;
}

.vv-kpi-value {
    font-size: 1.38rem;
    font-weight: 800;
    color: #0F172A;
    line-height: 1.2;
    margin-bottom: 0.35rem;
}

.vv-kpi-foot {
    font-size: 0.78rem;
    color: #475569;
}

/* Badges semánticos */
.vv-badge {
    display: inline-block;
    padding: 0.18rem 0.58rem;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}

.vv-badge-low {
    background: #DCFCE7;
    color: #166534;
    border: 1px solid #86EFAC;
}

.vv-badge-med {
    background: #FEF3C7;
    color: #9A3412;
    border: 1px solid #FDBA74;
}

.vv-badge-high {
    background: #FEE2E2;
    color: #991B1B;
    border: 1px solid #FCA5A5;
}

.vv-badge-info {
    background: #DBEAFE;
    color: #1E40AF;
    border: 1px solid #93C5FD;
}

/* Tarjetas de sección y chips de factores */
.vv-card {
    background: #FFFFFF;
    border: 1px solid #D8E2EE;
    border-radius: 12px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
}

.vv-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 0.55rem;
}

.vv-chip {
    background: #EEF3F9;
    color: #1C4587;
    border: 1px solid #CBD5E1;
    border-radius: 8px;
    padding: 0.28rem 0.65rem;
    font-size: 0.8rem;
    font-weight: 600;
}

/* Timeline LangGraph */
.vv-timeline {
    background: #FFFFFF;
    border: 1px solid #D8E2EE;
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    margin-top: 0.5rem;
}

.vv-step {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0.25rem;
    border-bottom: 1px dashed #E2E8F0;
}

.vv-step:last-child {
    border-bottom: none;
}

.vv-step-left {
    display: flex;
    align-items: center;
    gap: 0.65rem;
}

.vv-step-idx {
    width: 24px;
    height: 24px;
    border-radius: 999px;
    background: #1C4587;
    color: #FFFFFF;
    font-size: 0.74rem;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}

.vv-step-name {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.86rem;
    font-weight: 600;
    color: #0F172A;
}

.vv-step-ms {
    font-size: 0.76rem;
    font-weight: 700;
    color: #1C4587;
    background: #EEF3F9;
    padding: 0.15rem 0.5rem;
    border-radius: 999px;
}

/* Panel de revisión humana */
.vv-review-pending {
    background: #FFF8F2;
    border: 1.5px solid #C85103;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.9rem 0 0.6rem 0;
}

.vv-review-approved {
    background: #ECFDF5;
    border: 1.5px solid #0F766E;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.9rem 0 0.6rem 0;
}

.vv-review-title {
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
    color: #0F172A;
}

.vv-review-desc {
    font-size: 0.85rem;
    color: #334155;
    margin: 0;
}

/* Pestañas y botones */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.45rem;
    border-bottom: 2px solid #D8E2EE;
}

.stTabs [data-baseweb="tab"] {
    background-color: #EEF3F9;
    border-radius: 8px 8px 0 0;
    padding: 0.5rem 1rem;
    font-weight: 600;
    color: #334155;
}

.stTabs [aria-selected="true"] {
    background-color: #1C4587 !important;
    color: #FFFFFF !important;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #1C4587 0%, #2070C0 100%);
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 0.55rem 1.25rem;
}
</style>
"""


def inject_styles() -> None:
    """Inyecta la hoja de estilos ejecutiva en la página Streamlit."""
    st.markdown(_CSS, unsafe_allow_html=True)


def badge_class(level: str) -> str:
    lvl = (level or "").strip().lower()
    if lvl in {"baja", "low"}:
        return "vv-badge vv-badge-low"
    if lvl in {"alta", "high"}:
        return "vv-badge vv-badge-high"
    if lvl in {"media", "medium"}:
        return "vv-badge vv-badge-med"
    return "vv-badge vv-badge-info"
