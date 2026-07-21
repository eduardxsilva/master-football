from __future__ import annotations

import os
import re
import tempfile
from datetime import datetime, timezone
from html import escape
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from streamlit_option_menu import option_menu
except Exception:  # fallback para deploys sem o pacote opcional
    option_menu = None

from fifa2026_core import (
    InternetDataExtractor,
    MatchDataLoader,
    DynamicEloModel,
    BasePoissonModel,
    MLMatchOutcomeModel,
    EnsemblePredictionModel,
    MonteCarloChampionSimulator,
    WorldCupFormatSimulator,
    ModelBacktester,
    BettingOddsExtractor,
    combinar_previsao_com_odds,
)
from competition_data import (
    COMPETICOES,
    SeasonPlanError,
    THESPORTSDB_COMPETICOES,
    fetch_competition,
    fetch_thesportsdb_two_seasons,
    obter_api_key as obter_chave_futebol,
    predict_fixtures,
    temporadas_recentes_thesportsdb,
)


# ============================================================
# CONFIGURAÇÃO STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Football Analytics Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# ÍCONES OUTLINE SVG — estilo Lucide / liquid glass
# Sem emoji, sem preenchimento, apenas contorno.
# ============================================================

ICON_PATHS: dict[str, str] = {
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "arrow-up-right": '<path d="M7 7h10v10"/><path d="M7 17 17 7"/>',
    "bar-chart": '<path d="M3 3v18h18"/><path d="M7 16V8"/><path d="M12 16V5"/><path d="M17 16v-4"/>',
    "brain": '<path d="M12 5a3 3 0 0 0-5.83-1"/><path d="M12 5a3 3 0 0 1 5.83-1"/><path d="M7 4a4 4 0 0 0-2.5 7"/><path d="M17 4a4 4 0 0 1 2.5 7"/><path d="M5 11a4 4 0 0 0 1.5 7"/><path d="M19 11a4 4 0 0 1-1.5 7"/><path d="M8 18a4 4 0 0 0 4 3"/><path d="M16 18a4 4 0 0 1-4 3"/><path d="M12 5v16"/>',
    "calendar": '<path d="M8 2v4"/><path d="M16 2v4"/><rect x="3" y="4" width="18" height="18" rx="2"/><path d="M3 10h18"/>',
    "cloud-download": '<path d="M12 13v8"/><path d="m8 17 4 4 4-4"/><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5 5 5-5"/><path d="M12 15V3"/>',
    "gauge": '<path d="M12 14l4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "globe": '<circle cx="12" cy="12" r="10"/><path d="M2 12h20"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>',
    "home": '<path d="m3 10 9-7 9 7"/><path d="M5 10v10a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V10"/><path d="M9 21v-6h6v6"/>',
    "layers": '<path d="m12 2 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5"/><path d="m3 17 9 5 9-5"/>',
    "line-chart": '<path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/>',
    "medal": '<path d="M7.21 15 2.66 7.14A2 2 0 0 1 4.39 4h15.22a2 2 0 0 1 1.73 3.14L16.79 15"/><path d="M11 12 5.12 4"/><path d="m13 12 5.88-8"/><circle cx="12" cy="17" r="5"/>',
    "sparkles": '<path d="m12 3-1.9 5.8L4 11l6.1 2.2L12 19l1.9-5.8L20 11l-6.1-2.2L12 3Z"/><path d="M5 3v4"/><path d="M3 5h4"/><path d="M19 17v4"/><path d="M17 19h4"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "trophy": '<path d="M8 21h8"/><path d="M12 17v4"/><path d="M7 4h10v5a5 5 0 0 1-10 0V4Z"/><path d="M5 5H3v2a4 4 0 0 0 4 4"/><path d="M19 5h2v2a4 4 0 0 1-4 4"/>',
    "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M17 8l-5-5-5 5"/><path d="M12 3v12"/>',
    "wand": '<path d="M15 4V2"/><path d="M15 16v-2"/><path d="M8 9H6"/><path d="M20 9h-2"/><path d="M17.8 6.2 19 5"/><path d="M6.2 17.8 5 19"/><path d="m3 21 9-9"/><path d="M12.2 6.2 11 5"/>',
}


def icon_svg(name: str, size: int = 22, stroke: float = 1.8) -> str:
    paths = ICON_PATHS.get(name, ICON_PATHS["sparkles"])
    return (
        f'<svg class="line-icon" xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths}</svg>'
    )


# ============================================================
# CSS LIQUID GLASS
# ============================================================

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,300..500,0,0&display=swap');

    :root {
        --ink: #F7FAFC;
        --muted: rgba(247,250,252,0.72);
        --faint: rgba(247,250,252,0.52);
        --glass: rgba(14,27,46,0.68);
        --glass-strong: rgba(25,195,125,0.16);
        --glass-soft: rgba(14,27,46,0.44);
        --stroke: rgba(25,195,125,0.20);
        --stroke-strong: rgba(25,195,125,0.34);
        --accent: #19C37D;
        --accent-2: #10A37F;
        --accent-3: #34D399;
        --danger: #fb7185;
        --warning: #fbbf24;
        --shadow: rgba(0,0,0,0.42);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* Correção crítica: não deixe o CSS global transformar ícones internos do Streamlit
       em texto bruto como "keyboard_double_arrow_left". */
    .material-symbols-rounded,
    .material-symbols-outlined,
    .material-icons,
    span[data-testid="stIconMaterial"],
    [class*="material-symbols"] {
        font-family: 'Material Symbols Rounded' !important;
        font-weight: normal !important;
        font-style: normal !important;
        font-size: 20px !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        display: inline-flex !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        -webkit-font-feature-settings: 'liga' !important;
        -webkit-font-smoothing: antialiased !important;
        font-feature-settings: 'liga' !important;
        font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24 !important;
    }

    /* Correção para Bootstrap Icons usados pelo streamlit-option-menu. */
    .bi,
    i.bi,
    [class^="bi-"],
    [class*=" bi-"] {
        font-family: bootstrap-icons !important;
        font-style: normal !important;
        font-weight: normal !important;
        line-height: 1 !important;
        text-transform: none !important;
        letter-spacing: normal !important;
    }

    .stApp {
        color: var(--ink);
        background:
            radial-gradient(circle at 8% 5%, rgba(25,195,125,0.24), transparent 30%),
            radial-gradient(circle at 86% 12%, rgba(16,163,127,0.18), transparent 34%),
            radial-gradient(circle at 45% 90%, rgba(25,195,125,0.12), transparent 38%),
            linear-gradient(135deg, #07111F 0%, #0A1628 42%, #0E1B2E 100%);
        background-attachment: fixed;
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
            linear-gradient(rgba(255,255,255,0.028) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.024) 1px, transparent 1px);
        background-size: 38px 38px;
        mask-image: radial-gradient(circle at 50% 30%, black, transparent 72%);
        opacity: .5;
        z-index: 0;
    }

    .block-container {
        padding-top: 1.35rem;
        padding-bottom: 2.7rem;
        max-width: 1440px;
        position: relative;
        z-index: 1;
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(7,17,31,0.86), rgba(14,27,46,0.86)),
            radial-gradient(circle at 20% 0%, rgba(25,195,125,0.20), transparent 38%);
        border-right: 1px solid rgba(255,255,255,0.13);
        box-shadow: 18px 0 40px rgba(0,0,0,0.22);
        backdrop-filter: blur(28px) saturate(160%);
        -webkit-backdrop-filter: blur(28px) saturate(160%);
    }

    [data-testid="stSidebar"] :not(.material-symbols-rounded):not(.material-symbols-outlined):not(.material-icons):not(.bi):not(svg):not(path) {
        font-family: 'Inter', sans-serif;
    }

    .brand-card {
        padding: 18px 16px 14px 16px;
        margin-bottom: 14px;
        border-radius: 24px;
        background: linear-gradient(145deg, rgba(255,255,255,0.13), rgba(255,255,255,0.045));
        border: 1px solid var(--stroke);
        box-shadow: inset 0 1px rgba(255,255,255,0.22), 0 16px 38px rgba(0,0,0,0.22);
    }

    .brand-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.03rem;
        font-weight: 900;
        letter-spacing: -0.035em;
    }

    .brand-subtitle {
        color: var(--muted);
        font-size: .78rem;
        margin-top: 5px;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 34px 34px;
        border-radius: 34px;
        background:
            linear-gradient(135deg, rgba(14,27,46,0.88), rgba(7,17,31,0.70)),
            radial-gradient(circle at 12% 20%, rgba(25,195,125,0.34), transparent 26%),
            radial-gradient(circle at 80% 10%, rgba(16,163,127,0.24), transparent 32%),
            radial-gradient(circle at 62% 90%, rgba(25,195,125,0.14), transparent 36%);
        border: 1px solid rgba(255,255,255,0.18);
        box-shadow: inset 0 1px rgba(255,255,255,0.22), 0 30px 90px rgba(0,0,0,0.34);
        backdrop-filter: blur(24px) saturate(170%);
        -webkit-backdrop-filter: blur(24px) saturate(170%);
        margin-bottom: 24px;
    }

    .hero::after {
        content: "";
        position: absolute;
        inset: 1px;
        border-radius: 33px;
        background: linear-gradient(120deg, rgba(255,255,255,0.18), transparent 22%, transparent 70%, rgba(255,255,255,0.08));
        pointer-events: none;
    }

    .hero-title {
        display: flex;
        align-items: center;
        gap: 14px;
        margin: 0 0 10px 0;
        font-size: clamp(2rem, 4vw, 3.6rem);
        letter-spacing: -0.07em;
        line-height: .98;
        font-weight: 950;
    }

    .hero-title .line-icon {
        width: 44px;
        height: 44px;
        color: var(--accent);
        filter: drop-shadow(0 0 18px rgba(25,195,125,0.52));
    }

    .hero p {
        color: var(--muted);
        font-size: 1.03rem;
        max-width: 900px;
        margin: 0;
        line-height: 1.6;
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 9px;
        margin-top: 22px;
    }

    .chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 9px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.085);
        border: 1px solid rgba(255,255,255,0.15);
        color: rgba(255,255,255,0.86);
        font-size: 0.83rem;
        font-weight: 650;
        white-space: nowrap;
        box-shadow: inset 0 1px rgba(255,255,255,0.14);
    }

    .chip .line-icon {
        width: 17px;
        height: 17px;
        color: var(--accent);
    }

    .glass-card, .metric-card, div[data-testid="stMetric"] {
        border-radius: 24px !important;
        background: linear-gradient(145deg, rgba(255,255,255,0.12), rgba(255,255,255,0.045)) !important;
        border: 1px solid rgba(255,255,255,0.16) !important;
        box-shadow: inset 0 1px rgba(255,255,255,0.20), 0 18px 48px rgba(0,0,0,0.24) !important;
        backdrop-filter: blur(22px) saturate(165%);
        -webkit-backdrop-filter: blur(22px) saturate(165%);
    }

    .metric-card {
        padding: 20px 20px;
        height: 100%;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: "";
        position: absolute;
        inset: -40% -20% auto auto;
        width: 160px;
        height: 160px;
        border-radius: 999px;
        background: radial-gradient(circle, rgba(125,211,252,0.24), transparent 70%);
        pointer-events: none;
    }

    .metric-icon {
        width: 42px;
        height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 15px;
        background: rgba(25,195,125,0.12);
        border: 1px solid rgba(25,195,125,0.23);
        color: var(--accent);
        margin-bottom: 13px;
        box-shadow: inset 0 1px rgba(255,255,255,0.16);
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.82rem;
        font-weight: 680;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: .07em;
    }

    .metric-value {
        font-size: 1.72rem;
        font-weight: 900;
        letter-spacing: -0.055em;
        margin-bottom: 3px;
    }

    .metric-note {
        color: var(--faint);
        font-size: 0.80rem;
        line-height: 1.4;
    }

    .section-title {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 1.35rem;
        font-weight: 900;
        letter-spacing: -0.045em;
        margin: 26px 0 12px 0;
    }

    .section-title .line-icon {
        width: 24px;
        height: 24px;
        color: var(--accent);
    }

    .soft-text {
        color: var(--muted);
        font-size: 0.96rem;
        line-height: 1.6;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 16px !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        background: linear-gradient(145deg, rgba(255,255,255,0.13), rgba(255,255,255,0.055)) !important;
        color: rgba(255,255,255,0.94) !important;
        font-weight: 800 !important;
        min-height: 46px !important;
        box-shadow: inset 0 1px rgba(255,255,255,0.20), 0 14px 36px rgba(0,0,0,0.20) !important;
        backdrop-filter: blur(18px) saturate(155%);
        -webkit-backdrop-filter: blur(18px) saturate(155%);
        transition: transform .18s ease, border-color .18s ease, background .18s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        border-color: rgba(25,195,125,0.44) !important;
        background: linear-gradient(145deg, rgba(25,195,125,0.20), rgba(255,255,255,0.075)) !important;
    }

    button[kind="primary"],
    .stDownloadButton button[kind="primary"] {
        background: linear-gradient(135deg, rgba(25,195,125,0.35), rgba(16,163,127,0.22)) !important;
        border-color: rgba(25,195,125,0.42) !important;
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border-radius: 22px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 18px 48px rgba(0,0,0,0.18);
    }

    div[data-testid="stTabs"] button {
        border-radius: 999px;
        padding: 10px 16px;
        color: rgba(255,255,255,0.72);
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: rgba(255,255,255,0.96);
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.14);
    }

    [data-testid="stSelectbox"], [data-testid="stNumberInput"], [data-testid="stTextInput"], [data-testid="stFileUploader"] {
        border-radius: 18px;
    }

    .stAlert {
        border-radius: 18px;
        backdrop-filter: blur(18px);
    }


    /* Evita textos/ícones internos estourando a página em telas pequenas. */
    .main .block-container, [data-testid="stSidebar"] {
        overflow-x: hidden;
    }

    /* Mobile polish */
    @media (max-width: 900px) {
        .hero { padding: 24px 20px; border-radius: 28px; }
        .hero-title { font-size: 2.05rem; }
        .brand-card { border-radius: 20px; }
        .chip { font-size: .78rem; padding: 8px 10px; }
        .metric-value { font-size: 1.35rem; }
    }


    /* Tema solicitado: dark + verde OpenAI */
    [data-testid="stHeader"] {
        background: rgba(7,17,31,0.72) !important;
        border-bottom: 1px solid rgba(25,195,125,0.12) !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #0E1B2E !important;
    }

    .stTextInput input,
    .stNumberInput input,
    .stSelectbox [data-baseweb="select"],
    .stMultiSelect [data-baseweb="select"],
    .stTextArea textarea {
        background-color: rgba(14,27,46,0.92) !important;
        color: #F7FAFC !important;
        border-color: rgba(25,195,125,0.22) !important;
    }

    .stProgress > div > div > div > div {
        background-color: #19C37D !important;
    }

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# ESTADO E HELPERS
# ============================================================


def init_state() -> None:
    defaults = {
        "df_matches": None,
        "internet_extras": {},
        "internet_log": [],
        "elo_model": None,
        "poisson_model": None,
        "ml_model": None,
        "ensemble_model": None,
        "simulador_campeao": None,
        "simulador_copa": None,
        "equipes_copa_2026": [],
        "df_copa2026_classificacao": None,
        "df_copa2026_classificados": None,
        "ultima_previsao": None,
        "ultima_previsao_ml": None,
        "ultima_previsao_ensemble": None,
        "df_elo": None,
        "df_team_stats": None,
        "df_fifa_team_metrics": None,
        "df_simulacao_campeao": None,
        "df_simulacao_copa": None,
        "ultima_copa_simulada": None,
        "df_validacao_resumo": None,
        "df_validacao_previsoes": None,
        "fonte_base": None,
        "fifa_live_last_update": None,
        "df_odds_raw": None,
        "df_odds_consenso": None,
        "odds_log": [],
        "odds_last_update": None,
        "ultima_odds_partida": None,
        "ultima_previsao_modelo_mercado": None,
        "competicao_atual": None,
        "competicao_id": None,
        "competicao_temporada": None,
        "df_competicao_futuros": None,
        "df_competicao_bruto": None,
        "df_previsoes_calendario": None,
        "competicao_atualizada_utc": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def has_data() -> bool:
    return st.session_state.df_matches is not None and not st.session_state.df_matches.empty


def has_models() -> bool:
    return st.session_state.poisson_model is not None and st.session_state.df_team_stats is not None


def metric_card(icon_name: str, label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{icon_svg(icon_name, 23)}</div>
            <div class="metric-label">{escape(str(label))}</div>
            <div class="metric-value">{escape(str(value))}</div>
            <div class="metric-note">{escape(str(note))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">{icon_svg('sparkles', 44, 1.7)} Football Analytics Engine</div>
            <p>
                Plataforma premium para previsão de partidas, Elo dinâmico, xG,
                Poisson bivariada, Dixon-Coles, Machine Learning, Ensemble e simulações Monte Carlo.
            </p>
            <div class="chip-row">
                <span class="chip">{icon_svg('brain', 17)} Ensemble estatístico + ML</span>
                <span class="chip">{icon_svg('line-chart', 17)} Validação temporal</span>
                <span class="chip">{icon_svg('trophy', 17)} Simulação de campeão</span>
                <span class="chip">{icon_svg('cloud-download', 17)} Extração online</span>
                <span class="chip">{icon_svg('download', 17)} Exportação Excel</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, icon_name: str = "sparkles") -> None:
    st.markdown(
        f"<div class='section-title'>{icon_svg(icon_name, 24)} {escape(title)}</div>",
        unsafe_allow_html=True,
    )


def get_teams() -> list[str]:
    if not has_models():
        return []
    return sorted(st.session_state.df_team_stats["Equipe"].astype(str).tolist())


def get_copa_teams_for_model() -> list[str]:
    """Times da Copa 2026 detectados pela FIFA e presentes no modelo treinado."""
    if not has_models():
        return []
    model_teams = set(st.session_state.df_team_stats["Equipe"].astype(str).tolist())
    copa_teams = st.session_state.get("equipes_copa_2026", []) or []
    return sorted([team for team in copa_teams if team in model_teams])


def sync_copa_live_state_from_extras(extras: dict) -> None:
    """Atualiza o estado da Copa 2026 usando a última consulta solicitada à FIFA."""
    st.session_state.df_copa2026_classificacao = extras.get("copa2026_classificacao_atual")
    st.session_state.df_copa2026_classificados = extras.get("copa2026_classificados_atuais")

    df_update = extras.get("fifa_atualizacao")
    if isinstance(df_update, pd.DataFrame) and not df_update.empty and "Atualizado_UTC" in df_update.columns:
        st.session_state.fifa_live_last_update = str(df_update.iloc[0]["Atualizado_UTC"])

    equipes = []
    df_eq = extras.get("copa2026_equipes_oficiais")
    if isinstance(df_eq, pd.DataFrame) and not df_eq.empty and "Equipe" in df_eq.columns:
        equipes = df_eq["Equipe"].astype(str).dropna().drop_duplicates().tolist()
    elif isinstance(st.session_state.df_copa2026_classificacao, pd.DataFrame) and not st.session_state.df_copa2026_classificacao.empty:
        equipes = st.session_state.df_copa2026_classificacao["Equipe"].astype(str).dropna().drop_duplicates().tolist()

    st.session_state.equipes_copa_2026 = sorted([e for e in equipes if e and e.lower() != "nan"])




def _buscar_sheet_extra(extras: dict, termos: list[str]) -> pd.DataFrame:
    if not isinstance(extras, dict):
        return pd.DataFrame()
    termos_norm = [_normalizar_coluna_local(t) for t in termos]
    for key, df in extras.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        k = _normalizar_coluna_local(key)
        if all(t in k for t in termos_norm):
            return df.copy()
    for key, df in extras.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        k = _normalizar_coluna_local(key)
        if any(t in k for t in termos_norm):
            return df.copy()
    return pd.DataFrame()


def sync_odds_state_from_extras(extras: dict) -> None:
    """Carrega odds já salvas no Excel, quando existirem."""
    consenso = _buscar_sheet_extra(extras, ["odds", "consenso"])
    bruto = _buscar_sheet_extra(extras, ["odds", "bruto"])
    log_df = _buscar_sheet_extra(extras, ["odds", "log"])

    if isinstance(consenso, pd.DataFrame) and not consenso.empty:
        st.session_state.df_odds_consenso = consenso
        if "Atualizado_UTC" in consenso.columns:
            st.session_state.odds_last_update = str(consenso["Atualizado_UTC"].dropna().astype(str).max())
    if isinstance(bruto, pd.DataFrame) and not bruto.empty:
        st.session_state.df_odds_raw = bruto
    if isinstance(log_df, pd.DataFrame) and not log_df.empty:
        col = log_df.columns[0]
        st.session_state.odds_log = log_df[col].astype(str).tolist()


def obter_odds_api_key(valor_digitado: str = "") -> str:
    """Prioridade: campo digitado > secrets.toml > variável de ambiente."""
    valor_digitado = str(valor_digitado or "").strip()
    if valor_digitado:
        return valor_digitado
    try:
        secret = st.secrets.get("ODDS_API_KEY", "")
        if secret:
            return str(secret).strip()
    except Exception:
        pass
    return str(os.environ.get("ODDS_API_KEY", "")).strip()


def split_csv_local(valor: str) -> list[str]:
    return [x.strip() for x in str(valor or "").split(",") if x.strip()]


def obter_equipes_para_filtro_odds() -> list[str]:
    """Usa seleções da Copa 2026, se disponíveis; senão usa equipes treinadas no modelo."""
    equipes = []
    copa = st.session_state.get("equipes_copa_2026", []) or []
    if copa:
        equipes.extend([str(x) for x in copa if str(x).strip()])
    elif isinstance(st.session_state.get("df_team_stats"), pd.DataFrame) and not st.session_state.df_team_stats.empty:
        if "Equipe" in st.session_state.df_team_stats.columns:
            equipes.extend(st.session_state.df_team_stats["Equipe"].dropna().astype(str).tolist())
    return sorted(set([x for x in equipes if x and x.lower() != "nan"]))


def atualizar_odds_ao_vivo(
    api_key: str,
    sport_keys: str,
    regions: str,
    bookmakers: str = "",
    provider: str = "odds-api-io",
    data_partidas=None,
    filtrar_equipes: bool = True,
) -> None:
    equipes_filtro = obter_equipes_para_filtro_odds() if filtrar_equipes else []
    extractor = BettingOddsExtractor(
        api_key=api_key,
        sport_keys=split_csv_local(sport_keys),
        regions=regions,
        bookmakers=bookmakers.strip() or None,
        timeout=35,
        provider=provider,
        target_date=data_partidas,
        target_timezone="America/Sao_Paulo",
        team_filter=equipes_filtro,
    )
    result = extractor.extrair_ao_vivo()

    # Teste real em dois passos: primeiro com filtro de seleções; se vier vazio,
    # repete a mesma data/liga/bookmakers sem o filtro. Isso separa problema de
    # API vazia de problema de normalização de nomes.
    if (
        filtrar_equipes
        and (not isinstance(result.consensus, pd.DataFrame) or result.consensus.empty)
        and equipes_filtro
    ):
        log_primeira = list(result.log or [])
        log_primeira.append("--- RETESTE AUTOMÁTICO: sem filtro de seleções reconhecidas ---")
        extractor_sem_filtro = BettingOddsExtractor(
            api_key=api_key,
            sport_keys=split_csv_local(sport_keys),
            regions=regions,
            bookmakers=bookmakers.strip() or None,
            timeout=35,
            provider=provider,
            target_date=data_partidas,
            target_timezone="America/Sao_Paulo",
            team_filter=[],
        )
        result_sem_filtro = extractor_sem_filtro.extrair_ao_vivo()
        result = type(result)(
            raw=result_sem_filtro.raw,
            consensus=result_sem_filtro.consensus,
            log=log_primeira + list(result_sem_filtro.log or []),
        )

    st.session_state.df_odds_raw = result.raw.copy() if isinstance(result.raw, pd.DataFrame) else pd.DataFrame()
    st.session_state.df_odds_consenso = result.consensus.copy() if isinstance(result.consensus, pd.DataFrame) else pd.DataFrame()
    st.session_state.odds_log = result.log
    if isinstance(result.consensus, pd.DataFrame) and not result.consensus.empty and "Atualizado_UTC" in result.consensus.columns:
        st.session_state.odds_last_update = str(result.consensus["Atualizado_UTC"].dropna().astype(str).max())
    else:
        st.session_state.odds_last_update = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Também guarda como extras para exportação posterior.
    if not isinstance(st.session_state.internet_extras, dict):
        st.session_state.internet_extras = {}
    if isinstance(st.session_state.df_odds_raw, pd.DataFrame) and not st.session_state.df_odds_raw.empty:
        st.session_state.internet_extras["odds_mercado_bruto"] = st.session_state.df_odds_raw
    if isinstance(st.session_state.df_odds_consenso, pd.DataFrame) and not st.session_state.df_odds_consenso.empty:
        st.session_state.internet_extras["odds_consenso_partidas"] = st.session_state.df_odds_consenso
    st.session_state.internet_extras["odds_log"] = pd.DataFrame({"log": result.log})

    if not isinstance(st.session_state.df_odds_consenso, pd.DataFrame) or st.session_state.df_odds_consenso.empty:
        st.warning(
            "Odds não encontradas para os filtros atuais. Confira a data, league/sport key, bookmakers e o filtro de seleções."
        )
        with st.expander("Ver log técnico da consulta de odds"):
            st.code("\n".join(result.log[-80:]) if result.log else "Sem log retornado.")


def _normalizar_coluna_local(valor) -> str:
    texto = str(valor).strip().lower()
    substituicoes = {
        "ç": "c", "ã": "a", "á": "a", "à": "a", "â": "a",
        "é": "e", "ê": "e", "í": "i", "ó": "o", "ô": "o",
        "õ": "o", "ú": "u", "ü": "u",
    }
    for antigo, novo in substituicoes.items():
        texto = texto.replace(antigo, novo)
    texto = texto.replace(" ", "_").replace("-", "_").replace("/", "_")
    while "__" in texto:
        texto = texto.replace("__", "_")
    return texto.strip("_")


def _encontrar_coluna_local(df: pd.DataFrame, opcoes: list[str]) -> str | None:
    mapa = {_normalizar_coluna_local(col): col for col in df.columns}
    for opcao in opcoes:
        opcao_norm = _normalizar_coluna_local(opcao)
        if opcao_norm in mapa:
            return mapa[opcao_norm]
    for opcao in opcoes:
        opcao_norm = _normalizar_coluna_local(opcao)
        for col_norm, col_original in mapa.items():
            if opcao_norm in col_norm or col_norm in opcao_norm:
                return col_original
    return None


def _canonical_team_local(valor) -> str:
    texto = str(valor).replace("\xa0", " ").replace("*", "").strip()
    texto = " ".join(texto.split())
    if not texto or texto.lower() == "nan":
        return ""
    aliases = {
        "usa": "United States", "eua": "United States", "u_s_a": "United States",
        "united_states_of_america": "United States", "estados_unidos": "United States",
        "korea_republic": "South Korea", "republic_of_korea": "South Korea", "south_korea": "South Korea",
        "republica_da_coreia": "South Korea", "coreia_do_sul": "South Korea",
        "czechia": "Czech Republic", "czech_republic": "Czech Republic", "tchequia": "Czech Republic", "republica_tcheca": "Czech Republic",
        "turkiye": "Turkey", "turkey": "Turkey", "turquia": "Turkey",
        "cote_d_ivoire": "Ivory Coast", "ivory_coast": "Ivory Coast", "costa_do_marfim": "Ivory Coast",
        "curacao": "Curaçao", "curacau": "Curaçao",
        "ir_iran": "Iran", "ri_do_ira": "Iran", "iran": "Iran",
        "congo_dr": "DR Congo", "rd_do_congo": "DR Congo", "dr_congo": "DR Congo",
        "brasil": "Brazil", "brazil": "Brazil",
        "alemanha": "Germany", "germany": "Germany",
        "inglaterra": "England", "england": "England",
        "franca": "France", "france": "France",
        "espanha": "Spain", "spain": "Spain",
        "suica": "Switzerland", "switzerland": "Switzerland",
        "japao": "Japan", "japan": "Japan",
        "marrocos": "Morocco", "morocco": "Morocco",
        "mexico": "Mexico", "canada": "Canada", "canada": "Canada",
        "africa_do_sul": "South Africa", "south_africa": "South Africa",
        "bosnia_e_herzegovina": "Bosnia and Herzegovina", "bosnia_herzegovina": "Bosnia and Herzegovina", "bosnia_and_herzegovina": "Bosnia and Herzegovina",
        "catar": "Qatar", "qatar": "Qatar",
        "australia": "Australia", "paraguai": "Paraguay", "paraguay": "Paraguay",
        "equador": "Ecuador", "ecuador": "Ecuador",
        "holanda": "Netherlands", "paises_baixos": "Netherlands", "netherlands": "Netherlands",
        "suecia": "Sweden", "sweden": "Sweden",
        "tunisia": "Tunisia", "tunisia": "Tunisia",
        "egito": "Egypt", "egypt": "Egypt",
        "belgica": "Belgium", "belgium": "Belgium",
        "nova_zelandia": "New Zealand", "new_zealand": "New Zealand",
        "uruguai": "Uruguay", "uruguay": "Uruguay",
        "cabo_verde": "Cape Verde", "cape_verde": "Cape Verde",
        "arabia_saudita": "Saudi Arabia", "saudi_arabia": "Saudi Arabia",
        "noruega": "Norway", "norway": "Norway",
        "senegal": "Senegal", "iraque": "Iraq", "iraq": "Iraq",
        "austria": "Austria", "argelia": "Algeria", "algeria": "Algeria",
        "jordania": "Jordan", "jordan": "Jordan",
        "colombia": "Colombia", "gana": "Ghana", "ghana": "Ghana",
        "croacia": "Croatia", "croatia": "Croatia",
        "panama": "Panama", "haiti": "Haiti",
        "escocia": "Scotland", "scotland": "Scotland",
        "uzbequistao": "Uzbekistan", "uzbekistan": "Uzbekistan",
    }
    return aliases.get(_normalizar_coluna_local(texto), texto)



def _codigo_para_time_local(codigo: str) -> str:
    mapa = {
        "MEX": "Mexico", "RSA": "South Africa", "KOR": "South Korea", "CZE": "Czech Republic",
        "SUI": "Switzerland", "CAN": "Canada", "BIH": "Bosnia and Herzegovina", "QAT": "Qatar",
        "BRA": "Brazil", "MAR": "Morocco", "SCO": "Scotland", "HAI": "Haiti",
        "USA": "United States", "AUS": "Australia", "PAR": "Paraguay", "TUR": "Turkey",
        "GER": "Germany", "CIV": "Ivory Coast", "ECU": "Ecuador", "CUW": "Curaçao",
        "NED": "Netherlands", "JPN": "Japan", "SWE": "Sweden", "TUN": "Tunisia",
        "EGY": "Egypt", "IRN": "Iran", "BEL": "Belgium", "NZL": "New Zealand",
        "ESP": "Spain", "URU": "Uruguay", "CPV": "Cabo Verde", "KSA": "Saudi Arabia",
        "FRA": "France", "NOR": "Norway", "SEN": "Senegal", "IRQ": "Iraq",
        "ARG": "Argentina", "AUT": "Austria", "ALG": "Algeria", "JOR": "Jordan",
        "COL": "Colombia", "POR": "Portugal", "COD": "DR Congo", "UZB": "Uzbekistan",
        "ENG": "England", "GHA": "Ghana", "CRO": "Croatia", "PAN": "Panama",
    }
    return mapa.get(str(codigo).strip().upper(), "")


def _separar_nome_codigo_fifa_local(valor) -> tuple[str, str]:
    raw = str(valor).replace("\xa0", " ").replace("*", "").strip()
    raw = " ".join(raw.split())
    if not raw or raw.lower() == "nan":
        return "", ""
    m = re.match(r"^(.*?)([A-Z]{3})$", raw)
    if m:
        nome_bruto = m.group(1).strip()
        codigo = m.group(2).strip().upper()
        nome_codigo = _codigo_para_time_local(codigo)
        return (nome_codigo or _canonical_team_local(nome_bruto), codigo)
    return _canonical_team_local(raw), ""


def _inferir_grupo_de_standings_dom(sheet_name: str, columns: list) -> str:
    fonte = " ".join([str(sheet_name)] + [str(c) for c in columns[:3]])
    m = re.search(r"Grupo\s*([A-L])", fonte, flags=re.I)
    if not m:
        m = re.search(r"Group\s*([A-L])", fonte, flags=re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"dom_standings_(\d+)", str(sheet_name), flags=re.I)
    if m:
        idx = int(m.group(1))
        if 1 <= idx <= 12:
            return chr(ord("A") + idx - 1)
    return "LOCAL"


def _parse_dom_standings_fifa_local(xls: pd.ExcelFile) -> pd.DataFrame:
    """Reconstrói a classificação a partir das abas dom_standings_* da FIFA.

    Layout FIFA observado:
    coluna 0 = grupo/ícone visual, coluna 1 = posição, coluna 2 = equipe+sigla,
    Unnamed:3..11 = J, C, E, D, M, S, DG, PCE, Pts, Unnamed:12 = últimos resultados visual.
    """
    rows = []
    for sheet in xls.sheet_names:
        if not str(sheet).lower().startswith("dom_standings_"):
            continue
        try:
            d = pd.read_excel(xls, sheet_name=sheet)
        except Exception:
            continue
        if d is None or d.empty or d.shape[1] < 12:
            continue

        grupo = _inferir_grupo_de_standings_dom(sheet, list(d.columns))
        cols = list(d.columns)
        for _, r in d.iterrows():
            pos = pd.to_numeric(r.get(cols[1]), errors="coerce") if len(cols) > 1 else pd.NA
            equipe, codigo = _separar_nome_codigo_fifa_local(r.get(cols[2], "")) if len(cols) > 2 else ("", "")
            if not equipe or str(equipe).lower() == "nan":
                continue
            rows.append({
                "Grupo": grupo,
                "Posicao_Grupo": int(pos) if pd.notna(pos) else len(rows) + 1,
                "Equipe": equipe,
                "Codigo_FIFA": codigo,
                "Jogos": pd.to_numeric(r.get(cols[3]), errors="coerce") if len(cols) > 3 else 0,
                "Vitorias": pd.to_numeric(r.get(cols[4]), errors="coerce") if len(cols) > 4 else 0,
                "Empates": pd.to_numeric(r.get(cols[5]), errors="coerce") if len(cols) > 5 else 0,
                "Derrotas": pd.to_numeric(r.get(cols[6]), errors="coerce") if len(cols) > 6 else 0,
                "GP": pd.to_numeric(r.get(cols[7]), errors="coerce") if len(cols) > 7 else 0,
                "GC": pd.to_numeric(r.get(cols[8]), errors="coerce") if len(cols) > 8 else 0,
                "SG": pd.to_numeric(r.get(cols[9]), errors="coerce") if len(cols) > 9 else 0,
                "PCE": pd.to_numeric(r.get(cols[10]), errors="coerce") if len(cols) > 10 else 0,
                "Pontos": pd.to_numeric(r.get(cols[11]), errors="coerce") if len(cols) > 11 else 0,
                "Ultimos_Resultados": str(r.get(cols[12], "")).strip() if len(cols) > 12 else "",
                "Fonte_Tabela": sheet,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "Grupo", "Posicao_Grupo", "Equipe", "Codigo_FIFA", "Jogos", "Vitorias", "Empates", "Derrotas",
            "GP", "GC", "SG", "PCE", "Pontos", "Ultimos_Resultados", "Fonte_Tabela"
        ])
    out = pd.DataFrame(rows)
    for col in ["Jogos", "Vitorias", "Empates", "Derrotas", "GP", "GC", "SG", "PCE", "Pontos"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    out["Posicao_Grupo"] = pd.to_numeric(out["Posicao_Grupo"], errors="coerce").fillna(99).astype(int)
    out = out.drop_duplicates(subset=["Grupo", "Equipe"]).sort_values(["Grupo", "Posicao_Grupo"]).reset_index(drop=True)
    return out


def _padronizar_classificacao_copa_local(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Grupo", "Posicao_Grupo", "Equipe", "Pontos", "Jogos", "SG", "GP", "Fonte_Tabela"])
    d = df.copy()
    d.columns = [" ".join(str(c).split()) for c in d.columns]
    col_grupo = _encontrar_coluna_local(d, ["Grupo", "Group"])
    col_pos = _encontrar_coluna_local(d, ["Posicao_Grupo", "Posição", "Posicao", "Pos", "Rank", "Position"])
    col_eq = _encontrar_coluna_local(d, ["Equipe", "Seleção", "Selecao", "Team", "Country", "Nation"])
    col_codigo = _encontrar_coluna_local(d, ["Codigo_FIFA", "Código FIFA", "Code", "Codigo", "Sigla"])
    col_pts = _encontrar_coluna_local(d, ["Pontos", "Pts", "Points"])
    col_j = _encontrar_coluna_local(d, ["Jogos", "J", "Played", "MP", "Pld"])
    col_c = _encontrar_coluna_local(d, ["Vitorias", "Vitórias", "C", "Wins", "W"])
    col_e = _encontrar_coluna_local(d, ["Empates", "E", "Draws", "D"])
    col_d = _encontrar_coluna_local(d, ["Derrotas", "D", "Losses", "L"])
    col_sg = _encontrar_coluna_local(d, ["SG", "DG", "GD", "Saldo", "Goal Difference"])
    col_gp = _encontrar_coluna_local(d, ["GP", "M", "GF", "Gols Pro", "Goals For"])
    col_gc = _encontrar_coluna_local(d, ["GC", "S", "GA", "Gols Contra", "Gols Sofridos", "Goals Against"])
    col_pce = _encontrar_coluna_local(d, ["PCE"])
    if col_eq is None:
        return pd.DataFrame(columns=["Grupo", "Posicao_Grupo", "Equipe", "Pontos", "Jogos", "SG", "GP", "Fonte_Tabela"])
    out = pd.DataFrame()
    out["Grupo"] = d[col_grupo].astype(str).str.upper().str.replace("GROUP", "", regex=False).str.replace("GRUPO", "", regex=False).str.strip() if col_grupo else "LOCAL"
    out["Posicao_Grupo"] = pd.to_numeric(d[col_pos], errors="coerce") if col_pos else range(1, len(d) + 1)
    out["Equipe"] = d[col_eq].apply(_canonical_team_local)
    if col_codigo:
        out["Codigo_FIFA"] = d[col_codigo].astype(str).str.upper().str.strip()
    out["Pontos"] = pd.to_numeric(d[col_pts], errors="coerce").fillna(0) if col_pts else 0
    out["Jogos"] = pd.to_numeric(d[col_j], errors="coerce").fillna(0) if col_j else 0
    out["Vitorias"] = pd.to_numeric(d[col_c], errors="coerce").fillna(0) if col_c else 0
    out["Empates"] = pd.to_numeric(d[col_e], errors="coerce").fillna(0) if col_e else 0
    out["Derrotas"] = pd.to_numeric(d[col_d], errors="coerce").fillna(0) if col_d else 0
    out["SG"] = pd.to_numeric(d[col_sg], errors="coerce").fillna(0) if col_sg else 0
    out["GP"] = pd.to_numeric(d[col_gp], errors="coerce").fillna(0) if col_gp else 0
    out["GC"] = pd.to_numeric(d[col_gc], errors="coerce").fillna(0) if col_gc else 0
    out["PCE"] = pd.to_numeric(d[col_pce], errors="coerce").fillna(0) if col_pce else 0
    out["Fonte_Tabela"] = "arquivo_local"
    out = out[out["Equipe"].astype(str).str.strip().ne("")]
    return out.drop_duplicates(subset=["Grupo", "Equipe"]).reset_index(drop=True)


def _empty_jogos_copa_local() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "Data", "Hora", "Fase", "Grupo", "Time_A", "Codigo_A", "Time_B", "Codigo_B",
        "Gols_A", "Gols_B", "Status", "Estadio", "Cidade", "Fonte", "Ordem"
    ])


def _is_data_fifa_token_local(valor: str) -> bool:
    return bool(re.fullmatch(
        r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+\d{1,2}\s+[A-Za-z]+\s+2026",
        str(valor or "").strip(),
        flags=re.I,
    ))


def _parse_data_fifa_token_local(valor: str) -> str:
    meses = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    }
    m = re.fullmatch(
        r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})",
        str(valor or "").strip(),
        flags=re.I,
    )
    if not m:
        return str(valor or "").strip()
    dia = int(m.group(1))
    mes = meses.get(m.group(2).lower(), 1)
    ano = int(m.group(3))
    return datetime(ano, mes, dia).date().isoformat()


def _is_hora_fifa_token_local(valor: str) -> bool:
    return bool(re.fullmatch(r"\d{1,2}:\d{2}", str(valor or "").strip()))


def _is_int_fifa_token_local(valor: str) -> bool:
    return bool(re.fullmatch(r"\d{1,2}", str(valor or "").strip()))


def _is_fase_fifa_token_local(valor: str) -> bool:
    fases = {
        _normalizar_coluna_local(x) for x in [
            "First Stage", "Round of 32", "Round of 16", "Quarter-final",
            "Semi-final", "Final", "Play-off for third place",
        ]
    }
    return _normalizar_coluna_local(valor) in fases


def _slot_fifa_placeholder_local(valor) -> bool:
    s = str(valor or "").strip().upper().replace(" ", "")
    return bool(re.fullmatch(r"(?:[123][A-L]{1,5}|W\d+|RU\d+|TBD)", s))


def _extract_codigo_nome_fifa_local(valor) -> tuple[str, str]:
    s = str(valor or "").replace("\xa0", " ").strip()
    s = " ".join(s.split())
    m = re.match(r"^([A-Z]{3})\s+(.+)$", s)
    if m and not re.fullmatch(r"[A-Z0-9]+", m.group(2).strip()):
        return m.group(1).upper(), _canonical_team_local(m.group(2).strip())
    if _slot_fifa_placeholder_local(s):
        return "", s.upper().replace(" ", "")
    return "", _canonical_team_local(s)


def _tokenizar_texto_jogos_fifa_local(texto: str) -> list[str]:
    texto = str(texto or "").replace("\xa0", " ")
    texto = re.sub(r"\s+", " ", texto)
    fases = [
        "First Stage", "Round of 32", "Round of 16", "Quarter-final",
        "Semi-final", "Final", "Play-off for third place",
    ]
    # Corrige colagens comuns do DOM: "South Africa First Stage" e "New Zealand 00:00".
    for fase in fases:
        texto = re.sub(rf"\s+({re.escape(fase)})\b", r" | \1", texto)
    texto = re.sub(r"(?<=[A-Za-zÀ-ÖØ-öø-ÿ\)])\s+(\d{1,2}:\d{2})\b", r" | \1", texto)
    texto = texto.replace(" · ", " | · | ")
    texto = re.sub(
        r"\b((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+\d{1,2}\s+[A-Za-z]+\s+2026)\b",
        r"| \1 |",
        texto,
    )
    return [t.strip() for t in texto.split("|") if t and t.strip() and t.strip().lower() != "nan"]


def _parse_textos_jogos_fifa_local(textos: list[str], fonte: str = "copa2026_jogos_fifa") -> pd.DataFrame:
    if not textos:
        return _empty_jogos_copa_local()

    texto = " | ".join(str(t) for t in textos if str(t).strip() and str(t).strip().lower() != "nan")
    tokens = _tokenizar_texto_jogos_fifa_local(texto)
    if not tokens:
        return _empty_jogos_copa_local()

    rows = []
    data_atual = ""
    ordem = 0
    ignorar = {"view groups", "view brackets", "·"}

    for i, token in enumerate(tokens):
        token_limpo = str(token).strip()
        if _is_data_fifa_token_local(token_limpo):
            data_atual = _parse_data_fifa_token_local(token_limpo)
            continue
        if not data_atual:
            continue
        low = token_limpo.lower()
        if low in ignorar or low.startswith(("match fixtures", "sort by", "where to watch", "latest fifa")):
            continue

        if i + 5 < len(tokens) and _is_int_fifa_token_local(tokens[i + 1]) and str(tokens[i + 2]).upper() in {"FT", "AET", "PEN", "LIVE"} and _is_int_fifa_token_local(tokens[i + 3]):
            raw_a = token_limpo
            raw_b = tokens[i + 4]
            fase_idx = i + 5
            hora = ""
            gols_a = int(tokens[i + 1])
            gols_b = int(tokens[i + 3])
            status = str(tokens[i + 2]).upper()
        elif i + 3 < len(tokens) and _is_hora_fifa_token_local(tokens[i + 1]):
            raw_a = token_limpo
            raw_b = tokens[i + 2]
            fase_idx = i + 3
            hora = str(tokens[i + 1]).strip()
            gols_a = pd.NA
            gols_b = pd.NA
            status = "Scheduled"
        else:
            continue

        fase = str(tokens[fase_idx]).strip() if fase_idx < len(tokens) else ""
        if not _is_fase_fifa_token_local(fase):
            continue

        codigo_a, time_a = _extract_codigo_nome_fifa_local(raw_a)
        codigo_b, time_b = _extract_codigo_nome_fifa_local(raw_b)
        if not time_a or not time_b:
            continue

        grupo = ""
        estadio = ""
        cidade = ""
        busca = [str(x).strip() for x in tokens[fase_idx + 1:fase_idx + 9]]
        pos_depois_grupo = 0
        if _normalizar_coluna_local(fase) == "first_stage":
            for j, item in enumerate(busca):
                m_grupo = re.search(r"Group\s+([A-L])", item, flags=re.I)
                if m_grupo:
                    grupo = m_grupo.group(1).upper()
                    pos_depois_grupo = j + 1
                    break
        else:
            grupo = "Mata-mata"

        resto = [x for x in busca[pos_depois_grupo:] if x and x != "·"]
        if resto and re.match(r"Group\s+[A-L]", resto[0], flags=re.I):
            resto = resto[1:]
        if resto:
            estadio = resto[0]
        if len(resto) > 1 and re.match(r"^\(.+\)$", resto[1]):
            cidade = resto[1].strip("()")

        ordem += 1
        rows.append({
            "Data": data_atual,
            "Hora": hora,
            "Fase": fase,
            "Grupo": grupo,
            "Time_A": time_a,
            "Codigo_A": codigo_a,
            "Time_B": time_b,
            "Codigo_B": codigo_b,
            "Gols_A": gols_a,
            "Gols_B": gols_b,
            "Status": status,
            "Estadio": estadio,
            "Cidade": cidade,
            "Fonte": fonte,
            "Ordem": ordem,
        })

    if not rows:
        return _empty_jogos_copa_local()

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["Data", "Hora", "Fase", "Time_A", "Time_B"], keep="first").reset_index(drop=True)
    out["Ordem"] = range(1, len(out) + 1)
    return out


def _parse_jogos_fifa_de_dataframe_local(df: pd.DataFrame, fonte: str) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_jogos_copa_local()
    textos = []
    for col in ["trecho", "texto", "Texto", "text", "content", "Conteudo", "Conteúdo"]:
        if col in df.columns:
            textos.extend(df[col].dropna().astype(str).tolist())
    # Fallback: usa todas as células textuais se o nome das colunas mudou.
    if not textos:
        try:
            textos = df.astype(str).fillna("").agg(" | ".join, axis=1).tolist()
        except Exception:
            textos = []
    return _parse_textos_jogos_fifa_local(textos, fonte=fonte)


def _padronizar_jogos_copa_local(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_jogos_copa_local()
    d = df.copy()
    d.columns = [" ".join(str(c).split()) for c in d.columns]
    col_data = _encontrar_coluna_local(d, ["Data", "Date", "Match Date"])
    col_hora = _encontrar_coluna_local(d, ["Hora", "Time", "Kickoff", "Kick-off"])
    col_fase = _encontrar_coluna_local(d, ["Fase", "Round", "Stage"])
    col_grupo = _encontrar_coluna_local(d, ["Grupo", "Group"])
    col_a = _encontrar_coluna_local(d, ["Time_A", "Mandante", "Home", "Team A", "Equipe A"])
    col_b = _encontrar_coluna_local(d, ["Time_B", "Visitante", "Away", "Team B", "Equipe B"])
    col_ga = _encontrar_coluna_local(d, ["Gols_A", "Home Goals", "Gols Casa", "Score A"])
    col_gb = _encontrar_coluna_local(d, ["Gols_B", "Away Goals", "Gols Fora", "Score B"])
    col_status = _encontrar_coluna_local(d, ["Status", "State"])
    col_estadio = _encontrar_coluna_local(d, ["Estadio", "Estádio", "Stadium", "Venue"])
    col_cidade = _encontrar_coluna_local(d, ["Cidade", "City"])

    if col_a is None or col_b is None:
        return _parse_jogos_fifa_de_dataframe_local(d, fonte="copa2026_jogos_fifa_texto_bruto")

    out = pd.DataFrame()
    out["Data"] = pd.to_datetime(d[col_data], errors="coerce").dt.date.astype(str) if col_data else ""
    out["Hora"] = d[col_hora].astype(str).str.strip() if col_hora else ""
    out["Fase"] = d[col_fase].astype(str).str.strip() if col_fase else ""
    out["Grupo"] = d[col_grupo].astype(str).str.upper().str.replace("GROUP", "", regex=False).str.strip() if col_grupo else ""
    out["Time_A"] = d[col_a].apply(_canonical_team_local)
    out["Codigo_A"] = ""
    out["Time_B"] = d[col_b].apply(_canonical_team_local)
    out["Codigo_B"] = ""
    out["Gols_A"] = pd.to_numeric(d[col_ga], errors="coerce") if col_ga else pd.NA
    out["Gols_B"] = pd.to_numeric(d[col_gb], errors="coerce") if col_gb else pd.NA
    out["Status"] = d[col_status].astype(str) if col_status else ""
    out["Estadio"] = d[col_estadio].astype(str) if col_estadio else ""
    out["Cidade"] = d[col_cidade].astype(str) if col_cidade else ""
    out["Fonte"] = "arquivo_local"
    out = out[out["Time_A"].astype(str).str.strip().ne("") & out["Time_B"].astype(str).str.strip().ne("")]
    out = out.drop_duplicates(subset=["Data", "Hora", "Fase", "Time_A", "Time_B"]).reset_index(drop=True)
    out["Ordem"] = range(1, len(out) + 1)

    # Se a aba veio quase toda como texto bruto, o parser textual costuma ser mais completo.
    parsed = _parse_jogos_fifa_de_dataframe_local(d, fonte="copa2026_jogos_fifa_texto_bruto")
    if len(parsed) > len(out):
        return parsed
    return out


def _parse_page_text_scores_fixtures_local(xls: pd.ExcelFile) -> pd.DataFrame:
    textos = []
    for sheet in ["page_text_scores_fixtures", "copa2026_jogos_fifa"]:
        if sheet not in xls.sheet_names:
            continue
        try:
            d = pd.read_excel(xls, sheet_name=sheet)
        except Exception:
            continue
        for col in ["trecho", "texto", "Texto", "text", "content"]:
            if col in d.columns:
                textos.extend(d[col].dropna().astype(str).tolist())
    return _parse_textos_jogos_fifa_local(textos, fonte="page_text_scores_fixtures")


def _calcular_classificados_copa_local(standings: pd.DataFrame) -> pd.DataFrame:
    if standings.empty or "Grupo" not in standings.columns:
        return pd.DataFrame()
    parts = []
    sort_cols = [c for c in ["Pontos", "SG", "GP", "Posicao_Grupo"] if c in standings.columns]
    asc = [False, False, False, True][:len(sort_cols)]
    for grupo, gdf in standings.groupby("Grupo"):
        g = gdf.copy().sort_values(sort_cols, ascending=asc).reset_index(drop=True) if sort_cols else gdf.copy().reset_index(drop=True)
        g["Posicao_Calculada"] = range(1, len(g) + 1)
        parts.append(g)
    ranked = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if ranked.empty:
        return pd.DataFrame()
    first_second = ranked[ranked["Posicao_Calculada"].isin([1, 2])].copy()
    first_second["Tipo_Classificacao"] = "1º/2º do grupo"
    thirds = ranked[ranked["Posicao_Calculada"] == 3].copy()
    thirds_sort_cols = [c for c in ["Pontos", "SG", "GP"] if c in thirds.columns]
    if thirds_sort_cols:
        thirds = thirds.sort_values(thirds_sort_cols, ascending=[False] * len(thirds_sort_cols)).head(8)
    thirds["Tipo_Classificacao"] = "melhor terceiro atual"
    return pd.concat([first_second, thirds], ignore_index=True)

def carregar_estado_copa_2026_local(uploaded_file) -> dict[str, pd.DataFrame]:
    """Lê o Excel gerado pelo baixador_fifa_local.py e monta extras compatíveis com o app."""
    xls = pd.ExcelFile(uploaded_file)
    sheets = {name.lower(): name for name in xls.sheet_names}

    def ler_primeira(possiveis: list[str]) -> pd.DataFrame:
        for alvo in possiveis:
            alvo_norm = alvo.lower()
            for low, original in sheets.items():
                if alvo_norm == low or alvo_norm in low:
                    return pd.read_excel(xls, sheet_name=original)
        return pd.DataFrame()

    raw_cls = ler_primeira(["copa2026_classificacao_atual", "classificacao", "standings", "grupos"])
    raw_eq = ler_primeira(["copa2026_equipes_oficiais", "equipes", "teams"])
    raw_jogos = ler_primeira(["copa2026_jogos_fifa", "jogos", "fixtures", "matches", "scores"])
    raw_classificados = ler_primeira(["copa2026_classificados_atuais", "classificados"])
    raw_update = ler_primeira(["fifa_atualizacao", "atualizacao"])
    raw_log = ler_primeira(["fifa_log", "log"])

    # Guarda TODAS as abas do Excel local como extras. Assim, estatísticas de equipe,
    # jogador, power rankings, textos e JSONs capturados pelo Selenium não são perdidos.
    abas_extras_local: dict[str, pd.DataFrame] = {}
    for original_sheet in xls.sheet_names:
        try:
            safe_name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(original_sheet)).strip("_")
            key = f"fifa_local_{safe_name}"[:80]
            abas_extras_local[key] = pd.read_excel(xls, sheet_name=original_sheet)
        except Exception:
            pass

    standings = _padronizar_classificacao_copa_local(raw_cls)
    standings_dom = _parse_dom_standings_fifa_local(xls)
    # Se a aba principal estiver vazia/antiga, reconstrói pelas abas dom_standings_* da FIFA.
    if standings.empty and not standings_dom.empty:
        standings = standings_dom
    elif not standings.empty and "Codigo_FIFA" not in standings.columns and not standings_dom.empty:
        # Prefere o DOM quando ele traz Código FIFA e o layout completo J/C/E/D/M/S/DG/PCE/Pts.
        standings = standings_dom

    jogos = _padronizar_jogos_copa_local(raw_jogos)
    jogos_texto = _parse_page_text_scores_fixtures_local(xls)
    if jogos.empty or len(jogos_texto) > len(jogos):
        jogos = jogos_texto

    if standings.empty and not raw_eq.empty:
        col_eq = _encontrar_coluna_local(raw_eq, ["Equipe", "Seleção", "Selecao", "Team", "Country", "Nation"])
        if col_eq is not None:
            equipes_tmp = raw_eq[col_eq].apply(_canonical_team_local).dropna().drop_duplicates().tolist()
            standings = pd.DataFrame({
                "Grupo": "LOCAL",
                "Posicao_Grupo": range(1, len(equipes_tmp) + 1),
                "Equipe": equipes_tmp,
                "Pontos": 0,
                "Jogos": 0,
                "SG": 0,
                "GP": 0,
                "Fonte_Tabela": "arquivo_local_equipes",
            })

    equipes = []
    if not standings.empty:
        equipes += standings["Equipe"].astype(str).tolist()
    if not raw_eq.empty:
        col_eq = _encontrar_coluna_local(raw_eq, ["Equipe", "Seleção", "Selecao", "Team", "Country", "Nation"])
        if col_eq is not None:
            equipes += raw_eq[col_eq].apply(_canonical_team_local).tolist()
    if not jogos.empty:
        equipes += jogos["Time_A"].astype(str).tolist() + jogos["Time_B"].astype(str).tolist()
    equipes = sorted(set([_canonical_team_local(e) for e in equipes if _canonical_team_local(e)]))
    if not standings.empty and "Codigo_FIFA" in standings.columns:
        df_equipes = standings[["Equipe", "Codigo_FIFA"]].drop_duplicates().sort_values("Equipe").reset_index(drop=True)
        df_equipes["Fonte"] = "arquivo_local_standings"
    else:
        df_equipes = pd.DataFrame({"Equipe": equipes, "Fonte": "arquivo_local"})

    # Recalcula classificados pela classificação padronizada. Evita usar aba vazia/antiga.
    classificados = _calcular_classificados_copa_local(standings) if not standings.empty else raw_classificados
    if raw_update.empty:
        raw_update = pd.DataFrame([{
            "Atualizado_UTC": datetime.now(timezone.utc).isoformat(),
            "Origem": getattr(uploaded_file, "name", "arquivo_local"),
        }])
    if raw_log.empty:
        raw_log = pd.DataFrame({"log": [f"Arquivo local carregado: {getattr(uploaded_file, 'name', 'sem_nome')}"]})

    extras = {
        "copa2026_classificacao_atual": standings,
        "copa2026_equipes_oficiais": df_equipes,
        "copa2026_jogos_fifa": jogos,
        "copa2026_classificados_atuais": classificados,
        "fifa_atualizacao": raw_update,
        "fifa_log": raw_log,
    }
    # Mantém as abas brutas também, sem sobrescrever as abas padronizadas.
    for k, v in abas_extras_local.items():
        if k not in extras:
            extras[k] = v
    return extras




def obter_metricas_fifa_equipes_do_estado() -> pd.DataFrame:
    """Localiza a aba consolidada de estatísticas FIFA carregada no Excel local."""
    extras = st.session_state.get("internet_extras") or {}
    if not isinstance(extras, dict) or not extras:
        return pd.DataFrame()

    preferidos = [
        "relatorio_mestre_equipes",
        "fifa_local_relatorio_mestre_equipes",
        "relatorio_mestre_fifa",
        "fifa_local_relatorio_mestre_fifa",
        "team_stats_consolidado",
        "fifa_local_team_stats_consolidado",
    ]
    for key in preferidos:
        df = extras.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.copy()

    # Fallback: procura nomes prováveis entre todas as abas importadas.
    for key, df in extras.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        k = str(key).lower()
        if "relatorio_mestre" in k or "team_stats" in k or "estatisticas_equipe" in k or "statistics_team" in k:
            # Evita usar page_text/raw quando houver texto bruto, não tabela de métricas.
            if not any(bad in k for bad in ["page_text", "raw", "json", "log"]):
                return df.copy()

    return pd.DataFrame()


def carregar_excel_fifa_gerado_completo(uploaded_file) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Lê o Excel gerado pelo gerador local:
    - base_historica_2018 vira df_matches.
    - abas FIFA viram extras usados pelo modelo e pela Copa.
    """
    data = uploaded_file.getvalue()
    xls = pd.ExcelFile(BytesIO(data))
    sheets_lower = {s.lower(): s for s in xls.sheet_names}

    tmp_path = None
    try:
        if "base_historica_2018" in sheets_lower:
            df_base = pd.read_excel(xls, sheet_name=sheets_lower["base_historica_2018"])
            tmp_buffer = BytesIO()
            df_base.to_excel(tmp_buffer, index=False)
            tmp_buffer.seek(0)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                tmp.write(tmp_buffer.getvalue())
                tmp_path = tmp.name
        else:
            suffix = Path(uploaded_file.name).suffix.lower()
            if suffix not in {".xlsx", ".xls", ".csv"}:
                raise ValueError("Formato de arquivo não permitido.")
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

        loader = MatchDataLoader()
        df_matches = loader.carregar_arquivo(tmp_path)
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    extras = {}
    try:
        extras = carregar_estado_copa_2026_local(BytesIO(data))
    except Exception:
        extras = {}

    return df_matches, extras

def style_plotly(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.025)",
        font=dict(color="rgba(255,255,255,0.86)", family="Inter, Segoe UI, sans-serif"),
        title_font=dict(size=18, color="rgba(255,255,255,0.95)"),
        margin=dict(l=18, r=18, t=58, b=18),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.10)"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.075)", zerolinecolor="rgba(255,255,255,0.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.075)", zerolinecolor="rgba(255,255,255,0.12)")
    return fig


def reset_models() -> None:
    for key in [
        "elo_model",
        "poisson_model",
        "ml_model",
        "ensemble_model",
        "simulador_campeao",
        "simulador_copa",
        "df_elo",
        "df_team_stats",
        "ultima_previsao",
        "ultima_previsao_ml",
        "ultima_previsao_ensemble",
        "ultima_odds_partida",
        "ultima_previsao_modelo_mercado",
        "df_simulacao_campeao",
        "df_simulacao_copa",
        "ultima_copa_simulada",
        "df_validacao_resumo",
        "df_validacao_previsoes",
    ]:
        st.session_state[key] = None


def treinar_modelos() -> None:
    if not has_data():
        raise ValueError("Importe uma base antes de treinar.")

    df = st.session_state.df_matches.copy()
    df_metricas_fifa = obter_metricas_fifa_equipes_do_estado()
    st.session_state.df_fifa_team_metrics = df_metricas_fifa.copy() if isinstance(df_metricas_fifa, pd.DataFrame) and not df_metricas_fifa.empty else None

    elo_model = DynamicEloModel().treinar(df)
    poisson_model = BasePoissonModel(max_gols=10).treinar(
        df,
        elo_model=elo_model,
        df_fifa_team_metrics=df_metricas_fifa if isinstance(df_metricas_fifa, pd.DataFrame) and not df_metricas_fifa.empty else None,
    )

    ml_model = MLMatchOutcomeModel()
    try:
        ml_model.treinar(df, poisson_model=poisson_model, elo_model=elo_model)
    except Exception:
        ml_model.treinado = False

    ensemble_model = EnsemblePredictionModel()

    st.session_state.elo_model = elo_model
    st.session_state.poisson_model = poisson_model
    st.session_state.ml_model = ml_model
    st.session_state.ensemble_model = ensemble_model
    st.session_state.df_elo = elo_model.ranking()
    st.session_state.df_team_stats = poisson_model.df_team_stats.copy()

    st.session_state.simulador_campeao = MonteCarloChampionSimulator(
        poisson_model=poisson_model,
        ml_model=ml_model,
        ensemble_model=ensemble_model,
        random_state=42,
    )
    grupos_copa = st.session_state.get("df_copa2026_classificacao")
    equipes_copa = st.session_state.get("equipes_copa_2026", []) or []
    jogos_copa = None
    if isinstance(st.session_state.get("internet_extras"), dict):
        jogos_copa = st.session_state.internet_extras.get("copa2026_jogos_fifa")
    classificados_copa = st.session_state.get("df_copa2026_classificados")

    st.session_state.simulador_copa = WorldCupFormatSimulator(
        poisson_model=poisson_model,
        ml_model=ml_model,
        ensemble_model=ensemble_model,
        random_state=42,
        equipes_copa=equipes_copa,
        grupos_copa=grupos_copa if isinstance(grupos_copa, pd.DataFrame) and not grupos_copa.empty else None,
        jogos_copa=jogos_copa if isinstance(jogos_copa, pd.DataFrame) and not jogos_copa.empty else None,
        classificados_copa=classificados_copa if isinstance(classificados_copa, pd.DataFrame) and not classificados_copa.empty else None,
    )




def recriar_simulador_copa_com_estado_fifa() -> None:
    """Reaponta o simulador da Copa para as equipes/grupos da última atualização FIFA."""
    if not has_models():
        return

    grupos_copa = st.session_state.get("df_copa2026_classificacao")
    equipes_copa = st.session_state.get("equipes_copa_2026", []) or []
    jogos_copa = None
    if isinstance(st.session_state.get("internet_extras"), dict):
        jogos_copa = st.session_state.internet_extras.get("copa2026_jogos_fifa")
    classificados_copa = st.session_state.get("df_copa2026_classificados")

    st.session_state.simulador_copa = WorldCupFormatSimulator(
        poisson_model=st.session_state.poisson_model,
        ml_model=st.session_state.ml_model,
        ensemble_model=st.session_state.ensemble_model,
        random_state=42,
        equipes_copa=equipes_copa,
        grupos_copa=grupos_copa if isinstance(grupos_copa, pd.DataFrame) and not grupos_copa.empty else None,
        jogos_copa=jogos_copa if isinstance(jogos_copa, pd.DataFrame) and not jogos_copa.empty else None,
        classificados_copa=classificados_copa if isinstance(classificados_copa, pd.DataFrame) and not classificados_copa.empty else None,
    )

def dataframe_to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        used_names = set()
        for name, df in sheets.items():
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                continue
            clean = str(name)[:31] or "sheet"
            original = clean
            idx = 1
            while clean in used_names:
                suffix = f"_{idx}"
                clean = (original[: 31 - len(suffix)] + suffix)[:31]
                idx += 1
            used_names.add(clean)
            df.to_excel(writer, sheet_name=clean, index=False)
    return buffer.getvalue()


init_state()


# ============================================================
# SIDEBAR
# ============================================================

PAGES = [
    "Dashboard",
    "Campeonatos",
    "Importar dados",
    "Treinar modelos",
    "Prever partida",
    "Simulações",
    "Validação",
    "Exportar",
]

with st.sidebar:
    st.markdown(
        f"""
        <div class="brand-card">
            <div class="brand-title">{icon_svg('layers', 22)} Football Analytics</div>
            <div class="brand-subtitle">Liquid Glass Analytics Suite</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if option_menu is not None:
        page = option_menu(
            menu_title=None,
            options=PAGES,
            icons=["speedometer2", "globe-americas", "cloud-download", "cpu", "crosshair", "trophy", "graph-up-arrow", "box-arrow-down"],
            default_index=0,
            styles={
                "container": {"padding": "0", "background-color": "transparent"},
                "icon": {"color": "#19C37D", "font-size": "18px"},
                "nav-link": {
                    "font-size": "14px",
                    "font-weight": "700",
                    "color": "rgba(255,255,255,0.74)",
                    "border-radius": "14px",
                    "padding": "12px 12px",
                    "margin": "4px 0",
                    "background-color": "transparent",
                },
                "nav-link-selected": {
                    "background": "linear-gradient(135deg, rgba(125,211,252,0.22), rgba(167,139,250,0.16))",
                    "color": "white",
                    "border": "1px solid rgba(255,255,255,0.14)",
                },
            },
        )
    else:
        page = st.radio("Navegação", PAGES, label_visibility="collapsed")

    st.divider()
    if has_data():
        st.success("Base carregada")
        st.caption(st.session_state.fonte_base or "Fonte não informada")
    else:
        st.warning("Sem base")

    if has_models():
        st.success("Modelos treinados")
        if st.session_state.ml_model and getattr(st.session_state.ml_model, "treinado", False):
            st.caption("Machine Learning ativo")
        else:
            st.caption("Machine Learning indisponível ou base pequena")
    else:
        st.info("Modelos não treinados")


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":
    hero()

    if not has_data():
        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("cloud-download", "Passo 1", "Importe dados", "Use internet ou planilha local")
        with col2:
            metric_card("brain", "Passo 2", "Treine", "Elo, Poisson, ML e Ensemble")
        with col3:
            metric_card("trophy", "Passo 3", "Simule", "Partidas, Copa e campeão")

        st.info("Vá em Importar dados para começar.")
    else:
        df = st.session_state.df_matches.copy()
        teams = sorted(set(df["home_team"]).union(set(df["away_team"])))

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("calendar", "Partidas", f"{len(df):,}".replace(",", "."), "jogos válidos")
        with col2:
            metric_card("globe", "Equipes", str(len(teams)), "seleções/equipes únicas")
        with col3:
            metric_card("line-chart", "Primeiro jogo", str(df["date"].min().date()), "início da série")
        with col4:
            metric_card("activity", "Último jogo", str(df["date"].max().date()), "fim da série")

        if st.session_state.equipes_copa_2026:
            section("Estado Copa 2026 — última atualização FIFA", "globe")
            c_live1, c_live2 = st.columns([1, 2])
            with c_live1:
                nota_update = st.session_state.fifa_live_last_update or "sem horário registrado"
                metric_card("trophy", "Seleções da Copa", str(len(st.session_state.equipes_copa_2026)), f"última consulta: {nota_update}")
            with c_live2:
                if isinstance(st.session_state.df_copa2026_classificacao, pd.DataFrame) and not st.session_state.df_copa2026_classificacao.empty:
                    st.dataframe(st.session_state.df_copa2026_classificacao.head(60), width="stretch", hide_index=True)
                else:
                    st.dataframe(pd.DataFrame({"Seleção": st.session_state.equipes_copa_2026}), width="stretch", hide_index=True)

        section("Visão geral da base", "bar-chart")
        df_year = df.assign(ano=df["date"].dt.year).groupby("ano").size().reset_index(name="partidas")
        fig = px.line(df_year, x="ano", y="partidas", markers=True, title="Partidas por ano")
        fig.update_traces(line_width=3, marker_size=8)
        fig.update_layout(height=370)
        st.plotly_chart(style_plotly(fig), width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            top_home = df["home_team"].value_counts().head(15).reset_index()
            top_home.columns = ["Equipe", "Jogos como mandante"]
            fig_home = px.bar(top_home, x="Jogos como mandante", y="Equipe", orientation="h", title="Top mandantes")
            fig_home.update_layout(height=450, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(style_plotly(fig_home), width="stretch")
        with c2:
            st.dataframe(df.tail(20), width="stretch", hide_index=True)


# ============================================================
# CAMPEONATOS
# ============================================================

elif page == "Campeonatos":
    hero()
    section("Selecionar campeonato e temporada", "globe")
    st.markdown(
        "Baixe resultados encerrados e o calendário futuro da competição. "
        "Os jogos concluídos treinam o mesmo motor Elo, Poisson, Dixon–Coles, ML e Ensemble usado no restante do aplicativo."
    )

    provedor_campeonato = st.radio(
        "Fonte dos dados",
        ["TheSportsDB — gratuito, 2025 + 2026", "API-Football — requer plano para temporada atual"],
        horizontal=True,
    )
    usar_tsdb = provedor_campeonato.startswith("TheSportsDB")
    if usar_tsdb:
        st.info(
            "O modo gratuito junta automaticamente as duas temporadas recentes. "
            "Para ligas de ano-calendário usa 2025 e 2026; para ligas europeias usa 2025–2026 e 2026–2027. "
            "A base é colaborativa, portanto o aplicativo mostra a quantidade realmente encontrada."
        )
    else:
        st.warning("A chave gratuita da API-Football não libera 2025/2026. Use este modo somente com plano compatível.")

    c1, c2, c3 = st.columns([2.2, 1, 1])
    with c1:
        nome_competicao = st.selectbox("Campeonato", list(COMPETICOES.keys()))
    with c2:
        ano_final_dupla = st.number_input(
            "Ano mais recente", min_value=2001, max_value=2035,
            value=2026, step=1, disabled=not usar_tsdb,
            help="No modo gratuito, o app busca este ano e o anterior.",
        )
        temporada_competicao = st.number_input(
            "Temporada API-Football", min_value=2000, max_value=2035,
            value=2026, step=1, disabled=usar_tsdb,
        )
    with c3:
        id_padrao = int(THESPORTSDB_COMPETICOES[nome_competicao] if usar_tsdb else COMPETICOES[nome_competicao])
        id_competicao = st.number_input(
            "ID da competição", min_value=1, max_value=99999,
            value=id_padrao if id_padrao > 0 else 1, step=1,
            disabled=id_padrao > 0,
            help="Edite quando o campeonato ainda não possuir ID gratuito confirmado.",
        )

    try:
        secret_disponivel = bool(obter_chave_futebol("", st.secrets))
    except Exception:
        secret_disponivel = bool(obter_chave_futebol(""))
    api_football_digitada = ""
    if not usar_tsdb:
        api_football_digitada = st.text_input(
            "Chave API-Football", value="", type="password",
            placeholder="Configurada nos Secrets" if secret_disponivel else "Cole a chave somente para esta sessão",
            help="No Streamlit Cloud, prefira API_FOOTBALL_KEY nos Secrets.",
        )
    treinar_automaticamente = st.checkbox("Treinar todos os modelos após baixar", value=True)

    if st.button("Baixar campeonato", type="primary", width="stretch", icon=":material/cloud_download:"):
        try:
            league_id = int(id_padrao if id_padrao > 0 else id_competicao)
            with st.spinner("Consultando partidas, separando resultados e calendário..."):
                if usar_tsdb:
                    temporadas_usadas = temporadas_recentes_thesportsdb(nome_competicao, int(ano_final_dupla))
                    result = fetch_thesportsdb_two_seasons(nome_competicao, league_id, temporadas_usadas)
                else:
                    chave = obter_chave_futebol(api_football_digitada, st.secrets)
                    temporadas_usadas = [str(int(temporada_competicao))]
                    result = fetch_competition(chave, nome_competicao, league_id, int(temporada_competicao))
            if len(result.matches) < 20:
                st.warning(
                    f"A API retornou apenas {len(result.matches)} jogos encerrados. "
                    "O modelo pode ficar instável; confira a temporada ou aguarde mais rodadas."
                )
            st.session_state.df_matches = result.matches.copy()
            st.session_state.df_competicao_futuros = result.fixtures.copy()
            st.session_state.df_competicao_bruto = result.raw.copy()
            st.session_state.competicao_atual = result.competition
            st.session_state.competicao_id = result.league_id
            st.session_state.competicao_temporada = " + ".join(temporadas_usadas)
            st.session_state.competicao_atualizada_utc = result.fetched_at_utc
            st.session_state.df_previsoes_calendario = None
            st.session_state.fonte_base = f"{result.competition} — temporadas {' + '.join(temporadas_usadas)}"
            reset_models()
            if treinar_automaticamente:
                with st.spinner("Treinando Elo, Poisson, Dixon–Coles, ML e Ensemble..."):
                    treinar_modelos()
            st.success(
                f"{len(result.matches)} jogos encerrados e {len(result.fixtures)} jogos futuros carregados."
            )
        except SeasonPlanError as e:
            st.error(str(e))
            st.warning(
                "Selecione 2024, 2023 ou 2022 no modo gratuito. Para prever partidas atuais, "
                "use uma chave com acesso à temporada atual ou importe um CSV/XLSX atualizado em Importar dados."
            )
        except Exception as e:
            st.error(f"Não foi possível carregar o campeonato: {e}")

    if st.session_state.competicao_atual:
        st.divider()
        section(f"{st.session_state.competicao_atual} — {st.session_state.competicao_temporada}", "trophy")
        m1, m2, m3 = st.columns(3)
        with m1:
            jogos_estado = st.session_state.df_matches
            metric_card("calendar", "Jogos encerrados", str(len(jogos_estado)) if isinstance(jogos_estado, pd.DataFrame) else "0", "base de treino")
        with m2:
            futuros_estado = st.session_state.df_competicao_futuros
            metric_card("target", "Jogos futuros", str(len(futuros_estado)) if isinstance(futuros_estado, pd.DataFrame) else "0", "calendário disponível")
        with m3:
            metric_card("activity", "ID da competição", str(st.session_state.competicao_id), "API-Football")

        if isinstance(st.session_state.df_competicao_futuros, pd.DataFrame) and not st.session_state.df_competicao_futuros.empty:
            with st.expander("Ver próximos jogos", expanded=True):
                cols = [c for c in ["date", "round", "home_team", "away_team", "venue", "status"] if c in st.session_state.df_competicao_futuros.columns]
                st.dataframe(st.session_state.df_competicao_futuros[cols], width="stretch", hide_index=True)

            limite_previsoes = st.slider("Quantidade máxima de jogos para prever", 1, 200, 50)
            if st.button("Prever calendário futuro", width="stretch", icon=":material/analytics:"):
                if not has_models():
                    st.warning("Treine os modelos antes de prever o calendário.")
                else:
                    previsoes = predict_fixtures(
                        st.session_state.df_competicao_futuros,
                        st.session_state.poisson_model,
                        limit=limite_previsoes,
                    )
                    st.session_state.df_previsoes_calendario = previsoes
                    if previsoes.empty:
                        st.warning("Nenhum jogo pôde ser previsto. Os nomes das equipes podem não existir na base treinada.")
                    else:
                        st.success(f"{len(previsoes)} partidas previstas.")

        if isinstance(st.session_state.df_previsoes_calendario, pd.DataFrame) and not st.session_state.df_previsoes_calendario.empty:
            section("Previsões do calendário", "line-chart")
            st.dataframe(st.session_state.df_previsoes_calendario, width="stretch", hide_index=True)
            excel_comp = dataframe_to_excel_bytes({
                "previsoes": st.session_state.df_previsoes_calendario,
                "jogos_treino": st.session_state.df_matches,
                "calendario": st.session_state.df_competicao_futuros,
            })
            st.download_button(
                "Baixar previsões em Excel", data=excel_comp,
                file_name=f"previsoes_{st.session_state.competicao_id}_{st.session_state.competicao_temporada}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch", icon=":material/download:",
            )


# ============================================================
# IMPORTAR DADOS
# ============================================================

elif page == "Importar dados":
    hero()
    section("Importar arquivo único de treino", "upload")

    tab_unico, tab_net = st.tabs(["Arquivo único local", "Extração da internet"])

    with tab_unico:
        st.markdown(
            "Use **um único Excel** contendo a base histórica e as abas da Copa/FIFA. "
            "O arquivo ideal é o `copa2026_fifa_local.xlsx` gerado pelo extrator local. "
            "Ao carregar, o app importa a base, importa as métricas FIFA, importa a Copa 2026 e já pode treinar os modelos."
        )

        st.info(
            "Estrutura esperada no Excel: aba `base_historica_2018` para treino; "
            "abas `relatorio_mestre_equipes` ou `team_stats_*` para métricas FIFA; "
            "abas `copa2026_equipes_oficiais`, `copa2026_classificacao_atual` e `copa2026_jogos_fifa` para a Copa."
        )

        uploaded_unico = st.file_uploader(
            "Selecione o Excel único de treino + Copa 2026",
            type=["xlsx", "xls", "csv"],
            key="uploaded_arquivo_unico_treino",
        )

        treinar_apos_carregar = st.checkbox(
            "Treinar modelos automaticamente após carregar",
            value=True,
            help="Quando marcado, o app carrega o Excel, sincroniza a Copa 2026, incorpora as métricas FIFA e treina Elo, Poisson, ML e Ensemble no mesmo clique.",
        )

        st.divider()
        st.markdown("**Marcador de odds das casas de aposta — opcional**")
        ativar_odds_import = st.checkbox(
            "Buscar odds em tempo real durante este import",
            value=False,
            help="Consulta uma API de odds e salva as médias por partida no estado do app e na exportação. Não usa scraping direto de sites de aposta.",
        )
        with st.expander("Configurar API de odds"):
            odds_api_key_input = st.text_input(
                "API key das odds",
                value="",
                type="password",
                help="Pode deixar vazio se você configurou ODDS_API_KEY no secrets.toml ou nas variáveis de ambiente.",
            )
            odds_provider = st.selectbox(
                "Provedor",
                ["odds-api-io", "the-odds-api"],
                index=0,
                help="Use odds-api-io para chaves geradas em odds-api.io. Use the-odds-api só para chaves do the-odds-api.com.",
            )
            odds_sport_keys = st.text_input(
                "Esportes / sport keys",
                value="football:international-fifa-world-cup",
                help="Para odds-api.io use football:international-fifa-world-cup. Também aceita football ou football:league. Para the-odds-api use soccer_fifa_world_cup,soccer_international_friendlies.",
            )
            odds_regions = st.text_input(
                "Regiões",
                value="",
                help="Usado apenas no the-odds-api.com. No odds-api.io pode deixar vazio.",
            )
            odds_bookmakers = st.text_input(
                "Bookmakers específicos, opcional",
                value="Bet365,Unibet",
                help="No odds-api.io use nomes como Bet365,Unibet. Se deixar vazio, usa os bookmakers selecionados no dashboard da conta, quando o provedor permitir.",
            )
            odds_data_partidas = st.date_input(
                "Data das partidas para varredura",
                value=datetime.now().date(),
                help="O app busca eventos dessa data no fuso America/Sao_Paulo e só depois consulta as odds.",
                key="odds_data_import",
            )
            odds_filtrar_selecoes = st.checkbox(
                "Filtrar apenas seleções reconhecidas no Excel/modelo",
                value=True,
                help="Evita trazer odds de clubes quando o sport=football retorna muitos campeonatos.",
                key="odds_filtrar_selecoes_import",
            )

        if uploaded_unico is not None:
            if st.button("Carregar arquivo único e preparar modelos", type="primary", width="stretch", icon=":material/upload_file:"):
                suffix = Path(uploaded_unico.name).suffix.lower()
                try:
                    with st.spinner("Lendo arquivo único, separando base histórica e dados da Copa 2026..."):
                        data = uploaded_unico.getvalue()
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(data)
                            tmp_path = tmp.name

                        extras_local = {}
                        if suffix in [".xlsx", ".xls"]:
                            try:
                                df, extras_local = carregar_excel_fifa_gerado_completo(uploaded_unico)
                            except Exception:
                                loader = MatchDataLoader()
                                df = loader.carregar_arquivo(tmp_path)
                                extras_local = {}
                        else:
                            loader = MatchDataLoader()
                            df = loader.carregar_arquivo(tmp_path)
                            extras_local = {}

                        st.session_state.df_matches = df.copy()
                        st.session_state.internet_extras = extras_local if isinstance(extras_local, dict) else {}
                        st.session_state.internet_log = []
                        st.session_state.fonte_base = f"{uploaded_unico.name} (arquivo único)"

                        if extras_local:
                            sync_copa_live_state_from_extras(extras_local)
                            sync_odds_state_from_extras(extras_local)
                        else:
                            st.session_state.equipes_copa_2026 = []
                            st.session_state.df_copa2026_classificacao = None
                            st.session_state.df_copa2026_classificados = None
                            st.session_state.fifa_live_last_update = None

                        reset_models()

                        if ativar_odds_import:
                            api_key_odds = obter_odds_api_key(odds_api_key_input)
                            if not api_key_odds:
                                st.warning("Odds não atualizadas: informe ODDS_API_KEY ou digite a API key no campo de configuração.")
                            else:
                                with st.spinner("Consultando odds em tempo real e calculando consenso das casas..."):
                                    atualizar_odds_ao_vivo(
                                        api_key_odds,
                                        odds_sport_keys,
                                        odds_regions,
                                        odds_bookmakers,
                                        odds_provider,
                                        data_partidas=odds_data_partidas,
                                        filtrar_equipes=odds_filtrar_selecoes,
                                    )

                    metricas_tmp = obter_metricas_fifa_equipes_do_estado()
                    jogos_tmp = st.session_state.internet_extras.get("copa2026_jogos_fifa") if isinstance(st.session_state.internet_extras, dict) else None

                    st.success("Arquivo único carregado com sucesso.")

                    c1, c2, c3, c4, c5 = st.columns(5)
                    with c1:
                        metric_card("calendar", "Partidas para treino", f"{len(st.session_state.df_matches):,}".replace(",", "."), "base histórica")
                    with c2:
                        equipes_hist = len(set(st.session_state.df_matches["home_team"]).union(set(st.session_state.df_matches["away_team"])))
                        metric_card("globe", "Equipes históricas", str(equipes_hist), "presentes no treino")
                    with c3:
                        metric_card("trophy", "Seleções Copa", str(len(st.session_state.equipes_copa_2026)), "arquivo local")
                    with c4:
                        qtd_metricas = len(metricas_tmp) if isinstance(metricas_tmp, pd.DataFrame) and not metricas_tmp.empty else 0
                        metric_card("gauge", "Métricas FIFA", str(qtd_metricas), "linhas detectadas")
                    with c5:
                        qtd_odds = len(st.session_state.df_odds_consenso) if isinstance(st.session_state.df_odds_consenso, pd.DataFrame) and not st.session_state.df_odds_consenso.empty else 0
                        metric_card("line-chart", "Odds mercado", str(qtd_odds), "partidas com consenso")

                    if isinstance(metricas_tmp, pd.DataFrame) and not metricas_tmp.empty:
                        st.info("Métricas FIFA detectadas. Elas serão incorporadas ao Poisson, ML e força de campeão.")
                    else:
                        st.warning("Nenhuma métrica FIFA consolidada foi detectada. O modelo ainda treina pela base histórica.")

                    if st.session_state.equipes_copa_2026:
                        with st.expander("Ver seleções da Copa importadas"):
                            st.dataframe(pd.DataFrame({"Seleção": st.session_state.equipes_copa_2026}), width="stretch", hide_index=True)

                    if isinstance(st.session_state.df_copa2026_classificacao, pd.DataFrame) and not st.session_state.df_copa2026_classificacao.empty:
                        with st.expander("Ver classificação/grupos importados"):
                            st.dataframe(st.session_state.df_copa2026_classificacao, width="stretch", hide_index=True)

                    if isinstance(jogos_tmp, pd.DataFrame) and not jogos_tmp.empty:
                        with st.expander("Ver jogos FIFA importados"):
                            st.dataframe(jogos_tmp, width="stretch", hide_index=True)

                    if isinstance(st.session_state.df_odds_consenso, pd.DataFrame) and not st.session_state.df_odds_consenso.empty:
                        with st.expander("Ver odds médias das casas"):
                            st.dataframe(st.session_state.df_odds_consenso, width="stretch", hide_index=True)

                    with st.expander("Prévia da base histórica usada no treino"):
                        st.dataframe(st.session_state.df_matches.head(40), width="stretch", hide_index=True)

                    if treinar_apos_carregar:
                        try:
                            with st.spinner("Treinando Elo, Poisson, ML, Ensemble e simuladores com o arquivo único..."):
                                treinar_modelos()
                            st.success("Modelos treinados com sucesso usando o arquivo único.")
                            if isinstance(st.session_state.df_team_stats, pd.DataFrame) and not st.session_state.df_team_stats.empty:
                                cols_auditoria = [
                                    "Equipe", "Elo", "FIFA_Usado_No_Modelo", "FIFA_Overall_Index",
                                    "FIFA_Attack_Index", "FIFA_Defense_Index", "Ataque_Geral", "Defesa_Geral"
                                ]
                                cols_existentes = [c for c in cols_auditoria if c in st.session_state.df_team_stats.columns]
                                with st.expander("Auditoria das forças do modelo"):
                                    st.dataframe(st.session_state.df_team_stats[cols_existentes].head(60), width="stretch", hide_index=True)
                        except Exception as e:
                            st.error(f"Arquivo carregado, mas falhou ao treinar: {e}")

                except Exception as e:
                    st.error(f"Erro ao carregar arquivo único: {e}")

        st.divider()
        st.markdown(
            "Depois do carregamento, vá em **Prever partida** ou **Simulações**. "
            "Se desmarcar o treinamento automático, use a página **Treinar modelos**."
        )

    with tab_net:
        st.markdown(
            "Modo alternativo: baixar base histórica da internet. Para a Copa/FIFA, prefira o arquivo único local; é mais estável que scraping no Streamlit Cloud."
        )
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            ano_minimo = st.number_input("Ano mínimo", min_value=1872, max_value=2030, value=2018, step=1)
        with c2:
            incluir_fifa = st.checkbox("Consultar FIFA junto", value=False)
        with c3:
            renderizar_js = st.checkbox(
                "Renderizar JavaScript da FIFA com Selenium",
                value=False,
                help="Use somente se o ambiente tiver Selenium e navegador disponível. No Streamlit Cloud, pode falhar."
            )

        if st.button("Extrair base histórica da internet", type="secondary", width="stretch", icon=":material/cloud_download:"):
            try:
                with st.spinner("Extraindo e padronizando base histórica..."):
                    extractor = InternetDataExtractor(timeout=45, renderizar_js=bool(renderizar_js))
                    result = extractor.extrair(ano_minimo=int(ano_minimo), incluir_fifa=bool(incluir_fifa))

                st.session_state.df_matches = result.matches.copy()
                st.session_state.internet_extras = result.extras
                st.session_state.internet_log = result.log
                st.session_state.fonte_base = "internet"
                sync_copa_live_state_from_extras(result.extras)
                sync_odds_state_from_extras(result.extras)
                reset_models()

                st.success("Base histórica extraída. Treine os modelos novamente antes de prever ou simular.")
                st.dataframe(result.matches.head(30), width="stretch", hide_index=True)
                with st.expander("Ver log da extração"):
                    st.code("\n".join(result.log))
            except Exception as e:
                st.error(f"Erro na extração: {e}")


# ============================================================
# TREINAR MODELOS
# ============================================================

elif page == "Treinar modelos":
    hero()
    section("Treinamento", "brain")

    if not has_data():
        st.warning("Importe uma base primeiro.")
    else:
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.metric("Partidas", len(st.session_state.df_matches))
        with c2:
            st.metric("Equipes", len(set(st.session_state.df_matches["home_team"]).union(set(st.session_state.df_matches["away_team"]))))

        if st.button("Treinar todos os modelos", type="primary", width="stretch", icon=":material/rocket_launch:"):
            try:
                with st.spinner("Treinando Elo, xG, Poisson, Dixon-Coles, bivariada, ML e Ensemble..."):
                    treinar_modelos()
                st.success("Modelos treinados com sucesso.")
            except Exception as e:
                st.error(f"Erro no treinamento: {e}")

        if has_models():
            df_metricas_usadas = st.session_state.get("df_fifa_team_metrics")
            if isinstance(df_metricas_usadas, pd.DataFrame) and not df_metricas_usadas.empty:
                st.success(f"Métricas FIFA incorporadas ao modelo: {len(df_metricas_usadas)} seleções/linhas de métricas.")
            else:
                st.info("Modelo treinado sem métricas FIFA extras. Apenas histórico de partidas foi usado.")

            section("Ranking Elo", "medal")
            df_elo = st.session_state.df_elo.copy()
            c1, c2 = st.columns([1.25, 1])
            with c1:
                st.dataframe(df_elo.head(40), width="stretch", hide_index=True)
            with c2:
                top = df_elo.head(15).sort_values("Elo")
                fig = px.bar(top, x="Elo", y="Equipe", orientation="h", title="Top 15 por Elo")
                fig.update_layout(height=520)
                st.plotly_chart(style_plotly(fig), width="stretch")

            with st.expander("Ver forças ofensivas e defensivas"):
                st.dataframe(st.session_state.df_team_stats, width="stretch", hide_index=True)


# ============================================================
# PREVER PARTIDA
# ============================================================

elif page == "Prever partida":
    hero()
    section("Previsão de partida", "target")

    if not has_models():
        st.warning("Treine os modelos antes de prever.")
    else:
        all_teams = get_teams()
        copa_teams = get_copa_teams_for_model()
        usar_copa = False
        if copa_teams:
            usar_copa = st.toggle(
                "Modo Copa 2026: usar apenas seleções detectadas na última atualização FIFA",
                value=True,
                help="O histórico treina o modelo, mas a competição da Copa fica limitada às seleções da última consulta FIFA."
            )
        teams = copa_teams if usar_copa and copa_teams else all_teams

        if usar_copa and copa_teams:
            st.info(f"Modo Copa 2026 ativo: {len(copa_teams)} seleções disponíveis, vindas da última atualização FIFA.")

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            mandante = st.selectbox("Mandante / Time A", teams, index=0)
        with c2:
            default_idx = 1 if len(teams) > 1 else 0
            visitante = st.selectbox("Visitante / Time B", teams, index=default_idx)
        with c3:
            competicao = st.text_input("Competição", value="FIFA World Cup 2026" if usar_copa else "Não informado")

        with st.expander("Odds das casas de aposta"):
            st.caption("As odds são usadas como leitura de mercado. O placar provável continua vindo do modelo estatístico.")
            c_od1, c_od2, c_od3 = st.columns([1, 1, 1])
            with c_od1:
                peso_odds = st.slider("Peso das odds na combinação final", 0.0, 0.8, 0.35, 0.05)
            with c_od2:
                st.metric("Partidas com odds", len(st.session_state.df_odds_consenso) if isinstance(st.session_state.df_odds_consenso, pd.DataFrame) and not st.session_state.df_odds_consenso.empty else 0)
            with c_od3:
                st.caption(f"Última atualização: {st.session_state.odds_last_update or 'não atualizada'}")

            odds_data_pred = st.date_input(
                "Data das partidas para buscar/usar odds",
                value=datetime.now().date(),
                help="Filtra odds pela data local America/Sao_Paulo. Use a data do jogo, não a data da previsão.",
                key="odds_data_prever",
            )
            odds_filtrar_selecoes_pred = st.checkbox(
                "Filtrar odds apenas por seleções do Excel/modelo",
                value=True,
                key="odds_filtrar_selecoes_prever",
            )
            atualizar_agora = st.checkbox("Atualizar odds agora antes da previsão", value=False)
            if atualizar_agora:
                odds_api_key_pred = st.text_input("API key das odds", value="", type="password", key="odds_api_key_prever")
                odds_provider_pred = st.selectbox("Provedor", ["odds-api-io", "the-odds-api"], index=0, key="odds_provider_prever")
                odds_sport_keys_pred = st.text_input("Esportes / sport keys", value="football:international-fifa-world-cup", key="odds_sport_keys_prever")
                odds_regions_pred = st.text_input("Regiões", value="", key="odds_regions_prever")
                odds_bookmakers_pred = st.text_input("Bookmakers específicos, opcional", value="Bet365,Unibet", key="odds_books_prever")

            if st.session_state.odds_log:
                with st.expander("Último log técnico de odds"):
                    st.code("\n".join(st.session_state.odds_log[-80:]))

        if st.button("Calcular previsão", type="primary", width="stretch", icon=":material/bolt:"):
            if mandante == visitante:
                st.error("Escolha equipes diferentes.")
            else:
                try:
                    r = st.session_state.poisson_model.prever_partida(mandante, visitante)
                    r_ml = None
                    if st.session_state.ml_model and getattr(st.session_state.ml_model, "treinado", False):
                        try:
                            r_ml = st.session_state.ml_model.prever_partida(mandante, visitante, competicao=competicao)
                        except Exception:
                            r_ml = None
                    r_ens = st.session_state.ensemble_model.combinar(r, r_ml)

                    if 'atualizar_agora' in locals() and atualizar_agora:
                        api_key_odds = obter_odds_api_key(odds_api_key_pred)
                        if not api_key_odds:
                            st.warning("Odds não atualizadas: informe ODDS_API_KEY ou digite a API key.")
                        else:
                            with st.spinner("Atualizando odds em tempo real..."):
                                atualizar_odds_ao_vivo(
                                    api_key_odds,
                                    odds_sport_keys_pred,
                                    odds_regions_pred,
                                    odds_bookmakers_pred,
                                    odds_provider_pred,
                                    data_partidas=odds_data_pred,
                                    filtrar_equipes=odds_filtrar_selecoes_pred,
                                )

                    odds_partida = BettingOddsExtractor.buscar_odds_partida(
                        st.session_state.df_odds_consenso,
                        mandante,
                        visitante,
                        target_date=odds_data_pred if 'odds_data_pred' in locals() else None,
                        target_timezone="America/Sao_Paulo",
                    ) if isinstance(st.session_state.df_odds_consenso, pd.DataFrame) else None
                    previsao_mercado = combinar_previsao_com_odds(r_ens, odds_partida, peso_odds=peso_odds) if odds_partida else None

                    st.session_state.ultima_previsao = r
                    st.session_state.ultima_previsao_ml = r_ml
                    st.session_state.ultima_previsao_ensemble = r_ens
                    st.session_state.ultima_odds_partida = odds_partida
                    st.session_state.ultima_previsao_modelo_mercado = previsao_mercado
                    st.success("Previsão calculada.")
                except Exception as e:
                    st.error(f"Erro na previsão: {e}")

        r = st.session_state.ultima_previsao
        r_ml = st.session_state.ultima_previsao_ml
        r_ens = st.session_state.ultima_previsao_ensemble
        odds_partida = st.session_state.ultima_odds_partida
        previsao_mercado = st.session_state.ultima_previsao_modelo_mercado

        if r and r_ens:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card("home", f"Vitória {r_ens['mandante']}", f"{r_ens['ensemble_prob_mandante']}%", "Ensemble final")
            with c2:
                metric_card("layers", "Empate", f"{r_ens['ensemble_prob_empate']}%", "Ensemble final")
            with c3:
                metric_card("arrow-up-right", f"Vitória {r_ens['visitante']}", f"{r_ens['ensemble_prob_visitante']}%", "Ensemble final")
            with c4:
                metric_card("target", "Placar provável", r_ens["placar_provavel"], f"Confiança: {r_ens['confianca']}")

            if odds_partida:
                c_od1, c_od2, c_od3, c_od4 = st.columns(4)
                with c_od1:
                    metric_card("line-chart", "Favorito casas", str(odds_partida.get("Favorito_Casas", "n/d")), f"{odds_partida.get('Casas_Usadas', 0)} casa(s)")
                with c_od2:
                    metric_card("home", f"Mercado {r_ens['mandante']}", f"{odds_partida.get('Prob_Mercado_Mandante_pct', 0)}%", f"odd média {round(float(odds_partida.get('Odds_Media_Mandante', 0) or 0), 2)}")
                with c_od3:
                    metric_card("layers", "Mercado empate", f"{odds_partida.get('Prob_Mercado_Empate_pct', 0)}%", f"odd média {round(float(odds_partida.get('Odds_Media_Empate', 0) or 0), 2)}")
                with c_od4:
                    metric_card("arrow-up-right", f"Mercado {r_ens['visitante']}", f"{odds_partida.get('Prob_Mercado_Visitante_pct', 0)}%", f"odd média {round(float(odds_partida.get('Odds_Media_Visitante', 0) or 0), 2)}")

                if previsao_mercado:
                    st.info(
                        f"**Favorito combinado modelo + mercado:** {previsao_mercado['favorito_modelo_mercado']} "
                        f"| Modelo: {round(100 * previsao_mercado['peso_modelo'])}% "
                        f"| Odds: {round(100 * previsao_mercado['peso_odds'])}% "
                        f"| Placar provável estatístico: {previsao_mercado['placar_provavel_modelo']}"
                    )
            else:
                st.warning("Nenhuma odd correspondente foi encontrada para esta partida. A previsão abaixo usa apenas o modelo estatístico/ML.")

            prob_df = pd.DataFrame({
                "Resultado": [f"Vitória {r_ens['mandante']}", "Empate", f"Vitória {r_ens['visitante']}"],
                "Probabilidade": [r_ens["ensemble_prob_mandante"], r_ens["ensemble_prob_empate"], r_ens["ensemble_prob_visitante"]],
            })
            lambda_df = pd.DataFrame({
                "Equipe": [r["mandante"], r["visitante"]],
                "xG esperado": [r["lambda_mandante"], r["lambda_visitante"]],
            })
            if odds_partida:
                mercado_df = pd.DataFrame({
                    "Resultado": [f"Vitória {r_ens['mandante']}", "Empate", f"Vitória {r_ens['visitante']}"],
                    "Probabilidade mercado": [
                        odds_partida.get("Prob_Mercado_Mandante_pct", 0),
                        odds_partida.get("Prob_Mercado_Empate_pct", 0),
                        odds_partida.get("Prob_Mercado_Visitante_pct", 0),
                    ],
                    "Probabilidade modelo": [
                        r_ens["ensemble_prob_mandante"],
                        r_ens["ensemble_prob_empate"],
                        r_ens["ensemble_prob_visitante"],
                    ],
                })
                with st.expander("Comparativo modelo x casas"):
                    st.dataframe(mercado_df, width="stretch", hide_index=True)
                    st.dataframe(pd.DataFrame([odds_partida]), width="stretch", hide_index=True)

            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(prob_df, x="Resultado", y="Probabilidade", title="Probabilidades 1X2 — Ensemble", text="Probabilidade")
                fig.update_layout(height=400, yaxis_title="%")
                st.plotly_chart(style_plotly(fig), width="stretch")
            with c2:
                fig2 = px.bar(lambda_df, x="Equipe", y="xG esperado", title="xG esperado / intensidade ofensiva", text="xG esperado")
                fig2.update_layout(height=400)
                st.plotly_chart(style_plotly(fig2), width="stretch")

            c1, c2 = st.columns(2)
            with c1:
                section("Parâmetros do modelo", "gauge")
                df_parametros = pd.DataFrame([
                    {"Métrica": "Rho Dixon-Coles", "Valor": r["rho_dixon_coles"]},
                    {"Métrica": "Lambda compartilhado", "Valor": r["lambda_compartilhado"]},
                    {"Métrica": "Peso estatístico", "Valor": r_ens["peso_estatistico"]},
                    {"Métrica": "Peso ML", "Valor": r_ens["peso_ml"]},
                    {"Métrica": "Over 2.5", "Valor": f"{r['prob_over_25']}%"},
                    {"Métrica": "Ambas marcam", "Valor": f"{r['prob_btts']}%"},
                    {"Métrica": "Favorito casas", "Valor": odds_partida.get("Favorito_Casas", "sem odds") if odds_partida else "sem odds"},
                    {"Métrica": "Favorito combinado", "Valor": previsao_mercado.get("favorito_modelo_mercado", "sem odds") if previsao_mercado else "sem odds"},
                ])
                df_parametros["Valor"] = df_parametros["Valor"].astype(str)
                st.dataframe(df_parametros, width="stretch", hide_index=True)
            with c2:
                section("Placares mais prováveis", "bar-chart")
                st.dataframe(pd.DataFrame(r["top_placares"]), width="stretch", hide_index=True)

            if r_ml:
                with st.expander("Ver previsão separada do Machine Learning"):
                    st.dataframe(pd.DataFrame([r_ml]), width="stretch", hide_index=True)


# ============================================================
# SIMULAÇÕES
# ============================================================

elif page == "Simulações":
    hero()
    section("Simulações Monte Carlo", "trophy")

    if not has_models():
        st.warning("Treine os modelos antes de simular.")
    else:
        tab_ko, tab_wc = st.tabs(["Mata-mata aleatório", "Copa 2026 aproximada"])

        with tab_ko:
            iteracoes = st.number_input("Simulações", min_value=100, max_value=30000, value=3000, step=500, key="it_ko")
            if st.button("Simular mata-mata", type="primary", width="stretch", icon=":material/trophy:"):
                try:
                    with st.spinner("Simulando torneios..."):
                        equipes_copa = get_copa_teams_for_model()
                        df_sim = st.session_state.simulador_campeao.simular_campeao(
                            iteracoes=int(iteracoes),
                            equipes=equipes_copa if equipes_copa else None,
                        )
                    st.session_state.df_simulacao_campeao = df_sim.copy()
                    st.success("Simulação concluída.")
                except Exception as e:
                    st.error(f"Erro na simulação: {e}")

            if st.session_state.df_simulacao_campeao is not None:
                df_sim = st.session_state.df_simulacao_campeao.head(25).copy()
                fig = px.bar(df_sim.sort_values("Probabilidade_Titulo_%"), x="Probabilidade_Titulo_%", y="Equipe", orientation="h", title="Probabilidade de título — mata-mata")
                fig.update_layout(height=650)
                st.plotly_chart(style_plotly(fig), width="stretch")
                st.dataframe(st.session_state.df_simulacao_campeao, width="stretch", hide_index=True)

        with tab_wc:
            iteracoes_copa = st.number_input("Simulações", min_value=100, max_value=20000, value=1000, step=500, key="it_wc")
            if st.button("Simular formato Copa 2026", type="primary", width="stretch", icon=":material/public:"):
                try:
                    with st.spinner("Simulando fase de grupos e mata-mata..."):
                        df_wc, ultima = st.session_state.simulador_copa.simular_campeao_formato_copa(iteracoes=int(iteracoes_copa))
                    st.session_state.df_simulacao_copa = df_wc.copy()
                    st.session_state.ultima_copa_simulada = ultima
                    st.success("Simulação da Copa concluída.")
                except Exception as e:
                    st.error(f"Erro na simulação da Copa: {e}")

            if st.session_state.df_simulacao_copa is not None:
                top = st.session_state.df_simulacao_copa.head(25).copy()
                fig = px.bar(top.sort_values("Prob_Titulo_%"), x="Prob_Titulo_%", y="Equipe", orientation="h", title="Probabilidade de título — Copa aproximada")
                fig.update_layout(height=650)
                st.plotly_chart(style_plotly(fig), width="stretch")
                st.dataframe(st.session_state.df_simulacao_copa, width="stretch", hide_index=True)

                if st.session_state.ultima_copa_simulada is not None:
                    with st.expander("Ver última Copa simulada"):
                        st.dataframe(st.session_state.ultima_copa_simulada["classificados"], width="stretch", hide_index=True)
                        st.dataframe(st.session_state.ultima_copa_simulada["mata_mata"], width="stretch", hide_index=True)


# ============================================================
# VALIDAÇÃO
# ============================================================

elif page == "Validação":
    hero()
    section("Validação temporal", "line-chart")

    if not has_data():
        st.warning("Importe uma base antes de validar.")
    else:
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            min_treino = st.number_input("Mínimo de treino", min_value=30, max_value=5000, value=80, step=10)
        with c2:
            max_teste = st.number_input("Máximo de testes", min_value=10, max_value=500, value=100, step=10)
        with c3:
            st.markdown("<div class='soft-text'>A validação treina com jogos antigos e prevê jogos posteriores, evitando vazamento de dados.</div>", unsafe_allow_html=True)

        if st.button("Rodar validação", type="primary", width="stretch", icon=":material/monitoring:"):
            try:
                with st.spinner("Rodando backtest temporal..."):
                    validator = ModelBacktester()
                    resumo, previsoes = validator.validar(
                        st.session_state.df_matches,
                        min_treino=int(min_treino),
                        max_partidas_teste=int(max_teste),
                        usar_ml=True,
                    )
                st.session_state.df_validacao_resumo = resumo.copy()
                st.session_state.df_validacao_previsoes = previsoes.copy()
                st.success("Validação concluída.")
            except Exception as e:
                st.error(f"Erro na validação: {e}")

        if st.session_state.df_validacao_resumo is not None:
            section("Métricas", "activity")
            st.dataframe(st.session_state.df_validacao_resumo, width="stretch", hide_index=True)

            resumo_long = st.session_state.df_validacao_resumo.melt(
                id_vars=["Modelo", "Partidas_Avaliadas"],
                value_vars=["Acuracia_%", "Log_Loss", "Brier_Score"],
                var_name="Métrica",
                value_name="Valor",
            )
            fig = px.bar(resumo_long, x="Modelo", y="Valor", color="Métrica", barmode="group", title="Comparação de métricas")
            fig.update_layout(height=440)
            st.plotly_chart(style_plotly(fig), width="stretch")

            with st.expander("Ver previsões testadas"):
                st.dataframe(st.session_state.df_validacao_previsoes, width="stretch", hide_index=True)


# ============================================================
# EXPORTAR
# ============================================================

elif page == "Exportar":
    hero()
    section("Exportação", "download")

    sheets: dict[str, pd.DataFrame] = {}

    if st.session_state.df_matches is not None:
        sheets["base_padronizada"] = st.session_state.df_matches
    if st.session_state.df_elo is not None:
        sheets["ranking_elo"] = st.session_state.df_elo
    if st.session_state.df_team_stats is not None:
        sheets["forcas_modelo"] = st.session_state.df_team_stats
    if st.session_state.ultima_previsao is not None:
        prev = {k: v for k, v in st.session_state.ultima_previsao.items() if k != "matriz"}
        sheets["ultima_previsao_stat"] = pd.DataFrame([prev])
        sheets["matriz_placares"] = st.session_state.ultima_previsao["matriz"]
    if st.session_state.ultima_previsao_ml is not None:
        sheets["ultima_previsao_ml"] = pd.DataFrame([st.session_state.ultima_previsao_ml])
    if st.session_state.ultima_previsao_ensemble is not None:
        sheets["ultima_previsao_ensemble"] = pd.DataFrame([st.session_state.ultima_previsao_ensemble])
    if st.session_state.df_simulacao_campeao is not None:
        sheets["simulacao_mata_mata"] = st.session_state.df_simulacao_campeao
    if st.session_state.df_simulacao_copa is not None:
        sheets["simulacao_copa_2026"] = st.session_state.df_simulacao_copa
    if st.session_state.df_validacao_resumo is not None:
        sheets["validacao_resumo"] = st.session_state.df_validacao_resumo
    if st.session_state.df_validacao_previsoes is not None:
        sheets["validacao_previsoes"] = st.session_state.df_validacao_previsoes
    if isinstance(st.session_state.df_odds_consenso, pd.DataFrame) and not st.session_state.df_odds_consenso.empty:
        sheets["odds_consenso_partidas"] = st.session_state.df_odds_consenso
    if isinstance(st.session_state.df_odds_raw, pd.DataFrame) and not st.session_state.df_odds_raw.empty:
        sheets["odds_mercado_bruto"] = st.session_state.df_odds_raw
    if st.session_state.ultima_odds_partida is not None:
        sheets["ultima_odds_partida"] = pd.DataFrame([st.session_state.ultima_odds_partida])
    if st.session_state.ultima_previsao_modelo_mercado is not None:
        sheets["ultima_prev_modelo_mercado"] = pd.DataFrame([st.session_state.ultima_previsao_modelo_mercado])
    if isinstance(st.session_state.df_competicao_futuros, pd.DataFrame) and not st.session_state.df_competicao_futuros.empty:
        sheets["calendario_competicao"] = st.session_state.df_competicao_futuros
    if isinstance(st.session_state.df_competicao_bruto, pd.DataFrame) and not st.session_state.df_competicao_bruto.empty:
        sheets["dados_competicao_brutos"] = st.session_state.df_competicao_bruto
    if isinstance(st.session_state.df_previsoes_calendario, pd.DataFrame) and not st.session_state.df_previsoes_calendario.empty:
        sheets["previsoes_calendario"] = st.session_state.df_previsoes_calendario

    for name, df_extra in (st.session_state.internet_extras or {}).items():
        if isinstance(df_extra, pd.DataFrame) and not df_extra.empty:
            sheets[f"extra_{name}"] = df_extra

    if not sheets:
        st.warning("Ainda não há dados para exportar.")
    else:
        st.write(f"Serão exportadas **{len(sheets)}** abas.")
        excel_bytes = dataframe_to_excel_bytes(sheets)
        st.download_button(
            label="Baixar relatório Excel",
            data=excel_bytes,
            file_name="relatorio_football_analytics.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            width="stretch",
            icon=":material/download:",
        )

        with st.expander("Abas incluídas"):
            st.write(list(sheets.keys()))
