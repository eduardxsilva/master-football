
"""
FIFA 2026 Analytics Engine - versão completa corrigida.

Entrada principal:
    Base histórica em CSV/XLSX com colunas mínimas:
        date, home_team, away_team, home_goals, away_goals

Colunas opcionais:
        home_xg, away_xg, competition

Também aceita equivalentes em português:
        data, mandante, visitante, gols_casa, gols_fora, xg_casa, xg_fora, competicao

Dependências:
    pip install streamlit pandas numpy scipy scikit-learn openpyxl requests beautifulsoup4 lxml
"""

from __future__ import annotations

import os
import math
import traceback
import io
import threading
import time
import json
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import requests
from bs4 import BeautifulSoup


import numpy as np
import pandas as pd

from scipy.stats import poisson
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# ============================================================
# UTILITÁRIOS
# ============================================================

def normalizar_texto(valor: Any) -> str:
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
    return texto


def encontrar_coluna(df: pd.DataFrame, opcoes: List[str]) -> Optional[str]:
    mapa = {normalizar_texto(col): col for col in df.columns}

    for opcao in opcoes:
        opcao_norm = normalizar_texto(opcao)
        if opcao_norm in mapa:
            return mapa[opcao_norm]

    for opcao in opcoes:
        opcao_norm = normalizar_texto(opcao)
        for col_norm, col_original in mapa.items():
            if opcao_norm in col_norm:
                return col_original

    return None


def to_number(s: pd.Series, default: Optional[float] = None) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    if default is not None:
        out = out.fillna(default)
    return out


def clamp(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


# ============================================================
# MÉTRICAS FIFA DE EQUIPE — NORMALIZAÇÃO PARA O MODELO
# ============================================================

def _serie_numerica_fifa(s: pd.Series) -> pd.Series:
    """Converte colunas da FIFA para número, tolerando %, vírgula decimal e textos."""
    if s is None:
        return pd.Series(dtype=float)
    txt = s.astype(str).str.replace("%", "", regex=False).str.replace(",", ".", regex=False)
    txt = txt.str.replace(r"[^0-9\.\-]", "", regex=True)
    txt = txt.replace({"": np.nan, "-": np.nan, ".": np.nan})
    return pd.to_numeric(txt, errors="coerce")


def _coluna_equipe_metricas(df: pd.DataFrame) -> Optional[str]:
    return encontrar_coluna(df, [
        "Seleção", "Selecao", "Equipe", "Team", "Country", "Nation", "País", "Pais"
    ])


def _score_percentil_metricas(
    numericos: pd.DataFrame,
    incluir_tokens: List[str],
    inverter_tokens: Optional[List[str]] = None,
) -> pd.Series:
    """
    Gera um índice 0..1 por equipe a partir de colunas FIFA.
    0,5 = neutro. Para métricas ruins, inverte o percentil.
    """
    if numericos.empty:
        return pd.Series(0.5, index=numericos.index, dtype=float)

    inverter_tokens = inverter_tokens or []
    series_scores = []

    for col in numericos.columns:
        col_norm = normalizar_texto(col)
        if not any(tok in col_norm for tok in incluir_tokens):
            continue

        s = pd.to_numeric(numericos[col], errors="coerce")
        if s.notna().sum() < 2:
            continue

        # Remove colunas praticamente constantes, porque elas não ajudam o modelo.
        if float(s.max() - s.min()) == 0:
            continue

        score = s.rank(pct=True, method="average")
        if any(tok in col_norm for tok in inverter_tokens):
            score = 1.0 - score
        series_scores.append(score)

    if not series_scores:
        return pd.Series(0.5, index=numericos.index, dtype=float)

    return pd.concat(series_scores, axis=1).mean(axis=1).fillna(0.5).clip(0, 1)


def preparar_metricas_fifa_equipes(df_metricas: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Recebe a aba relatorio_mestre_equipes/team_stats_* do gerador local
    e devolve índices objetivos usados pelo Poisson, ML e simulação.

    Saída esperada:
        Equipe
        FIFA_Attack_Index
        FIFA_Defense_Index
        FIFA_Goalkeeper_Index
        FIFA_Distribution_Index
        FIFA_Discipline_Index
        FIFA_Physical_Index
        FIFA_Movement_Index
        FIFA_Overall_Index
        FIFA_Mult_Ataque
        FIFA_Mult_Defesa
        FIFA_Elo_Bonus
        FIFA_Usado_No_Modelo
    """
    if df_metricas is None or not isinstance(df_metricas, pd.DataFrame) or df_metricas.empty:
        return pd.DataFrame()

    df = df_metricas.copy()
    col_team = _coluna_equipe_metricas(df)
    if col_team is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["Equipe"] = df[col_team].apply(canonical_team_name)
    out = out[out["Equipe"].astype(str).str.strip().ne("")]
    if out.empty:
        return pd.DataFrame()

    numericos = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col == col_team:
            continue
        s = _serie_numerica_fifa(df[col])
        if s.notna().sum() >= 2:
            numericos[str(col)] = s

    if numericos.empty:
        out["FIFA_Attack_Index"] = 0.5
        out["FIFA_Defense_Index"] = 0.5
        out["FIFA_Goalkeeper_Index"] = 0.5
        out["FIFA_Distribution_Index"] = 0.5
        out["FIFA_Discipline_Index"] = 0.5
        out["FIFA_Physical_Index"] = 0.5
        out["FIFA_Movement_Index"] = 0.5
    else:
        bad_defense = ["sofr", "conced", "against", "gc", "goals_allowed"]
        bad_discipline = ["cart", "card", "falt", "foul", "vermel", "amarel", "yellow", "red"]

        out["FIFA_Attack_Index"] = _score_percentil_metricas(
            numericos,
            ["ataque", "attack", "gols_feitos", "goals_for", "goals", "xg_ataque", "shots", "finaliz", "chances"],
            ["sofr", "conced", "against", "gc"],
        ).reindex(out.index).fillna(0.5).values

        out["FIFA_Defense_Index"] = _score_percentil_metricas(
            numericos,
            ["defesa", "defense", "defence", "tackle", "intercept", "bloque", "gols_sofridos", "conced"],
            bad_defense,
        ).reindex(out.index).fillna(0.5).values

        out["FIFA_Goalkeeper_Index"] = _score_percentil_metricas(
            numericos,
            ["goleiro", "goalkeeper", "keeper", "save", "defesas"],
            bad_defense,
        ).reindex(out.index).fillna(0.5).values

        out["FIFA_Distribution_Index"] = _score_percentil_metricas(
            numericos,
            ["distribuicao", "distribuição", "distribution", "passes", "passing", "posse", "possession"],
            [],
        ).reindex(out.index).fillna(0.5).values

        out["FIFA_Discipline_Index"] = _score_percentil_metricas(
            numericos,
            ["disciplina", "discipline", "card", "cart", "foul", "falt", "yellow", "red"],
            bad_discipline,
        ).reindex(out.index).fillna(0.5).values

        out["FIFA_Physical_Index"] = _score_percentil_metricas(
            numericos,
            ["fisico", "físico", "physical", "distance", "distancia", "distância", "speed", "velocidade", "sprint"],
            [],
        ).reindex(out.index).fillna(0.5).values

        out["FIFA_Movement_Index"] = _score_percentil_metricas(
            numericos,
            ["moviment", "movement", "runs", "corrida", "press"],
            [],
        ).reindex(out.index).fillna(0.5).values

    for col in [
        "FIFA_Attack_Index", "FIFA_Defense_Index", "FIFA_Goalkeeper_Index",
        "FIFA_Distribution_Index", "FIFA_Discipline_Index", "FIFA_Physical_Index",
        "FIFA_Movement_Index"
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.5).clip(0, 1)

    out["FIFA_Overall_Index"] = (
        0.30 * out["FIFA_Attack_Index"] +
        0.25 * out["FIFA_Defense_Index"] +
        0.10 * out["FIFA_Goalkeeper_Index"] +
        0.15 * out["FIFA_Distribution_Index"] +
        0.05 * out["FIFA_Discipline_Index"] +
        0.10 * out["FIFA_Physical_Index"] +
        0.05 * out["FIFA_Movement_Index"]
    ).clip(0, 1)

    # Multiplicadores moderados para não deixar estatística de torneio curto dominar histórico.
    out["FIFA_Mult_Ataque"] = np.exp(
        0.32 * (out["FIFA_Attack_Index"] - 0.5) +
        0.08 * (out["FIFA_Distribution_Index"] - 0.5) +
        0.06 * (out["FIFA_Physical_Index"] - 0.5)
    ).clip(0.86, 1.18)

    # No modelo, Defesa_* é fator de gols sofridos. Menor = melhor.
    out["FIFA_Mult_Defesa"] = np.exp(
        -0.28 * (out["FIFA_Defense_Index"] - 0.5) +
        -0.08 * (out["FIFA_Goalkeeper_Index"] - 0.5) +
        -0.05 * (out["FIFA_Discipline_Index"] - 0.5) +
        -0.04 * (out["FIFA_Physical_Index"] - 0.5)
    ).clip(0.84, 1.16)

    out["FIFA_Elo_Bonus"] = (90.0 * (out["FIFA_Overall_Index"] - 0.5)).clip(-45, 45)
    out["FIFA_Usado_No_Modelo"] = True

    # Se houver duplicata de seleção, mantém a média dos índices.
    numeric_cols = [c for c in out.columns if c != "Equipe" and c != "FIFA_Usado_No_Modelo"]
    out = out.groupby("Equipe", as_index=False)[numeric_cols].mean()
    out["FIFA_Usado_No_Modelo"] = True

    return out


# ============================================================
# NORMALIZAÇÃO DE NOMES DE SELEÇÕES
# ============================================================

TEAM_ALIASES = {
    "usa": "United States",
    "u_s_a": "United States",
    "estados_unidos": "United States",
    "united_states_of_america": "United States",
    "korea_republic": "South Korea",
    "republic_of_korea": "South Korea",
    "coreia_do_sul": "South Korea",
    "south_korea": "South Korea",
    "czechia": "Czech Republic",
    "czech_republic": "Czech Republic",
    "republica_tcheca": "Czech Republic",
    "turkiye": "Turkey",
    "turkey": "Turkey",
    "turquia": "Turkey",
    "cote_d_ivoire": "Ivory Coast",
    "côte_d_ivoire": "Ivory Coast",
    "ivory_coast": "Ivory Coast",
    "costa_do_marfim": "Ivory Coast",
    "iran": "Iran",
    "ir_iran": "Iran",
    "islamic_republic_of_iran": "Iran",
    "curacao": "Curaçao",
    "curaçao": "Curaçao",
    "bosnia_herzegovina": "Bosnia and Herzegovina",
    "bosnia_and_herzegovina": "Bosnia and Herzegovina",
    "bosnia_e_herzegovina": "Bosnia and Herzegovina",
    "england": "England",
    "inglaterra": "England",
    "scotland": "Scotland",
    "escocia": "Scotland",
    "brazil": "Brazil",
    "brasil": "Brazil",
    "germany": "Germany",
    "alemanha": "Germany",
    "argentina": "Argentina",
    "france": "France",
    "franca": "France",
    "spain": "Spain",
    "espanha": "Spain",
    "portugal": "Portugal",
    "netherlands": "Netherlands",
    "holanda": "Netherlands",
    "mexico": "Mexico",
    "méxico": "Mexico",
    "canada": "Canada",
    "canadá": "Canada",
    "morocco": "Morocco",
    "marrocos": "Morocco",
    "japan": "Japan",
    "japao": "Japan",
    "japão": "Japan",
    "switzerland": "Switzerland",
    "suica": "Switzerland",
    "suíça": "Switzerland",
    "eua": "United States",
    "africa_do_sul": "South Africa",
    "república_da_coreia": "South Korea",
    "republica_da_coreia": "South Korea",
    "tchequia": "Czech Republic",
    "catar": "Qatar",
    "australia": "Australia",
    "paraguai": "Paraguay",
    "equador": "Ecuador",
    "curacau": "Curaçao",
    "suecia": "Sweden",
    "tunisia": "Tunisia",
    "egito": "Egypt",
    "ri_do_ira": "Iran",
    "belgica": "Belgium",
    "nova_zelandia": "New Zealand",
    "uruguai": "Uruguay",
    "cabo_verde": "Cape Verde",
    "arabia_saudita": "Saudi Arabia",
    "noruega": "Norway",
    "senegal": "Senegal",
    "iraque": "Iraq",
    "austria": "Austria",
    "argelia": "Algeria",
    "jordania": "Jordan",
    "colombia": "Colombia",
    "rd_do_congo": "DR Congo",
    "congo_dr": "DR Congo",
    "uzbequistao": "Uzbekistan",
    "gana": "Ghana",
    "croacia": "Croatia",
    "panama": "Panama",
    "haiti": "Haiti",
}


def canonical_team_name(value: Any) -> str:
    """Padroniza nomes da FIFA e da base histórica para permitir cruzamento."""
    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        return ""
    # remove sufixos frequentes em tabelas dinâmicas
    raw = raw.replace("\xa0", " ").strip()
    raw = raw.replace("*", "").strip()
    key = normalizar_texto(raw)
    return TEAM_ALIASES.get(key, raw)



# ============================================================
# EXTRAÇÃO AUTOMÁTICA DA INTERNET
# ============================================================

@dataclass
class InternetExtractionResult:
    matches: pd.DataFrame
    extras: Dict[str, pd.DataFrame]
    log: List[str]


class InternetDataExtractor:
    """
    Baixa uma base histórica internacional para treino e, quando solicitado,
    consulta as páginas públicas da FIFA no momento do clique.

    Importante:
    - A base de treino continua vindo do CSV histórico.
    - A FIFA é consultada sob demanda; não há processo em background.
    - Se a página da FIFA vier renderizada por JavaScript, o extrator tenta:
        1) tabelas HTML simples;
        2) JSON embutido no HTML/Next.js;
        3) Selenium opcional, caso renderizar_js=True e Selenium esteja instalado.
    """

    RESULTS_CSV_URLS = [
        "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
    ]

    FIFA_PAGES = {
        "fifa_worldcup_standings_en": "https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/standings",
        "fifa_worldcup_standings_pt": "https://www.fifa.com/pt/tournaments/mens/worldcup/canadamexicousa2026/standings",
        "fifa_worldcup_scores_en": "https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/scores-fixtures",
        "fifa_worldcup_scores_pt": "https://www.fifa.com/pt/tournaments/mens/worldcup/canadamexicousa2026/scores-fixtures?country=BR&wtw-filter=ALL",
        "fifa_worldcup_team_stats_en": "https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/statistics/team-statistics",
        "fifa_worldcup_player_stats_en": "https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/statistics/player-statistics",
        "fifa_qualifiers_pt": "https://www.fifa.com/pt/tournaments/mens/worldcup/canadamexicousa2026/qualifiers",
    }

    def __init__(self, timeout: int = 30, renderizar_js: bool = False):
        self.timeout = timeout
        self.renderizar_js = renderizar_js
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
        }
    def extrair(self, ano_minimo: int = 2018, incluir_fifa: bool = True) -> InternetExtractionResult:
        log: List[str] = []
        extras: Dict[str, pd.DataFrame] = {}

        log.append("Iniciando extração automática da internet.")
        base_bruta = self._baixar_resultados_internacionais(log)
        extras["internet_resultados_brutos"] = base_bruta.copy()

        matches = self._padronizar_resultados_internacionais(base_bruta, ano_minimo, log)

        if incluir_fifa:
            fifa_extras = self._baixar_tabelas_fifa(log, renderizar_js=self.renderizar_js)
            extras.update(fifa_extras)
            estado_copa = self._montar_estado_copa_2026(fifa_extras, log)
            extras.update(estado_copa)

        log.append(f"Base final padronizada: {len(matches)} partidas.")
        log.append(f"Equipes únicas: {len(set(matches['home_team']).union(set(matches['away_team'])))}.")
        log.append("Observação: quando não há xG público na fonte, home_xg e away_xg usam os gols como aproximação.")

        extras["internet_log"] = pd.DataFrame({"log": log})

        return InternetExtractionResult(matches=matches, extras=extras, log=log)

    def _baixar_resultados_internacionais(self, log: List[str]) -> pd.DataFrame:
        ultimo_erro = None

        for url in self.RESULTS_CSV_URLS:
            try:
                log.append(f"Baixando base histórica: {url}")
                resp = requests.get(url, headers=self.headers, timeout=self.timeout)
                resp.raise_for_status()
                df = pd.read_csv(io.StringIO(resp.text))
                if df.empty:
                    raise ValueError("CSV baixado está vazio.")
                log.append(f"Base histórica recebida com {len(df)} linhas.")
                return df
            except Exception as e:
                ultimo_erro = e
                log.append(f"Falha nessa fonte: {e}")

        raise RuntimeError(f"Não foi possível baixar a base histórica. Último erro: {ultimo_erro}")

    def _padronizar_resultados_internacionais(self, df: pd.DataFrame, ano_minimo: int, log: List[str]) -> pd.DataFrame:
        col_date = encontrar_coluna(df, ["date", "data"])
        col_home = encontrar_coluna(df, ["home_team", "mandante", "time_casa", "home"])
        col_away = encontrar_coluna(df, ["away_team", "visitante", "time_fora", "away"])
        col_hg = encontrar_coluna(df, ["home_score", "home_goals", "gols_casa", "score_home"])
        col_ag = encontrar_coluna(df, ["away_score", "away_goals", "gols_fora", "score_away"])
        col_comp = encontrar_coluna(df, ["tournament", "competition", "competicao", "competição"])

        obrigatorias = [col_date, col_home, col_away, col_hg, col_ag]
        if any(c is None for c in obrigatorias):
            raise ValueError(
                "A fonte online não possui as colunas mínimas esperadas: "
                "date, home_team, away_team, home_score, away_score."
            )

        out = pd.DataFrame()
        out["date"] = pd.to_datetime(df[col_date], errors="coerce")
        out["home_team"] = df[col_home].apply(canonical_team_name)
        out["away_team"] = df[col_away].apply(canonical_team_name)
        out["home_goals"] = pd.to_numeric(df[col_hg], errors="coerce")
        out["away_goals"] = pd.to_numeric(df[col_ag], errors="coerce")

        if col_comp:
            out["competition"] = df[col_comp].astype(str).str.strip()
        else:
            out["competition"] = "International"

        out = out.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
        out = out[out["home_team"].str.lower() != "nan"]
        out = out[out["away_team"].str.lower() != "nan"]
        out = out[out["home_team"] != ""]
        out = out[out["away_team"] != ""]

        out["home_goals"] = out["home_goals"].astype(int)
        out["away_goals"] = out["away_goals"].astype(int)

        out = out.sort_values("date").reset_index(drop=True)

        if ano_minimo:
            filtrado = out[out["date"].dt.year >= int(ano_minimo)].copy()
            if len(filtrado) >= 100:
                out = filtrado
                log.append(f"Filtro aplicado: jogos a partir de {ano_minimo}. Linhas: {len(out)}.")
            else:
                log.append("Filtro por ano não aplicado porque deixaria a base pequena demais.")

        # Não é xG real. É aproximação para manter compatibilidade com o motor xG.
        out["home_xg"] = out["home_goals"].astype(float)
        out["away_xg"] = out["away_goals"].astype(float)
        out["xg_origem"] = "aproximado_por_gols"
        out["fonte"] = "internet_international_results"

        if out.empty:
            raise ValueError("A base online foi baixada, mas nenhuma partida válida restou após limpeza.")

        return out.reset_index(drop=True)

    def extrair_fifa_ao_vivo(self, renderizar_js: Optional[bool] = None) -> InternetExtractionResult:
        """
        Consulta somente as páginas da FIFA no momento da solicitação.
        Não altera nem baixa a base histórica de treino.
        """
        log: List[str] = []
        log.append("Iniciando atualização FIFA sob demanda.")
        log.append("Sem cache interno: cada clique força nova requisição HTTP às páginas FIFA.")

        fifa_extras = self._baixar_tabelas_fifa(log, renderizar_js=renderizar_js)
        estado_copa = self._montar_estado_copa_2026(fifa_extras, log)

        extras: Dict[str, pd.DataFrame] = {}
        extras.update(fifa_extras)
        extras.update(estado_copa)

        atualizado_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        extras["fifa_atualizacao"] = pd.DataFrame([{
            "Atualizado_UTC": atualizado_utc,
            "Paginas_Consultadas": len(self.FIFA_PAGES),
            "Tabelas_Extraidas": len([v for v in fifa_extras.values() if isinstance(v, pd.DataFrame) and not v.empty]),
            "Selecoes_Detectadas": len(estado_copa.get("copa2026_equipes_oficiais", pd.DataFrame())),
            "Renderizacao_JS": bool(self.renderizar_js if renderizar_js is None else renderizar_js),
        }])
        extras["internet_log"] = pd.DataFrame({"log": log})
        log.append(f"Atualização FIFA concluída em {atualizado_utc}.")
        return InternetExtractionResult(matches=pd.DataFrame(), extras=extras, log=log)

    def _url_sem_cache(self, url: str) -> str:
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}_ts={int(time.time())}"

    def _baixar_html_fresh(self, url: str) -> str:
        resp = requests.get(self._url_sem_cache(url), headers=self.headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.text or ""

    def _renderizar_html_selenium(self, url: str, log: List[str]) -> str:
        """
        Fallback para páginas FIFA renderizadas por JavaScript.
        Requer: pip install selenium
        E um Chrome/Chromium disponível no ambiente.
        """
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
        except Exception as e:
            log.append(f"Selenium indisponível: {e}")
            return ""

        driver = None
        try:
            options = Options()
            options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(self.timeout)
            driver.get(self._url_sem_cache(url))
            time.sleep(4)
            return driver.page_source or ""
        except Exception as e:
            log.append(f"Falha no Selenium para {url}: {e}")
            return ""
        finally:
            try:
                if driver is not None:
                    driver.quit()
            except Exception:
                pass

    @staticmethod
    def _limpar_tabela_extraida(tabela: pd.DataFrame) -> pd.DataFrame:
        tabela = tabela.copy()
        tabela = tabela.dropna(how="all")
        tabela = tabela.dropna(axis=1, how="all")
        tabela.columns = [" ".join(str(c).split()) for c in tabela.columns]
        return tabela.reset_index(drop=True)

    def _extrair_tabelas_html(self, html: str, nome: str, log: List[str]) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        if not html.strip():
            return out
        try:
            tabelas = pd.read_html(io.StringIO(html))
        except ValueError:
            tabelas = []
        except Exception as e:
            log.append(f"{nome}: erro ao ler tabelas HTML: {e}")
            tabelas = []

        for i, tabela in enumerate(tabelas, start=1):
            tabela = self._limpar_tabela_extraida(tabela)
            if tabela.empty:
                continue
            out[f"{nome}_html_{i}"[:31]] = tabela
        if out:
            log.append(f"{nome}: {len(out)} tabela(s) HTML extraída(s).")
        return out

    def _parse_json_possivel(self, valor: Any) -> List[Any]:
        candidatos: List[Any] = []
        if not isinstance(valor, str):
            return candidatos
        texto = valor.strip()
        if not texto or len(texto) < 2:
            return candidatos

        if texto[0] in "[{":
            try:
                candidatos.append(json.loads(texto))
            except Exception:
                pass

        for m in re.finditer(r'(\{[^{}]{40,}\}|\[[^\[\]]{40,}\])', texto):
            frag = m.group(1)
            try:
                candidatos.append(json.loads(frag))
            except Exception:
                continue
        return candidatos

    def _coletar_json_embutido(self, html: str, nome: str, log: List[str]) -> List[Any]:
        objetos: List[Any] = []
        if not html.strip():
            return objetos

        soup = BeautifulSoup(html, "html.parser")

        next_data = soup.find("script", id="__NEXT_DATA__")
        if next_data:
            raw = next_data.string or next_data.get_text() or ""
            try:
                objetos.append(json.loads(raw))
                log.append(f"{nome}: __NEXT_DATA__ encontrado.")
            except Exception:
                pass

        for script in soup.find_all("script"):
            raw = script.string or script.get_text() or ""
            typ = (script.get("type") or "").lower()
            if "json" in typ:
                try:
                    objetos.append(json.loads(raw))
                except Exception:
                    pass

            if "__next_f.push" in raw:
                for m in re.finditer(r'self\.__next_f\.push\((.*?)\);', raw, flags=re.S):
                    payload = m.group(1).strip()
                    try:
                        obj = json.loads(payload)
                        objetos.append(obj)
                        if isinstance(obj, list):
                            for item in obj:
                                objetos.extend(self._parse_json_possivel(item))
                    except Exception:
                        pass

        return objetos

    def _achatar_json_em_tabelas(self, obj: Any, nome: str, caminho: str = "root") -> Dict[str, pd.DataFrame]:
        tabelas: Dict[str, pd.DataFrame] = {}

        if isinstance(obj, dict):
            for k, v in obj.items():
                sub = f"{caminho}_{normalizar_texto(k)[:18]}"
                tabelas.update(self._achatar_json_em_tabelas(v, nome, sub))
            return tabelas

        if isinstance(obj, list):
            dicts = [x for x in obj if isinstance(x, dict)]
            if len(dicts) >= 2:
                try:
                    df = pd.json_normalize(dicts, sep="_")
                    df = self._limpar_tabela_extraida(df)
                    cols_norm = " ".join(normalizar_texto(c) for c in df.columns)
                    tem_col_interessante = any(p in cols_norm for p in [
                        "team", "equipe", "country", "match", "fixture", "standing", "group",
                        "score", "points", "pontos", "home", "away", "competitor", "name", "winner",
                    ])
                    if not df.empty and len(df.columns) >= 2 and tem_col_interessante:
                        key = f"{nome}_json_{normalizar_texto(caminho)[-16:]}"[:31]
                        tabelas[key] = df
                except Exception:
                    pass

            for i, item in enumerate(obj[:200]):
                sub = f"{caminho}_{i}"
                tabelas.update(self._achatar_json_em_tabelas(item, nome, sub))
            return tabelas

        for sub_obj in self._parse_json_possivel(obj):
            tabelas.update(self._achatar_json_em_tabelas(sub_obj, nome, f"{caminho}_str"))

        return tabelas

    def _extrair_tabelas_json_embutido(self, html: str, nome: str, log: List[str]) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        objetos = self._coletar_json_embutido(html, nome, log)
        for obj in objetos:
            for key, df in self._achatar_json_em_tabelas(obj, nome).items():
                if key not in out and isinstance(df, pd.DataFrame) and not df.empty:
                    out[key] = df
        if out:
            log.append(f"{nome}: {len(out)} tabela(s) extraída(s) de JSON embutido.")
        return out

    def _baixar_tabelas_fifa(self, log: List[str], renderizar_js: Optional[bool] = None) -> Dict[str, pd.DataFrame]:
        extras: Dict[str, pd.DataFrame] = {}
        renderizar = self.renderizar_js if renderizar_js is None else bool(renderizar_js)

        for nome, url in self.FIFA_PAGES.items():
            try:
                log.append(f"Consultando FIFA agora: {nome}")
                html = self._baixar_html_fresh(url)
                log.append(f"{nome}: HTML recebido com {len(html):,} caracteres.".replace(",", "."))

                coletadas: Dict[str, pd.DataFrame] = {}
                coletadas.update(self._extrair_tabelas_html(html, nome, log))
                coletadas.update(self._extrair_tabelas_json_embutido(html, nome, log))

                if not coletadas and renderizar:
                    log.append(f"{nome}: tentando renderização JavaScript com Selenium.")
                    html_renderizado = self._renderizar_html_selenium(url, log)
                    coletadas.update(self._extrair_tabelas_html(html_renderizado, nome, log))
                    coletadas.update(self._extrair_tabelas_json_embutido(html_renderizado, nome, log))

                if not coletadas:
                    log.append(f"{nome}: nenhuma tabela útil encontrada na FIFA.")
                    continue

                for key, df in coletadas.items():
                    extras[key] = df

            except Exception as e:
                log.append(f"Falha ao consultar {nome}: {e}")

        return extras
    def _montar_estado_copa_2026(self, fifa_extras: Dict[str, pd.DataFrame], log: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Monta o estado vivo da Copa 2026 com base nas tabelas que a FIFA expuser.

        O histórico internacional continua sendo usado apenas para treinar o modelo.
        Para a competição da Copa, esta saída restringe simulações e seletores às
        seleções presentes na tabela oficial/ao vivo da FIFA, quando a tabela estiver
        disponível no HTML.
        """
        standings = self._extrair_standings_das_tabelas(fifa_extras)
        out: Dict[str, pd.DataFrame] = {}

        if standings.empty:
            log.append(
                "Atualização FIFA: não consegui extrair standings/grupos da FIFA por HTML/JSON. "
                "O app continuará com a base histórica para treino; para restringir a Copa, "
                "importe uma configuração de grupos."
            )
            return out

        standings = standings.drop_duplicates(subset=["Grupo", "Equipe"]).reset_index(drop=True)
        standings["Equipe"] = standings["Equipe"].apply(canonical_team_name)
        standings = standings[standings["Equipe"].ne("")]

        equipes = standings[["Equipe"]].drop_duplicates().sort_values("Equipe").reset_index(drop=True)
        equipes["Fonte"] = "FIFA World Cup 2026 - última atualização solicitada"

        classificados = self._calcular_classificados_atuais(standings)

        out["copa2026_classificacao_atual"] = standings
        out["copa2026_equipes_oficiais"] = equipes
        if not classificados.empty:
            out["copa2026_classificados_atuais"] = classificados

        log.append(f"Atualização FIFA: {len(equipes)} seleções detectadas nas tabelas FIFA.")
        if not classificados.empty:
            log.append(f"Atualização FIFA: {len(classificados)} classificados atuais calculados pela tabela.")

        return out

    def _extrair_standings_das_tabelas(self, fifa_extras: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        grupo_seq = 0

        for nome, df in fifa_extras.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            if "standing" not in nome.lower() and "standings" not in nome.lower():
                # Algumas páginas de fixtures também podem conter tabelas de grupos, mas
                # priorizamos a página de standings para evitar captar tabelas erradas.
                continue

            d = df.copy()
            d.columns = [" ".join(str(c).split()) for c in d.columns]

            col_team = encontrar_coluna(d, [
                "Equipe", "Seleção", "Selecao", "Team", "Country", "País", "Pais", "Nation"
            ])
            if col_team is None:
                # fallback: primeira coluna textual com nomes de times
                text_cols = [c for c in d.columns if d[c].dtype == "object"]
                col_team = text_cols[0] if text_cols else None
            if col_team is None:
                continue

            col_group = encontrar_coluna(d, ["Grupo", "Group", "Grp"])
            col_pos = encontrar_coluna(d, ["Posição", "Posicao", "Position", "Rank", "Pos"])
            col_pts = encontrar_coluna(d, ["Pontos", "Pts", "Points", "Pt"])
            col_sg = encontrar_coluna(d, ["SG", "Saldo", "Goal Difference", "GD", "+/-"])
            col_gp = encontrar_coluna(d, ["GP", "Gols Pró", "Gols Pro", "Goals For", "GF", "Gols"])
            col_j = encontrar_coluna(d, ["J", "Jogos", "Played", "Pld", "MP"])

            if col_group is None:
                grupo_seq += 1
                grupo_default = chr(ord("A") + ((grupo_seq - 1) % 12))
            else:
                grupo_default = None

            for idx, row in d.iterrows():
                equipe = canonical_team_name(row.get(col_team, ""))
                if not equipe or equipe.lower() in {"team", "equipe", "seleção", "selecao", "nan"}:
                    continue
                # Evita capturar linhas de cabeçalho ou textos agregados.
                if len(equipe) < 3 or equipe.replace(" ", "").isdigit():
                    continue

                grupo = str(row.get(col_group, grupo_default)).strip().upper() if col_group else grupo_default
                if not grupo or grupo.lower() == "nan":
                    grupo = grupo_default or "FIFA"
                grupo = grupo.replace("GROUP", "").replace("GRUPO", "").strip() or grupo

                rows.append({
                    "Grupo": grupo,
                    "Posicao_Grupo": int(pd.to_numeric(row.get(col_pos, idx + 1), errors="coerce") if pd.notna(pd.to_numeric(row.get(col_pos, idx + 1), errors="coerce")) else idx + 1),
                    "Equipe": equipe,
                    "Pontos": float(pd.to_numeric(row.get(col_pts, 0), errors="coerce") if pd.notna(pd.to_numeric(row.get(col_pts, 0), errors="coerce")) else 0),
                    "Jogos": float(pd.to_numeric(row.get(col_j, 0), errors="coerce") if pd.notna(pd.to_numeric(row.get(col_j, 0), errors="coerce")) else 0),
                    "SG": float(pd.to_numeric(row.get(col_sg, 0), errors="coerce") if pd.notna(pd.to_numeric(row.get(col_sg, 0), errors="coerce")) else 0),
                    "GP": float(pd.to_numeric(row.get(col_gp, 0), errors="coerce") if pd.notna(pd.to_numeric(row.get(col_gp, 0), errors="coerce")) else 0),
                    "Fonte_Tabela": nome,
                })

        if not rows:
            return pd.DataFrame()

        out = pd.DataFrame(rows)
        out = out.sort_values(["Grupo", "Posicao_Grupo", "Pontos", "SG", "GP"], ascending=[True, True, False, False, False])
        return out.reset_index(drop=True)

    def _calcular_classificados_atuais(self, standings: pd.DataFrame) -> pd.DataFrame:
        if standings.empty or "Grupo" not in standings.columns:
            return pd.DataFrame()

        parts = []
        for grupo, gdf in standings.groupby("Grupo"):
            g = gdf.copy().sort_values(
                ["Pontos", "SG", "GP", "Posicao_Grupo"],
                ascending=[False, False, False, True],
            ).reset_index(drop=True)
            g["Posicao_Calculada"] = range(1, len(g) + 1)
            parts.append(g)
        ranked = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if ranked.empty:
            return pd.DataFrame()

        first_second = ranked[ranked["Posicao_Calculada"].isin([1, 2])].copy()
        first_second["Tipo_Classificacao"] = "1º/2º do grupo"

        thirds = ranked[ranked["Posicao_Calculada"] == 3].copy()
        thirds = thirds.sort_values(["Pontos", "SG", "GP"], ascending=[False, False, False]).head(8)
        thirds["Tipo_Classificacao"] = "melhor terceiro atual"

        out = pd.concat([first_second, thirds], ignore_index=True)
        out = out.sort_values(["Tipo_Classificacao", "Grupo", "Posicao_Calculada"]).reset_index(drop=True)
        return out


# ============================================================
# ODDS DE CASAS DE APOSTA — CONSENSO DE MERCADO
# ============================================================

@dataclass
class OddsExtractionResult:
    raw: pd.DataFrame
    consensus: pd.DataFrame
    log: List[str]


class BettingOddsExtractor:
    """
    Integra odds 1X2 de futebol por API, sem scraping direto de sites de aposta.

    Suporta dois provedores:
    1) odds-api.io (recomendado aqui, porque sua chave é desse serviço)
       - Base URL: https://api.odds-api.io/v3
       - Fluxo: /events -> /odds ou /odds/multi
       - esporte padrão: football
       - mercado usado: ML / Match Result / 1X2, quando disponível

    2) the-odds-api.com
       - Base URL: https://api.the-odds-api.com/v4
       - Fluxo: /sports/{sport_key}/odds
       - sport_key padrão: soccer_fifa_world_cup, etc.

    A saída é padronizada:
    - odds brutas por casa
    - consenso médio sem vig por partida
    - favorito das casas
    """

    ODDS_API_IO_BASE_URL = "https://api.odds-api.io/v3"
    THE_ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

    DEFAULT_ODDS_API_IO_SPORTS = ["football:international-fifa-world-cup"]

    DEFAULT_THE_ODDS_API_SPORT_KEYS = [
        "soccer_fifa_world_cup",
        "soccer_international_friendlies",
        "soccer_conmebol_copa_america",
        "soccer_uefa_european_championship",
    ]

    def __init__(
        self,
        api_key: str,
        sport_keys: Optional[List[str]] = None,
        regions: str = "eu,uk,us,us2,au",
        markets: str = "h2h",
        odds_format: str = "decimal",
        bookmakers: Optional[str] = None,
        timeout: int = 30,
        provider: str = "odds-api-io",
        limit: int = 100,
        target_date: Optional[Any] = None,
        target_timezone: str = "America/Sao_Paulo",
        team_filter: Optional[List[str]] = None,
    ):
        self.api_key = str(api_key or "").strip()
        self.provider = self._normalizar_provider(provider)
        if sport_keys is None:
            sport_keys = self.DEFAULT_ODDS_API_IO_SPORTS if self.provider == "odds-api-io" else self.DEFAULT_THE_ODDS_API_SPORT_KEYS
        self.sport_keys = [str(x).strip() for x in (sport_keys or []) if str(x).strip()]
        if not self.sport_keys:
            self.sport_keys = self.DEFAULT_ODDS_API_IO_SPORTS if self.provider == "odds-api-io" else self.DEFAULT_THE_ODDS_API_SPORT_KEYS
        self.regions = str(regions or "eu,uk,us,us2,au").strip()
        self.markets = str(markets or "h2h").strip()
        self.odds_format = str(odds_format or "decimal").strip()
        self.bookmakers = str(bookmakers or "").strip() or None
        self.timeout = int(timeout or 30)
        self.limit = int(limit or 100)
        self.target_date = self._normalizar_target_date(target_date)
        self.target_timezone = str(target_timezone or "America/Sao_Paulo").strip() or "America/Sao_Paulo"
        self.team_filter = {
            normalizar_texto(canonical_team_name(x))
            for x in (team_filter or [])
            if str(x or "").strip()
        }
        self.team_filter.discard("")
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
            "Cache-Control": "no-cache",
        }

    @staticmethod
    def _normalizar_provider(provider: Any) -> str:
        p = normalizar_texto(provider or "odds-api-io")
        if p in {"the_odds_api", "theoddsapi", "the_odds_api_com", "the_odds"}:
            return "the-odds-api"
        return "odds-api-io"

    @staticmethod
    def _split_csv(valor: Any) -> List[str]:
        return [x.strip() for x in str(valor or "").split(",") if x.strip()]

    @staticmethod
    def _normalizar_target_date(valor: Any) -> Optional[Any]:
        """Converte uma data escolhida no Streamlit para date. Retorna None se vazio."""
        if valor in (None, "", "Todas", "todos", "all"):
            return None
        try:
            ts = pd.to_datetime(valor, errors="coerce")
            if pd.isna(ts):
                return None
            return ts.date()
        except Exception:
            return None

    @staticmethod
    def _datetime_utc(valor: Any) -> Optional[pd.Timestamp]:
        """Parser tolerante para datas vindas das APIs de odds."""
        if valor in (None, ""):
            return None
        try:
            if isinstance(valor, (int, float)) and not pd.isna(valor):
                # Alguns provedores retornam epoch em segundos ou milissegundos.
                unit = "ms" if float(valor) > 10_000_000_000 else "s"
                ts = pd.to_datetime(valor, unit=unit, utc=True, errors="coerce")
            else:
                ts = pd.to_datetime(valor, utc=True, errors="coerce")
            if pd.isna(ts):
                return None
            return ts
        except Exception:
            return None

    def _data_local_evento(self, valor: Any) -> str:
        ts = self._datetime_utc(valor)
        if ts is None:
            return ""
        try:
            return ts.tz_convert(self.target_timezone).date().isoformat()
        except Exception:
            return ts.date().isoformat()

    def _datetime_local_evento(self, valor: Any) -> str:
        ts = self._datetime_utc(valor)
        if ts is None:
            return ""
        try:
            return ts.tz_convert(self.target_timezone).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return ts.strftime("%Y-%m-%d %H:%M")

    def _evento_na_data_alvo(self, valor_data: Any) -> bool:
        if self.target_date is None:
            return True
        data_local = self._data_local_evento(valor_data)
        if not data_local:
            return False
        return data_local == self.target_date.isoformat()

    def _evento_no_filtro_de_equipes(self, home: Any, away: Any) -> bool:
        if not self.team_filter:
            return True
        h = normalizar_texto(canonical_team_name(home))
        a = normalizar_texto(canonical_team_name(away))
        # Para evitar odds de clubes, exige que os dois lados sejam seleções reconhecidas.
        return h in self.team_filter and a in self.team_filter

    @staticmethod
    def _filtrar_dataframe_por_data(
        df: pd.DataFrame,
        target_date: Optional[Any],
        target_timezone: str = "America/Sao_Paulo",
    ) -> pd.DataFrame:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return df
        target = BettingOddsExtractor._normalizar_target_date(target_date)
        if target is None:
            return df
        d = df.copy()
        target_str = target.isoformat()
        if "Data_Local_Data" in d.columns:
            return d[d["Data_Local_Data"].astype(str) == target_str].copy()
        if "Data_UTC" not in d.columns:
            return d.iloc[0:0].copy()
        datas = pd.to_datetime(d["Data_UTC"], utc=True, errors="coerce")
        try:
            datas_local = datas.dt.tz_convert(target_timezone).dt.date.astype(str)
        except Exception:
            datas_local = datas.dt.date.astype(str)
        return d[datas_local == target_str].copy()

    @staticmethod
    def _match_key(time_a: Any, time_b: Any) -> str:
        a = normalizar_texto(canonical_team_name(time_a))
        b = normalizar_texto(canonical_team_name(time_b))
        return f"{a}__vs__{b}"

    @staticmethod
    def _is_draw_name(nome: Any) -> bool:
        n = normalizar_texto(nome)
        return n in {"draw", "empate", "tie", "x", "the_draw"}

    @staticmethod
    def _to_float_odd(valor: Any) -> float:
        if valor is None:
            return np.nan
        try:
            txt = str(valor).strip().replace(",", ".")
            txt = re.sub(r"[^0-9.\-]", "", txt)
            if not txt:
                return np.nan
            out = float(txt)
            if out <= 1.0:
                return np.nan
            return out
        except Exception:
            return np.nan

    @staticmethod
    def _chunked(seq: List[Any], n: int) -> List[List[Any]]:
        return [seq[i:i+n] for i in range(0, len(seq), n)]

    def _get_json(self, url: str, params: Dict[str, Any], log: List[str]) -> Any:
        clean_params = {k: v for k, v in params.items() if v not in (None, "", [], {})}
        try:
            resp = requests.get(url, params=clean_params, headers=self.headers, timeout=self.timeout)
        except requests.RequestException as exc:
            # Não propaga URL/consulta completas: parâmetros podem conter a API key.
            raise RuntimeError("Falha de comunicação com o provedor de odds.") from exc
        remaining = resp.headers.get("x-requests-remaining") or resp.headers.get("x-ratelimit-remaining")
        used = resp.headers.get("x-requests-used") or resp.headers.get("x-ratelimit-used")
        if remaining is not None or used is not None:
            log.append(f"Cota odds: usadas={used or 'n/d'} restantes={remaining or 'n/d'}.")
        if resp.status_code >= 400:
            detalhe = (resp.text or "")[:600]
            if self.api_key:
                detalhe = detalhe.replace(self.api_key, "[REDACTED]")
            raise RuntimeError(f"HTTP {resp.status_code}: {detalhe}")
        return resp.json()

    def listar_esportes_disponiveis(self) -> pd.DataFrame:
        """Lista esportes disponíveis no provedor configurado."""
        if self.provider == "odds-api-io":
            data = self._get_json(f"{self.ODDS_API_IO_BASE_URL}/sports", {}, [])
        else:
            if not self.api_key:
                raise ValueError("Informe a API key das odds.")
            data = self._get_json(f"{self.THE_ODDS_API_BASE_URL}/sports", {"apiKey": self.api_key}, [])
        if isinstance(data, dict):
            for key in ["data", "sports", "items", "results"]:
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        return pd.DataFrame(data if isinstance(data, list) else [])

    def extrair_ao_vivo(self) -> OddsExtractionResult:
        if not self.api_key:
            raise ValueError("Informe a API key das odds.")
        if self.provider == "the-odds-api":
            return self._extrair_the_odds_api()
        return self._extrair_odds_api_io()

    # ------------------------------------------------------------
    # Provedor 1: odds-api.io
    # ------------------------------------------------------------

    def _parse_sport_league(self, item: str) -> Tuple[str, Optional[str]]:
        """
        Permite formatos:
        - football
        - football:international-fifa-world-cup
        - football|international-fifa-world-cup

        Para Copa do Mundo, use preferencialmente:
        football:international-fifa-world-cup
        """
        s = str(item or "").strip()
        for sep in [":", "|", "/"]:
            if sep in s:
                a, b = s.split(sep, 1)
                return a.strip() or "football", b.strip() or None
        return s or "football", None

    def _janela_data_alvo_utc(self) -> Dict[str, str]:
        """
        Retorna janela UTC RFC3339 correspondente ao dia escolhido no fuso local.
        Ex.: 2026-06-13 em America/Sao_Paulo vira 2026-06-13T03:00:00Z até 2026-06-14T02:59:59Z.
        """
        if self.target_date is None:
            return {}
        try:
            start_local = pd.Timestamp(self.target_date).tz_localize(self.target_timezone)
        except TypeError:
            start_local = pd.Timestamp(self.target_date).tz_convert(self.target_timezone)
        except Exception:
            start_local = pd.Timestamp(str(self.target_date)).tz_localize(self.target_timezone)
        end_local = start_local + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        return {
            "from": start_local.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": end_local.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    def _bookmakers_para_requisicao(self, log: Optional[List[str]] = None) -> str:
        """
        odds-api.io exige bookmakers em /odds e /odds/multi.
        Se o usuário deixou vazio, tenta os bookmakers selecionados na conta; se não vier nada, usa fallback comum.
        """
        if self.bookmakers:
            return self.bookmakers
        try:
            selected = self._get_json(f"{self.ODDS_API_IO_BASE_URL}/bookmakers/selected", {"apiKey": self.api_key}, log or [])
            nomes: List[str] = []
            if isinstance(selected, dict):
                for key in ["bookmakers", "selected", "data", "items", "results"]:
                    val = selected.get(key)
                    if isinstance(val, list):
                        for item in val:
                            if isinstance(item, dict):
                                nome = item.get("name") or item.get("title") or item.get("key") or item.get("bookmaker")
                            else:
                                nome = item
                            if nome:
                                nomes.append(str(nome))
                if not nomes:
                    for k, v in selected.items():
                        if isinstance(v, bool) and v:
                            nomes.append(str(k))
                        elif isinstance(v, str) and v.strip():
                            nomes.append(v.strip())
            elif isinstance(selected, list):
                for item in selected:
                    if isinstance(item, dict):
                        nome = item.get("name") or item.get("title") or item.get("key") or item.get("bookmaker")
                    else:
                        nome = item
                    if nome:
                        nomes.append(str(nome))
            nomes = sorted(set([n.strip() for n in nomes if n and str(n).strip()]))
            if nomes:
                out = ",".join(nomes[:30])
                if log is not None:
                    log.append(f"Bookmakers obtidos da conta odds-api.io: {out}.")
                return out
        except Exception as e:
            if log is not None:
                log.append(f"Não consegui ler bookmakers selecionados na odds-api.io: {e}")
        fallback = "Bet365,Unibet"
        if log is not None:
            log.append(f"Bookmakers não informados; usando fallback: {fallback}.")
        return fallback

    def _eventos_search_por_times_odds_api_io(self, log: List[str]) -> List[Dict[str, Any]]:
        """
        Fallback: quando /events por liga não retorna nada, procura eventos por nome das seleções.
        Isso ajuda quando a API cadastrou o jogo em outra liga/slug.
        """
        if not self.team_filter or self.target_date is None:
            return []
        encontrados: List[Dict[str, Any]] = []
        vistos = set()
        # Usa nomes canônicos aproximados vindos do filtro.
        termos = sorted(self.team_filter)[:80]
        log.append("Fallback ativo: buscando eventos por nome das seleções em /events/search.")
        for termo in termos:
            try:
                params = {"apiKey": self.api_key, "query": termo.replace("_", " ")}
                data = self._get_json(f"{self.ODDS_API_IO_BASE_URL}/events/search", params, log)
                if isinstance(data, dict):
                    for key in ["data", "events", "items", "results"]:
                        if isinstance(data.get(key), list):
                            data = data[key]
                            break
                if not isinstance(data, list):
                    continue
                for ev in data:
                    if not isinstance(ev, dict):
                        continue
                    event_id = self._extract_event_id(ev)
                    if not event_id or event_id in vistos:
                        continue
                    home, away, date = self._parse_event_home_away(ev)
                    if not home or not away:
                        continue
                    if not self._evento_na_data_alvo(date):
                        continue
                    if not self._evento_no_filtro_de_equipes(home, away):
                        continue
                    ev2 = dict(ev)
                    ev2["_id"] = event_id
                    ev2["_home"] = home
                    ev2["_away"] = away
                    ev2["_date"] = date
                    ev2["_sport_item"] = "events/search"
                    encontrados.append(ev2)
                    vistos.add(event_id)
            except Exception as e:
                log.append(f"Falha no /events/search para {termo}: {e}")
        log.append(f"Fallback /events/search encontrou {len(encontrados)} evento(s) aproveitável(is).")
        return encontrados

    def _extract_event_id(self, event: Dict[str, Any]) -> str:
        for key in ["id", "eventId", "event_id", "fixtureId", "fixture_id", "gameId", "game_id"]:
            v = event.get(key)
            if v not in (None, ""):
                return str(v)
        return ""

    def _parse_event_home_away(self, event: Dict[str, Any]) -> Tuple[str, str, str]:
        home = event.get("home") or event.get("homeTeam") or event.get("home_team") or event.get("homeName")
        away = event.get("away") or event.get("awayTeam") or event.get("away_team") or event.get("awayName")
        date = event.get("date") or event.get("startTime") or event.get("commence_time") or event.get("time") or event.get("startsAt") or ""

        if not home or not away:
            participants = event.get("participants") or event.get("competitors") or event.get("teams") or []
            if isinstance(participants, list):
                for p in participants:
                    if not isinstance(p, dict):
                        continue
                    name = p.get("name") or p.get("team") or p.get("title") or p.get("participantName")
                    role = normalizar_texto(p.get("role") or p.get("type") or p.get("side") or p.get("qualifier") or "")
                    is_home = bool(p.get("home") or p.get("isHome") or role in {"home", "mandante", "casa"})
                    is_away = bool(p.get("away") or p.get("isAway") or role in {"away", "visitante", "fora"})
                    if is_home and name:
                        home = name
                    elif is_away and name:
                        away = name
                if (not home or not away) and len(participants) >= 2:
                    p0, p1 = participants[0], participants[1]
                    if isinstance(p0, dict) and isinstance(p1, dict):
                        home = home or p0.get("name") or p0.get("team") or p0.get("title")
                        away = away or p1.get("name") or p1.get("team") or p1.get("title")
                    else:
                        home = home or str(p0)
                        away = away or str(p1)

        return canonical_team_name(home), canonical_team_name(away), str(date or "")

    def _extrair_eventos_odds_api_io(self, log: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        eventos: List[Dict[str, Any]] = []
        event_map: Dict[str, Dict[str, Any]] = {}
        vistos = set()

        pulados_data = 0
        pulados_equipes = 0
        amostras_puladas: List[str] = []

        janela = self._janela_data_alvo_utc()
        if janela:
            log.append(f"Janela de busca enviada à API: from={janela['from']}; to={janela['to']}.")

        bookmaker_param = None
        if self.bookmakers:
            # /events usa singular bookmaker para filtrar disponibilidade; usa o primeiro como filtro leve.
            bookmaker_param = self._split_csv(self.bookmakers)[0] if self._split_csv(self.bookmakers) else None

        for item in self.sport_keys:
            sport, league = self._parse_sport_league(item)
            params: Dict[str, Any] = {
                "apiKey": self.api_key,
                "sport": sport,
                "status": "pending,live",
                "limit": self.limit,
                "skip": 0,
            }
            params.update(janela)
            if league:
                params["league"] = league
            if bookmaker_param:
                params["bookmaker"] = bookmaker_param

            try:
                log.append(
                    "Consultando odds-api.io /events: "
                    f"sport={sport}; league={league or 'todas'}; "
                    f"status={params.get('status')}; limit={self.limit}; "
                    f"bookmaker={bookmaker_param or 'sem filtro'}"
                )
                data = self._get_json(f"{self.ODDS_API_IO_BASE_URL}/events", params, log)
            except Exception as e:
                log.append(f"Falha em /events para {item}: {e}")
                continue

            if isinstance(data, dict):
                for key in ["data", "events", "items", "results"]:
                    if isinstance(data.get(key), list):
                        data = data[key]
                        break
            if not isinstance(data, list) or not data:
                log.append(f"{item}: nenhum evento retornado.")
                continue

            log.append(f"{item}: {len(data)} evento(s) retornado(s).")
            for ev in data:
                if not isinstance(ev, dict):
                    continue
                event_id = self._extract_event_id(ev)
                if not event_id or event_id in vistos:
                    continue
                home, away, date = self._parse_event_home_away(ev)
                if not home or not away:
                    continue
                if not self._evento_na_data_alvo(date):
                    pulados_data += 1
                    if len(amostras_puladas) < 5:
                        amostras_puladas.append(f"data: {home} x {away} em {date}")
                    continue
                if not self._evento_no_filtro_de_equipes(home, away):
                    pulados_equipes += 1
                    if len(amostras_puladas) < 5:
                        amostras_puladas.append(f"equipe: {home} x {away} em {date}")
                    continue
                ev2 = dict(ev)
                ev2["_id"] = event_id
                ev2["_home"] = home
                ev2["_away"] = away
                ev2["_date"] = date
                ev2["_sport_item"] = item
                eventos.append(ev2)
                event_map[event_id] = ev2
                vistos.add(event_id)

        if not eventos and self.team_filter and self.target_date is not None:
            for ev in self._eventos_search_por_times_odds_api_io(log):
                event_id = ev.get("_id") or self._extract_event_id(ev)
                if event_id and event_id not in vistos:
                    eventos.append(ev)
                    event_map[event_id] = ev
                    vistos.add(event_id)

        if self.target_date is not None:
            log.append(f"Filtro de data aplicado: {self.target_date.isoformat()} em {self.target_timezone}; eventos ignorados por data={pulados_data}.")
        if self.team_filter:
            log.append(f"Filtro de seleções aplicado: {len(self.team_filter)} nome(s) conhecidos; eventos ignorados por equipe={pulados_equipes}.")
        if amostras_puladas:
            log.append("Amostras de eventos ignorados: " + " | ".join(amostras_puladas))
        if not eventos:
            log.append(
                "Diagnóstico: nenhum evento passou pelos filtros. "
                "Use sport key 'football:international-fifa-world-cup', confira a data local do jogo, "
                "ou desmarque temporariamente o filtro de seleções para testar a resposta bruta da API."
            )

        return eventos, event_map

    def _extract_ml_from_odds_api_io_bookmakers(
        self,
        odds_event: Dict[str, Any],
        event_map: Dict[str, Dict[str, Any]],
        started: str,
    ) -> List[Dict[str, Any]]:
        event_id = self._extract_event_id(odds_event) or str(odds_event.get("id", ""))
        mapped = event_map.get(event_id, {})
        home, away, date = self._parse_event_home_away(odds_event)
        home = home or mapped.get("_home", "")
        away = away or mapped.get("_away", "")
        date = date or mapped.get("_date", "")
        if not home or not away:
            return []

        rows: List[Dict[str, Any]] = []
        bookmakers = odds_event.get("bookmakers") or odds_event.get("books") or odds_event.get("sportsbooks") or {}

        book_items: List[Tuple[str, Any]] = []
        if isinstance(bookmakers, dict):
            book_items = list(bookmakers.items())
        elif isinstance(bookmakers, list):
            for b in bookmakers:
                if isinstance(b, dict):
                    name = b.get("name") or b.get("title") or b.get("key") or b.get("bookmaker") or "Bookmaker"
                    book_items.append((str(name), b.get("markets") or b.get("odds") or b))

        for bookmaker_name, markets_obj in book_items:
            markets = markets_obj
            if isinstance(markets_obj, dict):
                markets = markets_obj.get("markets") or markets_obj.get("odds") or markets_obj.get("data") or markets_obj
            if isinstance(markets, dict):
                markets = list(markets.values())
            if not isinstance(markets, list):
                continue

            odds_home = np.nan
            odds_draw = np.nan
            odds_away = np.nan
            last_update = ""

            for market in markets:
                if not isinstance(market, dict):
                    continue
                market_name = normalizar_texto(market.get("name") or market.get("key") or market.get("market") or "")
                if market_name not in {"ml", "moneyline", "match_result", "match_winner", "1x2", "h2h", "full_time_result", "winner"}:
                    # odds-api.io documenta ML para resultado da partida.
                    continue
                last_update = market.get("updatedAt") or market.get("updated_at") or market.get("lastUpdate") or last_update
                odds_list = market.get("odds") or market.get("outcomes") or []
                if isinstance(odds_list, dict):
                    odds_list = [odds_list]
                for odd_obj in odds_list:
                    if not isinstance(odd_obj, dict):
                        continue
                    # Formato documentado: {'home': '2.10', 'draw': '3.40', 'away': '3.20'}
                    h = self._to_float_odd(odd_obj.get("home"))
                    d = self._to_float_odd(odd_obj.get("draw"))
                    a = self._to_float_odd(odd_obj.get("away"))
                    if not pd.isna(h):
                        odds_home = h
                    if not pd.isna(d):
                        odds_draw = d
                    if not pd.isna(a):
                        odds_away = a

                    # Formato alternativo: outcome por nome.
                    name = odd_obj.get("name") or odd_obj.get("outcome") or odd_obj.get("label")
                    price = self._to_float_odd(odd_obj.get("price") or odd_obj.get("odds") or odd_obj.get("value"))
                    if not pd.isna(price) and name:
                        nome_canon = canonical_team_name(name)
                        if nome_canon == home:
                            odds_home = price
                        elif nome_canon == away:
                            odds_away = price
                        elif self._is_draw_name(name):
                            odds_draw = price

            if pd.isna(odds_home) or pd.isna(odds_away):
                continue
            rows.append(self._row_from_odds(
                provider="odds-api.io",
                sport_key=str(mapped.get("_sport_item", "football")),
                event_id=event_id,
                date=date,
                home=home,
                away=away,
                home_raw=home,
                away_raw=away,
                bookmaker_key=bookmaker_name,
                bookmaker_title=bookmaker_name,
                last_update=str(last_update or ""),
                odds_home=odds_home,
                odds_draw=odds_draw,
                odds_away=odds_away,
                started=started,
            ))
        return rows

    def _extrair_odds_api_io(self) -> OddsExtractionResult:
        log: List[str] = []
        rows: List[Dict[str, Any]] = []
        started = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        log.append(f"Iniciando consulta de odds em {started}.")
        log.append("Fonte configurada: odds-api.io v3; fluxo /events -> /odds/multi; mercado ML/1X2; odds decimais.")
        log.append(f"Data-alvo: {self.target_date.isoformat() if self.target_date else 'todas'}; fuso de comparação: {self.target_timezone}.")
        if self.team_filter:
            log.append(f"Filtro ativo: somente jogos em que as duas equipes existem na base/modelo ({len(self.team_filter)} seleção(ões)).")

        eventos, event_map = self._extrair_eventos_odds_api_io(log)
        bookmakers_to_use = self._bookmakers_para_requisicao(log)
        if not eventos:
            log.append("Nenhum evento aproveitável foi encontrado em odds-api.io.")
            return OddsExtractionResult(raw=pd.DataFrame(), consensus=pd.DataFrame(), log=log)

        ids = [ev["_id"] for ev in eventos if ev.get("_id")]
        for chunk in self._chunked(ids, 10):
            try:
                params: Dict[str, Any] = {
                    "apiKey": self.api_key,
                    "eventIds": ",".join(chunk),
                }
                params["bookmakers"] = bookmakers_to_use
                log.append(f"Consultando odds-api.io /odds/multi para {len(chunk)} evento(s); bookmakers={bookmakers_to_use}.")
                data = self._get_json(f"{self.ODDS_API_IO_BASE_URL}/odds/multi", params, log)
            except Exception as e_multi:
                log.append(f"Falha em /odds/multi: {e_multi}. Tentando /odds evento a evento.")
                data = []
                for event_id in chunk:
                    try:
                        params = {"apiKey": self.api_key, "eventId": event_id, "bookmakers": bookmakers_to_use}
                        one = self._get_json(f"{self.ODDS_API_IO_BASE_URL}/odds", params, log)
                        data.append(one)
                    except Exception as e_one:
                        log.append(f"Falha em /odds para eventId={event_id}: {e_one}")

            if isinstance(data, dict):
                for key in ["data", "odds", "events", "items", "results"]:
                    if isinstance(data.get(key), list):
                        data = data[key]
                        break
                if isinstance(data, dict):
                    data = [data]
            if not isinstance(data, list):
                continue
            for odds_event in data:
                if not isinstance(odds_event, dict):
                    continue
                rows.extend(self._extract_ml_from_odds_api_io_bookmakers(odds_event, event_map, started))

        raw = pd.DataFrame(rows)
        if raw.empty:
            log.append("Eventos foram encontrados, mas nenhuma odd ML/1X2 aproveitável retornou. Verifique se os bookmakers informados têm mercado ML para a partida. Teste também limpar o campo Bookmakers ou usar Bet365,Unibet.")
            return OddsExtractionResult(raw=raw, consensus=pd.DataFrame(), log=log)
        consensus = self.consolidar_consenso(raw)
        log.append(f"Odds brutas: {len(raw)} linha(s). Consenso: {len(consensus)} partida(s).")
        return OddsExtractionResult(raw=raw, consensus=consensus, log=log)

    # ------------------------------------------------------------
    # Provedor 2: the-odds-api.com
    # ------------------------------------------------------------

    def _extrair_the_odds_api(self) -> OddsExtractionResult:
        if not self.sport_keys:
            raise ValueError("Informe ao menos um sport_key para consulta de odds.")

        log: List[str] = []
        rows: List[Dict[str, Any]] = []
        started = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        log.append(f"Iniciando consulta de odds em {started}.")
        log.append("Fonte configurada: The Odds API v4; mercado h2h/1X2; odds decimais.")
        log.append(f"Data-alvo: {self.target_date.isoformat() if self.target_date else 'todas'}; fuso de comparação: {self.target_timezone}.")
        if self.team_filter:
            log.append(f"Filtro ativo: somente jogos em que as duas equipes existem na base/modelo ({len(self.team_filter)} seleção(ões)).")

        for sport_key in self.sport_keys:
            url = f"{self.THE_ODDS_API_BASE_URL}/sports/{sport_key}/odds/"
            params: Dict[str, Any] = {
                "apiKey": self.api_key,
                "regions": self.regions,
                "markets": self.markets,
                "oddsFormat": self.odds_format,
                "dateFormat": "iso",
            }
            if self.bookmakers:
                params["bookmakers"] = self.bookmakers

            try:
                log.append(f"Consultando The Odds API: sport_key={sport_key}; regions={self.regions}; bookmakers={self.bookmakers or 'todos da região'}.")
                data = self._get_json(url, params, log)
            except Exception as e:
                log.append(f"Falha ao consultar {sport_key}: {e}")
                continue

            if not isinstance(data, list) or not data:
                log.append(f"{sport_key}: nenhuma partida com odds retornada.")
                continue

            log.append(f"{sport_key}: {len(data)} partida(s) retornada(s).")
            for event in data:
                rows.extend(self._extract_rows_from_the_odds_api_event(event, sport_key, started))

        raw = pd.DataFrame(rows)
        if raw.empty:
            log.append("Nenhuma odd 1X2 aproveitável foi encontrada.")
            return OddsExtractionResult(raw=raw, consensus=pd.DataFrame(), log=log)

        consensus = self.consolidar_consenso(raw)
        log.append(f"Odds brutas: {len(raw)} linha(s). Consenso: {len(consensus)} partida(s).")
        return OddsExtractionResult(raw=raw, consensus=consensus, log=log)

    def _extract_rows_from_the_odds_api_event(self, event: Dict[str, Any], sport_key: str, started: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not isinstance(event, dict):
            return rows
        home_raw = event.get("home_team", "")
        away_raw = event.get("away_team", "")
        home = canonical_team_name(home_raw)
        away = canonical_team_name(away_raw)
        if not home or not away:
            return rows

        commence_time = event.get("commence_time", "")
        if not self._evento_na_data_alvo(commence_time):
            return rows
        if not self._evento_no_filtro_de_equipes(home, away):
            return rows

        bookmakers_data = event.get("bookmakers") or []
        for book in bookmakers_data:
            if not isinstance(book, dict):
                continue
            book_key = book.get("key", "")
            book_title = book.get("title", book_key)
            last_update = book.get("last_update", "")

            odds_home = np.nan
            odds_draw = np.nan
            odds_away = np.nan

            for market in book.get("markets") or []:
                if not isinstance(market, dict) or market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes") or []:
                    if not isinstance(outcome, dict):
                        continue
                    nome_out = outcome.get("name", "")
                    preco = self._to_float_odd(outcome.get("price"))
                    if pd.isna(preco):
                        continue
                    nome_canon = canonical_team_name(nome_out)
                    if nome_canon == home:
                        odds_home = float(preco)
                    elif nome_canon == away:
                        odds_away = float(preco)
                    elif self._is_draw_name(nome_out):
                        odds_draw = float(preco)

            if pd.isna(odds_home) or pd.isna(odds_away):
                continue
            rows.append(self._row_from_odds(
                provider="the-odds-api.com",
                sport_key=sport_key,
                event_id=event.get("id", ""),
                date=commence_time,
                home=home,
                away=away,
                home_raw=home_raw,
                away_raw=away_raw,
                bookmaker_key=book_key,
                bookmaker_title=book_title,
                last_update=last_update,
                odds_home=odds_home,
                odds_draw=odds_draw,
                odds_away=odds_away,
                started=started,
            ))
        return rows

    def _row_from_odds(
        self,
        provider: str,
        sport_key: str,
        event_id: Any,
        date: Any,
        home: str,
        away: str,
        home_raw: Any,
        away_raw: Any,
        bookmaker_key: Any,
        bookmaker_title: Any,
        last_update: Any,
        odds_home: float,
        odds_draw: float,
        odds_away: float,
        started: str,
    ) -> Dict[str, Any]:
        inv_home = 1.0 / float(odds_home)
        inv_draw = 1.0 / float(odds_draw) if not pd.isna(odds_draw) and float(odds_draw) > 1 else 0.0
        inv_away = 1.0 / float(odds_away)
        overround = inv_home + inv_draw + inv_away
        if overround <= 0:
            overround = np.nan

        return {
            "Provider": provider,
            "Sport_Key": sport_key,
            "Event_ID": event_id,
            "Data_UTC": date,
            "Data_Local": self._datetime_local_evento(date),
            "Data_Local_Data": self._data_local_evento(date),
            "Mandante": home,
            "Visitante": away,
            "Mandante_Original": home_raw,
            "Visitante_Original": away_raw,
            "Bookmaker_Key": bookmaker_key,
            "Bookmaker": bookmaker_title,
            "Bookmaker_Last_Update": last_update,
            "Odds_Mandante": odds_home,
            "Odds_Empate": odds_draw,
            "Odds_Visitante": odds_away,
            "Overround": overround,
            "Prob_Bruta_Mandante": inv_home,
            "Prob_Bruta_Empate": inv_draw,
            "Prob_Bruta_Visitante": inv_away,
            "Prob_SemVig_Mandante": inv_home / overround if overround and not pd.isna(overround) else np.nan,
            "Prob_SemVig_Empate": inv_draw / overround if inv_draw > 0 and overround and not pd.isna(overround) else 0.0,
            "Prob_SemVig_Visitante": inv_away / overround if overround and not pd.isna(overround) else np.nan,
            "Match_Key": self._match_key(home, away),
            "Extraido_UTC": started,
        }

    @staticmethod
    def consolidar_consenso(raw: pd.DataFrame) -> pd.DataFrame:
        if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
            return pd.DataFrame()

        d = raw.copy()
        required = ["Match_Key", "Mandante", "Visitante", "Bookmaker"]
        if any(c not in d.columns for c in required):
            return pd.DataFrame()

        agg = d.groupby(["Match_Key", "Mandante", "Visitante"], as_index=False).agg(
            Provider=("Provider", lambda x: ", ".join(sorted(set(map(str, x)))[:3]) if "Provider" in d.columns else ""),
            Data_UTC=("Data_UTC", "min"),
            Data_Local=("Data_Local", "min") if "Data_Local" in d.columns else ("Data_UTC", "min"),
            Data_Local_Data=("Data_Local_Data", "min") if "Data_Local_Data" in d.columns else ("Data_UTC", "min"),
            Casas_Usadas=("Bookmaker", "nunique"),
            Casas_Lista=("Bookmaker", lambda x: ", ".join(sorted(set(map(str, x)))[:18])),
            Odds_Media_Mandante=("Odds_Mandante", "mean"),
            Odds_Media_Empate=("Odds_Empate", "mean"),
            Odds_Media_Visitante=("Odds_Visitante", "mean"),
            Odds_Min_Mandante=("Odds_Mandante", "min"),
            Odds_Max_Mandante=("Odds_Mandante", "max"),
            Odds_Min_Visitante=("Odds_Visitante", "min"),
            Odds_Max_Visitante=("Odds_Visitante", "max"),
            Prob_SemVig_Mandante=("Prob_SemVig_Mandante", "mean"),
            Prob_SemVig_Empate=("Prob_SemVig_Empate", "mean"),
            Prob_SemVig_Visitante=("Prob_SemVig_Visitante", "mean"),
            Overround_Medio=("Overround", "mean"),
            Atualizado_UTC=("Extraido_UTC", "max"),
        )

        soma = agg[["Prob_SemVig_Mandante", "Prob_SemVig_Empate", "Prob_SemVig_Visitante"]].sum(axis=1).replace(0, np.nan)
        for col in ["Mandante", "Empate", "Visitante"]:
            agg[f"Prob_Mercado_{col}_pct"] = (100 * agg[f"Prob_SemVig_{col}"] / soma).round(2)

        favoritos = []
        confs = []
        for _, row in agg.iterrows():
            probs = {
                str(row["Mandante"]): float(row.get("Prob_Mercado_Mandante_pct", 0) or 0),
                "Empate": float(row.get("Prob_Mercado_Empate_pct", 0) or 0),
                str(row["Visitante"]): float(row.get("Prob_Mercado_Visitante_pct", 0) or 0),
            }
            ordenado = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            favoritos.append(ordenado[0][0])
            confs.append(round(ordenado[0][1] - ordenado[1][1], 2) if len(ordenado) >= 2 else 0.0)
        agg["Favorito_Casas"] = favoritos
        agg["Margem_Favorito_Mercado_p.p."] = confs

        cols = [
            "Provider", "Data_UTC", "Data_Local", "Data_Local_Data", "Mandante", "Visitante", "Casas_Usadas", "Casas_Lista",
            "Odds_Media_Mandante", "Odds_Media_Empate", "Odds_Media_Visitante",
            "Prob_Mercado_Mandante_pct", "Prob_Mercado_Empate_pct", "Prob_Mercado_Visitante_pct",
            "Favorito_Casas", "Margem_Favorito_Mercado_p.p.", "Overround_Medio", "Atualizado_UTC", "Match_Key",
            "Odds_Min_Mandante", "Odds_Max_Mandante", "Odds_Min_Visitante", "Odds_Max_Visitante",
        ]
        cols = [c for c in cols if c in agg.columns]
        return agg[cols].sort_values(["Data_UTC", "Mandante", "Visitante"]).reset_index(drop=True)

    @staticmethod
    def _similaridade(a: Any, b: Any) -> float:
        from difflib import SequenceMatcher
        aa = normalizar_texto(canonical_team_name(a))
        bb = normalizar_texto(canonical_team_name(b))
        if not aa or not bb:
            return 0.0
        if aa == bb:
            return 1.0
        return float(SequenceMatcher(None, aa, bb).ratio())

    @classmethod
    def buscar_odds_partida(
        cls,
        consensus: pd.DataFrame,
        mandante: Any,
        visitante: Any,
        minimo_score: float = 0.84,
        target_date: Optional[Any] = None,
        target_timezone: str = "America/Sao_Paulo",
    ) -> Optional[Dict[str, Any]]:
        if consensus is None or not isinstance(consensus, pd.DataFrame) or consensus.empty:
            return None

        home = canonical_team_name(mandante)
        away = canonical_team_name(visitante)
        if not home or not away:
            return None

        d = cls._filtrar_dataframe_por_data(consensus.copy(), target_date, target_timezone)
        if d.empty:
            return None
        if not {"Mandante", "Visitante"}.issubset(d.columns):
            return None

        home_key = normalizar_texto(home)
        away_key = normalizar_texto(away)
        d["_home_key"] = d["Mandante"].apply(lambda x: normalizar_texto(canonical_team_name(x)))
        d["_away_key"] = d["Visitante"].apply(lambda x: normalizar_texto(canonical_team_name(x)))

        exact = d[(d["_home_key"] == home_key) & (d["_away_key"] == away_key)]
        if not exact.empty:
            row = exact.iloc[0].to_dict()
            row["Orientacao_Modelo"] = "mesma"
            return row

        reversed_match = d[(d["_home_key"] == away_key) & (d["_away_key"] == home_key)]
        if not reversed_match.empty:
            row = reversed_match.iloc[0].to_dict()
            return cls._inverter_orientacao(row, home, away)

        best = None
        best_score = 0.0
        best_reverse = False
        for _, row in d.iterrows():
            score_direct = 0.5 * cls._similaridade(home, row["Mandante"]) + 0.5 * cls._similaridade(away, row["Visitante"])
            score_reverse = 0.5 * cls._similaridade(home, row["Visitante"]) + 0.5 * cls._similaridade(away, row["Mandante"])
            if score_direct > best_score:
                best = row.to_dict()
                best_score = score_direct
                best_reverse = False
            if score_reverse > best_score:
                best = row.to_dict()
                best_score = score_reverse
                best_reverse = True

        if best is None or best_score < minimo_score:
            return None
        if best_reverse:
            best = cls._inverter_orientacao(best, home, away)
        else:
            best["Orientacao_Modelo"] = "aproximada"
        best["Score_Match_Odds"] = round(best_score, 3)
        return best

    @staticmethod
    def _inverter_orientacao(row: Dict[str, Any], home_modelo: str, away_modelo: str) -> Dict[str, Any]:
        out = dict(row)
        out["Mandante_Original_Odds"] = row.get("Mandante")
        out["Visitante_Original_Odds"] = row.get("Visitante")
        out["Mandante"] = home_modelo
        out["Visitante"] = away_modelo

        swaps = [
            ("Odds_Media_Mandante", "Odds_Media_Visitante"),
            ("Prob_Mercado_Mandante_pct", "Prob_Mercado_Visitante_pct"),
            ("Odds_Min_Mandante", "Odds_Min_Visitante"),
            ("Odds_Max_Mandante", "Odds_Max_Visitante"),
        ]
        for a, b in swaps:
            if a in out and b in out:
                out[a], out[b] = out[b], out[a]

        probs = {
            home_modelo: float(out.get("Prob_Mercado_Mandante_pct", 0) or 0),
            "Empate": float(out.get("Prob_Mercado_Empate_pct", 0) or 0),
            away_modelo: float(out.get("Prob_Mercado_Visitante_pct", 0) or 0),
        }
        out["Favorito_Casas"] = max(probs, key=probs.get)
        out["Orientacao_Modelo"] = "invertida"
        return out

def combinar_previsao_com_odds(previsao_ensemble: Dict[str, Any], odds_partida: Optional[Dict[str, Any]], peso_odds: float = 0.35) -> Optional[Dict[str, Any]]:
    """
    Combina previsão estatística/ML com probabilidade implícita de mercado.
    Retorna None se não houver odds disponíveis.
    """
    if not previsao_ensemble or not odds_partida:
        return None

    peso_odds = clamp(float(peso_odds), 0.0, 1.0)
    peso_modelo = 1.0 - peso_odds

    p_model_home = float(previsao_ensemble.get("ensemble_prob_mandante", 0) or 0)
    p_model_draw = float(previsao_ensemble.get("ensemble_prob_empate", 0) or 0)
    p_model_away = float(previsao_ensemble.get("ensemble_prob_visitante", 0) or 0)

    p_market_home = float(odds_partida.get("Prob_Mercado_Mandante_pct", 0) or 0)
    p_market_draw = float(odds_partida.get("Prob_Mercado_Empate_pct", 0) or 0)
    p_market_away = float(odds_partida.get("Prob_Mercado_Visitante_pct", 0) or 0)

    p_home = peso_modelo * p_model_home + peso_odds * p_market_home
    p_draw = peso_modelo * p_model_draw + peso_odds * p_market_draw
    p_away = peso_modelo * p_model_away + peso_odds * p_market_away

    soma = p_home + p_draw + p_away
    if soma > 0:
        p_home, p_draw, p_away = 100 * p_home / soma, 100 * p_draw / soma, 100 * p_away / soma

    mandante = str(previsao_ensemble.get("mandante", odds_partida.get("Mandante", "Mandante")))
    visitante = str(previsao_ensemble.get("visitante", odds_partida.get("Visitante", "Visitante")))
    probs = {mandante: p_home, "Empate": p_draw, visitante: p_away}
    ordenado = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)

    return {
        "mandante": mandante,
        "visitante": visitante,
        "peso_modelo": round(peso_modelo, 2),
        "peso_odds": round(peso_odds, 2),
        "prob_final_mandante": round(p_home, 2),
        "prob_final_empate": round(p_draw, 2),
        "prob_final_visitante": round(p_away, 2),
        "favorito_modelo_mercado": ordenado[0][0],
        "margem_favorito_final_p.p.": round(ordenado[0][1] - ordenado[1][1], 2) if len(ordenado) >= 2 else 0.0,
        "favorito_casas": odds_partida.get("Favorito_Casas", ""),
        "casas_usadas": odds_partida.get("Casas_Usadas", 0),
        "placar_provavel_modelo": previsao_ensemble.get("placar_provavel", ""),
    }

# ============================================================
# 1. LEITOR E VALIDADOR DE BASE
# ============================================================

class MatchDataLoader:
    def __init__(self):
        self.df_original: Optional[pd.DataFrame] = None
        self.df_matches: Optional[pd.DataFrame] = None

    def carregar_arquivo(self, caminho: str) -> pd.DataFrame:
        if not os.path.exists(caminho):
            raise FileNotFoundError("Arquivo não encontrado.")

        ext = os.path.splitext(caminho)[1].lower()

        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(caminho)
        elif ext == ".csv":
            try:
                df = pd.read_csv(caminho, sep=None, engine="python", encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(caminho, sep=None, engine="python", encoding="latin1")
        else:
            raise ValueError("Formato não suportado. Use .xlsx, .xls ou .csv.")

        if df.empty:
            raise ValueError("A base está vazia.")

        self.df_original = df.copy()
        self.df_matches = self._padronizar_base(df)
        return self.df_matches

    def _padronizar_base(self, df: pd.DataFrame) -> pd.DataFrame:
        col_date = encontrar_coluna(df, ["date", "data", "match_date", "data_jogo"])
        col_home_team = encontrar_coluna(df, ["home_team", "mandante", "time_casa", "casa", "selecao_casa", "seleção_casa", "equipe_casa"])
        col_away_team = encontrar_coluna(df, ["away_team", "visitante", "time_fora", "fora", "selecao_fora", "seleção_fora", "equipe_fora"])
        col_home_goals = encontrar_coluna(df, ["home_goals", "gols_casa", "placar_casa", "goals_home", "gols_mandante"])
        col_away_goals = encontrar_coluna(df, ["away_goals", "gols_fora", "placar_fora", "goals_away", "gols_visitante"])
        col_home_xg = encontrar_coluna(df, ["home_xg", "xg_casa", "xg_home", "expected_goals_home"])
        col_away_xg = encontrar_coluna(df, ["away_xg", "xg_fora", "xg_away", "expected_goals_away"])
        col_comp = encontrar_coluna(df, ["competition", "competicao", "competição", "torneio", "campeonato"])

        obrigatorias = {
            "date/data": col_date,
            "home_team/mandante": col_home_team,
            "away_team/visitante": col_away_team,
            "home_goals/gols_casa": col_home_goals,
            "away_goals/gols_fora": col_away_goals,
        }
        ausentes = [nome for nome, col in obrigatorias.items() if col is None]
        if ausentes:
            raise ValueError(
                "A base não possui as colunas mínimas necessárias:\n\n"
                + "\n".join(ausentes)
                + "\n\nUse: date, home_team, away_team, home_goals, away_goals."
            )

        matches = pd.DataFrame()
        matches["date"] = pd.to_datetime(df[col_date], errors="coerce")
        matches["home_team"] = df[col_home_team].apply(canonical_team_name)
        matches["away_team"] = df[col_away_team].apply(canonical_team_name)
        matches["home_goals"] = to_number(df[col_home_goals])
        matches["away_goals"] = to_number(df[col_away_goals])

        if col_home_xg:
            matches["home_xg"] = to_number(df[col_home_xg])
        else:
            matches["home_xg"] = matches["home_goals"]

        if col_away_xg:
            matches["away_xg"] = to_number(df[col_away_xg])
        else:
            matches["away_xg"] = matches["away_goals"]

        if col_comp:
            matches["competition"] = df[col_comp].astype(str).str.strip()
        else:
            matches["competition"] = "Não informado"

        matches = matches.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
        matches = matches[matches["home_team"].str.lower().ne("nan")]
        matches = matches[matches["away_team"].str.lower().ne("nan")]
        matches = matches[matches["home_team"].str.strip().ne("")]
        matches = matches[matches["away_team"].str.strip().ne("")]

        matches["home_goals"] = matches["home_goals"].astype(int)
        matches["away_goals"] = matches["away_goals"].astype(int)
        matches["home_xg"] = matches["home_xg"].fillna(matches["home_goals"]).astype(float).clip(lower=0)
        matches["away_xg"] = matches["away_xg"].fillna(matches["away_goals"]).astype(float).clip(lower=0)

        matches = matches.sort_values("date").reset_index(drop=True)

        if matches.empty:
            raise ValueError("Após a limpeza, nenhuma partida válida restou na base.")

        return matches

    def resumo(self) -> str:
        if self.df_matches is None:
            return "Nenhuma base carregada."

        df = self.df_matches
        times = sorted(set(df["home_team"]).union(set(df["away_team"])))
        return (
            "BASE CARREGADA COM SUCESSO\n"
            "========================================\n\n"
            f"Partidas válidas: {len(df)}\n"
            f"Seleções/equipes únicas: {len(times)}\n"
            f"Primeira data: {df['date'].min().date()}\n"
            f"Última data: {df['date'].max().date()}\n\n"
            "Colunas padronizadas:\n"
            "- date\n- home_team\n- away_team\n- home_goals\n- away_goals\n"
            "- home_xg\n- away_xg\n- competition\n\n"
            "Prévia:\n"
            "========================================\n"
            + df.head(12).to_string(index=False)
        )


# ============================================================
# 2. ELO DINÂMICO
# ============================================================

class DynamicEloModel:
    def __init__(self, rating_inicial: float = 1500.0, k_base: float = 30.0, vantagem_casa: float = 55.0):
        self.rating_inicial = rating_inicial
        self.k_base = k_base
        self.vantagem_casa = vantagem_casa
        self.ratings: Dict[str, float] = {}
        self.jogos: Dict[str, int] = {}
        self.historico: List[Dict[str, Any]] = []

    def resetar(self):
        self.ratings.clear()
        self.jogos.clear()
        self.historico.clear()

    def obter_rating(self, equipe: str) -> float:
        equipe = str(equipe).strip()
        if equipe not in self.ratings:
            self.ratings[equipe] = self.rating_inicial
            self.jogos[equipe] = 0
        return self.ratings[equipe]

    def obter_elo(self, equipe: str) -> float:
        equipe = str(equipe).strip()
        return self.ratings.get(equipe, self.rating_inicial)

    @staticmethod
    def probabilidade_esperada(rating_a: float, rating_b: float) -> float:
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    @staticmethod
    def resultado_real(gols_a: int, gols_b: int) -> float:
        if gols_a > gols_b:
            return 1.0
        if gols_a == gols_b:
            return 0.5
        return 0.0

    def peso_competicao(self, competicao: str) -> float:
        c = normalizar_texto(competicao)
        if "world_cup" in c or "copa_do_mundo" in c:
            return 1.80
        if "qualifier" in c or "eliminatoria" in c:
            return 1.45
        if "euro" in c or "copa_america" in c or "copa_america" in c:
            return 1.45
        if "nations" in c:
            return 1.25
        if "friendly" in c or "amistoso" in c:
            return 0.60
        return 1.00

    @staticmethod
    def multiplicador_saldo_gols(gols_a: int, gols_b: int, diff_rating: float) -> float:
        saldo = abs(gols_a - gols_b)
        if saldo <= 1:
            return 1.0
        return float(np.log(saldo + 1) * (2.2 / ((abs(diff_rating) * 0.001) + 2.2)))

    def atualizar_partida(self, mandante: str, visitante: str, gols_mandante: int, gols_visitante: int, competicao: str):
        mandante = str(mandante).strip()
        visitante = str(visitante).strip()

        elo_h = self.obter_rating(mandante)
        elo_a = self.obter_rating(visitante)
        esperado_h = self.probabilidade_esperada(elo_h + self.vantagem_casa, elo_a)
        real_h = self.resultado_real(gols_mandante, gols_visitante)

        k = (
            self.k_base
            * self.peso_competicao(competicao)
            * self.multiplicador_saldo_gols(gols_mandante, gols_visitante, elo_h - elo_a)
        )
        delta = k * (real_h - esperado_h)

        self.ratings[mandante] = elo_h + delta
        self.ratings[visitante] = elo_a - delta
        self.jogos[mandante] = self.jogos.get(mandante, 0) + 1
        self.jogos[visitante] = self.jogos.get(visitante, 0) + 1

        self.historico.append({
            "mandante": mandante, "visitante": visitante,
            "gols_mandante": gols_mandante, "gols_visitante": gols_visitante,
            "competicao": competicao, "elo_mandante_antes": elo_h,
            "elo_visitante_antes": elo_a, "elo_mandante_depois": elo_h + delta,
            "elo_visitante_depois": elo_a - delta, "delta": delta, "k": k,
            "esperado_mandante": esperado_h,
        })

    def treinar(self, df_matches: pd.DataFrame) -> "DynamicEloModel":
        self.resetar()
        df = df_matches.sort_values("date").reset_index(drop=True)
        for _, row in df.iterrows():
            self.atualizar_partida(
                row["home_team"], row["away_team"],
                int(row["home_goals"]), int(row["away_goals"]),
                row.get("competition", "Não informado"),
            )
        return self

    def ranking(self) -> pd.DataFrame:
        if not self.ratings:
            return pd.DataFrame(columns=["Rank", "Equipe", "Elo", "Jogos"])
        df = pd.DataFrame([
            {"Equipe": equipe, "Elo": elo, "Jogos": self.jogos.get(equipe, 0)}
            for equipe, elo in self.ratings.items()
        ])
        df = df.sort_values("Elo", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))
        df["Elo"] = df["Elo"].round(2)
        return df


# ============================================================
# 3. POISSON + xG + DIXON-COLES + BIVARIADA
# ============================================================

class BasePoissonModel:
    def __init__(self, max_gols: int = 10, peso_xg: float = 0.70, peso_gols: float = 0.30):
        self.max_gols = max_gols
        self.peso_xg = peso_xg
        self.peso_gols = peso_gols

        self.df_team_stats: Optional[pd.DataFrame] = None
        self.media_gols_casa = 1.35
        self.media_gols_fora = 1.10
        self.media_xg_casa = 1.35
        self.media_xg_fora = 1.10
        self.elo_model: Optional[DynamicEloModel] = None

        self.rho = 0.0
        self.lambda_compartilhado = 0.0

    def treinar(self, df_matches: pd.DataFrame, elo_model: Optional[DynamicEloModel] = None, df_fifa_team_metrics: Optional[pd.DataFrame] = None) -> "BasePoissonModel":
        self.elo_model = elo_model
        df = df_matches.copy()

        self.media_gols_casa = max(float(df["home_goals"].mean()), 0.20)
        self.media_gols_fora = max(float(df["away_goals"].mean()), 0.20)
        self.media_xg_casa = max(float(df["home_xg"].mean()), 0.20)
        self.media_xg_fora = max(float(df["away_xg"].mean()), 0.20)

        equipes = sorted(set(df["home_team"]).union(set(df["away_team"])))
        linhas = []

        for equipe in equipes:
            hc = df[df["home_team"] == equipe]
            af = df[df["away_team"] == equipe]
            n_h, n_a = len(hc), len(af)
            n = n_h + n_a
            if n == 0:
                continue

            gfh = hc["home_goals"].sum() / n_h if n_h else self.media_gols_casa
            gfa = af["away_goals"].sum() / n_a if n_a else self.media_gols_fora
            gah = hc["away_goals"].sum() / n_h if n_h else self.media_gols_fora
            gaa = af["home_goals"].sum() / n_a if n_a else self.media_gols_casa

            xg_h = hc["home_xg"].sum() / n_h if n_h else self.media_xg_casa
            xg_a = af["away_xg"].sum() / n_a if n_a else self.media_xg_fora
            xga_h = hc["away_xg"].sum() / n_h if n_h else self.media_xg_fora
            xga_a = af["home_xg"].sum() / n_a if n_a else self.media_xg_casa

            at_g_h = gfh / self.media_gols_casa
            at_g_a = gfa / self.media_gols_fora
            de_g_h = gah / self.media_gols_fora
            de_g_a = gaa / self.media_gols_casa

            at_x_h = xg_h / self.media_xg_casa
            at_x_a = xg_a / self.media_xg_fora
            de_x_h = xga_h / self.media_xg_fora
            de_x_a = xga_a / self.media_xg_casa

            ataque_h = self.peso_xg * at_x_h + self.peso_gols * at_g_h
            ataque_a = self.peso_xg * at_x_a + self.peso_gols * at_g_a
            defesa_h = self.peso_xg * de_x_h + self.peso_gols * de_g_h
            defesa_a = self.peso_xg * de_x_a + self.peso_gols * de_g_a

            elo = elo_model.obter_elo(equipe) if elo_model is not None else 1500.0

            linhas.append({
                "Equipe": equipe, "Jogos": n, "Jogos_Casa": n_h, "Jogos_Fora": n_a,
                "Gols_Marcados_Casa_Media": round(gfh, 3),
                "Gols_Marcados_Fora_Media": round(gfa, 3),
                "Gols_Sofridos_Casa_Media": round(gah, 3),
                "Gols_Sofridos_Fora_Media": round(gaa, 3),
                "xG_Casa_Media": round(xg_h, 3), "xG_Fora_Media": round(xg_a, 3),
                "xGA_Casa_Media": round(xga_h, 3), "xGA_Fora_Media": round(xga_a, 3),
                "Ataque_Casa": ataque_h, "Ataque_Fora": ataque_a,
                "Defesa_Casa": defesa_h, "Defesa_Fora": defesa_a,
                "Ataque_Geral": np.average([ataque_h, ataque_a], weights=[max(n_h, 1), max(n_a, 1)]),
                "Defesa_Geral": np.average([defesa_h, defesa_a], weights=[max(n_h, 1), max(n_a, 1)]),
                "Elo": elo,
            })

        stats = pd.DataFrame(linhas)
        if stats.empty:
            raise ValueError("Não foi possível calcular estatísticas por equipe.")

        metricas_fifa = preparar_metricas_fifa_equipes(df_fifa_team_metrics)
        if not metricas_fifa.empty:
            stats = stats.merge(metricas_fifa, on="Equipe", how="left")
        else:
            stats["FIFA_Usado_No_Modelo"] = False

        # Preenche neutro para seleções sem métrica FIFA extraída.
        for col in [
            "FIFA_Attack_Index", "FIFA_Defense_Index", "FIFA_Goalkeeper_Index",
            "FIFA_Distribution_Index", "FIFA_Discipline_Index", "FIFA_Physical_Index",
            "FIFA_Movement_Index", "FIFA_Overall_Index"
        ]:
            if col not in stats.columns:
                stats[col] = 0.5
            stats[col] = pd.to_numeric(stats[col], errors="coerce").fillna(0.5).clip(0, 1)

        for col, default in [("FIFA_Mult_Ataque", 1.0), ("FIFA_Mult_Defesa", 1.0), ("FIFA_Elo_Bonus", 0.0)]:
            if col not in stats.columns:
                stats[col] = default
            stats[col] = pd.to_numeric(stats[col], errors="coerce").fillna(default)

        if "FIFA_Usado_No_Modelo" not in stats.columns:
            stats["FIFA_Usado_No_Modelo"] = False
        stats["FIFA_Usado_No_Modelo"] = stats["FIFA_Usado_No_Modelo"].fillna(False).astype(bool)

        # Guarda bases antes do ajuste para auditoria.
        stats["Ataque_Casa_Base"] = stats["Ataque_Casa"]
        stats["Ataque_Fora_Base"] = stats["Ataque_Fora"]
        stats["Defesa_Casa_Base"] = stats["Defesa_Casa"]
        stats["Defesa_Fora_Base"] = stats["Defesa_Fora"]
        stats["Elo_Base"] = stats["Elo"]

        # Aplica métricas FIFA de forma moderada.
        stats["Ataque_Casa"] = stats["Ataque_Casa"] * stats["FIFA_Mult_Ataque"]
        stats["Ataque_Fora"] = stats["Ataque_Fora"] * stats["FIFA_Mult_Ataque"]
        stats["Defesa_Casa"] = stats["Defesa_Casa"] * stats["FIFA_Mult_Defesa"]
        stats["Defesa_Fora"] = stats["Defesa_Fora"] * stats["FIFA_Mult_Defesa"]
        stats["Ataque_Geral"] = np.average(
            stats[["Ataque_Casa", "Ataque_Fora"]].to_numpy(),
            axis=1,
            weights=[1, 1],
        )
        stats["Defesa_Geral"] = np.average(
            stats[["Defesa_Casa", "Defesa_Fora"]].to_numpy(),
            axis=1,
            weights=[1, 1],
        )
        stats["Elo"] = stats["Elo"] + stats["FIFA_Elo_Bonus"]

        for col in [
            "Ataque_Casa", "Ataque_Fora", "Defesa_Casa", "Defesa_Fora", "Ataque_Geral", "Defesa_Geral", "Elo",
            "Ataque_Casa_Base", "Ataque_Fora_Base", "Defesa_Casa_Base", "Defesa_Fora_Base", "Elo_Base",
            "FIFA_Attack_Index", "FIFA_Defense_Index", "FIFA_Goalkeeper_Index", "FIFA_Distribution_Index",
            "FIFA_Discipline_Index", "FIFA_Physical_Index", "FIFA_Movement_Index", "FIFA_Overall_Index",
            "FIFA_Mult_Ataque", "FIFA_Mult_Defesa", "FIFA_Elo_Bonus"
        ]:
            stats[col] = stats[col].astype(float).round(4)

        self.df_team_stats = stats.sort_values("Elo", ascending=False).reset_index(drop=True)
        self.rho = self._estimar_rho_dixon_coles(df)
        self.lambda_compartilhado = self._estimar_lambda_compartilhado(df)
        return self

    def _obter_linha_equipe(self, equipe: str) -> pd.Series:
        if self.df_team_stats is None:
            raise ValueError("Modelo ainda não treinado.")
        equipe = str(equipe).strip()
        row = self.df_team_stats[self.df_team_stats["Equipe"] == equipe]
        if row.empty:
            raise ValueError(f"Equipe não encontrada no modelo: {equipe}")
        return row.iloc[0]

    def calcular_lambdas(self, mandante: str, visitante: str, usar_elo: bool = True) -> Tuple[float, float]:
        h = self._obter_linha_equipe(mandante)
        a = self._obter_linha_equipe(visitante)

        lam_h = self.media_xg_casa * h["Ataque_Casa"] * a["Defesa_Fora"]
        lam_a = self.media_xg_fora * a["Ataque_Fora"] * h["Defesa_Casa"]

        if usar_elo:
            diff = (float(h["Elo"]) - float(a["Elo"])) / 400
            lam_h *= np.exp(0.12 * diff)
            lam_a *= np.exp(-0.12 * diff)

        return clamp(lam_h, 0.10, 5.50), clamp(lam_a, 0.10, 5.50)

    @staticmethod
    def _tau_dc(x: int, y: int, lh: float, la: float, rho: float) -> float:
        if x == 0 and y == 0:
            return 1 - lh * la * rho
        if x == 0 and y == 1:
            return 1 + lh * rho
        if x == 1 and y == 0:
            return 1 + la * rho
        if x == 1 and y == 1:
            return 1 - rho
        return 1.0

    def _nll_rho(self, rho: float, df_matches: pd.DataFrame) -> float:
        nll = 0.0
        for _, r in df_matches.iterrows():
            x, y = int(r["home_goals"]), int(r["away_goals"])
            lh, la = self.calcular_lambdas(r["home_team"], r["away_team"], usar_elo=True)
            p = poisson.pmf(x, lh) * poisson.pmf(y, la) * self._tau_dc(x, y, lh, la, rho)
            if p <= 0 or not np.isfinite(p):
                return 1e12
            nll -= math.log(max(p, 1e-12))
        return nll

    def _estimar_rho_dixon_coles(self, df_matches: pd.DataFrame) -> float:
        if len(df_matches) < 20:
            return 0.0
        try:
            res = minimize_scalar(lambda r: self._nll_rho(r, df_matches), bounds=(-0.20, 0.20), method="bounded")
            return clamp(res.x, -0.20, 0.20) if res.success else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _bivar_pmf(x: int, y: int, lh_total: float, la_total: float, lc: float) -> float:
        lc = clamp(lc, 0.0, 0.95 * min(lh_total, la_total))
        lh = max(lh_total - lc, 1e-8)
        la = max(la_total - lc, 1e-8)
        log_base = -(lh + la + lc)
        s = 0.0
        for k in range(min(x, y) + 1):
            log_term = (
                log_base
                + (x - k) * math.log(lh)
                + (y - k) * math.log(la)
                + k * math.log(max(lc, 1e-12))
                - gammaln(x - k + 1)
                - gammaln(y - k + 1)
                - gammaln(k + 1)
            )
            s += math.exp(log_term)
        return float(max(s, 0.0))

    def _nll_bivar(self, lc: float, df_matches: pd.DataFrame) -> float:
        nll = 0.0
        for _, r in df_matches.iterrows():
            x, y = int(r["home_goals"]), int(r["away_goals"])
            lh, la = self.calcular_lambdas(r["home_team"], r["away_team"], usar_elo=True)
            p = self._bivar_pmf(x, y, lh, la, lc)
            if p <= 0 or not np.isfinite(p):
                return 1e12
            nll -= math.log(max(p, 1e-12))
        return nll

    def _estimar_lambda_compartilhado(self, df_matches: pd.DataFrame) -> float:
        if len(df_matches) < 30:
            return 0.0
        try:
            res = minimize_scalar(lambda lc: self._nll_bivar(lc, df_matches), bounds=(0.0, 0.60), method="bounded")
            return clamp(res.x, 0.0, 0.60) if res.success else 0.0
        except Exception:
            return 0.0

    def prever_partida(self, mandante: str, visitante: str) -> Dict[str, Any]:
        lh, la = self.calcular_lambdas(mandante, visitante)
        matriz = []
        placares: Dict[Tuple[int, int], float] = {}
        p_h = p_d = p_a = p_over25 = p_btts = 0.0

        for x in range(self.max_gols + 1):
            for y in range(self.max_gols + 1):
                p = self._bivar_pmf(x, y, lh, la, self.lambda_compartilhado)
                p *= self._tau_dc(x, y, lh, la, self.rho)
                p = max(float(p), 0.0) if np.isfinite(p) else 0.0

                placares[(x, y)] = p
                matriz.append({"Gols_Mandante": x, "Gols_Visitante": y, "Probabilidade": p})

                if x > y:
                    p_h += p
                elif x == y:
                    p_d += p
                else:
                    p_a += p

                if x + y > 2.5:
                    p_over25 += p
                if x > 0 and y > 0:
                    p_btts += p

        total = p_h + p_d + p_a
        if total <= 0:
            raise ValueError("Erro numérico: soma das probabilidades zerada.")

        p_h, p_d, p_a = p_h / total, p_d / total, p_a / total
        p_over25, p_btts = p_over25 / total, p_btts / total

        placar = max(placares.items(), key=lambda kv: kv[1])[0]
        top = sorted(placares.items(), key=lambda kv: kv[1], reverse=True)[:8]

        df_matriz = pd.DataFrame(matriz)
        df_matriz["Probabilidade"] = df_matriz["Probabilidade"] / total

        return {
            "mandante": mandante, "visitante": visitante,
            "lambda_mandante": round(lh, 3), "lambda_visitante": round(la, 3),
            "rho_dixon_coles": round(self.rho, 5),
            "lambda_compartilhado": round(self.lambda_compartilhado, 5),
            "peso_xg": self.peso_xg, "peso_gols": self.peso_gols,
            "placar_provavel": f"{placar[0]} x {placar[1]}",
            "prob_mandante": round(p_h * 100, 2),
            "prob_empate": round(p_d * 100, 2),
            "prob_visitante": round(p_a * 100, 2),
            "prob_over_25": round(p_over25 * 100, 2),
            "prob_btts": round(p_btts * 100, 2),
            "top_placares": [{"placar": f"{k[0]} x {k[1]}", "probabilidade": round(v / total * 100, 2)} for k, v in top],
            "matriz": df_matriz,
        }


# ============================================================
# 4. MACHINE LEARNING 1X2
# ============================================================

class MLMatchOutcomeModel:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.modelos = []
        self.feature_cols: Optional[List[str]] = None
        self.poisson_model: Optional[BasePoissonModel] = None
        self.elo_model: Optional[DynamicEloModel] = None
        self.treinado = False
        self.n_partidas_treino = 0

    @staticmethod
    def _resultado(x: int, y: int) -> str:
        if x > y:
            return "H"
        if x == y:
            return "D"
        return "A"

    @staticmethod
    def _valor(row: pd.Series, col: str, default: float) -> float:
        try:
            if col in row.index and pd.notna(row[col]):
                return float(row[col])
        except Exception:
            pass
        return float(default)

    def _peso_competicao(self, comp: str) -> float:
        if self.elo_model is not None:
            return self.elo_model.peso_competicao(comp)
        return 1.0

    def _features(self, mandante: str, visitante: str, competicao: str = "Não informado") -> Dict[str, float]:
        if self.poisson_model is None:
            raise ValueError("Modelo estatístico ausente no ML.")

        h = self.poisson_model._obter_linha_equipe(mandante)
        a = self.poisson_model._obter_linha_equipe(visitante)
        lh, la = self.poisson_model.calcular_lambdas(mandante, visitante)

        elo_h = self._valor(h, "Elo", 1500)
        elo_a = self._valor(a, "Elo", 1500)
        at_h = self._valor(h, "Ataque_Casa", 1)
        at_a = self._valor(a, "Ataque_Fora", 1)
        de_h = self._valor(h, "Defesa_Casa", 1)
        de_a = self._valor(a, "Defesa_Fora", 1)
        xg_h = self._valor(h, "xG_Casa_Media", lh)
        xg_a = self._valor(a, "xG_Fora_Media", la)
        xga_h = self._valor(h, "xGA_Casa_Media", la)
        xga_a = self._valor(a, "xGA_Fora_Media", lh)
        games_h = self._valor(h, "Jogos", 0)
        games_a = self._valor(a, "Jogos", 0)

        fifa_attack_h = self._valor(h, "FIFA_Attack_Index", 0.5)
        fifa_attack_a = self._valor(a, "FIFA_Attack_Index", 0.5)
        fifa_defense_h = self._valor(h, "FIFA_Defense_Index", 0.5)
        fifa_defense_a = self._valor(a, "FIFA_Defense_Index", 0.5)
        fifa_goalkeeper_h = self._valor(h, "FIFA_Goalkeeper_Index", 0.5)
        fifa_goalkeeper_a = self._valor(a, "FIFA_Goalkeeper_Index", 0.5)
        fifa_distribution_h = self._valor(h, "FIFA_Distribution_Index", 0.5)
        fifa_distribution_a = self._valor(a, "FIFA_Distribution_Index", 0.5)
        fifa_discipline_h = self._valor(h, "FIFA_Discipline_Index", 0.5)
        fifa_discipline_a = self._valor(a, "FIFA_Discipline_Index", 0.5)
        fifa_physical_h = self._valor(h, "FIFA_Physical_Index", 0.5)
        fifa_physical_a = self._valor(a, "FIFA_Physical_Index", 0.5)
        fifa_movement_h = self._valor(h, "FIFA_Movement_Index", 0.5)
        fifa_movement_a = self._valor(a, "FIFA_Movement_Index", 0.5)
        fifa_overall_h = self._valor(h, "FIFA_Overall_Index", 0.5)
        fifa_overall_a = self._valor(a, "FIFA_Overall_Index", 0.5)
        fifa_mult_ataque_h = self._valor(h, "FIFA_Mult_Ataque", 1.0)
        fifa_mult_ataque_a = self._valor(a, "FIFA_Mult_Ataque", 1.0)
        fifa_mult_defesa_h = self._valor(h, "FIFA_Mult_Defesa", 1.0)
        fifa_mult_defesa_a = self._valor(a, "FIFA_Mult_Defesa", 1.0)

        return {
            "elo_mandante": elo_h, "elo_visitante": elo_a,
            "elo_diff": elo_h - elo_a, "elo_abs_diff": abs(elo_h - elo_a),
            "lambda_mandante": lh, "lambda_visitante": la,
            "lambda_diff": lh - la, "lambda_total": lh + la,
            "ataque_mandante": at_h, "ataque_visitante": at_a, "ataque_diff": at_h - at_a,
            "defesa_mandante": de_h, "defesa_visitante": de_a, "defesa_diff": de_a - de_h,
            "xg_mandante_media": xg_h, "xg_visitante_media": xg_a, "xg_diff": xg_h - xg_a,
            "xga_mandante_media": xga_h, "xga_visitante_media": xga_a, "xga_diff": xga_a - xga_h,
            "jogos_mandante": games_h, "jogos_visitante": games_a, "jogos_diff": games_h - games_a,
            "fifa_attack_mandante": fifa_attack_h, "fifa_attack_visitante": fifa_attack_a, "fifa_attack_diff": fifa_attack_h - fifa_attack_a,
            "fifa_defense_mandante": fifa_defense_h, "fifa_defense_visitante": fifa_defense_a, "fifa_defense_diff": fifa_defense_h - fifa_defense_a,
            "fifa_goalkeeper_mandante": fifa_goalkeeper_h, "fifa_goalkeeper_visitante": fifa_goalkeeper_a, "fifa_goalkeeper_diff": fifa_goalkeeper_h - fifa_goalkeeper_a,
            "fifa_distribution_mandante": fifa_distribution_h, "fifa_distribution_visitante": fifa_distribution_a, "fifa_distribution_diff": fifa_distribution_h - fifa_distribution_a,
            "fifa_discipline_mandante": fifa_discipline_h, "fifa_discipline_visitante": fifa_discipline_a, "fifa_discipline_diff": fifa_discipline_h - fifa_discipline_a,
            "fifa_physical_mandante": fifa_physical_h, "fifa_physical_visitante": fifa_physical_a, "fifa_physical_diff": fifa_physical_h - fifa_physical_a,
            "fifa_movement_mandante": fifa_movement_h, "fifa_movement_visitante": fifa_movement_a, "fifa_movement_diff": fifa_movement_h - fifa_movement_a,
            "fifa_overall_mandante": fifa_overall_h, "fifa_overall_visitante": fifa_overall_a, "fifa_overall_diff": fifa_overall_h - fifa_overall_a,
            "fifa_mult_ataque_mandante": fifa_mult_ataque_h, "fifa_mult_ataque_visitante": fifa_mult_ataque_a,
            "fifa_mult_defesa_mandante": fifa_mult_defesa_h, "fifa_mult_defesa_visitante": fifa_mult_defesa_a,
            "peso_competicao": self._peso_competicao(competicao),
            "rho_dixon_coles": float(self.poisson_model.rho),
            "lambda_compartilhado": float(self.poisson_model.lambda_compartilhado),
        }

    def _dataset(self, df_matches: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        X_rows, y = [], []
        for _, r in df_matches.iterrows():
            try:
                X_rows.append(self._features(r["home_team"], r["away_team"], r.get("competition", "Não informado")))
                y.append(self._resultado(int(r["home_goals"]), int(r["away_goals"])))
            except Exception:
                continue
        if not X_rows:
            raise ValueError("Não foi possível montar dataset para ML.")
        return pd.DataFrame(X_rows), pd.Series(y)

    def treinar(self, df_matches: pd.DataFrame, poisson_model: BasePoissonModel, elo_model: Optional[DynamicEloModel] = None):
        self.poisson_model = poisson_model
        self.elo_model = elo_model

        X, y = self._dataset(df_matches)
        self.n_partidas_treino = len(X)

        if len(X) < 30:
            raise ValueError("Base pequena demais para ML. Use pelo menos 30 partidas.")
        if y.nunique() < 2:
            raise ValueError("A base precisa ter pelo menos 2 classes de resultado.")

        self.feature_cols = list(X.columns)
        candidatos = [
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=self.random_state)),
            make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3, class_weight="balanced", random_state=self.random_state)),
            make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.10, random_state=self.random_state)),
        ]

        self.modelos = []
        for m in candidatos:
            try:
                m.fit(X, y)
                self.modelos.append(m)
            except Exception:
                pass

        if not self.modelos:
            raise ValueError("Nenhum modelo de ML conseguiu ser treinado.")

        self.treinado = True
        return self

    def prever_partida(self, mandante: str, visitante: str, competicao: str = "Não informado") -> Dict[str, Any]:
        if not self.treinado or not self.modelos or self.feature_cols is None:
            raise ValueError("Modelo de ML ainda não treinado.")

        X = pd.DataFrame([self._features(mandante, visitante, competicao)])
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_cols]

        acc = {"H": 0.0, "D": 0.0, "A": 0.0}
        usados = 0
        for m in self.modelos:
            try:
                proba = m.predict_proba(X)[0]
                classes = list(m.classes_)
                for cls, p in zip(classes, proba):
                    if cls in acc:
                        acc[cls] += float(p)
                usados += 1
            except Exception:
                continue

        if usados == 0:
            raise ValueError("Falha ao calcular ML.")

        for k in acc:
            acc[k] /= usados
        s = sum(acc.values())
        if s <= 0:
            acc = {"H": 1/3, "D": 1/3, "A": 1/3}
        else:
            acc = {k: v / s for k, v in acc.items()}

        return {
            "mandante": mandante, "visitante": visitante,
            "ml_prob_mandante": round(acc["H"] * 100, 2),
            "ml_prob_empate": round(acc["D"] * 100, 2),
            "ml_prob_visitante": round(acc["A"] * 100, 2),
            "ml_modelos_usados": usados,
            "ml_partidas_treino": self.n_partidas_treino,
        }


# ============================================================
# 5. ENSEMBLE
# ============================================================

class EnsemblePredictionModel:
    @staticmethod
    def _normalizar(probs: Dict[str, float]) -> Dict[str, float]:
        s = float(sum(probs.values()))
        if s <= 0:
            return {"H": 1/3, "D": 1/3, "A": 1/3}
        return {k: float(v) / s for k, v in probs.items()}

    @staticmethod
    def _peso_ml(previsao_ml: Optional[Dict[str, Any]]) -> float:
        if previsao_ml is None:
            return 0.0
        n = int(previsao_ml.get("ml_partidas_treino", 0))
        usados = int(previsao_ml.get("ml_modelos_usados", 0))
        if n < 30:
            peso = 0.0
        elif n < 100:
            peso = 0.20
        elif n < 300:
            peso = 0.30
        elif n < 1000:
            peso = 0.40
        else:
            peso = 0.45
        if usados < 2:
            peso *= 0.65
        return clamp(peso, 0.0, 0.45)

    @staticmethod
    def _confianca(probs: Dict[str, float]) -> str:
        vals = sorted(probs.values(), reverse=True)
        margem = vals[0] - vals[1]
        if margem >= 0.18:
            return "alta"
        if margem >= 0.09:
            return "moderada"
        return "baixa"

    def combinar(self, stat: Dict[str, Any], ml: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        p_stat = self._normalizar({
            "H": stat["prob_mandante"] / 100,
            "D": stat["prob_empate"] / 100,
            "A": stat["prob_visitante"] / 100,
        })

        if ml is not None:
            p_ml = self._normalizar({
                "H": ml["ml_prob_mandante"] / 100,
                "D": ml["ml_prob_empate"] / 100,
                "A": ml["ml_prob_visitante"] / 100,
            })
        else:
            p_ml = p_stat.copy()

        w_ml = self._peso_ml(ml)
        w_stat = 1 - w_ml
        p = self._normalizar({k: w_stat * p_stat[k] + w_ml * p_ml[k] for k in ["H", "D", "A"]})

        winner = max(p, key=p.get)
        resultado = "Empate" if winner == "D" else f"Vitória {stat['mandante'] if winner == 'H' else stat['visitante']}"

        return {
            "mandante": stat["mandante"], "visitante": stat["visitante"],
            "ensemble_prob_mandante": round(p["H"] * 100, 2),
            "ensemble_prob_empate": round(p["D"] * 100, 2),
            "ensemble_prob_visitante": round(p["A"] * 100, 2),
            "peso_estatistico": round(w_stat, 3), "peso_ml": round(w_ml, 3),
            "resultado_mais_provavel": resultado, "confianca": self._confianca(p),
            "placar_provavel": stat["placar_provavel"],
            "lambda_mandante": stat["lambda_mandante"], "lambda_visitante": stat["lambda_visitante"],
            "prob_over_25": stat["prob_over_25"], "prob_btts": stat["prob_btts"],
            "rho_dixon_coles": stat["rho_dixon_coles"], "lambda_compartilhado": stat["lambda_compartilhado"],
        }


# ============================================================
# 6. SIMULAÇÕES
# ============================================================

class MonteCarloChampionSimulator:
    def __init__(self, poisson_model: BasePoissonModel, ml_model: Optional[MLMatchOutcomeModel] = None, ensemble_model: Optional[EnsemblePredictionModel] = None, random_state: int = 42):
        self.poisson_model = poisson_model
        self.ml_model = ml_model
        self.ensemble_model = ensemble_model or EnsemblePredictionModel()
        self.rng = np.random.default_rng(random_state)
        self.cache: Dict[Tuple[str, str], Tuple[str, str, float, float]] = {}

    def _forca_time(self, equipe: str) -> float:
        row = self.poisson_model._obter_linha_equipe(equipe)
        elo = float(row.get("Elo", 1500.0))
        ataque = float(row.get("Ataque_Geral", 1.0))
        defesa = float(row.get("Defesa_Geral", 1.0))
        fifa_overall = float(row.get("FIFA_Overall_Index", 0.5))
        fifa_attack = float(row.get("FIFA_Attack_Index", 0.5))
        fifa_defense = float(row.get("FIFA_Defense_Index", 0.5))
        return (
            0.40 * ((elo - 1500) / 400)
            + 0.27 * ataque
            + 0.18 * (1 / max(defesa, 0.10))
            + 0.10 * (fifa_overall - 0.5)
            + 0.03 * (fifa_attack - 0.5)
            + 0.02 * (fifa_defense - 0.5)
        )

    def _previsao_neutra(self, a: str, b: str) -> Tuple[float, float]:
        key = tuple(sorted([a, b]))
        if key in self.cache:
            t1, t2, p1, p2 = self.cache[key]
            return (p1, p2) if t1 == a else (p2, p1)

        stat_ab = self.poisson_model.prever_partida(a, b)
        ml_ab = None
        if self.ml_model is not None and getattr(self.ml_model, "treinado", False):
            try:
                ml_ab = self.ml_model.prever_partida(a, b)
            except Exception:
                ml_ab = None
        ens_ab = self.ensemble_model.combinar(stat_ab, ml_ab)

        stat_ba = self.poisson_model.prever_partida(b, a)
        ml_ba = None
        if self.ml_model is not None and getattr(self.ml_model, "treinado", False):
            try:
                ml_ba = self.ml_model.prever_partida(b, a)
            except Exception:
                ml_ba = None
        ens_ba = self.ensemble_model.combinar(stat_ba, ml_ba)

        p_a_win = (ens_ab["ensemble_prob_mandante"] + ens_ba["ensemble_prob_visitante"]) / 200
        p_b_win = (ens_ab["ensemble_prob_visitante"] + ens_ba["ensemble_prob_mandante"]) / 200
        p_draw = (ens_ab["ensemble_prob_empate"] + ens_ba["ensemble_prob_empate"]) / 200

        sw = p_a_win + p_b_win
        if sw <= 0:
            p_a_pen = p_b_pen = 0.5
        else:
            p_a_pen = p_a_win / sw
            p_b_pen = p_b_win / sw

        p_a = p_a_win + p_draw * p_a_pen
        p_b = p_b_win + p_draw * p_b_pen
        total = p_a + p_b
        if total <= 0:
            p_a = p_b = 0.5
        else:
            p_a, p_b = p_a / total, p_b / total

        self.cache[key] = (a, b, p_a, p_b)
        return p_a, p_b

    def simular_confronto(self, a: str, b: str) -> str:
        p_a, _ = self._previsao_neutra(a, b)
        return a if self.rng.random() <= p_a else b

    def simular_campeao(self, iteracoes: int = 10000, equipes: Optional[List[str]] = None) -> pd.DataFrame:
        if self.poisson_model.df_team_stats is None:
            raise ValueError("Modelo ainda não treinado.")

        if equipes is None:
            equipes = self.poisson_model.df_team_stats["Equipe"].tolist()
        equipes = [str(e).strip() for e in equipes if str(e).strip()]
        if len(equipes) < 2:
            raise ValueError("É preciso ter pelo menos 2 equipes.")

        counts = {e: 0 for e in equipes}
        for _ in range(iteracoes):
            participantes = equipes.copy()
            while len(participantes) > 1:
                self.rng.shuffle(participantes)
                next_round = []
                if len(participantes) % 2 == 1:
                    bye = max(participantes, key=self._forca_time)
                    participantes.remove(bye)
                    next_round.append(bye)
                for i in range(0, len(participantes), 2):
                    next_round.append(self.simular_confronto(participantes[i], participantes[i + 1]))
                participantes = next_round
            counts[participantes[0]] += 1

        out = pd.DataFrame([
            {"Equipe": e, "Titulos_Simulados": c, "Probabilidade_Titulo_%": c / iteracoes * 100, "Forca_Modelo": self._forca_time(e)}
            for e, c in counts.items()
        ]).sort_values("Probabilidade_Titulo_%", ascending=False).reset_index(drop=True)
        out.insert(0, "Rank", range(1, len(out) + 1))
        out["Probabilidade_Titulo_%"] = out["Probabilidade_Titulo_%"].round(2)
        out["Forca_Modelo"] = out["Forca_Modelo"].round(4)
        return out


class WorldCupFormatSimulator:
    def __init__(
        self,
        poisson_model: BasePoissonModel,
        ml_model: Optional[MLMatchOutcomeModel] = None,
        ensemble_model: Optional[EnsemblePredictionModel] = None,
        random_state: int = 42,
        equipes_copa: Optional[List[str]] = None,
        grupos_copa: Optional[pd.DataFrame] = None,
        jogos_copa: Optional[pd.DataFrame] = None,
        classificados_copa: Optional[pd.DataFrame] = None,
    ):
        self.poisson_model = poisson_model
        self.ml_model = ml_model
        self.ensemble_model = ensemble_model or EnsemblePredictionModel()
        self.rng = np.random.default_rng(random_state)
        self.base = MonteCarloChampionSimulator(poisson_model, ml_model, self.ensemble_model, random_state)
        self.equipes_copa = [canonical_team_name(e) for e in (equipes_copa or []) if canonical_team_name(e)]
        self.grupos_copa = grupos_copa.copy() if isinstance(grupos_copa, pd.DataFrame) and not grupos_copa.empty else None
        if self.grupos_copa is not None and "Equipe" in self.grupos_copa.columns:
            self.grupos_copa["Equipe"] = self.grupos_copa["Equipe"].apply(canonical_team_name)
            if "Grupo" in self.grupos_copa.columns:
                self.grupos_copa["Grupo"] = self.grupos_copa["Grupo"].astype(str).str.upper().str.strip()

        self.jogos_copa = jogos_copa.copy() if isinstance(jogos_copa, pd.DataFrame) and not jogos_copa.empty else None
        if self.jogos_copa is not None:
            if "Time_A" in self.jogos_copa.columns:
                self.jogos_copa["Time_A"] = self.jogos_copa["Time_A"].apply(canonical_team_name)
            if "Time_B" in self.jogos_copa.columns:
                self.jogos_copa["Time_B"] = self.jogos_copa["Time_B"].apply(canonical_team_name)
            if "Grupo" in self.jogos_copa.columns:
                self.jogos_copa["Grupo"] = self.jogos_copa["Grupo"].astype(str).str.upper().str.strip()
            if "Fase" in self.jogos_copa.columns:
                self.jogos_copa["Fase"] = self.jogos_copa["Fase"].astype(str).str.strip()

        self.classificados_copa = classificados_copa.copy() if isinstance(classificados_copa, pd.DataFrame) and not classificados_copa.empty else None
        if self.classificados_copa is not None and "Equipe" in self.classificados_copa.columns:
            self.classificados_copa["Equipe"] = self.classificados_copa["Equipe"].apply(canonical_team_name)

    def _forca_time(self, equipe: str) -> float:
        return self.base._forca_time(equipe)

    def _equipes(self) -> List[str]:
        if self.poisson_model.df_team_stats is None:
            raise ValueError("Modelo ainda não treinado.")

        model_teams = set(self.poisson_model.df_team_stats["Equipe"].astype(str))

        if self.grupos_copa is not None and "Equipe" in self.grupos_copa.columns:
            equipes = [e for e in self.grupos_copa["Equipe"].astype(str).tolist() if e in model_teams]
            if len(equipes) >= 4:
                return list(dict.fromkeys(equipes))

        if self.equipes_copa:
            equipes = [e for e in self.equipes_copa if e in model_teams]
            if len(equipes) >= 4:
                return list(dict.fromkeys(equipes))

        teams = self.poisson_model.df_team_stats.sort_values("Elo", ascending=False)["Equipe"].tolist()
        return teams[:48] if len(teams) >= 48 else teams

    def _grupos_serpentina(self, equipes: List[str]) -> Dict[str, List[str]]:
        equipes = sorted(equipes, key=self._forca_time, reverse=True)
        n_grupos = min(12, max(1, math.ceil(len(equipes) / 4)))
        grupos = {chr(ord("A") + i): [] for i in range(n_grupos)}
        idx, step = 0, 1
        for e in equipes:
            grupos[chr(ord("A") + idx)].append(e)
            idx += step
            if idx >= n_grupos:
                idx, step = n_grupos - 1, -1
            elif idx < 0:
                idx, step = 0, 1
        return {g: t for g, t in grupos.items() if t}

    def _simular_placar_neutro(self, a: str, b: str) -> Tuple[int, int]:
        pab = self.poisson_model.prever_partida(a, b)["matriz"].rename(columns={"Gols_Mandante": "Gols_A", "Gols_Visitante": "Gols_B", "Probabilidade": "P1"})
        pba = self.poisson_model.prever_partida(b, a)["matriz"].rename(columns={"Gols_Mandante": "Gols_B", "Gols_Visitante": "Gols_A", "Probabilidade": "P2"})
        m = pd.merge(pab, pba, on=["Gols_A", "Gols_B"], how="inner")
        m["P"] = (m["P1"] + m["P2"]) / 2
        m["P"] = m["P"] / m["P"].sum()
        row = m.loc[self.rng.choice(m.index.to_numpy(), p=m["P"].to_numpy())]
        return int(row["Gols_A"]), int(row["Gols_B"])

    @staticmethod
    def _slot_placeholder(valor: Any) -> bool:
        """Detecta slots de mata-mata da FIFA: 1H, 3CEFHI, W73, RU101 etc."""
        s = str(valor or "").strip().upper().replace(" ", "")
        return bool(re.fullmatch(r"(?:[123][A-L]{1,5}|W\d+|RU\d+|TBD)", s))

    @staticmethod
    def _status_jogo_concluido(valor: Any) -> bool:
        s = normalizar_texto(valor or "")
        return any(tok in s for tok in ["ft", "final", "complete", "completed", "encerrado", "aet", "pen"])

    def _tabela_inicial_grupo(self, grupo: str, equipes: List[str]) -> Tuple[Dict[str, Dict[str, Any]], bool]:
        """Usa a classificação FIFA atual como ponto de partida, quando ela existe."""
        tabela = {
            e: {
                "Equipe": e,
                "Pontos": 0,
                "Jogos": 0,
                "Vitorias": 0,
                "Empates": 0,
                "Derrotas": 0,
                "GP": 0,
                "GC": 0,
                "SG": 0,
                "Forca": self._forca_time(e),
            }
            for e in equipes
        }
        usou_estado_real = False
        if self.grupos_copa is None or "Equipe" not in self.grupos_copa.columns:
            return tabela, usou_estado_real

        gdf = self.grupos_copa.copy()
        if "Grupo" in gdf.columns:
            gdf = gdf[gdf["Grupo"].astype(str).str.upper().str.strip() == str(grupo).upper().strip()]
        gdf = gdf[gdf["Equipe"].astype(str).isin(equipes)]
        if gdf.empty:
            return tabela, usou_estado_real

        for _, r in gdf.iterrows():
            e = str(r.get("Equipe", ""))
            if e not in tabela:
                continue
            for col, default in [
                ("Pontos", 0), ("Jogos", 0), ("Vitorias", 0), ("Empates", 0),
                ("Derrotas", 0), ("GP", 0), ("GC", 0), ("SG", 0),
            ]:
                if col in gdf.columns:
                    val = pd.to_numeric(r.get(col, default), errors="coerce")
                    tabela[e][col] = int(val) if pd.notna(val) else default
            if "SG" not in gdf.columns or pd.isna(pd.to_numeric(r.get("SG", np.nan), errors="coerce")):
                tabela[e]["SG"] = tabela[e]["GP"] - tabela[e]["GC"]
            if tabela[e]["Jogos"] > 0 or tabela[e]["Pontos"] > 0 or tabela[e]["GP"] > 0 or tabela[e]["GC"] > 0:
                usou_estado_real = True
        return tabela, usou_estado_real

    def _jogos_reais_grupo(self, grupo: str, equipes: List[str]) -> pd.DataFrame:
        if self.jogos_copa is None or self.jogos_copa.empty:
            return pd.DataFrame()
        jogos = self.jogos_copa.copy()
        if "Time_A" not in jogos.columns or "Time_B" not in jogos.columns:
            return pd.DataFrame()
        jogos = jogos[jogos["Time_A"].astype(str).isin(equipes) & jogos["Time_B"].astype(str).isin(equipes)]
        if "Grupo" in jogos.columns:
            jogos = jogos[(jogos["Grupo"].astype(str).str.upper().str.strip() == str(grupo).upper().strip()) | jogos["Grupo"].astype(str).str.strip().eq("")]
        if "Fase" in jogos.columns:
            fase_norm = jogos["Fase"].astype(str).apply(normalizar_texto)
            jogos = jogos[fase_norm.str.contains("first_stage|grupo|group", regex=True, na=False) | fase_norm.eq("")]
        return jogos.reset_index(drop=True)

    def _aplicar_resultado_tabela(self, tabela: Dict[str, Dict[str, Any]], a: str, b: str, ga: int, gb: int) -> None:
        for team, gf, gc in [(a, ga, gb), (b, gb, ga)]:
            tabela[team]["Jogos"] += 1
            tabela[team]["GP"] += gf
            tabela[team]["GC"] += gc
            tabela[team]["SG"] = tabela[team]["GP"] - tabela[team]["GC"]
        if ga > gb:
            tabela[a]["Pontos"] += 3
            tabela[a]["Vitorias"] += 1
            tabela[b]["Derrotas"] += 1
        elif gb > ga:
            tabela[b]["Pontos"] += 3
            tabela[b]["Vitorias"] += 1
            tabela[a]["Derrotas"] += 1
        else:
            tabela[a]["Pontos"] += 1
            tabela[b]["Pontos"] += 1
            tabela[a]["Empates"] += 1
            tabela[b]["Empates"] += 1

    def _simular_grupo(self, grupo: str, equipes: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simula apenas o que falta da fase de grupos.

        Se a classificação FIFA já tem pontos/jogos, ela vira o estado inicial.
        Jogos concluídos da aba copa2026_jogos_fifa entram no log, mas não são
        somados de novo quando a tabela oficial já traz esses pontos. Assim uma
        seleção já classificada não volta a ser tratada como grupo zerado.
        """
        tabela, usou_estado_real = self._tabela_inicial_grupo(grupo, equipes)
        jogos = []
        pares_concluidos = set()
        jogos_reais = self._jogos_reais_grupo(grupo, equipes)

        if not jogos_reais.empty:
            for _, r in jogos_reais.iterrows():
                a, b = str(r.get("Time_A", "")), str(r.get("Time_B", ""))
                if a not in tabela or b not in tabela:
                    continue
                ga = pd.to_numeric(r.get("Gols_A", np.nan), errors="coerce")
                gb = pd.to_numeric(r.get("Gols_B", np.nan), errors="coerce")
                status = r.get("Status", "")
                concluido = self._status_jogo_concluido(status) or (pd.notna(ga) and pd.notna(gb))
                if concluido:
                    pares_concluidos.add(tuple(sorted([a, b])))
                    ga_i = int(ga) if pd.notna(ga) else 0
                    gb_i = int(gb) if pd.notna(gb) else 0
                    if not usou_estado_real:
                        self._aplicar_resultado_tabela(tabela, a, b, ga_i, gb_i)
                    jogos.append({
                        "Grupo": grupo, "Time_A": a, "Time_B": b,
                        "Gols_A": ga_i, "Gols_B": gb_i,
                        "Status": "Real/Concluído", "Fonte": r.get("Fonte", "copa2026_jogos_fifa"),
                    })

        for i in range(len(equipes)):
            for j in range(i + 1, len(equipes)):
                a, b = equipes[i], equipes[j]
                par = tuple(sorted([a, b]))
                if par in pares_concluidos:
                    continue
                # Se todos já têm 3 jogos na tabela oficial, o grupo está encerrado.
                if usou_estado_real and tabela[a].get("Jogos", 0) >= 3 and tabela[b].get("Jogos", 0) >= 3:
                    continue
                ga, gb = self._simular_placar_neutro(a, b)
                self._aplicar_resultado_tabela(tabela, a, b, ga, gb)
                jogos.append({"Grupo": grupo, "Time_A": a, "Time_B": b, "Gols_A": ga, "Gols_B": gb, "Status": "Simulado", "Fonte": "modelo"})

        tab = pd.DataFrame(tabela.values())
        tab["Grupo"] = grupo
        tab = tab.sort_values(["Pontos", "SG", "GP", "Forca"], ascending=[False, False, False, False]).reset_index(drop=True)
        tab.insert(0, "Posicao_Grupo", range(1, len(tab) + 1))
        return tab, pd.DataFrame(jogos)

    def simular_uma_copa(self) -> Dict[str, Any]:
        if self.grupos_copa is not None and "Grupo" in self.grupos_copa.columns and "Equipe" in self.grupos_copa.columns:
            grupos = {
                str(grupo).upper(): gdf["Equipe"].astype(str).tolist()
                for grupo, gdf in self.grupos_copa.groupby("Grupo")
            }
        else:
            grupos = self._grupos_serpentina(self._equipes())
        tabs, jogos = [], []
        for g, teams in grupos.items():
            tab, jog = self._simular_grupo(g, teams)
            tabs.append(tab); jogos.append(jog)
        df_tabs = pd.concat(tabs, ignore_index=True)
        df_jogos = pd.concat(jogos, ignore_index=True) if jogos else pd.DataFrame()

        first_second = df_tabs[df_tabs["Posicao_Grupo"].isin([1, 2])]
        thirds = df_tabs[df_tabs["Posicao_Grupo"] == 3].sort_values(["Pontos", "SG", "GP", "Forca"], ascending=[False, False, False, False]).head(8)
        classificados = pd.concat([first_second, thirds], ignore_index=True)
        classificados = classificados.sort_values(["Pontos", "SG", "GP", "Forca"], ascending=[False, False, False, False]).reset_index(drop=True)
        classificados.insert(0, "Seed_Mata_Mata", range(1, len(classificados) + 1))

        participantes = classificados["Equipe"].tolist()
        mata = []
        fase = 1
        while len(participantes) > 1:
            next_round = []
            if len(participantes) % 2 == 1:
                bye = max(participantes, key=self._forca_time)
                participantes.remove(bye)
                next_round.append(bye)
            for i in range(0, len(participantes), 2):
                a, b = participantes[i], participantes[i + 1]
                vencedor = self.base.simular_confronto(a, b)
                mata.append({"Fase": fase, "Time_A": a, "Time_B": b, "Vencedor": vencedor})
                next_round.append(vencedor)
            participantes = next_round
            fase += 1

        return {"campeao": participantes[0], "classificados": classificados, "tabela_grupos": df_tabs, "jogos_grupos": df_jogos, "mata_mata": pd.DataFrame(mata)}

    def simular_campeao_formato_copa(self, iteracoes: int = 5000) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        equipes = self._equipes()
        titles = {e: 0 for e in equipes}
        quals = {e: 0 for e in equipes}
        last = None
        for _ in range(iteracoes):
            copa = self.simular_uma_copa()
            last = copa
            titles[copa["campeao"]] += 1
            for e in copa["classificados"]["Equipe"].tolist():
                if e in quals:
                    quals[e] += 1

        res = pd.DataFrame([
            {"Equipe": e, "Prob_Classificar_Mata_Mata_%": quals[e] / iteracoes * 100, "Titulos_Simulados": titles[e], "Prob_Titulo_%": titles[e] / iteracoes * 100, "Forca_Modelo": self._forca_time(e)}
            for e in equipes
        ]).sort_values("Prob_Titulo_%", ascending=False).reset_index(drop=True)
        res.insert(0, "Rank", range(1, len(res) + 1))
        for c in ["Prob_Classificar_Mata_Mata_%", "Prob_Titulo_%"]:
            res[c] = res[c].round(2)
        res["Forca_Modelo"] = res["Forca_Modelo"].round(4)
        return res, last


# ============================================================
# 7. CONFIGURAÇÃO CUSTOMIZADA DE COPA
# ============================================================

class TournamentConfigLoader:
    def __init__(self):
        self.grupos: Optional[pd.DataFrame] = None
        self.mata_mata: Optional[pd.DataFrame] = None

    def carregar(self, caminho: str) -> Dict[str, pd.DataFrame]:
        if not os.path.exists(caminho):
            raise FileNotFoundError("Arquivo de configuração não encontrado.")

        abas = pd.read_excel(caminho, sheet_name=None)
        if not abas:
            raise ValueError("A planilha de configuração está vazia.")

        nome_grupos = None
        nome_mata = None
        for nome in abas:
            n = normalizar_texto(nome)
            if "grupo" in n or "group" in n:
                nome_grupos = nome
            if "mata" in n or "bracket" in n or "chave" in n or "knockout" in n:
                nome_mata = nome

        if nome_grupos is None:
            raise ValueError("Não encontrei a aba de grupos.")
        if nome_mata is None:
            raise ValueError("Não encontrei a aba de mata-mata/chaveamento.")

        self.grupos = self._padronizar_grupos(abas[nome_grupos])
        self.mata_mata = self._padronizar_mata(abas[nome_mata])
        return {"grupos": self.grupos, "mata_mata": self.mata_mata}

    def _padronizar_grupos(self, df: pd.DataFrame) -> pd.DataFrame:
        cg = encontrar_coluna(df, ["grupo", "group"])
        ce = encontrar_coluna(df, ["equipe", "time", "team", "selecao", "seleção"])
        cs = encontrar_coluna(df, ["seed", "pote", "ranking"])
        if cg is None or ce is None:
            raise ValueError("A aba grupos precisa ter Grupo e Equipe.")

        out = pd.DataFrame()
        out["Grupo"] = df[cg].astype(str).str.strip().str.upper()
        out["Equipe"] = df[ce].astype(str).str.strip()
        out["Seed"] = to_number(df[cs], 99).astype(int) if cs else 99
        out = out[out["Grupo"].ne("") & out["Equipe"].ne("") & out["Equipe"].str.lower().ne("nan")]
        if out.empty:
            raise ValueError("A aba grupos não possui equipes válidas.")
        return out.reset_index(drop=True)

    def _ordem_fase(self, fase: str) -> int:
        f = normalizar_texto(fase)
        if "32" in f or "r32" in f:
            return 1
        if "16" in f or "r16" in f or "oitava" in f:
            return 2
        if "quarta" in f or "qf" in f:
            return 3
        if "semi" in f or "sf" in f:
            return 4
        if "final" in f:
            return 5
        return 99

    def _padronizar_mata(self, df: pd.DataFrame) -> pd.DataFrame:
        cf = encontrar_coluna(df, ["fase", "round"])
        cj = encontrar_coluna(df, ["jogo", "match", "id"])
        ca = encontrar_coluna(df, ["slot_a", "time_a", "equipe_a", "mandante"])
        cb = encontrar_coluna(df, ["slot_b", "time_b", "equipe_b", "visitante"])
        if cf is None or cj is None or ca is None or cb is None:
            raise ValueError("A aba mata_mata precisa ter Fase, Jogo, Slot_A e Slot_B.")

        out = pd.DataFrame()
        out["Fase"] = df[cf].astype(str).str.strip()
        out["Jogo"] = df[cj].astype(str).str.strip()
        out["Slot_A"] = df[ca].astype(str).str.strip().str.upper()
        out["Slot_B"] = df[cb].astype(str).str.strip().str.upper()
        out["Ordem_Fase"] = out["Fase"].apply(self._ordem_fase)
        out = out[out["Jogo"].ne("") & out["Slot_A"].ne("") & out["Slot_B"].ne("")]
        return out.sort_values(["Ordem_Fase", "Jogo"]).reset_index(drop=True)

    def resumo(self) -> str:
        if self.grupos is None or self.mata_mata is None:
            return "Nenhuma configuração carregada."
        return (
            "CONFIGURAÇÃO DE TORNEIO CARREGADA\n"
            "========================================\n\n"
            f"Equipes nos grupos: {len(self.grupos)}\n"
            f"Grupos: {self.grupos['Grupo'].nunique()}\n"
            f"Jogos de mata-mata: {len(self.mata_mata)}\n\n"
            "GRUPOS\n========================================\n"
            + self.grupos.to_string(index=False)
            + "\n\nMATA-MATA\n========================================\n"
            + self.mata_mata[["Fase", "Jogo", "Slot_A", "Slot_B"]].to_string(index=False)
        )


class ConfiguredWorldCupSimulator(WorldCupFormatSimulator):
    def __init__(self, poisson_model: BasePoissonModel, tournament_config: Dict[str, pd.DataFrame], ml_model: Optional[MLMatchOutcomeModel] = None, ensemble_model: Optional[EnsemblePredictionModel] = None, random_state: int = 42):
        super().__init__(poisson_model, ml_model, ensemble_model, random_state)
        self.config = tournament_config

    def _simular_fase_grupos_config(self):
        tabs, jogos = [], []
        for grupo, df_g in self.config["grupos"].sort_values(["Grupo", "Seed"]).groupby("Grupo"):
            teams = df_g["Equipe"].tolist()
            tab, jog = self._simular_grupo(grupo, teams)
            tabs.append(tab)
            jogos.append(jog)

        df_tabs = pd.concat(tabs, ignore_index=True)
        df_jogos = pd.concat(jogos, ignore_index=True) if jogos else pd.DataFrame()

        first_second = df_tabs[df_tabs["Posicao_Grupo"].isin([1, 2])]
        thirds = df_tabs[df_tabs["Posicao_Grupo"] == 3].sort_values(["Pontos", "SG", "GP", "Forca"], ascending=[False, False, False, False]).head(8)
        classificados = pd.concat([first_second, thirds], ignore_index=True)
        classificados = classificados.sort_values(["Pontos", "SG", "GP", "Forca"], ascending=[False, False, False, False]).reset_index(drop=True)
        classificados.insert(0, "Seed_Mata_Mata", range(1, len(classificados) + 1))

        slots = {}
        for _, r in df_tabs.iterrows():
            g, p, e = str(r["Grupo"]).upper(), int(r["Posicao_Grupo"]), r["Equipe"]
            slots[f"{g}{p}"] = e
            slots[f"{p}{g}"] = e

        thirds_rank = thirds.reset_index(drop=True)
        for i, (_, r) in enumerate(thirds_rank.iterrows(), start=1):
            slots[f"T{i}"] = r["Equipe"]
            slots[f"BEST3_{i}"] = r["Equipe"]
            slots[f"{str(r['Grupo']).upper()}3"] = r["Equipe"]
            slots[f"3{str(r['Grupo']).upper()}"] = r["Equipe"]

        for _, r in classificados.iterrows():
            slots[f"S{int(r['Seed_Mata_Mata'])}"] = r["Equipe"]

        return classificados, df_tabs, df_jogos, slots

    def _resolver_slot(self, slot: str, slots: Dict[str, str], winners: Dict[str, str]) -> str:
        s = str(slot).strip().upper().replace(" ", "")
        if s.startswith("W_"):
            return winners[s[2:]]
        if s.startswith("W") and s[1:] in winners:
            return winners[s[1:]]
        if "/" in s:
            for part in s.split("/"):
                if part in slots:
                    return slots[part]
            raise ValueError(f"Slot não resolvido: {slot}")
        if s not in slots:
            raise ValueError(f"Slot não encontrado: {slot}")
        return slots[s]

    def simular_uma_copa(self) -> Dict[str, Any]:
        classificados, tabs, jogos, slots = self._simular_fase_grupos_config()
        winners = {}
        mata_rows = []

        for _, r in self.config["mata_mata"].sort_values(["Ordem_Fase", "Jogo"]).iterrows():
            jogo = str(r["Jogo"]).strip().upper()
            a = self._resolver_slot(r["Slot_A"], slots, winners)
            b = self._resolver_slot(r["Slot_B"], slots, winners)
            vencedor = self.base.simular_confronto(a, b)
            winners[jogo] = vencedor
            mata_rows.append({"Fase": r["Fase"], "Jogo": jogo, "Slot_A": r["Slot_A"], "Slot_B": r["Slot_B"], "Time_A": a, "Time_B": b, "Vencedor": vencedor})

        mata = pd.DataFrame(mata_rows)
        if mata.empty:
            raise ValueError("Nenhum jogo de mata-mata foi simulado.")

        return {"campeao": mata.iloc[-1]["Vencedor"], "classificados": classificados, "tabela_grupos": tabs, "jogos_grupos": jogos, "mata_mata": mata}

    def simular(self, iteracoes: int = 5000) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        equipes = self.config["grupos"]["Equipe"].tolist()
        titles = {e: 0 for e in equipes}
        quals = {e: 0 for e in equipes}
        last = None
        for _ in range(iteracoes):
            copa = self.simular_uma_copa()
            last = copa
            titles[copa["campeao"]] += 1
            for e in copa["classificados"]["Equipe"].tolist():
                if e in quals:
                    quals[e] += 1
        res = pd.DataFrame([
            {"Equipe": e, "Prob_Classificar_Mata_Mata_%": quals[e] / iteracoes * 100, "Titulos_Simulados": titles[e], "Prob_Titulo_%": titles[e] / iteracoes * 100, "Forca_Modelo": self._forca_time(e)}
            for e in equipes
        ]).sort_values("Prob_Titulo_%", ascending=False).reset_index(drop=True)
        res.insert(0, "Rank", range(1, len(res) + 1))
        res["Prob_Classificar_Mata_Mata_%"] = res["Prob_Classificar_Mata_Mata_%"].round(2)
        res["Prob_Titulo_%"] = res["Prob_Titulo_%"].round(2)
        res["Forca_Modelo"] = res["Forca_Modelo"].round(4)
        return res, last


# ============================================================
# 8. VALIDAÇÃO TEMPORAL
# ============================================================

class ModelBacktester:
    labels = ["H", "D", "A"]

    @staticmethod
    def _resultado(x: int, y: int) -> str:
        if x > y:
            return "H"
        if x == y:
            return "D"
        return "A"

    @staticmethod
    def _norm(arr: List[float]) -> np.ndarray:
        a = np.array(arr, dtype=float)
        a = np.clip(a, 1e-9, 1.0)
        return a / a.sum() if a.sum() > 0 else np.array([1/3, 1/3, 1/3])

    def _metricas(self, df: pd.DataFrame, prefix: str, name: str) -> Dict[str, Any]:
        cols = [f"{prefix}_H", f"{prefix}_D", f"{prefix}_A"]
        d = df[df["Valida"] == True].dropna(subset=cols)
        if d.empty:
            return {"Modelo": name, "Partidas_Avaliadas": 0, "Acuracia_%": np.nan, "Log_Loss": np.nan, "Brier_Score": np.nan}
        correct = 0
        losses, briers = [], []
        for _, r in d.iterrows():
            p = self._norm([r[cols[0]], r[cols[1]], r[cols[2]]])
            real = r["Resultado_Real"]
            idx = self.labels.index(real)
            pred = self.labels[int(np.argmax(p))]
            correct += int(pred == real)
            y = np.zeros(3); y[idx] = 1
            losses.append(-np.log(p[idx]))
            briers.append(np.sum((p - y) ** 2))
        n = len(d)
        return {"Modelo": name, "Partidas_Avaliadas": n, "Acuracia_%": round(correct / n * 100, 2), "Log_Loss": round(float(np.mean(losses)), 4), "Brier_Score": round(float(np.mean(briers)), 4)}

    def validar(self, df_matches: pd.DataFrame, min_treino: int = 80, max_partidas_teste: int = 150, usar_ml: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = df_matches.sort_values("date").reset_index(drop=True)
        if len(df) < min_treino + 10:
            raise ValueError("Base pequena demais para validação temporal.")

        indices = list(range(min_treino, len(df)))
        if len(indices) > max_partidas_teste:
            indices = indices[-max_partidas_teste:]

        rows = []
        for idx in indices:
            match = df.iloc[idx]
            train = df.iloc[:idx].copy()
            base_row = {
                "Data": match["date"], "Mandante": match["home_team"], "Visitante": match["away_team"],
                "Gols_Mandante": int(match["home_goals"]), "Gols_Visitante": int(match["away_goals"]),
                "Resultado_Real": self._resultado(int(match["home_goals"]), int(match["away_goals"])),
                "stat_H": np.nan, "stat_D": np.nan, "stat_A": np.nan,
                "ml_H": np.nan, "ml_D": np.nan, "ml_A": np.nan,
                "ensemble_H": np.nan, "ensemble_D": np.nan, "ensemble_A": np.nan,
                "Pred_Stat": None, "Pred_ML": None, "Pred_Ensemble": None,
                "Peso_ML_Ensemble": np.nan, "ML_Treinado": False, "Valida": False, "Erro": "",
            }
            try:
                elo = DynamicEloModel().treinar(train)
                pm = BasePoissonModel().treinar(train, elo)
                ml_model = MLMatchOutcomeModel()
                ml_pred = None
                if usar_ml:
                    try:
                        ml_model.treinar(train, pm, elo)
                        ml_pred = ml_model.prever_partida(match["home_team"], match["away_team"])
                        base_row["ML_Treinado"] = True
                    except Exception:
                        ml_pred = None

                stat = pm.prever_partida(match["home_team"], match["away_team"])
                ens = EnsemblePredictionModel().combinar(stat, ml_pred)

                vals = {
                    "stat": [stat["prob_mandante"]/100, stat["prob_empate"]/100, stat["prob_visitante"]/100],
                    "ensemble": [ens["ensemble_prob_mandante"]/100, ens["ensemble_prob_empate"]/100, ens["ensemble_prob_visitante"]/100],
                }
                if ml_pred:
                    vals["ml"] = [ml_pred["ml_prob_mandante"]/100, ml_pred["ml_prob_empate"]/100, ml_pred["ml_prob_visitante"]/100]

                for pref, probs in vals.items():
                    p = self._norm(probs)
                    base_row[f"{pref}_H"], base_row[f"{pref}_D"], base_row[f"{pref}_A"] = p
                    base_row[f"Pred_{'Stat' if pref=='stat' else 'ML' if pref=='ml' else 'Ensemble'}"] = self.labels[int(np.argmax(p))]
                base_row["Peso_ML_Ensemble"] = ens["peso_ml"]
                base_row["Valida"] = True
            except Exception as e:
                base_row["Erro"] = str(e)
            rows.append(base_row)

        pred = pd.DataFrame(rows)
        resumo = pd.DataFrame([
            self._metricas(pred, "stat", "Estatístico"),
            self._metricas(pred, "ml", "Machine Learning"),
            self._metricas(pred, "ensemble", "Ensemble Final"),
        ])
        return resumo, pred


# ============================================================
# 9. INTERFACE GRÁFICA
# ============================================================
