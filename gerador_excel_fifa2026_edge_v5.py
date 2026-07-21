# -*- coding: utf-8 -*-
"""
GERADOR LOCAL DE EXCEL - FIFA 2026 / HISTÓRICO DESDE 2018 - v5

Roda no Windows com Microsoft Edge visível via Selenium.

O programa gera um Excel único para ser importado depois no app FIFA 2026 Analytics Engine.

Abas principais geradas:
    - base_historica_2018
    - relatorio_mestre_equipes
    - player_stats_consolidado
    - copa2026_classificacao_atual
    - copa2026_equipes_oficiais
    - copa2026_jogos_fifa
    - copa2026_classificados_atuais
    - fifa_log

Dependências:
    pip install pandas requests beautifulsoup4 lxml openpyxl selenium webdriver-manager
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
import queue
import traceback
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext
except Exception:
    tk = None

from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.microsoft import EdgeChromiumDriverManager


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

URLS_FIFA = {
    "scores_fixtures": "https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/scores-fixtures?country=BR&wtw-filter=ALL",
    "match_schedule_article": "https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/match-schedule-fixtures-results-teams-stadiums",
    "teams": "https://www.fifa.com/pt/tournaments/mens/worldcup/canadamexicousa2026/teams",
    "team_statistics": "https://www.fifa.com/pt/tournaments/mens/worldcup/canadamexicousa2026/statistics/team-statistics",
    "player_statistics": "https://www.fifa.com/pt/tournaments/mens/worldcup/canadamexicousa2026/statistics/player-statistics",
    "statistics_home": "https://www.fifa.com/pt/tournaments/mens/worldcup/canadamexicousa2026/statistics",
    "power_rankings": "https://www.fifa.com/pt/tournaments/mens/worldcup/canadamexicousa2026/power-rankings",
    "standings": "https://www.fifa.com/pt/tournaments/mens/worldcup/canadamexicousa2026/standings",
    # Páginas oficiais de Copas anteriores. A FIFA pode redirecionar/alterar estrutura.
    "wc2022_scores_fixtures": "https://www.fifa.com/en/tournaments/mens/worldcup/qatar2022/scores-fixtures",
    "wc2018_scores_fixtures": "https://www.fifa.com/en/tournaments/mens/worldcup/2018russia/scores-fixtures",
}

# Fonte pública estável para base histórica de partidas internacionais.
# Não é API oficial da FIFA. É usada para montar base de treino padronizada desde 2018.
HISTORICO_INTERNACIONAL_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

XPATHS_TEAM_ABAS = {
    "Ataque": "/html/body/div[1]/div/div[2]/div/main/div/div[1]/div/div/div/div/section/div/div[1]/div/div/div/div[1]/button/span",
    "Distribuição": "/html/body/div[1]/div/div[2]/div/main/div/div[1]/div/div/div/div/section/div/div[1]/div/div/div/div[2]/button/span",
    "Defesa": "/html/body/div[1]/div/div[2]/div/main/div/div[1]/div/div/div/div/section/div/div[1]/div/div/div/div[3]/button/span",
    "Disciplina": "/html/body/div[1]/div/div[2]/div/main/div/div[1]/div/div/div/div/section/div/div[1]/div/div/div/div[4]/button/span",
    "Goleiro": "/html/body/div[1]/div/div[2]/div/main/div/div[1]/div/div/div/div/section/div/div[1]/div/div/div/div[5]/button/span",
    "Movimentação": "/html/body/div[1]/div/div[2]/div/main/div/div[1]/div/div/div/div/section/div/div[1]/div/div/div/div[6]/button/span",
    "Físico": "/html/body/div[1]/div/div[2]/div/main/div/div[1]/div/div/div/div/section/div/div[1]/div/div/div/div[7]/button/span",
}

XPATHS_PLAYER_ABAS = {
    "O Artilheiro": "/html/body/div[1]/div/div[2]/div/main/div/div[1]/div/div/div/div/section/div/div[1]/div/div/div/div[1]/button/span",
    "Ataque": "/html/body/div[1]/div/div[2]/div/main/div/div[1]/div/div/div/div/section/div/div[1]/div/div/div/div[2]/button/span",
    "Distribuição": "/html/body/div[1]/div/div[2]/div/main/div/div[1]/div/div/div/div/section/div/div[1]/div/div/div/div[3]/button/span",
    "Defesa": "/html/body/div[1]/div/div[2]/div/main/div/div[1]/div/div/div/div/section/div/div[1]/div/div/div/div[4]/button/span",
    "Disciplina": "/html/body/div[1]/div/div[2]/div/main/div/div[1]/div/div/div/div/section/div/div[1]/div/div/div/div[5]/button/span",
    "Goleiro": "/html/body/div[1]/div/div[2]/div/main/div/div[1]/div/div/div/div/section/div/div[1]/div/div/div/div[6]/button/span",
    "Movimentação": "/html/body/div[1]/div/div[2]/div/main/div/div[1]/div/div/div/div/section/div/div[1]/div/div/div/div[7]/button/span",
    "Físico": "/html/body/div[1]/div/div[2]/div/main/div/div[1]/div/div/div/div/section/div/div[1]/div/div/div/div[8]/button/span",
}


# =============================================================================
# LOG
# =============================================================================

@dataclass
class Logger:
    q: Optional[queue.Queue] = None
    lines: Optional[List[str]] = None

    def __post_init__(self):
        if self.lines is None:
            self.lines = []

    def log(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.lines.append(line)
        if self.q is not None:
            self.q.put(line)


# =============================================================================
# UTILITÁRIOS
# =============================================================================

def limpar_texto(x: Any) -> str:
    if x is None:
        return ""
    txt = str(x).replace("\xa0", " ").replace("\n", " ").strip()
    txt = " ".join(txt.split())
    return txt


def normalizar_coluna(c: Any) -> str:
    txt = limpar_texto(c)
    txt = re.sub(r"^Unnamed:\s*\d+_level_\d+$", "", txt, flags=re.I)
    return limpar_texto(txt)


def normalizar_time(valor: Any) -> str:
    txt = limpar_texto(valor)
    txt = txt.replace("*", "").strip()
    # Remove trechos comuns grudados pela renderização: "Argentina ARG" vira "Argentina ARG";
    # mantemos porque às vezes a sigla ajuda. A coluna Seleção limpa é tratada depois.
    return txt


def to_numero(valor: Any) -> Any:
    if pd.isna(valor):
        return valor
    if isinstance(valor, (int, float)):
        return valor
    txt = limpar_texto(valor)
    txt = txt.replace("%", "").replace("−", "-")
    # Padrão BR/PT: 60,48 -> 60.48
    if re.search(r"\d,\d", txt):
        txt = txt.replace(".", "").replace(",", ".")
    # Remove textos residuais não numéricos quando a célula é só número com símbolo.
    cand = re.sub(r"[^0-9.\-]", "", txt)
    if cand in {"", ".", "-", "-."}:
        return valor
    try:
        if "." in cand:
            return float(cand)
        return int(cand)
    except Exception:
        return valor


def safe_sheet_name(name: str, used: set) -> str:
    raw = limpar_texto(name) or "sheet"
    raw = re.sub(r"[\\/*?:\[\]]", "_", raw)[:31]
    if not raw:
        raw = "sheet"
    base = raw
    i = 1
    while raw in used:
        suffix = f"_{i}"
        raw = (base[:31 - len(suffix)] + suffix)[:31]
        i += 1
    used.add(raw)
    return raw


def limpar_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalizar_coluna(c) or f"col_{i+1}" for i, c in enumerate(out.columns)]
    # Evita colunas duplicadas
    counts = {}
    new_cols = []
    for c in out.columns:
        counts[c] = counts.get(c, 0) + 1
        new_cols.append(c if counts[c] == 1 else f"{c}_{counts[c]}")
    out.columns = new_cols
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].map(limpar_texto)
    return out


def encontrar_coluna(df: pd.DataFrame, termos: List[str]) -> Optional[str]:
    for c in df.columns:
        cl = limpar_texto(c).lower()
        for termo in termos:
            if termo.lower() in cl:
                return c
    return None


def chave_alias_nome(txt: Any) -> str:
    """Normaliza chave de seleção para cruzar nomes em PT/EN com a base histórica."""
    s = limpar_texto(txt).lower()
    mapa = {
        "á": "a", "à": "a", "â": "a", "ã": "a",
        "é": "e", "ê": "e", "í": "i",
        "ó": "o", "ô": "o", "õ": "o",
        "ú": "u", "ü": "u", "ç": "c",
    }
    for a, b in mapa.items():
        s = s.replace(a, b)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


PT_TEAM_ALIASES = {
    "mexico": "Mexico",
    "africa_do_sul": "South Africa",
    "republica_da_coreia": "South Korea",
    "coreia_do_sul": "South Korea",
    "tchequia": "Czech Republic",
    "republica_tcheca": "Czech Republic",
    "suica": "Switzerland",
    "canada": "Canada",
    "bosnia_e_herzegovina": "Bosnia and Herzegovina",
    "catar": "Qatar",
    "qatar": "Qatar",
    "brasil": "Brazil",
    "marrocos": "Morocco",
    "escocia": "Scotland",
    "haiti": "Haiti",
    "eua": "United States",
    "estados_unidos": "United States",
    "australia": "Australia",
    "paraguai": "Paraguay",
    "turquia": "Turkey",
    "alemanha": "Germany",
    "costa_do_marfim": "Ivory Coast",
    "equador": "Ecuador",
    "curacau": "Curaçao",
    "curacao": "Curaçao",
    "holanda": "Netherlands",
    "paises_baixos": "Netherlands",
    "japao": "Japan",
    "suecia": "Sweden",
    "tunisia": "Tunisia",
    "egito": "Egypt",
    "ri_do_ira": "Iran",
    "ira": "Iran",
    "irao": "Iran",
    "iran": "Iran",
    "belgica": "Belgium",
    "nova_zelandia": "New Zealand",
    "espanha": "Spain",
    "uruguai": "Uruguay",
    "cabo_verde": "Cape Verde",
    "arabia_saudita": "Saudi Arabia",
    "franca": "France",
    "noruega": "Norway",
    "senegal": "Senegal",
    "iraque": "Iraq",
    "argentina": "Argentina",
    "austria": "Austria",
    "argelia": "Algeria",
    "jordania": "Jordan",
    "colombia": "Colombia",
    "portugal": "Portugal",
    "rd_do_congo": "DR Congo",
    "republica_democratica_do_congo": "DR Congo",
    "uzbequistao": "Uzbekistan",
    "inglaterra": "England",
    "gana": "Ghana",
    "croacia": "Croatia",
    "panama": "Panama",
}


def canonicalizar_nome_selecao(nome: Any) -> str:
    s = limpar_texto(nome)
    if not s:
        return ""
    return PT_TEAM_ALIASES.get(chave_alias_nome(s), s)


def extrair_nome_codigo_selecao(txt: Any) -> Tuple[str, str]:
    """Separa casos como 'MéxicoMEX' em ('Mexico', 'MEX')."""
    s = limpar_texto(txt)
    if not s:
        return "", ""
    s = re.sub(r"^\d+\s+", "", s).strip()
    s = s.replace("*", "").strip()

    codigo = ""
    nome = s

    # Caso típico FIFA em PT: BrasilBRA, República da CoreiaKOR, RI do IrãIRN.
    m = re.match(r"^(.+?)([A-Z]{3})$", s)
    if m and len(m.group(1).strip()) >= 2:
        nome = m.group(1).strip()
        codigo = m.group(2).strip()
    else:
        # Caso com espaço antes da sigla: Brazil BRA.
        m = re.match(r"^(.+?)\s+([A-Z]{3})$", s)
        if m and len(m.group(1).strip()) >= 2:
            nome = m.group(1).strip()
            codigo = m.group(2).strip()

    nome = canonicalizar_nome_selecao(nome)
    return nome, codigo


def reduzir_nome_selecao(txt: Any) -> str:
    """Retorna nome canônico da seleção para cruzar com a base histórica."""
    nome, _codigo = extrair_nome_codigo_selecao(txt)
    return nome


# =============================================================================
# EDGE / SELENIUM
# =============================================================================

def criar_driver(log: Logger) -> webdriver.Edge:
    options = webdriver.EdgeOptions()
    # Não usar headless. A FIFA tende a bloquear/limitar renderização sem navegador visível.
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")
    options.add_argument(f"user-agent={USER_AGENT}")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    log.log("Abrindo Microsoft Edge local via Selenium em modo visível.")
    driver = webdriver.Edge(service=Service(EdgeChromiumDriverManager().install()), options=options)
    try:
        driver.maximize_window()
    except Exception:
        pass
    return driver


def aceitar_cookies(driver: webdriver.Edge, log: Logger) -> None:
    textos = [
        "Accept", "Accept All", "I Accept", "Agree", "Allow all",
        "Aceitar", "Aceitar todos", "Concordo", "Permitir todos", "OK"
    ]
    for txt in textos:
        try:
            xpath = (
                f"//button[contains(translate(normalize-space(.), "
                f"'ABCDEFGHIJKLMNOPQRSTUVWXYZÁÀÂÃÉÊÍÓÔÕÚÇ', "
                f"'abcdefghijklmnopqrstuvwxyzáàâãéêíóôõúç'), "
                f"'{txt.lower()}')]"
            )
            btns = driver.find_elements(By.XPATH, xpath)
            for b in btns[:3]:
                if b.is_displayed() and b.is_enabled():
                    driver.execute_script("arguments[0].click();", b)
                    log.log(f"Cookie/consentimento clicado: {txt}")
                    time.sleep(1)
                    return
        except Exception:
            pass


def rolar_pagina(driver: webdriver.Edge, pausas: int = 8, intervalo: float = 0.7) -> None:
    try:
        altura_ant = 0
        for _ in range(pausas):
            driver.execute_script("window.scrollBy(0, Math.floor(window.innerHeight * 0.85));")
            time.sleep(intervalo)
            altura = driver.execute_script("return window.scrollY + window.innerHeight")
            if altura == altura_ant:
                break
            altura_ant = altura
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(0.5)
    except Exception:
        pass


def esperar_tabela_ou_corpo(driver: webdriver.Edge, timeout: int = 35) -> None:
    wait = WebDriverWait(driver, timeout)
    try:
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    except Exception:
        return
    try:
        WebDriverWait(driver, 12).until(EC.presence_of_element_located((By.TAG_NAME, "table")))
    except Exception:
        # Algumas páginas não têm table real; capturamos texto bruto mesmo assim.
        pass


def clicar_aba_por_xpath(driver: webdriver.Edge, wait: WebDriverWait, xpath_exato: str, nome_aba: str, log: Logger) -> bool:
    try:
        elemento_span = wait.until(EC.presence_of_element_located((By.XPATH, xpath_exato)))
        try:
            botao = elemento_span.find_element(By.XPATH, "..")
        except Exception:
            botao = elemento_span
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", botao)
        time.sleep(0.7)
        driver.execute_script("arguments[0].click();", botao)
        time.sleep(3.5)
        return True
    except Exception as e:
        log.log(f"Aviso: falha ao abrir aba {nome_aba}: {str(e)[:180]}")
        return False


def extrair_primeira_tabela(driver: webdriver.Edge) -> Optional[pd.DataFrame]:
    soup = BeautifulSoup(driver.page_source, "html.parser")
    tabela_html = soup.find("table")
    if not tabela_html:
        return None
    try:
        df = pd.read_html(io.StringIO(str(tabela_html)))[0]
        return limpar_df(df)
    except Exception:
        return None


def extrair_todas_tabelas(driver: webdriver.Edge) -> List[pd.DataFrame]:
    try:
        dfs = pd.read_html(io.StringIO(driver.page_source))
        return [limpar_df(df) for df in dfs if isinstance(df, pd.DataFrame) and not df.empty]
    except Exception:
        return []


def texto_visivel(driver: webdriver.Edge) -> str:
    try:
        body = driver.find_element(By.TAG_NAME, "body")
        return limpar_texto(body.text)
    except Exception:
        return ""


def _debug_dir_fifa() -> Path:
    """Pasta local para auditoria das páginas sem tabela DOM."""
    path = Path.cwd() / "debug_fifa_pages"
    path.mkdir(parents=True, exist_ok=True)
    return path


def salvar_debug_pagina(driver: webdriver.Edge, nome: str, log: Logger) -> Dict[str, str]:
    """
    Salva HTML/TXT/PNG para investigar quando a FIFA não expõe <table>.
    Esses arquivos podem registrar conteúdo exibido no navegador. A pasta é
    ignorada pelo Git e deve ser revisada antes de compartilhamento manual.
    Não interrompe o fluxo se screenshot ou escrita falhar.
    """
    debug_dir = _debug_dir_fifa()
    safe = re.sub(r"[^a-zA-Z0-9_\-]+", "_", nome)[:80] or "pagina"
    html_path = debug_dir / f"{safe}.html"
    txt_path = debug_dir / f"{safe}.txt"
    png_path = debug_dir / f"{safe}.png"

    try:
        html_path.write_text(driver.page_source or "", encoding="utf-8", errors="ignore")
    except Exception as e:
        log.log(f"Aviso: não foi possível salvar HTML debug de {nome}: {str(e)[:120]}")
    try:
        txt = driver.execute_script("return document.body ? document.body.innerText : ''") or ""
        txt_path.write_text(txt, encoding="utf-8", errors="ignore")
    except Exception as e:
        log.log(f"Aviso: não foi possível salvar TXT debug de {nome}: {str(e)[:120]}")
    try:
        driver.save_screenshot(str(png_path))
    except Exception:
        pass

    return {
        "debug_html": str(html_path),
        "debug_txt": str(txt_path),
        "debug_png": str(png_path),
    }


def esperar_renderizacao_basica(driver: webdriver.Edge, timeout: int = 25, min_chars: int = 400) -> None:
    """
    Espera o corpo da página ganhar texto. Isso é mais adequado que esperar <table>
    em páginas modernas da FIFA, porque scores/teams podem vir como cards.
    """
    start = time.time()
    ultimo_len = -1
    estavel = 0
    while time.time() - start < timeout:
        try:
            texto = driver.execute_script("return document.body ? document.body.innerText : ''") or ""
            tamanho = len(texto.strip())
            if tamanho >= min_chars:
                if abs(tamanho - ultimo_len) < 40:
                    estavel += 1
                else:
                    estavel = 0
                ultimo_len = tamanho
                if estavel >= 2:
                    return
        except Exception:
            pass
        time.sleep(1)


def texto_em_chunks(driver: webdriver.Edge, nome: str, max_chars: int = 1800) -> pd.DataFrame:
    """Converte texto visível em blocos auditáveis no Excel."""
    try:
        bruto = driver.execute_script("return document.body ? document.body.innerText : ''") or ""
    except Exception:
        bruto = texto_visivel(driver)

    bruto = bruto.replace("\xa0", " ").replace("\r", "\n")
    linhas = [limpar_texto(x) for x in bruto.split("\n")]
    linhas = [x for x in linhas if x]

    chunks: List[str] = []
    atual = ""
    for linha in linhas:
        if not atual:
            atual = linha
        elif len(atual) + len(linha) + 1 <= max_chars:
            atual += " | " + linha
        else:
            chunks.append(atual)
            atual = linha
    if atual:
        chunks.append(atual)

    if not chunks and bruto.strip():
        palavras = bruto.split()
        step = 140
        chunks = [" ".join(palavras[i:i + step]) for i in range(0, len(palavras), step)]

    return pd.DataFrame({
        "pagina": nome,
        "ordem": range(1, len(chunks) + 1),
        "trecho": chunks,
    })


def extrair_links_pagina(driver: webdriver.Edge, nome: str, limite: int = 1200) -> pd.DataFrame:
    registros = []
    vistos = set()
    try:
        links = driver.find_elements(By.TAG_NAME, "a")
    except Exception:
        links = []

    for a in links:
        try:
            texto = limpar_texto(a.text)
            href = a.get_attribute("href") or ""
            if not href:
                continue
            chave = (texto, href)
            if chave in vistos:
                continue
            vistos.add(chave)
            registros.append({"pagina": nome, "texto": texto, "url": href})
            if len(registros) >= limite:
                break
        except Exception:
            continue
    return pd.DataFrame(registros)


def extrair_blocos_visiveis(driver: webdriver.Edge, nome: str, limite: int = 1800) -> pd.DataFrame:
    """
    Fallback para componentes/cards sem <table>.
    Extrai blocos semânticos e elimina duplicatas muito óbvias.
    """
    soup = BeautifulSoup(driver.page_source or "", "html.parser")
    seletores = ["article", "section", "li", "a", "button", "h1", "h2", "h3"]
    registros: List[Dict[str, Any]] = []
    vistos = set()

    for seletor in seletores:
        for el in soup.select(seletor):
            texto = limpar_texto(el.get_text(" ", strip=True))
            if len(texto) < 18:
                continue
            if len(texto) > 3500:
                texto = texto[:3500]
            chave = re.sub(r"\W+", "", texto.lower())[:500]
            if chave in vistos:
                continue
            vistos.add(chave)
            href = el.get("href", "") if el.name == "a" else ""
            registros.append({
                "pagina": nome,
                "tipo_bloco": seletor,
                "texto": texto,
                "link": href,
            })
            if len(registros) >= limite:
                return pd.DataFrame(registros)

    return pd.DataFrame(registros)


def _valor_escalar_json(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool)) or v is None


def _achatar_json_escalar(obj: Any, prefixo: str = "", nivel: int = 0, limite_nivel: int = 2) -> Dict[str, Any]:
    """Achata campos escalares de um dict JSON para linhas de Excel."""
    out: Dict[str, Any] = {}
    if not isinstance(obj, dict):
        return out
    for k, v in obj.items():
        k2 = re.sub(r"[^a-zA-Z0-9_]+", "_", str(k)).strip("_")[:60] or "campo"
        nome = f"{prefixo}_{k2}" if prefixo else k2
        if _valor_escalar_json(v):
            if isinstance(v, str) and len(v) > 500:
                v = v[:500]
            out[nome[:120]] = v
        elif isinstance(v, dict) and nivel < limite_nivel:
            out.update(_achatar_json_escalar(v, nome, nivel + 1, limite_nivel))
    return out


def _coletar_registros_json(obj: Any, caminho: str, saida: List[Dict[str, Any]], limite: int = 2500) -> None:
    if len(saida) >= limite:
        return

    termos_interesse = (
        "match", "fixture", "score", "team", "teams", "country", "stadium",
        "date", "group", "stage", "home", "away", "competitor", "winner",
        "player", "ranking", "standing", "result", "name", "code"
    )

    if isinstance(obj, dict):
        flat = _achatar_json_escalar(obj)
        chaves = " ".join(flat.keys()).lower()
        if len(flat) >= 2 and any(t in chaves for t in termos_interesse):
            row = {"json_path": caminho}
            row.update(flat)
            saida.append(row)
            if len(saida) >= limite:
                return
        for k, v in obj.items():
            _coletar_registros_json(v, f"{caminho}.{k}" if caminho else str(k), saida, limite)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if len(saida) >= limite:
                return
            _coletar_registros_json(item, f"{caminho}[{i}]", saida, limite)


def extrair_json_embutido(driver: webdriver.Edge, nome: str, log: Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tenta ler JSON embutido em páginas Next.js/FIFA.
    Retorna: registros extraídos + índice de scripts encontrados.
    """
    soup = BeautifulSoup(driver.page_source or "", "html.parser")
    registros: List[Dict[str, Any]] = []
    scripts_idx: List[Dict[str, Any]] = []

    for idx, script in enumerate(soup.find_all("script"), start=1):
        sid = script.get("id", "")
        stype = script.get("type", "")
        txt = script.string if script.string is not None else script.get_text("", strip=False)
        txt = txt or ""
        amostra = limpar_texto(txt[:600])
        if txt:
            scripts_idx.append({
                "pagina": nome,
                "script_index": idx,
                "id": sid,
                "type": stype,
                "tamanho_chars": len(txt),
                "amostra": amostra,
            })

        candidato = txt.strip()
        if not candidato:
            continue
        # JSON puro: __NEXT_DATA__ ou scripts application/json.
        if sid == "__NEXT_DATA__" or "json" in stype.lower() or candidato[:1] in {"{", "["}:
            try:
                obj = json.loads(candidato)
            except Exception:
                continue
            try:
                _coletar_registros_json(obj, sid or f"script_{idx}", registros)
            except Exception as e:
                log.log(f"Aviso: falha ao achatar JSON em {nome}: {str(e)[:120]}")

    df_json = pd.DataFrame(registros)
    df_scripts = pd.DataFrame(scripts_idx)
    return df_json, df_scripts


def normalizar_colunas_para_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Evita colunas gigantes/duplicadas causadas por JSON/capturas genéricas."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    cols = []
    usados = {}
    for c in out.columns:
        c2 = limpar_texto(c)
        c2 = re.sub(r"[\r\n\t]+", " ", c2)
        c2 = c2[:120] or "coluna"
        usados[c2] = usados.get(c2, 0) + 1
        cols.append(c2 if usados[c2] == 1 else f"{c2}_{usados[c2]}")
    out.columns = cols
    return out


# =============================================================================
# EXTRAÇÕES ESPECÍFICAS
# =============================================================================

def extrair_historico_partidas_desde_2018(log: Logger) -> pd.DataFrame:
    """
    Baixa base histórica pública de partidas internacionais e padroniza para o app.
    Observação: fonte pública estável, não endpoint oficial FIFA.
    """
    log.log("Baixando histórico de partidas internacionais desde 2018 para base de treino.")
    try:
        resp = requests.get(HISTORICO_INTERNACIONAL_URL, timeout=45, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"] >= pd.Timestamp("2018-01-01")].copy()

        out = pd.DataFrame()
        out["date"] = df["date"]
        out["home_team"] = df["home_team"].astype(str).map(limpar_texto)
        out["away_team"] = df["away_team"].astype(str).map(limpar_texto)
        out["home_goals"] = pd.to_numeric(df["home_score"], errors="coerce")
        out["away_goals"] = pd.to_numeric(df["away_score"], errors="coerce")
        out["competition"] = df.get("tournament", "International").astype(str).map(limpar_texto)
        out["city"] = df.get("city", "").astype(str).map(limpar_texto) if "city" in df.columns else ""
        out["country"] = df.get("country", "").astype(str).map(limpar_texto) if "country" in df.columns else ""
        out["neutral"] = df.get("neutral", "").astype(str).map(limpar_texto) if "neutral" in df.columns else ""
        out["home_xg"] = out["home_goals"]
        out["away_xg"] = out["away_goals"]
        out["xg_origem"] = "aproximado_por_gols"
        out["fonte"] = "international_results_public_dataset"

        out = out.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
        out["home_goals"] = out["home_goals"].astype(int)
        out["away_goals"] = out["away_goals"].astype(int)
        out = out.sort_values("date").reset_index(drop=True)
        log.log(f"Histórico padronizado: {len(out)} partidas desde 2018.")
        return out
    except Exception as e:
        log.log(f"Falha ao baixar histórico: {e}")
        return pd.DataFrame(columns=[
            "date", "home_team", "away_team", "home_goals", "away_goals",
            "competition", "home_xg", "away_xg", "xg_origem", "fonte"
        ])


def extrair_team_statistics(driver: webdriver.Edge, log: Logger) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    url = URLS_FIFA["team_statistics"]
    log.log("Acessando estatísticas de equipe da FIFA.")
    driver.get(url)
    esperar_tabela_ou_corpo(driver, 40)
    aceitar_cookies(driver, log)
    rolar_pagina(driver, pausas=3)

    wait = WebDriverWait(driver, 30)
    dados_consolidados: Dict[str, Dict[str, Any]] = {}
    abas: Dict[str, pd.DataFrame] = {}

    for nome_aba, xpath_exato in XPATHS_TEAM_ABAS.items():
        log.log(f"Raspando métricas de equipe: {nome_aba}")
        if nome_aba != "Ataque":
            clicar_aba_por_xpath(driver, wait, xpath_exato, nome_aba, log)
        else:
            # Mesmo na primeira aba, tentamos clicar se possível para garantir estado.
            clicar_aba_por_xpath(driver, wait, xpath_exato, nome_aba, log)

        df_tabela = extrair_primeira_tabela(driver)
        if df_tabela is None or df_tabela.empty:
            log.log(f"Tabela vazia/não encontrada na aba de equipe: {nome_aba}")
            continue

        abas[f"team_stats_{nome_aba}"] = df_tabela.copy()
        nome_coluna_time = encontrar_coluna(df_tabela, ["equipa", "equipe", "sele", "team", "país", "pais", "country"])
        if not nome_coluna_time:
            nome_coluna_time = df_tabela.columns[1] if len(df_tabela.columns) > 1 else df_tabela.columns[0]

        for _, linha in df_tabela.iterrows():
            time_nome = reduzir_nome_selecao(linha.get(nome_coluna_time, ""))
            if not time_nome or time_nome.lower() in {"nan", "equipa", "equipe", "team"}:
                continue

            if time_nome not in dados_consolidados:
                dados_consolidados[time_nome] = {"Seleção": time_nome}

            for col in df_tabela.columns:
                col_clean = limpar_texto(col)
                if col == nome_coluna_time:
                    continue
                if "posição" in col_clean.lower() or "posicao" in col_clean.lower() or "position" in col_clean.lower():
                    continue
                if "unnamed" in col_clean.lower():
                    continue
                valor_bruto = to_numero(linha.get(col))
                nome_coluna_formatada = f"[{nome_aba}] {col_clean}"
                dados_consolidados[time_nome][nome_coluna_formatada] = valor_bruto

                low = col_clean.lower()
                if nome_aba == "Ataque" and low in {"gols", "goals"}:
                    dados_consolidados[time_nome]["Gols_Feitos"] = valor_bruto
                if nome_aba == "Defesa" and ("gols sofridos" in low or "conceded" in low or "goals conceded" in low):
                    dados_consolidados[time_nome]["Gols_Sofridos"] = valor_bruto

    df_final = pd.DataFrame(list(dados_consolidados.values()))
    if df_final.empty:
        df_final = pd.DataFrame(columns=["Seleção", "Gols_Feitos", "Gols_Sofridos", "xG_Ataque", "Rating_Fifa"])
    if "Gols_Feitos" not in df_final.columns:
        df_final["Gols_Feitos"] = 0
    if "Gols_Sofridos" not in df_final.columns:
        df_final["Gols_Sofridos"] = 0

    df_final["Gols_Feitos"] = pd.to_numeric(df_final["Gols_Feitos"], errors="coerce").fillna(0)
    df_final["Gols_Sofridos"] = pd.to_numeric(df_final["Gols_Sofridos"], errors="coerce").fillna(0)
    df_final["xG_Ataque"] = df_final["Gols_Feitos"] * 0.95
    df_final["Rating_Fifa"] = 1000
    df_final = df_final.sort_values("Seleção").reset_index(drop=True)
    log.log(f"Relatório mestre de equipes: {len(df_final)} seleção(ões), {len(df_final.columns)} coluna(s).")
    return df_final, abas


def extrair_player_statistics(driver: webdriver.Edge, log: Logger) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    url = URLS_FIFA["player_statistics"]
    log.log("Acessando estatísticas de jogador da FIFA.")
    driver.get(url)
    esperar_tabela_ou_corpo(driver, 40)
    aceitar_cookies(driver, log)
    rolar_pagina(driver, pausas=3)

    wait = WebDriverWait(driver, 30)
    abas: Dict[str, pd.DataFrame] = {}
    consolidados = []

    for nome_aba, xpath_exato in XPATHS_PLAYER_ABAS.items():
        log.log(f"Raspando métricas de jogador: {nome_aba}")
        if nome_aba != "O Artilheiro":
            clicar_aba_por_xpath(driver, wait, xpath_exato, nome_aba, log)
        else:
            clicar_aba_por_xpath(driver, wait, xpath_exato, nome_aba, log)

        df_tabela = extrair_primeira_tabela(driver)
        if df_tabela is None or df_tabela.empty:
            log.log(f"Tabela vazia/não encontrada na aba de jogador: {nome_aba}")
            continue

        df_tabela.insert(0, "Aba", nome_aba)
        abas[f"player_stats_{nome_aba}"] = df_tabela.copy()
        consolidados.append(df_tabela)

    df_all = pd.concat(consolidados, ignore_index=True, sort=False) if consolidados else pd.DataFrame()
    log.log(f"Estatísticas de jogadores consolidadas: {len(df_all)} linha(s).")
    return df_all, abas


def extrair_pagina_generica(driver: webdriver.Edge, nome: str, url: str, log: Logger) -> Dict[str, pd.DataFrame]:
    """
    Extração robusta para páginas FIFA modernas.

    Primeiro tenta tabelas DOM. Quando a página não expõe <table>, não retorna vazio:
    salva texto, cards/blocos, links, JSON embutido e arquivos de debug.
    """
    log.log(f"Abrindo página FIFA: {nome}")
    driver.get(url)
    esperar_tabela_ou_corpo(driver, 35)
    aceitar_cookies(driver, log)
    esperar_renderizacao_basica(driver, timeout=20)
    rolar_pagina(driver, pausas=12, intervalo=0.9)
    esperar_renderizacao_basica(driver, timeout=8, min_chars=200)

    out: Dict[str, pd.DataFrame] = {}

    tabelas = extrair_todas_tabelas(driver)
    for i, df in enumerate(tabelas, start=1):
        out[f"dom_{nome}_{i}"] = normalizar_colunas_para_excel(df)

    # Sempre salva texto bruto, porque ajuda a auditar páginas com DOM dinâmico.
    df_texto = texto_em_chunks(driver, nome)
    if not df_texto.empty:
        out[f"page_text_{nome}"] = df_texto

    # Fallbacks quando não houver table ou quando a página for card-based.
    df_blocos = extrair_blocos_visiveis(driver, nome)
    if not df_blocos.empty:
        out[f"page_cards_{nome}"] = df_blocos

    df_links = extrair_links_pagina(driver, nome)
    if not df_links.empty:
        out[f"page_links_{nome}"] = df_links

    df_json, df_scripts = extrair_json_embutido(driver, nome, log)
    if not df_json.empty:
        out[f"page_json_{nome}"] = normalizar_colunas_para_excel(df_json)
    if not df_scripts.empty:
        out[f"page_scripts_{nome}"] = normalizar_colunas_para_excel(df_scripts)

    debug_info = salvar_debug_pagina(driver, nome, log)
    out[f"debug_{nome}"] = pd.DataFrame([{
        "pagina": nome,
        "url": url,
        "tabelas_dom": len(tabelas),
        "blocos_cards": len(df_blocos),
        "links": len(df_links),
        "json_registros": len(df_json),
        **debug_info,
        "observacao": "Se tabelas_dom=0, a página provavelmente renderiza dados como cards/JSON dinâmico, não como <table> HTML.",
    }])

    if len(tabelas) == 0:
        log.log(
            f"{nome}: 0 tabela(s) DOM; fallback gerado com "
            f"{len(df_blocos)} bloco(s), {len(df_links)} link(s), {len(df_json)} registro(s) JSON."
        )
    else:
        log.log(
            f"{nome}: {len(tabelas)} tabela(s) DOM capturada(s); "
            f"fallback/auditoria também salvo."
        )
    return out


def montar_copa2026_jogos(tabelas_scores: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Monta a aba de jogos.

    Prioridade: tabelas DOM de scores_fixtures.
    Fallback: JSON/cards/texto da página scores_fixtures e do artigo oficial de calendário.
    Assim a aba não fica vazia quando a FIFA não usa <table>.
    """
    frames = []

    prioridades = [
        "dom_scores_fixtures",
        "page_json_scores_fixtures",
        "page_cards_scores_fixtures",
        "page_text_scores_fixtures",
        "dom_match_schedule_article",
        "page_json_match_schedule_article",
        "page_cards_match_schedule_article",
        "page_text_match_schedule_article",
    ]

    for prefixo in prioridades:
        for name, df in tabelas_scores.items():
            if not name.startswith(prefixo):
                continue
            if df is None or df.empty:
                continue
            d = df.copy()
            d.insert(0, "Fonte_Tabela", name)
            d.insert(1, "Tipo_Extracao", "DOM" if name.startswith("dom_") else "FALLBACK")
            frames.append(d)

        # Se achou DOM real de scores_fixtures, não mistura fallback mais ruidoso.
        if prefixo == "dom_scores_fixtures" and frames:
            return pd.concat(frames, ignore_index=True, sort=False)

    if frames:
        return pd.concat(frames, ignore_index=True, sort=False)

    return pd.DataFrame(columns=[
        "Fonte_Tabela", "Tipo_Extracao", "observacao"
    ])


def _grupo_da_tabela_standings(name: str, df: pd.DataFrame) -> str:
    """Identifica o grupo pelo cabeçalho 'Grupo B' ou pelo sufixo dom_standings_2."""
    for c in df.columns[:3]:
        m = re.search(r"grupo\s+([a-z])", limpar_texto(c), flags=re.I)
        if m:
            return m.group(1).upper()
    m = re.search(r"_(\d+)$", name)
    if m:
        return chr(ord("A") + int(m.group(1)) - 1)
    return name


def _linha_standings_fifa_grupo(row: pd.Series, cols: List[str], grupo: str, name: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Parser específico para tabelas FIFA renderizadas com cabeçalhos assim:
    Grupo B | Grupo B.1 | Grupo B.2 | Unnamed: 3 ... Unnamed: 12

    Mapeamento observado na FIFA:
      Grupo X.1 = posição
      Grupo X.2 = seleção grudada com código FIFA, ex.: BrasilBRA
      Unnamed: 3 = J  jogos disputados
      Unnamed: 4 = C  vitórias
      Unnamed: 5 = E  empates
      Unnamed: 6 = D  derrotas
      Unnamed: 7 = M  gols pró
      Unnamed: 8 = S  gols contra
      Unnamed: 9 = DG saldo de gols
      Unnamed: 10 = PCE pontuação de conduta
      Unnamed: 11 = Pts pontos
      Unnamed: 12 = últimos resultados visual/textual, pode vir '--'
    """
    if len(cols) < 12:
        return None

    col_pos = cols[1]
    col_team = cols[2]
    nome, codigo = extrair_nome_codigo_selecao(row.get(col_team, ""))
    if not nome or nome.lower() in {"nan", "equipa", "equipe", "team"}:
        return None

    return {
        "Grupo": grupo,
        "Posicao_Grupo": pd.to_numeric(row.get(col_pos, idx + 1), errors="coerce"),
        "Equipe": nome,
        "Codigo_FIFA": codigo,
        "Jogos": pd.to_numeric(row.get(cols[3], 0), errors="coerce"),
        "Vitorias": pd.to_numeric(row.get(cols[4], 0), errors="coerce"),
        "Empates": pd.to_numeric(row.get(cols[5], 0), errors="coerce"),
        "Derrotas": pd.to_numeric(row.get(cols[6], 0), errors="coerce"),
        "GP": pd.to_numeric(row.get(cols[7], 0), errors="coerce"),
        "GC": pd.to_numeric(row.get(cols[8], 0), errors="coerce"),
        "SG": pd.to_numeric(row.get(cols[9], 0), errors="coerce"),
        "PCE": pd.to_numeric(row.get(cols[10], 0), errors="coerce"),
        "Pontos": pd.to_numeric(row.get(cols[11], 0), errors="coerce"),
        "Ultimos_Resultados": limpar_texto(row.get(cols[12], "")) if len(cols) > 12 else "",
        "Fonte_Tabela": name,
    }


def montar_standings(tabelas: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Transforma as tabelas DOM de standings da FIFA no formato usado pelo simulador."""
    linhas = []
    for name, df in tabelas.items():
        if not name.startswith("dom_standings"):
            continue
        d = limpar_df(df.copy())
        if d.empty:
            continue

        grupo = _grupo_da_tabela_standings(name, d)
        cols = list(d.columns)

        layout_grupo_fifa = (
            len(cols) >= 12
            and re.search(r"grupo\s+[a-z]", limpar_texto(cols[0]), flags=re.I)
            and re.search(r"grupo\s+[a-z]\.1", limpar_texto(cols[1]), flags=re.I)
            and re.search(r"grupo\s+[a-z]\.2", limpar_texto(cols[2]), flags=re.I)
        )

        if layout_grupo_fifa:
            for idx, row in d.iterrows():
                item = _linha_standings_fifa_grupo(row, cols, grupo, name, idx)
                if item is not None:
                    linhas.append(item)
            continue

        # Fallback genérico, para caso a FIFA mude o cabeçalho.
        col_team = encontrar_coluna(d, ["equipa", "equipe", "sele", "team", "country", "país", "pais"])
        if not col_team:
            col_team = cols[2] if len(cols) > 2 else (cols[1] if len(cols) > 1 else cols[0])
        col_pos = encontrar_coluna(d, ["posição", "posicao", "position", "pos"]) or (cols[1] if len(cols) > 1 else None)
        col_pts = encontrar_coluna(d, ["pts", "pontos", "points"])
        col_jogos = encontrar_coluna(d, ["j", "jogos", "played", "pld", "mp"])
        col_sg = encontrar_coluna(d, ["sg", "saldo", "gd", "+/-", "goal difference"])
        col_gp = encontrar_coluna(d, ["gp", "gols pró", "gols pro", "gf", "goals for"])

        for idx, row in d.iterrows():
            nome, codigo = extrair_nome_codigo_selecao(row.get(col_team, ""))
            if not nome or nome.lower() in {"nan", "equipa", "equipe", "team"}:
                continue
            if len(nome) < 3:
                continue
            linhas.append({
                "Grupo": grupo,
                "Posicao_Grupo": pd.to_numeric(row.get(col_pos, idx + 1), errors="coerce") if col_pos else idx + 1,
                "Equipe": nome,
                "Codigo_FIFA": codigo,
                "Jogos": pd.to_numeric(row.get(col_jogos, 0), errors="coerce") if col_jogos else 0,
                "Vitorias": 0,
                "Empates": 0,
                "Derrotas": 0,
                "GP": pd.to_numeric(row.get(col_gp, 0), errors="coerce") if col_gp else 0,
                "GC": 0,
                "SG": pd.to_numeric(row.get(col_sg, 0), errors="coerce") if col_sg else 0,
                "PCE": 0,
                "Pontos": pd.to_numeric(row.get(col_pts, 0), errors="coerce") if col_pts else 0,
                "Ultimos_Resultados": "",
                "Fonte_Tabela": name,
            })

    cols_out = [
        "Grupo", "Posicao_Grupo", "Equipe", "Codigo_FIFA", "Jogos",
        "Vitorias", "Empates", "Derrotas", "GP", "GC", "SG", "PCE",
        "Pontos", "Ultimos_Resultados", "Fonte_Tabela"
    ]
    out = pd.DataFrame(linhas)
    if out.empty:
        return pd.DataFrame(columns=cols_out)

    out = out[cols_out]
    out["Posicao_Grupo"] = pd.to_numeric(out["Posicao_Grupo"], errors="coerce").fillna(0).astype(int)
    for c in ["Jogos", "Vitorias", "Empates", "Derrotas", "GP", "GC", "SG", "PCE", "Pontos"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    return out.drop_duplicates(subset=["Grupo", "Equipe"]).sort_values(["Grupo", "Posicao_Grupo"]).reset_index(drop=True)



def calcular_classificados(standings: pd.DataFrame) -> pd.DataFrame:
    if standings.empty or "Grupo" not in standings.columns or "Equipe" not in standings.columns:
        return pd.DataFrame()
    parts = []
    for grupo, gdf in standings.groupby("Grupo"):
        g = gdf.copy().sort_values(["Pontos", "SG", "GP", "Posicao_Grupo"], ascending=[False, False, False, True]).reset_index(drop=True)
        g["Posicao_Calculada"] = range(1, len(g) + 1)
        parts.append(g)
    ranked = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if ranked.empty:
        return pd.DataFrame()
    fs = ranked[ranked["Posicao_Calculada"].isin([1, 2])].copy()
    fs["Tipo_Classificacao"] = "1º/2º do grupo"
    thirds = ranked[ranked["Posicao_Calculada"] == 3].copy()
    thirds = thirds.sort_values(["Pontos", "SG", "GP"], ascending=[False, False, False]).head(8)
    thirds["Tipo_Classificacao"] = "melhor terceiro atual"
    return pd.concat([fs, thirds], ignore_index=True).reset_index(drop=True)


def montar_equipes_oficiais(relatorio_equipes: pd.DataFrame, standings: pd.DataFrame, tabelas_teams: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    equipes = []
    if isinstance(relatorio_equipes, pd.DataFrame) and not relatorio_equipes.empty and "Seleção" in relatorio_equipes.columns:
        equipes += relatorio_equipes["Seleção"].dropna().astype(str).map(reduzir_nome_selecao).tolist()
    if isinstance(standings, pd.DataFrame) and not standings.empty and "Equipe" in standings.columns:
        equipes += standings["Equipe"].dropna().astype(str).map(reduzir_nome_selecao).tolist()
    for name, df in tabelas_teams.items():
        if not name.startswith("dom_teams"):
            continue
        col = encontrar_coluna(df, ["equipa", "equipe", "sele", "team", "country", "país", "pais"])
        if col:
            equipes += df[col].dropna().astype(str).map(reduzir_nome_selecao).tolist()

    cleaned = []
    for e in equipes:
        e = reduzir_nome_selecao(e)
        if e and e.lower() not in {"nan", "team", "equipa", "equipe", "seleção", "selecao"} and len(e) >= 3:
            cleaned.append(e)
    out = pd.DataFrame({"Equipe": sorted(set(cleaned))})
    out["Fonte"] = "FIFA/Selenium local"
    return out


# =============================================================================
# MOTOR PRINCIPAL
# =============================================================================

def gerar_excel_fifa2026(caminho_saida: str, log: Optional[Logger] = None) -> str:
    log = log or Logger()
    inicio = time.time()
    caminho_saida = str(Path(caminho_saida).resolve())
    sheets: Dict[str, pd.DataFrame] = {}

    log.log("Iniciando geração do Excel FIFA 2026.")

    # 1) Histórico padronizado desde 2018, primeira aba para facilitar importação como base local.
    hist = extrair_historico_partidas_desde_2018(log)
    sheets["base_historica_2018"] = hist

    driver = None
    try:
        driver = criar_driver(log)

        # 2) Team statistics seguindo a lógica exata por abas/XPaths.
        relatorio_equipes, team_abas = extrair_team_statistics(driver, log)
        sheets["relatorio_mestre_equipes"] = relatorio_equipes
        sheets.update(team_abas)

        # 3) Player statistics seguindo XPaths por abas.
        player_all, player_abas = extrair_player_statistics(driver, log)
        sheets["player_stats_consolidado"] = player_all
        sheets.update(player_abas)

        # 4) Páginas gerais: scores, standings, teams, rankings, páginas antigas.
        gerais: Dict[str, pd.DataFrame] = {}
        for nome in [
            "scores_fixtures", "match_schedule_article", "teams", "standings", "statistics_home",
            "power_rankings", "wc2022_scores_fixtures", "wc2018_scores_fixtures"
        ]:
            try:
                gerais.update(extrair_pagina_generica(driver, nome, URLS_FIFA[nome], log))
            except Exception as e:
                log.log(f"Falha parcial em {nome}: {str(e)[:250]}")
        sheets.update(gerais)

        # 5) Abas compatíveis com o app.
        jogos = montar_copa2026_jogos(gerais)
        sheets["copa2026_jogos_fifa"] = jogos

        standings = montar_standings(gerais)
        sheets["copa2026_classificacao_atual"] = standings

        equipes_oficiais = montar_equipes_oficiais(relatorio_equipes, standings, gerais)
        sheets["copa2026_equipes_oficiais"] = equipes_oficiais

        classificados = calcular_classificados(standings)
        sheets["copa2026_classificados_atuais"] = classificados

    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass

    log.log(f"Gerando arquivo Excel: {caminho_saida}")
    sheets["fifa_log"] = pd.DataFrame({"log": log.lines})

    # Escreve Excel. Mantemos a primeira aba como base_historica_2018.
    used = set()
    with pd.ExcelWriter(caminho_saida, engine="openpyxl") as writer:
        ordered = ["base_historica_2018"] + [k for k in sheets.keys() if k != "base_historica_2018"]
        for name in ordered:
            df = sheets.get(name)
            if df is None:
                continue
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            # Excel não aceita timezone em datetime.
            df = df.copy()
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    try:
                        df[col] = df[col].dt.tz_localize(None)
                    except Exception:
                        pass
            sheet = safe_sheet_name(name, used)
            df.to_excel(writer, sheet_name=sheet, index=False)

    segundos = round(time.time() - inicio, 1)
    log.log(f"Arquivo gerado com sucesso em {segundos}s: {caminho_saida}")
    return caminho_saida


# =============================================================================
# INTERFACE DESKTOP TKINTER
# =============================================================================

class AppGerador:
    def __init__(self, root):
        self.root = root
        self.root.title("Gerador Excel FIFA 2026 - Edge/Selenium v5")
        self.root.geometry("920x620")
        self.root.minsize(780, 500)

        self.log_queue = queue.Queue()
        self.pasta_saida = tk.StringVar(value=os.getcwd())
        self.nome_arquivo = tk.StringVar(value="copa2026_fifa_local.xlsx")
        self.running = False

        frame_top = tk.Frame(root, padx=12, pady=10)
        frame_top.pack(fill="x")

        tk.Label(frame_top, text="Pasta de saída:").grid(row=0, column=0, sticky="w")
        tk.Entry(frame_top, textvariable=self.pasta_saida, width=78).grid(row=0, column=1, sticky="ew", padx=6)
        tk.Button(frame_top, text="Escolher...", command=self.escolher_pasta).grid(row=0, column=2)

        tk.Label(frame_top, text="Arquivo:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        tk.Entry(frame_top, textvariable=self.nome_arquivo, width=78).grid(row=1, column=1, sticky="ew", padx=6, pady=(8, 0))

        frame_top.grid_columnconfigure(1, weight=1)

        frame_btn = tk.Frame(root, padx=12, pady=6)
        frame_btn.pack(fill="x")
        self.btn_gerar = tk.Button(frame_btn, text="Gerar Excel local agora", command=self.iniciar, height=2, bg="#0b5", fg="white")
        self.btn_gerar.pack(side="left")
        tk.Button(frame_btn, text="Sair", command=root.destroy, height=2).pack(side="right")

        info = (
            "Este gerador abre o Microsoft Edge visível, clica nas abas de estatísticas da FIFA, "
            "extrai tabelas e também baixa a base histórica de partidas internacionais desde 2018.\n"
            "Não feche o Edge durante a extração. Aceite cookies se aparecer."
        )
        tk.Label(root, text=info, justify="left", anchor="w", padx=12).pack(fill="x")

        self.txt = scrolledtext.ScrolledText(root, wrap="word", height=24)
        self.txt.pack(fill="both", expand=True, padx=12, pady=10)
        self.log("Pronto. Clique em 'Gerar Excel local agora'.")
        self.root.after(200, self.processar_logs)

    def log(self, msg: str) -> None:
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")

    def escolher_pasta(self):
        pasta = filedialog.askdirectory(initialdir=self.pasta_saida.get() or os.getcwd())
        if pasta:
            self.pasta_saida.set(pasta)

    def iniciar(self):
        if self.running:
            return
        pasta = self.pasta_saida.get().strip()
        nome = self.nome_arquivo.get().strip() or "copa2026_fifa_local.xlsx"
        if not nome.lower().endswith(".xlsx"):
            nome += ".xlsx"
        Path(pasta).mkdir(parents=True, exist_ok=True)
        caminho = str(Path(pasta) / nome)

        self.running = True
        self.btn_gerar.config(state="disabled", text="Gerando...")
        self.log("=" * 90)
        self.log(f"Iniciando em thread. Saída: {caminho}")

        def worker():
            logger = Logger(q=self.log_queue)
            try:
                gerar_excel_fifa2026(caminho, logger)
                self.log_queue.put(f"FINAL_OK|{caminho}")
            except Exception as e:
                self.log_queue.put("ERRO_CRITICO|" + str(e))
                self.log_queue.put(traceback.format_exc())

        threading.Thread(target=worker, daemon=True).start()

    def processar_logs(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                if item.startswith("FINAL_OK|"):
                    caminho = item.split("|", 1)[1]
                    self.log(f"\n✅ Excel gerado: {caminho}")
                    self.running = False
                    self.btn_gerar.config(state="normal", text="Gerar Excel local agora")
                    try:
                        messagebox.showinfo("Concluído", f"Excel gerado com sucesso:\n{caminho}")
                    except Exception:
                        pass
                elif item.startswith("ERRO_CRITICO|"):
                    self.log("\n❌ Erro crítico: " + item.split("|", 1)[1])
                    self.running = False
                    self.btn_gerar.config(state="normal", text="Gerar Excel local agora")
                    try:
                        messagebox.showerror("Erro", item.split("|", 1)[1])
                    except Exception:
                        pass
                else:
                    self.log(item)
        except queue.Empty:
            pass
        self.root.after(200, self.processar_logs)


def main():
    if len(sys.argv) >= 2 and sys.argv[1].lower() in {"--cli", "cli"}:
        caminho = sys.argv[2] if len(sys.argv) >= 3 else str(Path.cwd() / "copa2026_fifa_local.xlsx")
        gerar_excel_fifa2026(caminho, Logger())
        return

    if tk is None:
        caminho = str(Path.cwd() / "copa2026_fifa_local.xlsx")
        gerar_excel_fifa2026(caminho, Logger())
        return

    root = tk.Tk()
    AppGerador(root)
    root.mainloop()


if __name__ == "__main__":
    main()
