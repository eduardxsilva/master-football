# -*- coding: utf-8 -*-
"""
Corrige um Excel FIFA já gerado, reconstruindo as abas:
- copa2026_classificacao_atual
- copa2026_classificados_atuais
- copa2026_equipes_oficiais

Uso:
    python corrigir_excel_fifa2026_standings.py entrada.xlsx saida_corrigida.xlsx
"""
from __future__ import annotations
import re, sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


def limpar_texto(x: Any) -> str:
    if x is None:
        return ""
    txt = str(x).replace("\xa0", " ").replace("\n", " ").strip()
    return " ".join(txt.split())


def normalizar_coluna(c: Any) -> str:
    txt = limpar_texto(c)
    txt = re.sub(r"^Unnamed:\s*\d+_level_\d+$", "", txt, flags=re.I)
    return limpar_texto(txt)


def limpar_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalizar_coluna(c) or f"col_{i+1}" for i, c in enumerate(out.columns)]
    return out


def chave_alias_nome(txt: Any) -> str:
    s = limpar_texto(txt).lower()
    for a, b in {"á":"a","à":"a","â":"a","ã":"a","é":"e","ê":"e","í":"i","ó":"o","ô":"o","õ":"o","ú":"u","ü":"u","ç":"c"}.items():
        s = s.replace(a, b)
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")

PT_TEAM_ALIASES = {
    "mexico":"Mexico","africa_do_sul":"South Africa","republica_da_coreia":"South Korea","coreia_do_sul":"South Korea",
    "tchequia":"Czech Republic","republica_tcheca":"Czech Republic","suica":"Switzerland","canada":"Canada",
    "bosnia_e_herzegovina":"Bosnia and Herzegovina","catar":"Qatar","qatar":"Qatar","brasil":"Brazil",
    "marrocos":"Morocco","escocia":"Scotland","haiti":"Haiti","eua":"United States","estados_unidos":"United States",
    "australia":"Australia","paraguai":"Paraguay","turquia":"Turkey","alemanha":"Germany","costa_do_marfim":"Ivory Coast",
    "equador":"Ecuador","curacau":"Curaçao","curacao":"Curaçao","holanda":"Netherlands","paises_baixos":"Netherlands",
    "japao":"Japan","suecia":"Sweden","tunisia":"Tunisia","egito":"Egypt","ri_do_ira":"Iran","ira":"Iran","iran":"Iran",
    "belgica":"Belgium","nova_zelandia":"New Zealand","espanha":"Spain","uruguai":"Uruguay","cabo_verde":"Cape Verde",
    "arabia_saudita":"Saudi Arabia","franca":"France","noruega":"Norway","senegal":"Senegal","iraque":"Iraq","argentina":"Argentina",
    "austria":"Austria","argelia":"Algeria","jordania":"Jordan","colombia":"Colombia","portugal":"Portugal","rd_do_congo":"DR Congo",
    "republica_democratica_do_congo":"DR Congo","uzbequistao":"Uzbekistan","inglaterra":"England","gana":"Ghana","croacia":"Croatia","panama":"Panama",
}

def canonicalizar_nome_selecao(nome: Any) -> str:
    s = limpar_texto(nome)
    return PT_TEAM_ALIASES.get(chave_alias_nome(s), s) if s else ""


def extrair_nome_codigo_selecao(txt: Any) -> Tuple[str, str]:
    s = limpar_texto(txt)
    if not s:
        return "", ""
    s = re.sub(r"^\d+\s+", "", s).replace("*", "").strip()
    nome, codigo = s, ""
    m = re.match(r"^(.+?)([A-Z]{3})$", s)
    if m and len(m.group(1).strip()) >= 2:
        nome, codigo = m.group(1).strip(), m.group(2).strip()
    else:
        m = re.match(r"^(.+?)\s+([A-Z]{3})$", s)
        if m and len(m.group(1).strip()) >= 2:
            nome, codigo = m.group(1).strip(), m.group(2).strip()
    return canonicalizar_nome_selecao(nome), codigo


def grupo_da_tabela(name: str, df: pd.DataFrame) -> str:
    for c in df.columns[:3]:
        m = re.search(r"grupo\s+([a-z])", limpar_texto(c), flags=re.I)
        if m:
            return m.group(1).upper()
    m = re.search(r"_(\d+)$", name)
    return chr(ord("A") + int(m.group(1)) - 1) if m else name


def montar_standings(sheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    linhas = []
    for name, df in sheets.items():
        if not name.startswith("dom_standings"):
            continue
        d = limpar_df(df)
        if d.empty:
            continue
        cols = list(d.columns)
        if len(cols) < 12:
            continue
        grupo = grupo_da_tabela(name, d)
        for idx, row in d.iterrows():
            equipe, codigo = extrair_nome_codigo_selecao(row.get(cols[2], ""))
            if not equipe:
                continue
            linhas.append({
                "Grupo": grupo,
                "Posicao_Grupo": pd.to_numeric(row.get(cols[1], idx + 1), errors="coerce"),
                "Equipe": equipe,
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
            })
    out = pd.DataFrame(linhas)
    if out.empty:
        return pd.DataFrame(columns=["Grupo","Posicao_Grupo","Equipe","Codigo_FIFA","Jogos","Vitorias","Empates","Derrotas","GP","GC","SG","PCE","Pontos","Ultimos_Resultados","Fonte_Tabela"])
    out["Posicao_Grupo"] = pd.to_numeric(out["Posicao_Grupo"], errors="coerce").fillna(0).astype(int)
    for c in ["Jogos","Vitorias","Empates","Derrotas","GP","GC","SG","PCE","Pontos"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    return out.sort_values(["Grupo", "Posicao_Grupo"]).reset_index(drop=True)


def calcular_classificados(standings: pd.DataFrame) -> pd.DataFrame:
    if standings.empty:
        return pd.DataFrame()
    parts = []
    for grupo, gdf in standings.groupby("Grupo"):
        g = gdf.sort_values(["Pontos", "SG", "GP", "Posicao_Grupo"], ascending=[False, False, False, True]).copy().reset_index(drop=True)
        g["Posicao_Calculada"] = range(1, len(g) + 1)
        parts.append(g)
    ranked = pd.concat(parts, ignore_index=True)
    fs = ranked[ranked["Posicao_Calculada"].isin([1, 2])].copy()
    fs["Tipo_Classificacao"] = "1º/2º do grupo"
    thirds = ranked[ranked["Posicao_Calculada"] == 3].sort_values(["Pontos", "SG", "GP"], ascending=[False, False, False]).head(8).copy()
    thirds["Tipo_Classificacao"] = "melhor terceiro atual"
    return pd.concat([fs, thirds], ignore_index=True).reset_index(drop=True)


def main():
    if len(sys.argv) < 2:
        print("Uso: python corrigir_excel_fifa2026_standings.py entrada.xlsx [saida.xlsx]")
        sys.exit(1)
    entrada = Path(sys.argv[1]).resolve()
    saida = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else entrada.with_name(entrada.stem + "_corrigido.xlsx")
    xl = pd.ExcelFile(entrada)
    sheets = {s: pd.read_excel(entrada, sheet_name=s) for s in xl.sheet_names}
    standings = montar_standings(sheets)
    sheets["copa2026_classificacao_atual"] = standings
    sheets["copa2026_classificados_atuais"] = calcular_classificados(standings)
    equipes = standings[["Equipe", "Codigo_FIFA"]].drop_duplicates().sort_values("Equipe").reset_index(drop=True) if not standings.empty else pd.DataFrame(columns=["Equipe", "Codigo_FIFA"])
    equipes["Fonte"] = "dom_standings corrigido"
    sheets["copa2026_equipes_oficiais"] = equipes

    with pd.ExcelWriter(saida, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe = re.sub(r"[\\/*?:\[\]]", "_", name)[:31]
            df.to_excel(writer, sheet_name=safe, index=False)
    print(f"Arquivo corrigido salvo em: {saida}")
    print(f"Standings: {len(standings)} linhas")

if __name__ == "__main__":
    main()
