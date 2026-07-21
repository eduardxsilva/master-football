"""Coleta de campeonatos pela API-Football v3; credenciais nunca sao salvas."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import os

import pandas as pd
import requests

API_URL = "https://v3.football.api-sports.io/fixtures"
THESPORTSDB_URL = "https://www.thesportsdb.com/api/v1/json/123/eventsseason.php"
COMPETICOES = {
    "Brasileirão Série A": 71, "Brasileirão Série B": 72,
    "CONMEBOL Libertadores": 13, "Premier League": 39,
    "Championship (Inglaterra)": 40, "La Liga": 140,
    "Serie A (Itália)": 135, "Bundesliga": 78, "Ligue 1": 61,
    "Champions League": 2, "Europa League": 3, "Liga Portugal": 94,
    "Eredivisie": 88, "MLS": 253, "Liga Argentina": 128,
    "Outro — informar ID": 0,
}
# IDs publicos do TheSportsDB. Zero significa que a cobertura gratuita nao foi
# confirmada; nesses casos o app permite informar um ID manual.
THESPORTSDB_COMPETICOES = {
    "Brasileirão Série A": 4351,
    "Brasileirão Série B": 0,
    "CONMEBOL Libertadores": 0,
    "Premier League": 4328,
    "Championship (Inglaterra)": 4329,
    "La Liga": 4335,
    "Serie A (Itália)": 4332,
    "Bundesliga": 4331,
    "Ligue 1": 4334,
    "Champions League": 4480,
    "Europa League": 4481,
    "Liga Portugal": 4344,
    "Eredivisie": 4337,
    "MLS": 4346,
    "Liga Argentina": 4406,
    "Outro — informar ID": 0,
}
TEMPORADA_ANO_CALENDARIO = {"Brasileirão Série A", "Brasileirão Série B", "CONMEBOL Libertadores", "MLS", "Liga Argentina"}
FINALIZADOS = {"FT", "AET", "PEN"}
DESCARTADOS = {"CANC", "ABD", "AWD", "WO", "PST"}


class SeasonPlanError(RuntimeError):
    """A temporada solicitada nao esta incluida no plano da chave."""


@dataclass
class CompetitionResult:
    matches: pd.DataFrame
    fixtures: pd.DataFrame
    raw: pd.DataFrame
    competition: str
    league_id: int
    season: int
    fetched_at_utc: str


def temporadas_recentes_thesportsdb(competition: str, ano_final: int = 2026) -> list[str]:
    """Duas temporadas recentes: 2025/2026 ou 2025-26/2026-27."""
    if competition in TEMPORADA_ANO_CALENDARIO:
        return [str(ano_final - 1), str(ano_final)]
    return [f"{ano_final - 1}-{ano_final}", f"{ano_final}-{ano_final + 1}"]


def _tsdb_eventos(league_id: int, season: str) -> list[dict]:
    if int(league_id) <= 0:
        raise ValueError("Informe um ID válido do TheSportsDB para essa competição.")
    response = requests.get(THESPORTSDB_URL, params={"id": int(league_id), "s": season}, timeout=45)
    response.raise_for_status()
    payload = response.json()
    return payload.get("events") or []


def _tsdb_to_frame(events: list[dict], competition: str, season: str) -> pd.DataFrame:
    rows = []
    for event in events:
        timestamp = event.get("strTimestamp")
        if not timestamp:
            data, hora = event.get("dateEvent"), event.get("strTime") or "00:00:00"
            timestamp = f"{data}T{hora}" if data else None
        rows.append({
            "fixture_id": event.get("idEvent"), "date": timestamp,
            "venue": event.get("strVenue"), "status": event.get("strStatus"),
            "status_long": event.get("strStatus"), "round": event.get("intRound"),
            "competition": event.get("strLeague") or competition,
            "country": event.get("strCountry"), "season": event.get("strSeason") or season,
            "home_team": event.get("strHomeTeam"), "away_team": event.get("strAwayTeam"),
            "home_goals": event.get("intHomeScore"), "away_goals": event.get("intAwayScore"),
        })
    return pd.DataFrame(rows)


def fetch_thesportsdb_two_seasons(competition: str, league_id: int, seasons: list[str]) -> CompetitionResult:
    """Une duas temporadas recentes e separa treino de calendario futuro."""
    frames, cobertura = [], []
    for season in seasons:
        events = _tsdb_eventos(league_id, str(season))
        cobertura.append({"Temporada": str(season), "Eventos": len(events)})
        if events:
            frames.append(_tsdb_to_frame(events, competition, str(season)))
    if not frames:
        raise RuntimeError(
            "TheSportsDB não retornou jogos nas duas temporadas. "
            "A competição pode não ter cobertura gratuita ou o ID pode estar incorreto."
        )
    raw = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["fixture_id"], keep="last")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce", utc=True).dt.tz_convert(None)
    raw["home_goals"] = pd.to_numeric(raw.home_goals, errors="coerce")
    raw["away_goals"] = pd.to_numeric(raw.away_goals, errors="coerce")
    done = raw.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"]).copy()
    done[["home_goals", "away_goals"]] = done[["home_goals", "away_goals"]].astype(int)
    done["home_xg"], done["away_xg"] = done.home_goals.astype(float), done.away_goals.astype(float)
    matches = done[["date", "home_team", "away_team", "home_goals", "away_goals",
                    "home_xg", "away_xg", "competition"]].sort_values("date").reset_index(drop=True)
    latest = str(seasons[-1])
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    fixtures = raw[(raw.season.astype(str) == latest) & raw.home_goals.isna() & raw.away_goals.isna()
                   & raw.date.notna() & (raw.date >= now)].sort_values("date").reset_index(drop=True)
    raw.attrs["cobertura"] = cobertura
    return CompetitionResult(matches, fixtures, raw, competition, int(league_id),
                             int(str(seasons[-1])[:4]), datetime.now(timezone.utc).isoformat())


def obter_api_key(valor_digitado: str = "", secrets: Any = None) -> str:
    valor = str(valor_digitado or "").strip()
    if valor:
        return valor
    try:
        valor = str((secrets or {}).get("API_FOOTBALL_KEY", "")).strip()
        if valor:
            return valor
    except Exception:
        pass
    return str(os.environ.get("API_FOOTBALL_KEY", "")).strip()


def _request(api_key: str, league_id: int, season: int) -> dict:
    if not api_key:
        raise ValueError("Informe API_FOOTBALL_KEY na tela ou nos Secrets do Streamlit.")
    if int(league_id) <= 0:
        raise ValueError("Informe um ID de competição maior que zero.")
    response = requests.get(API_URL, headers={"x-apisports-key": api_key},
                            params={"league": int(league_id), "season": int(season)}, timeout=45)
    if response.status_code in {401, 403}:
        raise RuntimeError("Chave recusada ou competição indisponível no seu plano.")
    if response.status_code == 429:
        raise RuntimeError("Limite de requisições atingido.")
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        errors = payload["errors"]
        texto = str(errors)
        if "plan" in errors or "do not have access to this season" in texto.lower():
            detalhe = errors.get("plan", texto) if isinstance(errors, dict) else texto
            raise SeasonPlanError(
                "Seu plano da API não permite essa temporada. "
                f"Resposta do provedor: {detalhe}"
            )
        raise RuntimeError(f"Erro da API: {errors}")
    return payload


def _flatten(payload: dict, nome: str) -> pd.DataFrame:
    rows = []
    for item in payload.get("response", []):
        fixture, league = item.get("fixture") or {}, item.get("league") or {}
        teams, goals = item.get("teams") or {}, item.get("goals") or {}
        home, away = teams.get("home") or {}, teams.get("away") or {}
        status = fixture.get("status") or {}
        rows.append({
            "fixture_id": fixture.get("id"), "date": fixture.get("date"),
            "venue": (fixture.get("venue") or {}).get("name"),
            "status": status.get("short"), "status_long": status.get("long"),
            "round": league.get("round"), "competition": league.get("name") or nome,
            "country": league.get("country"), "season": league.get("season"),
            "home_team": home.get("name"), "away_team": away.get("name"),
            "home_goals": goals.get("home"), "away_goals": goals.get("away"),
        })
    return pd.DataFrame(rows)


def fetch_competition(api_key: str, competition: str, league_id: int, season: int) -> CompetitionResult:
    raw = _flatten(_request(api_key, league_id, season), competition)
    if raw.empty:
        raise RuntimeError("Nenhuma partida retornada. Confira ID, temporada e cobertura do plano.")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce", utc=True).dt.tz_convert(None)
    done = raw[raw.status.isin(FINALIZADOS)].copy()
    done["home_goals"] = pd.to_numeric(done.home_goals, errors="coerce")
    done["away_goals"] = pd.to_numeric(done.away_goals, errors="coerce")
    done = done.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    done[["home_goals", "away_goals"]] = done[["home_goals", "away_goals"]].astype(int)
    done["home_xg"], done["away_xg"] = done.home_goals.astype(float), done.away_goals.astype(float)
    matches = done[["date", "home_team", "away_team", "home_goals", "away_goals",
                    "home_xg", "away_xg", "competition"]].sort_values("date").reset_index(drop=True)
    fixtures = raw[~raw.status.isin(FINALIZADOS | DESCARTADOS) & raw.home_goals.isna() & raw.away_goals.isna()]
    return CompetitionResult(matches, fixtures.sort_values("date").reset_index(drop=True), raw,
                             competition, int(league_id), int(season), datetime.now(timezone.utc).isoformat())


def predict_fixtures(fixtures: pd.DataFrame, poisson_model, limit: int = 100) -> pd.DataFrame:
    rows = []
    if fixtures is None or fixtures.empty:
        return pd.DataFrame()
    for _, match in fixtures.head(max(int(limit), 1)).iterrows():
        try:
            p = poisson_model.prever_partida(str(match.home_team), str(match.away_team))
        except (ValueError, KeyError):
            continue
        rows.append({"Data": match.date, "Rodada": match.get("round"), "Mandante": match.home_team,
                     "Visitante": match.away_team, "Placar provável": p["placar_provavel"],
                     "xG mandante": p["lambda_mandante"], "xG visitante": p["lambda_visitante"],
                     "Vitória mandante (%)": p["prob_mandante"], "Empate (%)": p["prob_empate"],
                     "Vitória visitante (%)": p["prob_visitante"], "Mais de 2,5 (%)": p["prob_over_25"],
                     "Ambas marcam (%)": p["prob_btts"], "Fixture ID": match.fixture_id})
    return pd.DataFrame(rows)
