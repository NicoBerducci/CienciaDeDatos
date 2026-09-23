"""Ingesta diaria de partidos de Counter-Strike 2 desde Bo3.gg.

El DAG implementa dos capas:

* Bronze append-only: conserva por separado la respuesta de cada endpoint.
* Silver canónico: un partido competitivo finalizado por fila.

Silver contiene observaciones del partido. Las variables históricas para el
modelo (winrates, rachas, forma, H2H, etc.) pertenecen a Gold y deliberadamente
no se calculan aquí.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as py_date
from datetime import datetime as py_datetime
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from airflow.sdk import dag, get_current_context, task
from pendulum import datetime


BASE_URL = "https://api.bo3.gg/api"
DISCOVERY_URL = f"{BASE_URL}/v1/matches"
CS2_DISCIPLINE_ID = 1
CS2_START_DATE = py_date(2023, 9, 27)

DISCOVERY_PAGE_SIZE = 100
DISCOVERY_DATES_PER_TASK = 30
ENRICHMENT_MATCHES_PER_TASK = 25
MAX_ENRICHMENT_TASKS = 500
MAX_PARALLEL_API_TASKS = 4
REQUEST_TIMEOUT_SECONDS = 45
MAX_HTTP_ATTEMPTS = 4
MAX_GAMES_PER_MATCH = 5
MIN_SILVER_ROWS = 1_000
RECENT_RETRY_OFFSETS_DAYS = (1, 3, 7)
CONTROL_IO_WORKERS = 8
CONTROL_LOG_EVERY_BATCHES = 10

OUTPUT_DIR = Path("/usr/local/airflow/include/output")
BRONZE_DIR = OUTPUT_DIR / "bronze"
BRONZE_RUNS_DIR = BRONZE_DIR / "runs"
CONTROL_DIR = OUTPUT_DIR / "control"
STATE_PATH = CONTROL_DIR / "pipeline_state.json"
REGISTRY_PATH = CONTROL_DIR / "match_registry.json"
BRONZE_INDEX_PATH = CONTROL_DIR / "bronze_index.json"
STAGING_DIR = OUTPUT_DIR / "staging"
QUALITY_DIR = OUTPUT_DIR / "quality"
SILVER_DIR = OUTPUT_DIR / "silver"

LOGGER = logging.getLogger(__name__)

HTTP_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "utn-cs2-data-project/1.0",
}

ROUND_STAT_FIELDS = {
    "kills": "kills",
    "death": "deaths",
    "assists": "assists",
    "headshots": "headshots",
    "first_kills": "first_kills",
    "first_death": "first_deaths",
    "trade_kills": "trade_kills",
    "trade_death": "trade_deaths",
    "damage": "damage",
    "got_damage": "got_damage",
    "hits": "hits",
    "shots": "shots",
    "flash_assists": "flash_assists",
    "grenades_damage": "grenades_damage",
    "clutches": "clutches",
    "money_spent": "money_spent",
    "equipment_value": "equipment_value",
}


def utc_now() -> py_datetime:
    return py_datetime.now(timezone.utc)


def iso_utc(value: py_datetime | None = None) -> str:
    current = value or utc_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_datetime(value: Any) -> py_datetime | None:
    if not value:
        return None
    try:
        parsed = py_datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl_atomic(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_gzip_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"), default=str)
    temporary.replace(path)


def write_dataframe_csv_atomic(path: Path, dataframe: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    dataframe.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_batches(values: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise ValueError("El tamaño del lote debe ser positivo.")
    return [values[index : index + size] for index in range(0, len(values), size)]


def daterange(start: py_date, end: py_date) -> list[str]:
    if end < start:
        return []
    days = (end - start).days
    return [(start + timedelta(days=offset)).isoformat() for offset in range(days + 1)]


def extraction_identity() -> tuple[str, str]:
    context = get_current_context()
    logical_date = context["logical_date"].in_timezone("UTC")
    run_id = context["run_id"]
    run_suffix = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:8]
    return f"{logical_date.format('YYYYMMDDTHHmmss[Z]')}__{run_suffix}", run_id


def fetch_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    fail_on_error: bool = False,
) -> dict[str, Any]:
    """Obtiene JSON y conserva el payload sin modificar dentro de un sobre."""

    last_error: str | None = None
    for attempt in range(1, MAX_HTTP_ATTEMPTS + 1):
        try:
            response = requests.get(
                url,
                headers=HTTP_HEADERS,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < MAX_HTTP_ATTEMPTS:
                time.sleep(min(2**attempt, 30))
                continue
            if fail_on_error:
                raise RuntimeError(f"No se pudo consultar {url}: {last_error}") from exc
            return {
                "ok": False,
                "status_code": None,
                "fetched_at_utc": iso_utc(),
                "url": url,
                "params": params or {},
                "payload": None,
                "error": last_error,
            }

        retryable = response.status_code == 429 or 500 <= response.status_code < 600
        if retryable and attempt < MAX_HTTP_ATTEMPTS:
            retry_after = response.headers.get("Retry-After", "")
            seconds = int(retry_after) if retry_after.isdigit() else 2**attempt
            time.sleep(min(seconds, 60))
            continue

        try:
            payload = response.json()
        except ValueError:
            payload = None
            last_error = f"Respuesta no JSON: {response.text[:300]}"

        envelope = {
            "ok": response.status_code == 200 and payload is not None,
            "status_code": response.status_code,
            "fetched_at_utc": iso_utc(),
            "url": response.url,
            "params": params or {},
            "payload": payload,
            "error": last_error,
        }
        if not envelope["ok"] and fail_on_error:
            raise RuntimeError(
                f"Bo3.gg devolvió HTTP {response.status_code} para {response.url}: "
                f"{response.text[:300]}"
            )
        return envelope

    raise RuntimeError(f"Se agotaron los intentos para {url}.")


def envelope_payload(path_value: str | None, default: Any = None) -> Any:
    if not path_value:
        return default
    path = Path(path_value)
    if not path.exists():
        return default
    document = read_gzip_json(path)
    return document.get("payload", default) if isinstance(document, dict) else default


def payload_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("results", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def discovery_summary(match: dict[str, Any]) -> dict[str, Any] | None:
    match_id = match.get("id")
    slug = match.get("slug")
    if match_id is None or not slug:
        return None
    return {
        "match_id": int(match_id),
        "slug": str(slug),
        "status": match.get("status"),
        "parsed_status": match.get("parsed_status"),
        "start_date": match.get("start_date") or match.get("begin_at"),
        "end_date": match.get("end_date"),
        "winner_team_id": match.get("winner_team_id") or match.get("winner_id"),
        "team1_id": match.get("team1_id"),
        "team2_id": match.get("team2_id"),
        "forfeit": bool(match.get("forfeit") or match.get("defwin")),
    }


def is_finished(status: Any) -> bool:
    return str(status or "").strip().lower() in {"finished", "done"}


def is_forfeit(detail: dict[str, Any]) -> bool:
    status = str(detail.get("status") or "").strip().lower()
    return bool(
        detail.get("forfeit")
        or detail.get("defwin")
        or detail.get("walkover")
        or status in {"defwin", "forfeit", "walkover", "w/o"}
    )


def match_result_valid(detail: dict[str, Any]) -> bool:
    team1 = detail.get("team1_id") or (detail.get("team1") or {}).get("id")
    team2 = detail.get("team2_id") or (detail.get("team2") or {}).get("id")
    winner = detail.get("winner_team_id") or detail.get("winner_id")
    return bool(
        is_finished(detail.get("status"))
        and team1 is not None
        and team2 is not None
        and team1 != team2
        and winner in {team1, team2}
        and not is_forfeit(detail)
    )


def assess_enrichment(
    detail: dict[str, Any],
    games: list[dict[str, Any]],
    full_stats: list[dict[str, Any]],
    short_stats: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    issues: list[str] = []
    valid_result = match_result_valid(detail)
    if not valid_result:
        issues.append("invalid_or_non_competitive_result")

    complete_games = 0
    rounds_reported = 0
    rounds_received = 0
    for game in games:
        has_score = game.get("winner_clan_score") is not None and game.get("loser_clan_score") is not None
        if game.get("map_name") and has_score:
            complete_games += 1
        expected = game.get("rounds_count")
        received = len(game.get("game_rounds") or [])
        if expected is not None:
            rounds_reported += int(expected)
            rounds_received += received
            if int(expected) != received:
                issues.append(f"game_{game.get('id')}_rounds_{received}_of_{expected}")

    if not games:
        issues.append("games_missing")
    elif complete_games != len(games):
        issues.append("games_incomplete")
    if not full_stats:
        issues.append("full_player_stats_missing")
    if not short_stats:
        issues.append("short_player_stats_missing")
    if not profiles:
        issues.append("historical_lineup_missing")

    rounds_complete = rounds_reported > 0 and rounds_reported == rounds_received
    if valid_result and games and complete_games == len(games) and full_stats and profiles and rounds_complete:
        completeness = "complete"
    elif valid_result and (games or full_stats or short_stats or profiles):
        completeness = "partial"
    elif valid_result:
        completeness = "minimal"
    else:
        completeness = "invalid"

    return {
        "result_valid": valid_result,
        "data_completeness": completeness,
        "games_count": len(games),
        "complete_games_count": complete_games,
        "rounds_reported_count": rounds_reported,
        "rounds_received_count": rounds_received,
        "full_stats_rows": len(full_stats),
        "short_stats_rows": len(short_stats),
        "profiles_rows": len(profiles),
        "issues": sorted(set(issues)),
    }


def team_aliases(team: dict[str, Any]) -> set[str]:
    aliases = {str(team.get("name") or "").casefold().strip()}
    aliases.update(
        str(clan.get("clan_name") or "").casefold().strip()
        for clan in team.get("team_clans") or []
        if isinstance(clan, dict)
    )
    return {alias for alias in aliases if alias}


def resolve_team_id(
    *,
    explicit_team_id: Any = None,
    clan_name: Any = None,
    team_a_id: int,
    team_b_id: int,
    aliases_a: set[str],
    aliases_b: set[str],
) -> int | None:
    if explicit_team_id is not None:
        try:
            candidate = int(explicit_team_id)
        except (TypeError, ValueError):
            candidate = None
        if candidate in {team_a_id, team_b_id}:
            return candidate
    normalized = str(clan_name or "").casefold().strip()
    if normalized in aliases_a and normalized not in aliases_b:
        return team_a_id
    if normalized in aliases_b and normalized not in aliases_a:
        return team_b_id
    return None


def safe_mean(values: Iterable[Any]) -> float | None:
    numeric: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    return sum(numeric) / len(numeric) if numeric else None


def build_silver_columns() -> list[str]:
    columns = [
        "match_id", "match_slug", "start_at_utc", "end_at_utc", "duration_seconds",
        "bo_type", "tier", "game_version", "tournament_id", "tournament_name",
        "stage_id", "stage_name", "team_a_id", "team_a_name", "team_a_country_code",
        "team_b_id", "team_b_name", "team_b_country_code", "team_a_rank_at_extraction",
        "team_b_rank_at_extraction", "rank_extracted_at_utc", "winner_team_id",
        "loser_team_id", "target_team_a_won", "post_team_a_maps_won",
        "post_team_b_maps_won", "post_maps_played",
    ]
    for number in range(1, MAX_GAMES_PER_MATCH + 1):
        columns.extend([
            f"post_map_{number}_game_id", f"post_map_{number}_name",
            f"post_map_{number}_winner_team_id", f"post_map_{number}_team_a_score",
            f"post_map_{number}_team_b_score", f"post_map_{number}_rounds_count",
            f"post_map_{number}_overtime",
        ])
    columns.extend([
        "post_team_a_rounds_won", "post_team_b_rounds_won", "post_round_difference_a",
        "post_total_rounds", "post_team_a_ct_rounds_won", "post_team_a_t_rounds_won",
        "post_team_b_ct_rounds_won", "post_team_b_t_rounds_won",
        "post_team_a_overtime_rounds_won", "post_team_b_overtime_rounds_won",
    ])
    for side in ("a", "b"):
        for output_name in ROUND_STAT_FIELDS.values():
            columns.append(f"post_team_{side}_{output_name}")
        columns.extend([
            f"post_team_{side}_player_rating_mean", f"post_team_{side}_adr_mean",
            f"post_team_{side}_kast_mean", f"post_team_{side}_players_stats_count",
        ])
    columns.extend([
        "team_a_lineup_profile_ids", "team_b_lineup_profile_ids", "team_a_lineup_size",
        "team_b_lineup_size", "parsed_status", "data_completeness",
    ])
    return columns


SILVER_COLUMNS = build_silver_columns()


def build_match_row(
    detail: dict[str, Any],
    games: list[dict[str, Any]],
    full_stats: list[dict[str, Any]],
    short_stats: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    *,
    data_completeness: str,
    rank_extracted_at_utc: str | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Construye una fila por partido sin calcular información histórica."""

    issues: list[str] = []
    if not match_result_valid(detail):
        return None, ["invalid_or_non_competitive_result"]

    source_team1 = detail.get("team1") or {}
    source_team2 = detail.get("team2") or {}
    source_team1_id = int(detail.get("team1_id") or source_team1.get("id"))
    source_team2_id = int(detail.get("team2_id") or source_team2.get("id"))
    if source_team1_id < source_team2_id:
        team_a, team_b = source_team1, source_team2
        team_a_id, team_b_id = source_team1_id, source_team2_id
    else:
        team_a, team_b = source_team2, source_team1
        team_a_id, team_b_id = source_team2_id, source_team1_id

    aliases_a = team_aliases(team_a)
    aliases_b = team_aliases(team_b)
    winner_id = int(detail.get("winner_team_id") or detail.get("winner_id"))
    loser_id = team_b_id if winner_id == team_a_id else team_a_id
    start_at = parse_datetime(detail.get("start_date") or detail.get("begin_at"))
    end_at = parse_datetime(detail.get("end_date") or detail.get("end_at"))
    duration = int((end_at - start_at).total_seconds()) if start_at and end_at and end_at >= start_at else None

    tournament = detail.get("tournament") or detail.get("tournament_deep") or {}
    stage = detail.get("stage") or {}
    row = {column: None for column in SILVER_COLUMNS}
    row.update({
        "match_id": int(detail["id"]), "match_slug": detail.get("slug"),
        "start_at_utc": iso_utc(start_at) if start_at else None,
        "end_at_utc": iso_utc(end_at) if end_at else None,
        "duration_seconds": duration, "bo_type": detail.get("bo_type"), "tier": detail.get("tier"),
        "game_version": detail.get("game_version"),
        "tournament_id": detail.get("tournament_id") or tournament.get("id"),
        "tournament_name": tournament.get("name"), "stage_id": detail.get("stage_id") or stage.get("id"),
        "stage_name": stage.get("title") or stage.get("name"),
        "team_a_id": team_a_id, "team_a_name": team_a.get("name"),
        "team_a_country_code": (team_a.get("country") or {}).get("code"), "team_b_id": team_b_id,
        "team_b_name": team_b.get("name"), "team_b_country_code": (team_b.get("country") or {}).get("code"),
        "team_a_rank_at_extraction": team_a.get("rank"), "team_b_rank_at_extraction": team_b.get("rank"),
        "rank_extracted_at_utc": rank_extracted_at_utc, "winner_team_id": winner_id,
        "loser_team_id": loser_id, "target_team_a_won": int(winner_id == team_a_id),
        "parsed_status": detail.get("parsed_status"), "data_completeness": data_completeness,
    })

    ordered_games = sorted(games, key=lambda game: (
        int(game.get("number")) if str(game.get("number", "")).isdigit() else 999,
        int(game.get("id") or 0),
    ))
    maps_won = {team_a_id: 0, team_b_id: 0}
    rounds_won = {team_a_id: 0, team_b_id: 0}
    side_rounds = {team_a_id: {"CT": 0, "T": 0, "OT": 0}, team_b_id: {"CT": 0, "T": 0, "OT": 0}}
    round_totals: dict[int, dict[str, float]] = {team_a_id: defaultdict(float), team_b_id: defaultdict(float)}
    round_stat_rows = Counter({team_a_id: 0, team_b_id: 0})

    for slot, game in enumerate(ordered_games[:MAX_GAMES_PER_MATCH], start=1):
        winner_clan = game.get("winner_team_clan") or {}
        loser_clan = game.get("loser_team_clan") or {}
        game_winner = resolve_team_id(
            explicit_team_id=game.get("winner_team_id") or winner_clan.get("team_id"),
            clan_name=game.get("winner_clan_name") or winner_clan.get("clan_name"),
            team_a_id=team_a_id, team_b_id=team_b_id, aliases_a=aliases_a, aliases_b=aliases_b,
        )
        game_loser = resolve_team_id(
            explicit_team_id=game.get("loser_team_id") or loser_clan.get("team_id"),
            clan_name=game.get("loser_clan_name") or loser_clan.get("clan_name"),
            team_a_id=team_a_id, team_b_id=team_b_id, aliases_a=aliases_a, aliases_b=aliases_b,
        )
        winner_score = game.get("winner_clan_score")
        loser_score = game.get("loser_clan_score")
        score_a = winner_score if game_winner == team_a_id else loser_score if game_loser == team_a_id else None
        score_b = winner_score if game_winner == team_b_id else loser_score if game_loser == team_b_id else None
        overtime = any(bool(segment.get("overtime")) for segment in game.get("game_side_results") or [])
        row.update({
            f"post_map_{slot}_game_id": game.get("id"), f"post_map_{slot}_name": game.get("map_name"),
            f"post_map_{slot}_winner_team_id": game_winner, f"post_map_{slot}_team_a_score": score_a,
            f"post_map_{slot}_team_b_score": score_b, f"post_map_{slot}_rounds_count": game.get("rounds_count"),
            f"post_map_{slot}_overtime": overtime,
        })
        if game_winner in maps_won:
            maps_won[game_winner] += 1
        if score_a is not None:
            rounds_won[team_a_id] += int(score_a)
        if score_b is not None:
            rounds_won[team_b_id] += int(score_b)

        for segment in game.get("game_side_results") or []:
            segment_is_ot = bool(segment.get("overtime"))
            for prefix in ("winner", "loser"):
                segment_team = resolve_team_id(
                    clan_name=segment.get(f"{prefix}_clan_name"), team_a_id=team_a_id, team_b_id=team_b_id,
                    aliases_a=aliases_a, aliases_b=aliases_b,
                )
                score = segment.get(f"{prefix}_clan_score")
                side = str(segment.get(f"{prefix}_clan_side") or "").upper()
                if segment_team in side_rounds and score is not None:
                    if side in {"CT", "T"}:
                        side_rounds[segment_team][side] += int(score)
                    if segment_is_ot:
                        side_rounds[segment_team]["OT"] += int(score)

        for game_round in game.get("game_rounds") or []:
            for team_round in game_round.get("game_round_team_clans") or []:
                round_team = resolve_team_id(
                    clan_name=team_round.get("clan_name"), team_a_id=team_a_id, team_b_id=team_b_id,
                    aliases_a=aliases_a, aliases_b=aliases_b,
                )
                if round_team not in round_totals:
                    continue
                round_stat_rows[round_team] += 1
                for source_name, output_name in ROUND_STAT_FIELDS.items():
                    value = team_round.get(source_name)
                    if value is not None:
                        round_totals[round_team][output_name] += float(value)

    if len(ordered_games) > MAX_GAMES_PER_MATCH:
        issues.append("more_than_five_games")

    row.update({
        "post_team_a_maps_won": maps_won[team_a_id], "post_team_b_maps_won": maps_won[team_b_id],
        "post_maps_played": len(ordered_games),
        "post_team_a_rounds_won": rounds_won[team_a_id] if ordered_games else None,
        "post_team_b_rounds_won": rounds_won[team_b_id] if ordered_games else None,
        "post_round_difference_a": rounds_won[team_a_id] - rounds_won[team_b_id] if ordered_games else None,
        "post_total_rounds": rounds_won[team_a_id] + rounds_won[team_b_id] if ordered_games else None,
        "post_team_a_ct_rounds_won": side_rounds[team_a_id]["CT"] if ordered_games else None,
        "post_team_a_t_rounds_won": side_rounds[team_a_id]["T"] if ordered_games else None,
        "post_team_b_ct_rounds_won": side_rounds[team_b_id]["CT"] if ordered_games else None,
        "post_team_b_t_rounds_won": side_rounds[team_b_id]["T"] if ordered_games else None,
        "post_team_a_overtime_rounds_won": side_rounds[team_a_id]["OT"] if ordered_games else None,
        "post_team_b_overtime_rounds_won": side_rounds[team_b_id]["OT"] if ordered_games else None,
    })
    for side, team_id in (("a", team_a_id), ("b", team_b_id)):
        for output_name in ROUND_STAT_FIELDS.values():
            value = round_totals[team_id].get(output_name)
            row[f"post_team_{side}_{output_name}"] = (
                int(value) if value is not None and float(value).is_integer() else value
            ) if round_stat_rows[team_id] else None

    profile_team_votes: dict[int, set[int]] = defaultdict(set)
    lineup_ids: dict[int, set[int]] = {team_a_id: set(), team_b_id: set()}
    for profile in profiles:
        profile_id = profile.get("steam_profile_id") or (profile.get("steam_profile") or {}).get("id")
        profile_team = resolve_team_id(
            explicit_team_id=(profile.get("team_clan") or {}).get("team_id"),
            clan_name=profile.get("clan_name") or (profile.get("team_clan") or {}).get("clan_name"),
            team_a_id=team_a_id, team_b_id=team_b_id, aliases_a=aliases_a, aliases_b=aliases_b,
        )
        if profile_id is None or profile_team is None:
            continue
        profile_id = int(profile_id)
        lineup_ids[profile_team].add(profile_id)
        profile_team_votes[profile_id].add(profile_team)

    switched_profiles = {pid for pid, teams in profile_team_votes.items() if len(teams) > 1}
    if switched_profiles:
        issues.append("lineup_team_switch_detected")
        # La fuente asignó el mismo perfil a ambos clanes en distintos mapas.
        # No inventamos una afiliación: esos perfiles quedan fuera del resumen
        # de alineación y de los promedios de jugadores, pero permanecen en
        # Bronze y se documentan en match_quality.jsonl.
        for team_lineup in lineup_ids.values():
            team_lineup.difference_update(switched_profiles)

    player_rows_by_team: dict[int, dict[int, dict[str, Any]]] = {team_a_id: {}, team_b_id: {}}
    for stat in full_stats:
        profile_id = stat.get("steam_profile_id") or (stat.get("steam_profile") or {}).get("id")
        if profile_id is None:
            continue
        profile_id = int(profile_id)
        if profile_id in switched_profiles:
            continue
        historical_teams = profile_team_votes.get(profile_id, set())
        explicit_team = resolve_team_id(
            explicit_team_id=(stat.get("team_clan") or {}).get("team_id"),
            clan_name=stat.get("clan_name") or (stat.get("team_clan") or {}).get("clan_name"),
            team_a_id=team_a_id, team_b_id=team_b_id, aliases_a=aliases_a, aliases_b=aliases_b,
        )
        if len(historical_teams) == 1:
            stats_team = next(iter(historical_teams))
            if explicit_team is not None and explicit_team != stats_team:
                continue
        elif explicit_team is not None:
            stats_team = explicit_team
        else:
            continue
        existing = player_rows_by_team[stats_team].get(profile_id)
        if existing is None or int(stat.get("id") or 0) > int(existing.get("id") or 0):
            player_rows_by_team[stats_team][profile_id] = stat

    for side, team_id in (("a", team_a_id), ("b", team_b_id)):
        player_rows = list(player_rows_by_team[team_id].values())
        row[f"post_team_{side}_player_rating_mean"] = safe_mean(stat.get("player_rating") for stat in player_rows)
        row[f"post_team_{side}_adr_mean"] = safe_mean(stat.get("adr") for stat in player_rows)
        row[f"post_team_{side}_kast_mean"] = safe_mean(stat.get("kast") for stat in player_rows)
        row[f"post_team_{side}_players_stats_count"] = len(player_rows) if player_rows else None

        # Si no hay rondas detalladas, las estadísticas completas mantienen
        # algunos totales observados. Short stats es el último respaldo y no
        # reemplaza ratings/KAST, porque sus definiciones no son equivalentes.
        if not round_stat_rows[team_id] and player_rows:
            fallback_fields = {
                "kills": "kills", "death": "deaths", "assists": "assists",
                "headshots": "headshots", "first_kills": "first_kills",
                "first_death": "first_deaths", "trade_kills": "trade_kills",
                "trade_death": "trade_deaths", "damage": "damage",
                "got_damage": "got_damage", "hits": "hits", "shots": "shots",
                "flash_assists": "flash_assists", "grenades_damage": "grenades_damage",
                "clutches": "clutches", "money_spent": "money_spent",
                "total_equipment_value": "equipment_value",
            }
            for source_name, output_name in fallback_fields.items():
                values = [stat.get(source_name) for stat in player_rows if stat.get(source_name) is not None]
                if values:
                    row[f"post_team_{side}_{output_name}"] = sum(values)
            issues.append(f"team_{side}_round_stats_fallback_full_players")

        if not player_rows:
            short_rows = []
            for stat in short_stats:
                try:
                    stat_team_id = int(stat.get("team_id"))
                except (TypeError, ValueError):
                    continue
                if stat_team_id == team_id:
                    short_rows.append(stat)
            if short_rows:
                row[f"post_team_{side}_players_stats_count"] = len(short_rows)
                adr_values = []
                for stat in short_rows:
                    games_count = stat.get("games_count")
                    adr_sum = stat.get("adr_sum")
                    if games_count and adr_sum is not None:
                        adr_values.append(float(adr_sum) / int(games_count))
                row[f"post_team_{side}_adr_mean"] = safe_mean(adr_values)
                if not round_stat_rows[team_id]:
                    short_fields = {
                        "kills_sum": "kills", "deaths_sum": "deaths",
                        "assists_sum": "assists", "headshots_sum": "headshots",
                        "flash_assists_sum": "flash_assists",
                    }
                    for source_name, output_name in short_fields.items():
                        values = [stat.get(source_name) for stat in short_rows if stat.get(source_name) is not None]
                        if values:
                            row[f"post_team_{side}_{output_name}"] = sum(values)
                issues.append(f"team_{side}_player_stats_fallback_short")

    row["team_a_lineup_profile_ids"] = ";".join(str(value) for value in sorted(lineup_ids[team_a_id])) or None
    row["team_b_lineup_profile_ids"] = ";".join(str(value) for value in sorted(lineup_ids[team_b_id])) or None
    row["team_a_lineup_size"] = len(lineup_ids[team_a_id]) if lineup_ids[team_a_id] else None
    row["team_b_lineup_size"] = len(lineup_ids[team_b_id]) if lineup_ids[team_b_id] else None
    return row, sorted(set(issues))


def next_retry_at(*, first_finished_at: py_datetime, attempts: int, historical: bool) -> str | None:
    if historical or attempts > len(RECENT_RETRY_OFFSETS_DAYS):
        return None
    target = first_finished_at + timedelta(days=RECENT_RETRY_OFFSETS_DAYS[attempts - 1])
    return iso_utc(target)


@dag(
    dag_id="bo3_cs2_ingest",
    description="Bronze y Silver diario de partidos históricos de CS2 desde Bo3.gg",
    start_date=datetime(2026, 9, 21, tz="UTC"),
    schedule="0 0 * * *",
    catchup=False,
    max_active_runs=1,
    max_active_tasks=MAX_PARALLEL_API_TASKS,
    default_args={"owner": "grupo_5K10_07", "retries": 2, "retry_delay": timedelta(minutes=2)},
    tags=["bo3", "cs2", "bronze", "silver", "incremental"],
)
def bo3_cs2_ingest():
    @task
    def build_plan() -> dict[str, Any]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        CONTROL_DIR.mkdir(parents=True, exist_ok=True)
        extraction_id, airflow_run_id = extraction_identity()
        context = get_current_context()
        interval_end = context.get("data_interval_end")
        today_utc = interval_end.in_timezone("UTC").date() if interval_end is not None else utc_now().date()
        today_utc = min(today_utc, utc_now().date())
        last_complete_date = today_utc - timedelta(days=1)
        state = read_json(STATE_PATH, {}) or {}
        last_discovered = state.get("last_discovered_date")
        start_date = py_date.fromisoformat(last_discovered) + timedelta(days=1) if last_discovered else CS2_START_DATE
        dates = daterange(start_date, last_complete_date)
        run_directory = BRONZE_RUNS_DIR / f"extraction={extraction_id}"
        run_directory.mkdir(parents=True, exist_ok=True)
        return {
            "extraction_id": extraction_id, "airflow_run_id": airflow_run_id,
            "created_at_utc": iso_utc(), "run_directory": str(run_directory),
            "discovery_start_date": start_date.isoformat() if dates else None,
            "discovery_end_date": last_complete_date.isoformat() if dates else None,
            "dates": dates, "date_batches": split_batches(dates, DISCOVERY_DATES_PER_TASK),
            "bootstrap": not bool(last_discovered),
        }

    @task
    def date_batches(plan: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {"batch_number": number, "dates": dates, "run_directory": plan["run_directory"]}
            for number, dates in enumerate(plan["date_batches"], start=1)
        ]

    @task(max_active_tis_per_dag=MAX_PARALLEL_API_TASKS)
    def discover_batch(batch: dict[str, Any]) -> dict[str, Any]:
        run_directory = Path(batch["run_directory"])
        summaries: list[dict[str, Any]] = []
        page_files: list[dict[str, Any]] = []
        for date_str in batch["dates"]:
            day = py_date.fromisoformat(date_str)
            lower_bound = py_datetime.combine(day - timedelta(days=1), py_datetime.max.time(), tzinfo=timezone.utc).replace(microsecond=0)
            upper_bound = py_datetime.combine(day + timedelta(days=1), py_datetime.min.time(), tzinfo=timezone.utc)
            offset, page_number, expected_pages = 0, 1, 1
            while page_number <= expected_pages:
                params = {
                    "page[offset]": offset, "page[limit]": DISCOVERY_PAGE_SIZE, "sort": "start_date",
                    "filter[matches.start_date][gt]": iso_utc(lower_bound),
                    "filter[matches.start_date][lt]": iso_utc(upper_bound),
                    "filter[matches.discipline_id][eq]": CS2_DISCIPLINE_ID,
                }
                envelope = fetch_json(DISCOVERY_URL, params=params, fail_on_error=True)
                destination = run_directory / "discovery" / f"date={date_str}" / f"page={page_number:03d}.json.gz"
                write_gzip_json_atomic(destination, envelope)
                payload = envelope.get("payload") or {}
                records = payload_records(payload)
                for match in records:
                    summary = discovery_summary(match)
                    if summary:
                        summary.update({"discovery_date": date_str, "discovery_path": str(destination)})
                        summaries.append(summary)
                total = payload.get("total") or {}
                expected_pages = max(int(total.get("pages") or 1), 1)
                page_files.append({
                    "date": date_str, "page": page_number, "path": str(destination),
                    "rows": len(records), "sha256": file_sha256(destination),
                })
                page_number += 1
                offset += DISCOVERY_PAGE_SIZE
        summary_path = run_directory / "manifests" / f"discovery_batch={batch['batch_number']:03d}.json"
        write_json_atomic(summary_path, {"matches": summaries, "files": page_files})
        return {"batch_number": batch["batch_number"], "dates": batch["dates"], "matches": len(summaries), "summary_path": str(summary_path)}

    @task(trigger_rule="none_failed")
    def prepare_enrichment(plan: dict[str, Any], discovery_results: list[dict[str, Any]]) -> list[str]:
        registry_document = read_json(REGISTRY_PATH, {"matches": {}}) or {"matches": {}}
        registry = registry_document.setdefault("matches", {})
        now_value = utc_now()
        newly_discovered: set[str] = set()
        for result in discovery_results or []:
            document = read_json(Path(result["summary_path"]), {}) or {}
            for match in document.get("matches", []):
                key = str(match["match_id"])
                entry = registry.setdefault(key, {"match_id": match["match_id"]})
                entry.update({
                    "slug": match["slug"], "source_status": match.get("status"),
                    "source_parsed_status": match.get("parsed_status"), "scheduled_at": match.get("start_date"),
                    "last_discovered_at_utc": iso_utc(now_value), "discovery_path": match.get("discovery_path"),
                })
                entry.setdefault("lifecycle_state", "discovered")
                entry.setdefault("enrichment_attempts", 0)
                newly_discovered.add(key)
        queue: list[dict[str, Any]] = []
        for key, entry in registry.items():
            if not entry.get("slug") or entry.get("lifecycle_state") in {
                "ready", "final_partial", "quarantined", "excluded_forfeit",
            }:
                continue
            retry_at = parse_datetime(entry.get("next_retry_at_utc"))
            if key not in newly_discovered and retry_at is not None and retry_at > now_value:
                continue
            scheduled = parse_datetime(entry.get("scheduled_at"))
            queue.append({
                "match_id": int(entry["match_id"]), "slug": entry["slug"],
                "scheduled_at": entry.get("scheduled_at"), "previous_status": entry.get("source_status"),
                "historical": bool((scheduled or now_value) < now_value - timedelta(days=7)),
            })
        registry_document["updated_at_utc"] = iso_utc(now_value)
        write_json_atomic(REGISTRY_PATH, registry_document)
        batch_paths: list[str] = []
        batch_size = max(ENRICHMENT_MATCHES_PER_TASK, math.ceil(len(queue) / MAX_ENRICHMENT_TASKS),)
        for number, matches in enumerate(split_batches(queue, batch_size), start=1):
            path = (
                Path(plan["run_directory"])
                / "queues"
                / f"enrichment_batch={number:04d}.json"
            )
            write_json_atomic(path, {"matches": matches})
            batch_paths.append(str(path))
        return batch_paths

    @task(max_active_tis_per_dag=MAX_PARALLEL_API_TASKS)
    def enrich_batch(queue_path: str) -> str:
        queue_document = read_json(Path(queue_path), {}) or {}
        run_directory = Path(queue_path).parents[1]
        results: list[dict[str, Any]] = []
        for item in queue_document.get("matches", []):
            match_id, slug = int(item["match_id"]), item["slug"]
            match_root = run_directory / "matches" / f"match_id={match_id}"
            endpoint_paths: dict[str, str] = {}
            endpoint_status: dict[str, Any] = {}
            detail_envelope = fetch_json(
                f"{BASE_URL}/v1/matches/{slug}",
                params={"scope": "show-match", "prefer_locale": "en", "with": "games,teams,tournament_deep,stage"},
            )
            detail_path = match_root / "match_detail.json.gz"
            write_gzip_json_atomic(detail_path, detail_envelope)
            endpoint_paths["detail"], endpoint_status["detail"] = str(detail_path), detail_envelope.get("status_code")
            detail = detail_envelope.get("payload") if detail_envelope.get("ok") else {}
            if isinstance(detail, dict) and is_finished(detail.get("status")) and not is_forfeit(detail):
                endpoints = {
                    "games": (f"{BASE_URL}/v1/games", {
                        "page[offset]": 0, "page[limit]": 10, "sort": "number",
                        "filter[games.match_id][eq]": match_id,
                        "with": "winner_team_clan,loser_team_clan,game_side_results,game_rounds",
                    }),
                    "players_stats": (f"{BASE_URL}/v1/matches/{slug}/players_stats", None),
                    "short_players_stats": (f"{BASE_URL}/v1/matches/{slug}/short_players_stats", None),
                    "game_steam_profiles": (f"{BASE_URL}/v1/matches/{slug}/game_steam_profiles", None),
                }
                for endpoint_name, (url, params) in endpoints.items():
                    envelope = fetch_json(url, params=params)
                    path = match_root / f"{endpoint_name}.json.gz"
                    write_gzip_json_atomic(path, envelope)
                    endpoint_paths[endpoint_name], endpoint_status[endpoint_name] = str(path), envelope.get("status_code")
            results.append({**item, "checked_at_utc": iso_utc(), "endpoint_paths": endpoint_paths, "endpoint_status": endpoint_status})
        result_path = Path(queue_path).with_name(Path(queue_path).stem + "_result.json")
        write_json_atomic(result_path, {"matches": results})
        return str(result_path)

    @task(trigger_rule="none_failed")
    def update_control(
        plan: dict[str, Any], discovery_results: list[dict[str, Any]], enrichment_result_paths: list[str]
    ) -> dict[str, Any]:
        registry_document = read_json(REGISTRY_PATH, {"matches": {}}) or {"matches": {}}
        registry = registry_document.setdefault("matches", {})
        index_document = read_json(BRONZE_INDEX_PATH, {"matches": {}}) or {"matches": {}}
        bronze_index = index_document.setdefault("matches", {})
        match_quality: list[dict[str, Any]] = []
        discarded = Counter()
        now_value = utc_now()
        enrichment_matches_processed = 0
        partial_directory = STAGING_DIR / f"extraction={plan['extraction_id']}" / "partials"

        def process_result_batch(result_path_value: str) -> dict[str, Any]:
            """Lee cada payload Bronze una sola vez y genera un parcial Silver atómico."""
            result_path = Path(result_path_value)
            result_document = read_json(result_path, {}) or {}
            processed: list[dict[str, Any]] = []
            rows: list[dict[str, Any]] = []
            batch_discarded = Counter()
            for result in result_document.get("matches", []):
                key = str(result["match_id"])
                # Cada match pertenece a un solo lote. Se copia el índice para que
                # los workers nunca muten estructuras compartidas entre threads.
                indexed = dict(bronze_index.get(key, {"match_id": result["match_id"]}))
                indexed.update({"slug": result["slug"], "updated_at_utc": result["checked_at_utc"]})
                # Un intento fallido no puede reemplazar una respuesta válida
                # de una corrida anterior en el índice de última versión.
                for endpoint_name, endpoint_path in result.get("endpoint_paths", {}).items():
                    if result.get("endpoint_status", {}).get(endpoint_name) == 200:
                        indexed[endpoint_name] = endpoint_path

                detail_document = (
                    read_gzip_json(Path(indexed["detail"]))
                    if indexed.get("detail") and Path(indexed["detail"]).exists()
                    else {}
                )
                detail_document = detail_document if isinstance(detail_document, dict) else {}
                detail = detail_document.get("payload", {})
                detail = detail if isinstance(detail, dict) else {}
                games = payload_records(envelope_payload(indexed.get("games"), []))
                full_stats = payload_records(envelope_payload(indexed.get("players_stats"), []))
                short_stats = payload_records(envelope_payload(indexed.get("short_players_stats"), []))
                profiles = payload_records(envelope_payload(indexed.get("game_steam_profiles"), []))
                assessment = assess_enrichment(detail, games, full_stats, short_stats, profiles)
                row = None
                row_issues: list[str] = []
                if assessment["result_valid"]:
                    row, row_issues = build_match_row(
                        detail, games, full_stats, short_stats, profiles,
                        data_completeness=assessment["data_completeness"],
                        rank_extracted_at_utc=detail_document.get("fetched_at_utc"),
                    )
                if row is not None:
                    rows.append(row)
                elif not detail:
                    batch_discarded["detail_missing"] += 1
                elif is_forfeit(detail):
                    batch_discarded["forfeit"] += 1
                elif not assessment["result_valid"]:
                    batch_discarded["invalid_or_unfinished_result"] += 1
                else:
                    batch_discarded["normalization_failed"] += 1

                processed.append({
                    "key": key,
                    "result": result,
                    "indexed": indexed,
                    "detail_status": detail.get("status"),
                    "detail_parsed_status": detail.get("parsed_status"),
                    "detail_slug": detail.get("slug"),
                    "detail_is_forfeit": bool(detail and is_forfeit(detail)),
                    "detail_is_finished": bool(detail and is_finished(detail.get("status"))),
                    "assessment": assessment,
                    "normalization_issues": row_issues,
                })

            partial_path = partial_directory / f"{result_path.stem}.csv"
            write_dataframe_csv_atomic(partial_path, pd.DataFrame(rows, columns=SILVER_COLUMNS))
            return {
                "processed": processed,
                "partial_path": str(partial_path),
                "discarded": dict(batch_discarded),
            }

        result_paths = list(enrichment_result_paths or [])
        partial_paths: list[str] = []
        worker_count = min(CONTROL_IO_WORKERS, max(1, len(result_paths)))
        LOGGER.info(
            "update_control procesará %s lotes con %s workers de E/S",
            len(result_paths), worker_count,
        )
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="control-io") as executor:
            futures = {
                executor.submit(process_result_batch, result_path): result_path
                for result_path in result_paths
            }
            for completed_batches, future in enumerate(as_completed(futures), start=1):
                source_path = futures.pop(future)
                try:
                    batch = future.result()
                except Exception:
                    LOGGER.exception("Falló el procesamiento del lote %s", source_path)
                    raise
                partial_paths.append(batch["partial_path"])
                discarded.update(batch["discarded"])
                for processed_match in batch["processed"]:
                    enrichment_matches_processed += 1
                    key = processed_match["key"]
                    result = processed_match["result"]
                    assessment = processed_match["assessment"]
                    entry = registry.setdefault(key, {"match_id": result["match_id"]})
                    already_processed_attempt = entry.get("last_checked_at_utc") == result["checked_at_utc"]
                    bronze_index[key] = processed_match["indexed"]
                    entry.update({
                        "last_checked_at_utc": result["checked_at_utc"],
                        "source_status": processed_match["detail_status"] or entry.get("source_status"),
                        "source_parsed_status": processed_match["detail_parsed_status"],
                        "slug": processed_match["detail_slug"] or result["slug"],
                    })
                    status_codes = list(result.get("endpoint_status", {}).values())
                    technical_failure = any(
                        code is None or code == 429 or (isinstance(code, int) and code >= 500)
                        for code in status_codes
                    )
                    if technical_failure:
                        technical_attempts = int(entry.get("technical_attempts") or 0)
                        if not already_processed_attempt:
                            technical_attempts += 1
                        entry["technical_attempts"] = technical_attempts
                        if technical_attempts < 3:
                            entry.update({
                                "lifecycle_state": "pending_request",
                                "data_completeness": assessment["data_completeness"],
                                "next_retry_at_utc": iso_utc(now_value + timedelta(days=1)),
                            })
                        elif assessment["result_valid"]:
                            entry.update({
                                "lifecycle_state": "final_partial",
                                "data_completeness": assessment["data_completeness"],
                                "next_retry_at_utc": None,
                            })
                        else:
                            entry.update({
                                "lifecycle_state": "quarantined",
                                "data_completeness": "invalid",
                                "next_retry_at_utc": None,
                            })
                        assessment["issues"] = sorted(set(assessment["issues"] + ["technical_endpoint_failure"]))
                    elif processed_match["detail_is_forfeit"]:
                        entry.update({"lifecycle_state": "excluded_forfeit", "data_completeness": "excluded", "next_retry_at_utc": None})
                    elif not processed_match["detail_is_finished"]:
                        scheduled = parse_datetime(entry.get("scheduled_at"))
                        stale = bool(scheduled and scheduled < now_value - timedelta(days=7))
                        entry.update({
                            "lifecycle_state": "quarantined" if stale else "pending_result",
                            "data_completeness": "invalid" if stale else "pending",
                            "next_retry_at_utc": None if stale else iso_utc(now_value + timedelta(days=1)),
                        })
                    else:
                        attempts = int(entry.get("enrichment_attempts") or 0)
                        if not already_processed_attempt:
                            attempts += 1
                        entry["enrichment_attempts"] = attempts
                        first_finished = parse_datetime(entry.get("first_finished_seen_at_utc")) or now_value
                        entry["first_finished_seen_at_utc"] = iso_utc(first_finished)
                        if not assessment["result_valid"]:
                            retry = next_retry_at(first_finished_at=first_finished, attempts=attempts, historical=bool(result.get("historical")))
                            entry.update({"lifecycle_state": "pending_result" if retry else "quarantined", "next_retry_at_utc": retry})
                        elif assessment["data_completeness"] == "complete":
                            entry.update({"lifecycle_state": "ready", "next_retry_at_utc": None})
                        else:
                            retry = next_retry_at(first_finished_at=first_finished, attempts=attempts, historical=bool(result.get("historical")))
                            entry.update({"lifecycle_state": "pending_enrichment" if retry else "final_partial", "next_retry_at_utc": retry})
                        entry["data_completeness"] = assessment["data_completeness"]
                    entry["quality_issues"] = assessment["issues"]
                    match_quality.append({
                        "match_id": result["match_id"], "slug": result["slug"],
                        "source_status": entry.get("source_status"), "parsed_status": entry.get("source_parsed_status"),
                        "lifecycle_state": entry.get("lifecycle_state"), **assessment,
                        "normalization_issues": processed_match["normalization_issues"],
                        "endpoint_status": result.get("endpoint_status", {}),
                    })
                if completed_batches % CONTROL_LOG_EVERY_BATCHES == 0 or completed_batches == len(result_paths):
                    LOGGER.info(
                        "update_control: %s/%s lotes, %s matches procesados",
                        completed_batches, len(result_paths), enrichment_matches_processed,
                    )

        partial_paths.sort()
        match_quality.sort(key=lambda value: int(value["match_id"]))
        if plan.get("discovery_end_date"):
            state = {
                "last_discovered_date": plan["discovery_end_date"], "last_successful_extraction": plan["extraction_id"],
                "last_successful_run_id": plan["airflow_run_id"], "updated_at_utc": iso_utc(now_value),
            }
        else:
            state = read_json(STATE_PATH, {}) or {}
            state.update({
                "last_successful_extraction": plan["extraction_id"], "last_successful_run_id": plan["airflow_run_id"],
                "updated_at_utc": iso_utc(now_value),
            })
        registry_document["updated_at_utc"] = iso_utc(now_value)
        index_document["updated_at_utc"] = iso_utc(now_value)
        write_json_atomic(REGISTRY_PATH, registry_document)
        write_json_atomic(BRONZE_INDEX_PATH, index_document)
        write_json_atomic(STATE_PATH, state)
        run_quality_path = Path(plan["run_directory"]) / "manifests" / "match_quality.jsonl"
        write_jsonl_atomic(run_quality_path, match_quality)
        quarantine = [{
            "match_id": value.get("match_id"), "slug": value.get("slug"),
            "reason": value.get("quality_issues", []), "last_checked_at_utc": value.get("last_checked_at_utc"),
        } for value in registry.values() if value.get("lifecycle_state") == "quarantined"]
        write_jsonl_atomic(QUALITY_DIR / "quarantine.jsonl", quarantine)
        return {
            **state, "extraction_id": plan["extraction_id"], "airflow_run_id": plan["airflow_run_id"],
            "run_directory": plan["run_directory"], "discovery_batches": len(discovery_results or []),
            "enrichment_batches": len(enrichment_result_paths or []), "matches_in_registry": len(registry),
            "dates_discovered": sum(len(result.get("dates", [])) for result in discovery_results or []),
            "matches_discovered": sum(int(result.get("matches", 0)) for result in discovery_results or []),
            "matches_enriched": enrichment_matches_processed,
            "match_quality_path": str(run_quality_path),
            "silver_partial_paths": partial_paths,
            "discarded": dict(discarded),
        }

    @task
    def transform_to_silver(control: dict[str, Any]) -> dict[str, Any]:
        partial_paths = [Path(value) for value in control.get("silver_partial_paths", [])]
        frames: list[pd.DataFrame] = []
        latest_path = SILVER_DIR / "matches_latest.csv"
        if latest_path.exists():
            latest = pd.read_csv(latest_path, low_memory=False)
            if list(latest.columns) != SILVER_COLUMNS:
                raise ValueError("El Silver existente no tiene el esquema canónico esperado.")
            frames.append(latest)

        for partial_path in partial_paths:
            partial = pd.read_csv(partial_path, low_memory=False)
            if list(partial.columns) != SILVER_COLUMNS:
                raise ValueError(f"El parcial Silver tiene un esquema inválido: {partial_path}")
            if not partial.empty:
                frames.append(partial)
        if not frames:
            raise RuntimeError("No se pudo construir ninguna fila Silver válida.")

        # El Silver previo preserva el histórico en corridas incrementales; los
        # parciales actuales van después y reemplazan el match por clave primaria.
        dataframe = pd.concat(frames, ignore_index=True)[SILVER_COLUMNS]
        dataframe.drop_duplicates("match_id", keep="last", inplace=True)
        dataframe.sort_values(["start_at_utc", "match_id"], inplace=True, na_position="last")
        dataframe.reset_index(drop=True, inplace=True)
        extraction_id = control["extraction_id"]
        staging_directory = STAGING_DIR / f"extraction={extraction_id}"
        staging_directory.mkdir(parents=True, exist_ok=True)
        candidate_path = staging_directory / "matches_candidate.csv"
        write_dataframe_csv_atomic(candidate_path, dataframe)
        match_quality_path = QUALITY_DIR / f"extraction={extraction_id}" / "match_quality.jsonl"
        copy_file_atomic(Path(control["match_quality_path"]), match_quality_path)
        return {
            **control, "candidate_path": str(candidate_path), "match_quality_path": str(match_quality_path),
            "rows": len(dataframe), "columns": len(dataframe.columns),
        }

    @task
    def validate_and_report(transformation: dict[str, Any]) -> dict[str, Any]:
        dataframe = pd.read_csv(transformation["candidate_path"], low_memory=False)
        duplicate_ids = int(dataframe["match_id"].duplicated().sum())
        all_null_columns = dataframe.columns[dataframe.isna().all()].tolist()
        target_values = set(dataframe["target_team_a_won"].dropna().astype(int).unique())
        winner_valid = ((dataframe["winner_team_id"] == dataframe["team_a_id"]) | (dataframe["winner_team_id"] == dataframe["team_b_id"])).all()
        checks = {
            "schema_exact": list(dataframe.columns) == SILVER_COLUMNS,
            "at_least_1000_rows": len(dataframe) >= MIN_SILVER_ROWS,
            "primary_key_not_null": not dataframe["match_id"].isna().any(),
            "primary_key_unique": duplicate_ids == 0,
            "teams_are_distinct": bool((dataframe["team_a_id"] != dataframe["team_b_id"]).all()),
            "winner_is_participant": bool(winner_valid),
            "binary_target": target_values.issubset({0, 1}) and bool(target_values),
            "target_not_null": not dataframe["target_team_a_won"].isna().any(),
            "no_all_null_columns": not all_null_columns,
            "at_least_5_useful_columns": int(dataframe.notna().any().sum()) >= 5,
        }
        errors = [name for name, passed in checks.items() if not passed]
        registry = (read_json(REGISTRY_PATH, {"matches": {}}) or {}).get("matches", {})
        lifecycle_counts = Counter(value.get("lifecycle_state", "unknown") for value in registry.values())
        report = {
            "extraction_id": transformation["extraction_id"], "generated_at_utc": iso_utc(),
            "valid": not errors, "rows": len(dataframe), "columns": len(dataframe.columns),
            "primary_key": "match_id", "target": "target_team_a_won",
            "duplicate_match_ids": duplicate_ids, "all_null_columns": all_null_columns,
            "target_distribution": dataframe["target_team_a_won"].value_counts(dropna=False).to_dict(),
            "parsed_status_distribution": dataframe["parsed_status"].value_counts(dropna=False).to_dict(),
            "data_completeness_distribution": dataframe["data_completeness"].value_counts(dropna=False).to_dict(),
            "lifecycle_distribution": dict(lifecycle_counts),
            "null_fraction_by_column": dataframe.isna().mean().sort_values(ascending=False).to_dict(),
            "discarded": transformation["discarded"], "checks": checks, "errors": errors,
        }
        report_directory = QUALITY_DIR / f"extraction={transformation['extraction_id']}"
        report_path = report_directory / "quality_report.json"
        write_json_atomic(report_path, report)
        if errors:
            raise ValueError("Fallaron validaciones de Silver: " + ", ".join(errors))
        return {**transformation, "quality_report_path": str(report_path)}

    @task
    def publish_dataset(validation: dict[str, Any]) -> dict[str, Any]:
        extraction_id = validation["extraction_id"]
        version_directory = SILVER_DIR / f"extraction={extraction_id}"
        version_directory.mkdir(parents=True, exist_ok=True)
        version_path = version_directory / "matches.csv"
        copy_file_atomic(Path(validation["candidate_path"]), version_path)
        metadata = {
            "dataset": "Bo3.gg CS2 matches",
            "unit_of_analysis": "one finished competitive match between two teams",
            "primary_key": "match_id", "target": "target_team_a_won",
            "extraction_id": extraction_id, "airflow_run_id": validation["airflow_run_id"],
            "rows": validation["rows"], "columns": validation["columns"],
            "published_at_utc": iso_utc(), "quality_report_path": validation["quality_report_path"],
            "notes": [
                "post_* columns describe the observed match and must not be used to predict that same match",
                "team ranks are values observed at extraction time, not guaranteed historical ranks",
                "Gold historical features must use only matches with start_at_utc earlier than the predicted match",
            ],
        }
        write_json_atomic(version_directory / "metadata.json", metadata)
        SILVER_DIR.mkdir(parents=True, exist_ok=True)
        latest_path = SILVER_DIR / "matches_latest.csv"
        copy_file_atomic(version_path, latest_path)
        write_json_atomic(SILVER_DIR / "metadata_latest.json", metadata)
        return {**validation, "dataset_path": str(version_path), "latest_path": str(latest_path), "metadata_path": str(version_directory / "metadata.json")}

    @task
    def finalize_run(publication: dict[str, Any]) -> str:
        run_directory = Path(publication["run_directory"])
        manifest_path = run_directory / "manifest.json"
        manifest = {
            "extraction_id": publication["extraction_id"], "airflow_run_id": publication["airflow_run_id"],
            "completed_at_utc": iso_utc(), "dataset_path": publication["dataset_path"],
            "latest_path": publication["latest_path"], "quality_report_path": publication["quality_report_path"],
            "rows": publication["rows"], "columns": publication["columns"],
            "dates_discovered": publication["dates_discovered"],
            "matches_discovered": publication["matches_discovered"],
            "matches_enriched": publication["matches_enriched"],
            "discarded": publication["discarded"],
        }
        write_json_atomic(manifest_path, manifest)
        success_path = run_directory / "_SUCCESS"
        write_json_atomic(success_path, {**manifest, "manifest_path": str(manifest_path)})
        return str(success_path)

    plan = build_plan()
    batches = date_batches(plan)
    discoveries = discover_batch.expand(batch=batches)
    enrichment_queues = prepare_enrichment(plan, discoveries)
    enrichment_results = enrich_batch.expand(queue_path=enrichment_queues)
    control = update_control(plan, discoveries, enrichment_results)
    candidate = transform_to_silver(control)
    validation = validate_and_report(candidate)
    publication = publish_dataset(validation)
    finalize_run(publication)


bo3_cs2_ingest()
