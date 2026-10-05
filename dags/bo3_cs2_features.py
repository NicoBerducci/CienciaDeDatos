"""Transformaciones puras para las features point-in-time de CS2.

Este modulo no importa Airflow ni pandas. De ese modo, las reglas temporales y
estadisticas pueden probarse sin levantar el scheduler.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable


INITIAL_ELO = 1500.0
ELO_K_FACTOR = 32.0
MIN_ROLLING_OBSERVATIONS = 5


IDENTITY_COLUMNS = [
    "match_id", "match_slug", "start_at_utc", "bo_type", "tier", "game_version",
    "tournament_id", "tournament_name", "stage_id", "stage_name",
    "team_a_id", "team_a_name", "team_a_country_code",
    "team_b_id", "team_b_name", "team_b_country_code", "target_team_a_won",
]

RANKING_COLUMNS = [
    "team_a_rank", "team_b_rank", "rank_advantage_a",
    "team_a_ranking_score", "team_b_ranking_score", "ranking_score_advantage_a",
    "ranking_age_days",
]

TEAM_FEATURE_SUFFIXES = [
    "matches_prior",
    "elo",
    "winrate_last10", "winrate_last10_count",
    "winrate_last20", "winrate_last20_count",
    "form_last5", "form_last5_count", "form_trend_5v5",
    "kda_last20", "kda_last20_count",
    "kd_last20", "kd_last20_count",
    "rating_last20", "rating_last20_count",
    "adr_last20", "adr_last20_count",
    "kast_last20", "kast_last20_count",
    "map_winrate_last20", "map_winrate_last20_count",
    "round_winrate_last20", "round_winrate_last20_count",
    "round_diff_per_match_last20", "round_diff_last20_count",
    "same_bo_winrate_last20", "same_bo_winrate_last20_count",
    "same_tier_winrate_last20", "same_tier_winrate_last20_count",
    "rest_days", "matches_last7d", "matches_last30d",
    "lineup_stability_last5", "lineup_stability_last5_count",
]

ADVANTAGE_COLUMNS = [
    "elo_advantage_a", "winrate_last10_advantage_a", "winrate_last20_advantage_a",
    "form_last5_advantage_a", "kda_last20_advantage_a", "kd_last20_advantage_a",
    "rating_last20_advantage_a", "adr_last20_advantage_a",
    "kast_last20_advantage_a", "map_winrate_last20_advantage_a",
    "round_winrate_last20_advantage_a", "round_diff_last20_advantage_a",
    "same_bo_winrate_last20_advantage_a", "same_tier_winrate_last20_advantage_a",
    "rest_days_advantage_a", "lineup_stability_last5_advantage_a",
]

SILVER_COLUMNS = (
    IDENTITY_COLUMNS
    + RANKING_COLUMNS
    + [f"team_{side}_{suffix}" for side in ("a", "b") for suffix in TEAM_FEATURE_SUFFIXES]
    + ["h2h_team_a_winrate_last5", "h2h_last5_count"]
    + ADVANTAGE_COLUMNS
)


def parse_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric == numeric else None


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value != value:
            return False
        return bool(value)
    return str(value or "").strip().casefold() in {"1", "true", "yes", "si", "sí"}


def parse_official_ranking_dates(payload: Any) -> list[str]:
    """Acepta las variantes observadas del endpoint de fechas oficiales."""

    values = payload.get("data", []) if isinstance(payload, dict) else payload
    dates: set[str] = set()
    if not isinstance(values, list):
        return []
    for value in values:
        candidate = value
        if isinstance(value, dict):
            candidate = value.get("ranking_date") or value.get("date") or value.get("value")
        if candidate:
            try:
                dates.add(datetime.fromisoformat(str(candidate)[:10]).date().isoformat())
            except ValueError:
                continue
    return sorted(dates)


def normalize_ranking_payloads(payloads: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Normaliza paginas V2 como ``fecha -> {teams, meta}``."""

    normalized: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        meta = payload.get("meta") or {}
        records = payload.get("data") or payload.get("results") or []
        if not isinstance(records, list):
            continue
        ranking_date = str(meta.get("ranking_date") or "")[:10]
        for record in records:
            if not isinstance(record, dict):
                continue
            record_date = str(record.get("ranking_date") or ranking_date)[:10]
            team = record.get("team") or {}
            team_id = record.get("team_id") or team.get("id")
            if not record_date or team_id is None:
                continue
            snapshot = normalized.setdefault(record_date, {
                "teams": {},
                "is_official": bool(meta.get("is_official", True)),
                "source": meta.get("source"),
                "source_updated_at": meta.get("updated_at"),
            })
            snapshot["teams"][int(team_id)] = {
                "rank": as_float(record.get("rank")),
                "score": as_float(record.get("score")),
            }
    return normalized


def attach_historical_rankings(
    facts: Iterable[dict[str, Any]],
    rankings_by_date: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Une el ultimo ranking oficial estrictamente anterior al partido."""

    official_dates = sorted(
        date for date, snapshot in rankings_by_date.items()
        if snapshot.get("is_official", True)
    )
    output: list[dict[str, Any]] = []
    for original in facts:
        fact = dict(original)
        start = parse_datetime(fact.get("start_at_utc"))
        snapshot_date = None
        if start is not None:
            position = bisect_left(official_dates, start.date().isoformat()) - 1
            if position >= 0:
                snapshot_date = official_dates[position]
        snapshot = rankings_by_date.get(snapshot_date, {}) if snapshot_date else {}
        teams = snapshot.get("teams", {})
        team_a = teams.get(int(fact["team_a_id"]), {})
        team_b = teams.get(int(fact["team_b_id"]), {})
        fact.update({
            "team_a_rank": team_a.get("rank"),
            "team_b_rank": team_b.get("rank"),
            "team_a_ranking_score": team_a.get("score"),
            "team_b_ranking_score": team_b.get("score"),
            "ranking_snapshot_date": snapshot_date,
            "ranking_age_days": (start.date() - datetime.fromisoformat(snapshot_date).date()).days
            if start is not None and snapshot_date else None,
            "ranking_is_official": snapshot.get("is_official") if snapshot else None,
            "ranking_source": snapshot.get("source") if snapshot else None,
            "ranking_source_updated_at": snapshot.get("source_updated_at") if snapshot else None,
        })
        output.append(fact)
    return output


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _published(value: float | None, count: int, minimum: int = MIN_ROLLING_OBSERVATIONS) -> float | None:
    return value if count >= minimum else None


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def _lineup(value: Any) -> frozenset[int] | None:
    if value is None or value == "":
        return None
    try:
        result = frozenset(int(item) for item in str(value).split(";") if item)
    except ValueError:
        return None
    return result or None


def _team_observation(fact: dict[str, Any], side: str) -> dict[str, Any]:
    opponent = "b" if side == "a" else "a"
    target_a_won = int(as_float(fact.get("target_team_a_won")) or 0)
    won = int(target_a_won == (1 if side == "a" else 0))
    return {
        "match_id": int(fact["match_id"]),
        "end": parse_datetime(fact.get("end_at_utc")),
        "opponent_id": int(fact[f"team_{opponent}_id"]),
        "won": won,
        "bo_type": fact.get("bo_type"),
        "tier": fact.get("tier"),
        "maps_won": as_float(fact.get(f"post_team_{side}_maps_won")) if as_bool(fact.get("maps_valid")) else None,
        "maps_lost": as_float(fact.get(f"post_team_{opponent}_maps_won")) if as_bool(fact.get("maps_valid")) else None,
        "rounds_won": as_float(fact.get(f"post_team_{side}_rounds_won")) if as_bool(fact.get("round_scores_valid")) else None,
        "rounds_lost": as_float(fact.get(f"post_team_{opponent}_rounds_won")) if as_bool(fact.get("round_scores_valid")) else None,
        "kills": as_float(fact.get(f"post_team_{side}_kills")) if as_bool(fact.get(f"team_{side}_mechanical_totals_valid")) else None,
        "deaths": as_float(fact.get(f"post_team_{side}_deaths")) if as_bool(fact.get(f"team_{side}_mechanical_totals_valid")) else None,
        "assists": as_float(fact.get(f"post_team_{side}_assists")) if as_bool(fact.get(f"team_{side}_mechanical_totals_valid")) else None,
        "rating": as_float(fact.get(f"post_team_{side}_player_rating_mean")) if as_bool(fact.get(f"team_{side}_rating_valid")) else None,
        "adr": as_float(fact.get(f"post_team_{side}_adr_mean")) if as_bool(fact.get(f"team_{side}_adr_valid")) else None,
        "kast": as_float(fact.get(f"post_team_{side}_kast_mean")) if as_bool(fact.get(f"team_{side}_kast_valid")) else None,
        "lineup": _lineup(fact.get(f"team_{side}_lineup_profile_ids")) if as_bool(fact.get(f"team_{side}_lineup_valid")) else None,
    }


def _team_features(
    history: list[dict[str, Any]],
    *,
    start: datetime,
    bo_type: Any,
    tier: Any,
    elo: float,
) -> dict[str, Any]:
    last20 = history[-20:]
    last10 = history[-10:]
    last5 = history[-5:]
    previous5 = history[-10:-5]

    result10 = [item["won"] for item in last10]
    result20 = [item["won"] for item in last20]
    result5 = [item["won"] for item in last5]
    prior5 = [item["won"] for item in previous5]

    mechanics = [item for item in last20 if None not in (item["kills"], item["deaths"], item["assists"])]
    kills = sum(item["kills"] for item in mechanics)
    deaths = sum(item["deaths"] for item in mechanics)
    assists = sum(item["assists"] for item in mechanics)
    kda = _ratio(kills + assists, deaths)
    kd = _ratio(kills, deaths)

    ratings = [item["rating"] for item in last20 if item["rating"] is not None]
    adrs = [item["adr"] for item in last20 if item["adr"] is not None]
    kasts = [item["kast"] for item in last20 if item["kast"] is not None]
    map_rows = [item for item in last20 if item["maps_won"] is not None and item["maps_lost"] is not None]
    round_rows = [item for item in last20 if item["rounds_won"] is not None and item["rounds_lost"] is not None]
    same_bo = [item["won"] for item in last20 if bo_type is not None and item["bo_type"] == bo_type]
    same_tier = [item["won"] for item in last20 if tier is not None and item["tier"] == tier]

    map_won = sum(item["maps_won"] for item in map_rows)
    map_total = map_won + sum(item["maps_lost"] for item in map_rows)
    round_won = sum(item["rounds_won"] for item in round_rows)
    round_lost = sum(item["rounds_lost"] for item in round_rows)

    lineups = [item["lineup"] for item in last5 if item["lineup"]]
    similarities: list[float] = []
    for previous, current in zip(lineups, lineups[1:]):
        union = previous | current
        if union:
            similarities.append(len(previous & current) / len(union))

    last_end = history[-1]["end"] if history else None

    def matches_since(days: int) -> int:
        count = 0
        for item in reversed(history):
            if (start - item["end"]).total_seconds() > days * 86400:
                break
            count += 1
        return count

    return {
        "matches_prior": len(history),
        "elo": elo,
        "winrate_last10": _published(_mean(result10), len(result10)),
        "winrate_last10_count": len(result10),
        "winrate_last20": _published(_mean(result20), len(result20)),
        "winrate_last20_count": len(result20),
        "form_last5": _published(_mean(result5), len(result5)),
        "form_last5_count": len(result5),
        "form_trend_5v5": (_mean(result5) - _mean(prior5)) if len(result5) == 5 and len(prior5) == 5 else None,
        "kda_last20": _published(kda, len(mechanics)),
        "kda_last20_count": len(mechanics),
        "kd_last20": _published(kd, len(mechanics)),
        "kd_last20_count": len(mechanics),
        "rating_last20": _published(_mean(ratings), len(ratings)),
        "rating_last20_count": len(ratings),
        "adr_last20": _published(_mean(adrs), len(adrs)),
        "adr_last20_count": len(adrs),
        "kast_last20": _published(_mean(kasts), len(kasts)),
        "kast_last20_count": len(kasts),
        "map_winrate_last20": _published(_ratio(map_won, map_total), len(map_rows)),
        "map_winrate_last20_count": len(map_rows),
        "round_winrate_last20": _published(_ratio(round_won, round_won + round_lost), len(round_rows)),
        "round_winrate_last20_count": len(round_rows),
        "round_diff_per_match_last20": _published(
            _mean(item["rounds_won"] - item["rounds_lost"] for item in round_rows), len(round_rows)
        ),
        "round_diff_last20_count": len(round_rows),
        "same_bo_winrate_last20": _published(_mean(same_bo), len(same_bo)),
        "same_bo_winrate_last20_count": len(same_bo),
        "same_tier_winrate_last20": _published(_mean(same_tier), len(same_tier)),
        "same_tier_winrate_last20_count": len(same_tier),
        "rest_days": (start - last_end).total_seconds() / 86400 if last_end else None,
        "matches_last7d": matches_since(7),
        "matches_last30d": matches_since(30),
        "lineup_stability_last5": _mean(similarities),
        "lineup_stability_last5_count": len(similarities),
    }


def _difference(left: Any, right: Any) -> float | None:
    left_number, right_number = as_float(left), as_float(right)
    return left_number - right_number if left_number is not None and right_number is not None else None


def build_point_in_time_silver(facts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Calcula features usando exclusivamente partidos terminados antes del actual."""

    eligible = [
        dict(fact) for fact in facts
        if parse_datetime(fact.get("start_at_utc")) is not None and as_bool(fact.get("result_valid", True))
    ]
    eligible.sort(key=lambda fact: (parse_datetime(fact["start_at_utc"]), int(fact["match_id"])))
    completed = sorted(
        [fact for fact in eligible if as_bool(fact.get("end_time_valid")) and parse_datetime(fact.get("end_at_utc"))],
        key=lambda fact: (parse_datetime(fact["end_at_utc"]), int(fact["match_id"])),
    )

    histories: dict[int, list[dict[str, Any]]] = defaultdict(list)
    elo: dict[int, float] = defaultdict(lambda: INITIAL_ELO)
    h2h: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    completed_position = 0
    output: list[dict[str, Any]] = []

    def add_completed(fact: dict[str, Any]) -> None:
        team_a, team_b = int(fact["team_a_id"]), int(fact["team_b_id"])
        obs_a, obs_b = _team_observation(fact, "a"), _team_observation(fact, "b")
        histories[team_a].append(obs_a)
        histories[team_b].append(obs_b)
        pair = tuple(sorted((team_a, team_b)))
        target_a_won = int(as_float(fact.get("target_team_a_won")) or 0)
        h2h[pair].append({"winner_id": team_a if target_a_won == 1 else team_b})
        expected_a = 1.0 / (1.0 + 10.0 ** ((elo[team_b] - elo[team_a]) / 400.0))
        actual_a = float(target_a_won)
        delta = ELO_K_FACTOR * (actual_a - expected_a)
        elo[team_a] += delta
        elo[team_b] -= delta

    for fact in eligible:
        start = parse_datetime(fact["start_at_utc"])
        while completed_position < len(completed):
            prior = completed[completed_position]
            if parse_datetime(prior["end_at_utc"]) >= start:
                break
            add_completed(prior)
            completed_position += 1

        team_a, team_b = int(fact["team_a_id"]), int(fact["team_b_id"])
        features_a = _team_features(
            histories[team_a], start=start, bo_type=fact.get("bo_type"), tier=fact.get("tier"), elo=elo[team_a]
        )
        features_b = _team_features(
            histories[team_b], start=start, bo_type=fact.get("bo_type"), tier=fact.get("tier"), elo=elo[team_b]
        )
        row = {column: fact.get(column) for column in IDENTITY_COLUMNS}
        row.update({column: fact.get(column) for column in RANKING_COLUMNS})
        row["rank_advantage_a"] = _difference(fact.get("team_b_rank"), fact.get("team_a_rank"))
        row["ranking_score_advantage_a"] = _difference(
            fact.get("team_a_ranking_score"), fact.get("team_b_ranking_score")
        )
        for side, features in (("a", features_a), ("b", features_b)):
            row.update({f"team_{side}_{name}": value for name, value in features.items()})

        pair_history = h2h[tuple(sorted((team_a, team_b)))][-5:]
        row["h2h_last5_count"] = len(pair_history)
        row["h2h_team_a_winrate_last5"] = (
            _mean(int(item["winner_id"] == team_a) for item in pair_history) if pair_history else None
        )
        comparison_sources = {
            "elo_advantage_a": "elo",
            "winrate_last10_advantage_a": "winrate_last10",
            "winrate_last20_advantage_a": "winrate_last20",
            "form_last5_advantage_a": "form_last5",
            "kda_last20_advantage_a": "kda_last20",
            "kd_last20_advantage_a": "kd_last20",
            "rating_last20_advantage_a": "rating_last20",
            "adr_last20_advantage_a": "adr_last20",
            "kast_last20_advantage_a": "kast_last20",
            "map_winrate_last20_advantage_a": "map_winrate_last20",
            "round_winrate_last20_advantage_a": "round_winrate_last20",
            "round_diff_last20_advantage_a": "round_diff_per_match_last20",
            "same_bo_winrate_last20_advantage_a": "same_bo_winrate_last20",
            "same_tier_winrate_last20_advantage_a": "same_tier_winrate_last20",
            "rest_days_advantage_a": "rest_days",
            "lineup_stability_last5_advantage_a": "lineup_stability_last5",
        }
        for output_name, source_name in comparison_sources.items():
            row[output_name] = _difference(features_a.get(source_name), features_b.get(source_name))
        output.append({column: row.get(column) for column in SILVER_COLUMNS})
    return output
