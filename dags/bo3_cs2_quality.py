"""Controles conservadores sobre Bronze; no corrigen ni inventan resultados.

Los resultados incoherentes quedan en apoyo, fuera de Silver e historiales.
Un marcador de serie fiable puede sobrevivir a estadisticas de mapas invalidas.
"""
from datetime import datetime, timezone
from math import isfinite

QUALITY_POLICY_VERSION = "series_temporal_family_v3"

# Excepciones revisadas: evidencia explicita, no inferencias a partir de nulos.
# La identidad completa impide aplicar la decision a otro encuentro.
REVIEWED_EXCLUSIONS = {
    100556: {
        "slug": "1win-vs-m1x-cs-25-09-2025",
        "teams": {238, 22604},
        "reason": "confirmed_forfeit",
        "reviewed_at": "2026-10-16",
        "sources": [
            "https://escorenews.com/en/go?match=616102",
            "https://liquipedia.net/counterstrike/1w_Team/Matches",
        ],
    },
}


def integer(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        n = float(value)
        return int(n) if isfinite(n) and n >= 0 and n.is_integer() else None
    except (ValueError, TypeError, OverflowError):
        return None


def timestamp(value):
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def reviewed_exclusion(detail):
    decision = REVIEWED_EXCLUSIONS.get(integer(detail.get("id")))
    teams = {integer(detail.get("team1_id")), integer(detail.get("team2_id"))}
    if decision and detail.get("slug") == decision["slug"] and teams == decision["teams"]:
        return decision
    return None


def series_score_valid(a, b, best_of):
    """BO2 admite 2-0, pero un empate no tiene target binario valido."""
    if a is None or b is None or a == b or best_of not in {1, 2, 3, 5}:
        return False
    required = best_of // 2 + 1
    return max(a, b) == required and min(a, b) < required and a + b <= best_of


def map_score_valid(won, lost, version):
    # MR12 CS2 / MR15 CS:GO, overtime MR3. Formatos especiales no inferidos.
    if won is None or lost is None or version not in {1, 2}:
        return False
    regulation = 13 if version == 2 else 16
    return ((won == regulation and lost <= regulation - 2)
            or (won >= regulation + 3 and (won - regulation) % 3 == 0
                and 2 <= won - lost <= 4))


def evaluate_match_quality(detail, games):
    t1 = integer(detail.get("team1_id") or (detail.get("team1") or {}).get("id"))
    t2 = integer(detail.get("team2_id") or (detail.get("team2") or {}).get("id"))
    winner = integer(detail.get("winner_team_id") or detail.get("winner_id"))
    teams = {t1, t2}
    bo = integer(detail.get("bo_type"))
    version = integer(detail.get("game_version"))
    a, b = integer(detail.get("team1_score")), integer(detail.get("team2_score"))
    issues = []
    decision = reviewed_exclusion(detail)
    forfeit = bool(decision or any(detail.get(k) for k in ("forfeit", "defwin", "walkover"))
                   or str(detail.get("status", "")).lower() in {"defwin", "forfeit", "walkover", "w/o"})
    base_result = (str(detail.get("status", "")).lower() in {"finished", "done"}
                   and None not in teams and t1 != t2 and winner in teams and not forfeit)
    series_valid = series_score_valid(a, b, bo)
    result_valid = bool(base_result and series_valid and winner == (t1 if a > b else t2))
    if forfeit:
        issues.append("confirmed_forfeit" if decision else "source_forfeit")
    elif not result_valid:
        issues.append("series_result_unverified")
        if a is not None and b is not None and a == b:
            issues.append("series_tied_or_incomplete")
        elif series_valid and winner != (t1 if a > b else t2):
            issues.append("series_score_winner_conflict")

    aliases = {}
    for tid, key in ((t1, "team1"), (t2, "team2")):
        team = detail.get(key) or {}
        names = [team.get("name"), team.get("slug"), team.get("acronym")]
        names += [c.get("clan_name") for c in team.get("team_clans") or [] if isinstance(c, dict)]
        aliases[tid] = {str(n).casefold().strip() for n in names if n}

    def game_team(game, role):
        clan = game.get(role + "_team_clan") or {}
        tid = integer(game.get(role + "_team_id") or clan.get("team_id"))
        if tid in teams and tid is not None:
            return tid
        name = str(game.get(role + "_clan_name") or clan.get("clan_name") or "").casefold().strip()
        found = [tid for tid, names in aliases.items() if name and name in names]
        return found[0] if len(found) == 1 else None

    counted = {t1: 0, t2: 0}
    ids, numbers = [], []
    score_checks = []
    starts = []
    for game in games:
        gw, gl = game_team(game, "winner"), game_team(game, "loser")
        sw, sl = integer(game.get("winner_clan_score")), integer(game.get("loser_clan_score"))
        identities_valid = gw in teams and gl in teams and None not in (gw, gl) and gw != gl
        valid = bool(identities_valid and game.get("map_name")
                     and game.get("status") in {"finished", "done"}
                     and map_score_valid(sw, sl, version))
        if game.get("rounds_count") is not None:
            valid = valid and sw is not None and sl is not None and integer(game["rounds_count"]) == sw + sl
        # No invertir automaticamente un supuesto winner con menos rondas.
        score_checks.append(valid)
        if identities_valid:
            counted[gw] += 1
        ids.append(integer(game.get("id")))
        numbers.append(integer(game.get("number")))
        begin = timestamp(game.get("begin_at"))
        if begin:
            starts.append(begin)

    expected_maps = a + b if result_valid else None
    unique_games = bool(games) and None not in ids and len(ids) == len(set(ids))
    continuous = bool(games) and None not in numbers and sorted(numbers) == list(range(1, len(games) + 1))
    maps_complete = bool(result_valid and unique_games and continuous and len(games) == expected_maps)
    agrees = bool(result_valid and counted[t1] == a and counted[t2] == b)
    map_stats_valid = bool(maps_complete and agrees and all(score_checks))
    if not games:
        issues.append("games_missing")
    else:
        if not maps_complete:
            issues.append("series_maps_incomplete_or_duplicate")
        if result_valid and not agrees:
            issues.append("series_map_winners_conflict")
        if not all(score_checks):
            issues.append("map_score_or_identity_invalid")

    start = timestamp(detail.get("start_date") or detail.get("begin_at"))
    raw_end = detail.get("end_date") or detail.get("end_at")
    end = timestamp(raw_end)
    start_valid = start is not None
    chronology_conflict = bool(start and end and end < start)
    # Una fecha de fin anterior al comienzo de un mapa tampoco es fiable.
    chronology_conflict |= bool(end and starts and max(starts) > end)
    chronology_conflict |= bool(start and starts and min(starts) < start)
    if chronology_conflict:
        start_valid = False
        issues.append("chronology_conflict")
    if start is None:
        issues.append("start_time_missing_or_invalid")
    end_valid = bool(start_valid and end and end >= start)
    if not end_valid:
        issues.append("end_time_missing_or_invalid")

    state = ("excluded_forfeit" if forfeit else "quarantined_result" if not result_valid
             else "quarantined_temporal" if not start_valid
             else "eligible_complete_maps" if map_stats_valid and end_valid else "eligible_partial")
    return {
        "quality_policy_version": QUALITY_POLICY_VERSION,
        "quality_state": state,
        "result_valid": result_valid,
        "start_time_valid": start_valid,
        "end_time_valid": end_valid,
        "series_score_valid": result_valid,
        "maps_complete": maps_complete,
        "map_stats_valid": map_stats_valid,
        "statistics_blocked": not map_stats_valid,
        "source_team1_score": detail.get("team1_score"),
        "source_team2_score": detail.get("team2_score"),
        "series_score_source": "match_detail" if result_valid else None,
        "reviewed_decision_source": ";".join(decision["sources"]) if decision else None,
        "issues": sorted(set(issues)),
    }
