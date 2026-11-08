from datetime import datetime, timedelta, timezone
from copy import deepcopy

from dags.bo3_cs2_features import (
    POST_MATCH_METADATA_COLUMNS,
    PREDICTOR_EXCLUDED_COLUMNS,
    SILVER_COLUMNS,
    attach_historical_rankings,
    build_point_in_time_silver,
    iter_point_in_time_silver,
    parse_official_ranking_dates,
)


def _fact(match_id, start, end, winner_a=1, *, stats_valid=True):
    return {
        "match_id": match_id,
        "game_version": 2,
        "match_slug": f"match-{match_id}",
        "start_at_utc": start.isoformat(),
        "end_at_utc": end.isoformat() if end else None,
        "end_time_valid": end is not None and end >= start,
        "result_valid": True,
        "team_a_id": 1,
        "team_b_id": 2,
        "target_team_a_won": winner_a,
        "maps_valid": True,
        "round_scores_valid": True,
        "post_team_a_maps_won": 2 if winner_a else 1,
        "post_team_b_maps_won": 1 if winner_a else 2,
        "post_team_a_rounds_won": 30 if winner_a else 20,
        "post_team_b_rounds_won": 20 if winner_a else 30,
        "team_a_mechanical_totals_valid": stats_valid,
        "team_b_mechanical_totals_valid": stats_valid,
        "post_team_a_kills": 100,
        "post_team_a_deaths": 80,
        "post_team_a_assists": 20,
        "post_team_b_kills": 80,
        "post_team_b_deaths": 100,
        "post_team_b_assists": 20,
    }


def test_ranking_join_is_strictly_before_match_date():
    fact = {"match_id": 1, "start_at_utc": "2026-01-10T12:00:00Z", "team_a_id": 1, "team_b_id": 2}
    rankings = {
        "2026-01-09": {"is_official": True, "teams": {1: {"rank": 3}, 2: {"rank": 7}}},
        "2026-01-10": {"is_official": True, "teams": {1: {"rank": 1}, 2: {"rank": 2}}},
    }
    joined = attach_historical_rankings([fact], rankings)[0]
    assert joined["ranking_snapshot_date"] == "2026-01-09"
    assert joined["team_a_rank"] == 3


def test_match_only_feeds_rows_after_its_end():
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first = _fact(1, origin, origin + timedelta(hours=2))
    overlapping = _fact(2, origin + timedelta(hours=1), origin + timedelta(hours=3))
    later = _fact(3, origin + timedelta(hours=4), origin + timedelta(hours=6))
    rows = build_point_in_time_silver([first, overlapping, later])
    assert rows[1]["team_a_matches_prior"] == 0
    assert rows[2]["team_a_matches_prior"] == 2


def test_invalid_end_never_feeds_future_history():
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    invalid = _fact(1, origin, None)
    later = _fact(2, origin + timedelta(days=1), origin + timedelta(days=1, hours=2))
    rows = build_point_in_time_silver([invalid, later])
    assert rows[1]["team_a_matches_prior"] == 0


def test_minimum_observations_keeps_count_and_nulls_value():
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    facts = [
        _fact(index, origin + timedelta(days=index), origin + timedelta(days=index, hours=2))
        for index in range(1, 6)
    ]
    rows = build_point_in_time_silver(facts)
    assert rows[4]["team_a_kda_last20_count"] == 4
    assert rows[4]["team_a_kda_last20"] is None


def test_partial_match_feeds_result_but_not_kda():
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    facts = [
        _fact(index, origin + timedelta(days=index), origin + timedelta(days=index, hours=2), stats_valid=False)
        for index in range(1, 7)
    ]
    rows = build_point_in_time_silver(facts)
    assert rows[-1]["team_a_winrate_last10_count"] == 5
    assert rows[-1]["team_a_winrate_last10"] == 1.0
    assert rows[-1]["team_a_kda_last20_count"] == 0
    assert rows[-1]["team_a_kda_last20"] is None


def test_metric_window_does_not_reach_past_twenty_matches():
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    facts = [_fact(1, origin, origin + timedelta(hours=2), stats_valid=True)]
    facts.extend(
        _fact(
            index,
            origin + timedelta(days=index),
            origin + timedelta(days=index, hours=2),
            stats_valid=False,
        )
        for index in range(2, 23)
    )
    rows = build_point_in_time_silver(facts)
    assert rows[-1]["team_a_matches_prior"] == 21
    assert rows[-1]["team_a_kda_last20_count"] == 0


def test_official_dates_parser_accepts_strings_and_objects():
    assert parse_official_ranking_dates({"data": ["2026-01-01", {"ranking_date": "2026-02-01"}]}) == [
        "2026-01-01", "2026-02-01"
    ]


def test_non_cs2_matches_do_not_change_any_cs2_feature():
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    cs2 = [
        _fact(i, origin + timedelta(days=i), origin + timedelta(days=i, hours=2))
        for i in range(1, 9)
    ]
    excluded = []
    for i, version in enumerate((1, None, "", 3, float("nan"), "unknown"), start=100):
        fact = _fact(i, origin, origin + timedelta(hours=1), winner_a=0)
        fact["game_version"] = version
        excluded.append(fact)
    # Equality across every column covers Elo, H2H, counts, rest and all windows.
    assert build_point_in_time_silver(excluded + cs2) == build_point_in_time_silver(cs2)
    assert build_point_in_time_silver(excluded) == []


def test_cs2_version_accepts_numeric_serializations_not_missing_version():
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for version in (2, 2.0, "2", "2.0"):
        fact = _fact(1, origin, origin + timedelta(hours=2))
        fact["game_version"] = version
        assert len(build_point_in_time_silver([fact])) == 1
    del fact["game_version"]
    assert build_point_in_time_silver([fact]) == []


def test_result_metadata_is_preserved_but_current_result_does_not_change_features():
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    prior = _fact(1, origin, origin + timedelta(hours=2))
    current = _fact(2, origin + timedelta(days=1), origin + timedelta(days=1, hours=2))
    row = build_point_in_time_silver([prior, current])[-1]
    assert list(row) == SILVER_COLUMNS
    assert len(SILVER_COLUMNS) == 113
    for column in POST_MATCH_METADATA_COLUMNS:
        assert row[column] == current[column]
        assert column in PREDICTOR_EXCLUDED_COLUMNS
    modified = dict(current, target_team_a_won=0, post_team_a_maps_won=0,
                    post_team_b_maps_won=2, end_at_utc=None, end_time_valid=False)
    changed_row = build_point_in_time_silver([prior, modified])[-1]
    for column in SILVER_COLUMNS:
        if column not in PREDICTOR_EXCLUDED_COLUMNS:
            assert changed_row[column] == row[column]


def test_missing_result_metadata_stays_null_without_dropping_match():
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fact = _fact(1, origin, None)
    fact.update(post_team_a_maps_won=None, post_team_b_maps_won=None, maps_valid=False)
    row = build_point_in_time_silver([fact])[0]
    assert all(row[column] is None for column in POST_MATCH_METADATA_COLUMNS)


def test_streamed_silver_does_not_mutate_input_and_keeps_time_order():
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first = _fact(9, origin, origin + timedelta(hours=2))
    second = _fact(1, origin + timedelta(days=1), origin + timedelta(days=1, hours=2))
    facts = [second, first]
    original = deepcopy(facts)
    rows = list(iter_point_in_time_silver(facts))
    assert facts == original
    assert [row["match_id"] for row in rows] == [9, 1]
    assert [row["team_a_matches_prior"] for row in rows] == [0, 1]
    assert rows[0]["team_a_elo"] == 1500
    assert rows[1]["team_a_elo"] == 1516
