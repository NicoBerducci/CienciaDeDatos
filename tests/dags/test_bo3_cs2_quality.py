"""Casos sinteticos acotados. Sin red ni reconstruccion del Bronze real."""
from copy import deepcopy
from pathlib import Path
import runpy

from dags.bo3_cs2_quality import evaluate_match_quality, map_score_valid, series_score_valid
from dags.bo3_cs2_features import build_point_in_time_silver


def task_namespace():
    return runpy.run_path(str(Path(__file__).with_name("test_bo3_cs2_incremental.py")))["load_task_functions"]()


def detail(**changes):
    d = {"id": 1, "slug": "fixture", "game_version": 2, "status": "finished", "bo_type": 3,
         "team1_id": 20, "team2_id": 10, "winner_team_id": 20, "team1_score": 2, "team2_score": 0,
         "team1": {"id": 20, "name": "Twenty"}, "team2": {"id": 10, "name": "Ten"},
         "start_date": "2026-01-01T12:00:00Z", "end_date": "2026-01-01T14:00:00Z"}
    d.update(changes)
    return d


def games():
    return [{"id": i, "number": i, "map_name": "de_mirage", "status": "finished",
             "winner_team_id": 20, "loser_team_id": 10, "winner_clan_score": 13, "loser_clan_score": 5,
             "rounds_count": 18, "begin_at": f"2026-01-01T{11+i}:05:00Z"} for i in (1, 2)]


def players():
    return [{"id": tid * 100 + i, "steam_profile_id": tid * 100 + i,
             "team_clan": {"team_id": tid}, "kills": 10, "death": 9, "assists": 3,
             "player_rating": 6, "adr": 70, "kast": 0.7} for tid in (10, 20) for i in range(5)]


def normalize(d, gs, ps=None):
    return task_namespace()["build_match_row"](d, gs, ps or [], [], [],
        data_completeness="partial", rank_extracted_at_utc=None)[0]


def test_series_score_is_mapped_by_team_id_not_source_order():
    row = normalize(detail(), games())
    assert row["team_a_id"] == 10 and row["post_team_a_maps_won"] == 0
    assert row["post_team_b_maps_won"] == 2 and row["target_team_a_won"] == 0
    assert row["maps_valid"] and row["map_stats_valid"]


def test_tied_count_preserves_valid_series_but_blocks_dependent_stats():
    gs = games()
    gs[1].update(winner_team_id=10, loser_team_id=20)
    row = normalize(detail(), gs, players())
    assert row["result_valid"] and row["maps_valid"]
    assert row["post_team_b_maps_won"] == 2
    assert not row["map_stats_valid"] and not row["round_scores_valid"]
    assert not row["team_a_rating_valid"] and row["post_team_a_kills"] is None
    assert row["post_team_a_player_rating_mean"] is None


def test_internal_detail_tie_is_not_a_binary_training_label():
    row = normalize(detail(team1_score=1, team2_score=1), games())
    assert row is not None and not row["result_valid"]
    assert row["quality_state"] == "quarantined_result"
    assert build_point_in_time_silver([row]) == []


def test_winner_conflict_is_not_repaired_by_flipping_target():
    row = normalize(detail(winner_team_id=10), games())
    assert row["target_team_a_won"] == 1  # original source kept for audit
    assert not row["result_valid"]
    assert build_point_in_time_silver([row]) == []


def test_missing_map_does_not_make_partial_totals_complete():
    row = normalize(detail(), games()[:1], players())
    assert row["result_valid"] and row["maps_valid"]
    assert not row["maps_complete"] and not row["map_stats_valid"]
    assert not row["team_b_mechanical_totals_valid"]


def test_valid_maps_can_use_full_players_when_round_details_are_missing():
    row = normalize(detail(), games(), players())
    assert row["map_stats_valid"] and not row["round_details_complete"]
    assert row["team_a_mechanical_stats_source"] == "full_players"
    assert row["team_a_rating_valid"] and row["post_team_a_kills"] == 50


def test_finished_map_with_incomplete_score_is_not_valid():
    gs = games()
    gs[0].update(winner_clan_score=9, loser_clan_score=12, rounds_count=21)
    q = evaluate_match_quality(detail(), gs)
    assert q["result_valid"] and not q["map_stats_valid"]


def test_duplicate_or_skipped_game_numbers_are_incomplete():
    for field, value in (("id", 1), ("number", 1), ("number", 3)):
        gs = games()
        gs[1][field] = value
        assert not evaluate_match_quality(detail(), gs)["map_stats_valid"]


def test_round_count_must_match_reported_score():
    gs = games()
    gs[0]["rounds_count"] = 17
    assert not evaluate_match_quality(detail(), gs)["map_stats_valid"]


def test_missing_end_stays_null_and_never_feeds_next_match():
    row = normalize(detail(end_date=None), games())
    later = normalize(detail(id=2, start_date="2026-01-03T12:00:00Z", end_date="2026-01-03T14:00:00Z"), [])
    assert row["start_time_valid"] and not row["end_time_valid"]
    silver = build_point_in_time_silver([row, later])
    assert len(silver) == 2 and silver[0]["end_at_utc"] is None
    assert silver[1]["team_a_matches_prior"] == 0


def test_conflicting_chronology_excludes_row_and_all_history():
    cases = [detail(end_date="2026-01-01T11:00:00Z"),
             detail(end_date="2026-01-01T13:00:00Z"),
             detail(start_date="2026-01-01T12:10:00Z")]
    for d in cases:
        row = normalize(d, games())
        assert not row["start_time_valid"]
        assert row["quality_state"] == "quarantined_temporal"
        assert build_point_in_time_silver([row]) == []


def test_confirmed_forfeit_requires_exact_reviewed_identity():
    d = detail(id=100556, slug="1win-vs-m1x-cs-25-09-2025", team1_id=238, team2_id=22604, winner_team_id=238)
    q = evaluate_match_quality(d, [])
    assert q["quality_state"] == "excluded_forfeit" and q["reviewed_decision_source"]
    assert normalize(d, []) is None
    d["slug"] = "another-match"
    assert evaluate_match_quality(d, [])["quality_state"] != "excluded_forfeit"


def test_missing_statistics_alone_do_not_imply_forfeit():
    q = evaluate_match_quality(detail(end_date=None), [])
    assert q["result_valid"] and q["quality_state"] == "eligible_partial"


def test_score_format_bounds_and_overtime():
    assert series_score_valid(2, 0, 2)
    assert not series_score_valid(1, 1, 2)
    assert not series_score_valid(1, 4, 5)
    for a, b in ((13, 0), (13, 11), (16, 12), (16, 14), (19, 17)):
        assert map_score_valid(a, b, 2)
    for a, b in ((12, 11), (13, 12), (15, 13), (16, 15), (9, 12)):
        assert not map_score_valid(a, b, 2)


def test_result_quarantine_does_not_update_elo_or_counts():
    bad = normalize(detail(team1_score=1, team2_score=1), games())
    later = normalize(detail(id=2, start_date="2026-01-03T12:00:00Z", end_date="2026-01-03T14:00:00Z"), [])
    row = build_point_in_time_silver([bad, later])[0]
    assert row["match_id"] == 2 and row["team_a_elo"] == 1500 and row["team_a_matches_prior"] == 0


def test_quality_functions_do_not_mutate_bronze_payload():
    d, gs = detail(), games()
    original = deepcopy((d, gs))
    normalize(d, gs, players())
    assert (d, gs) == original
