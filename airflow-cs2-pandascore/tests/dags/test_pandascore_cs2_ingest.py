"""Pruebas pequeñas para las decisiones más importantes del pipeline."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "dags"))

from pandascore_cs2_ingest import extraction_pages, normalize_match  # noqa: E402


def test_bootstrap_downloads_every_page():
    assert extraction_pages(0, 250) == ("bootstrap", [1, 2, 3])


def test_incremental_refreshes_an_incomplete_last_page():
    assert extraction_pages(250, 270) == ("incremental", [3])


def test_incremental_starts_a_new_page_after_a_complete_one():
    assert extraction_pages(300, 320) == ("incremental", [4])


def test_no_changes_downloads_no_pages():
    assert extraction_pages(320, 320) == ("no_changes", [])


def test_remote_total_cannot_be_smaller_than_bronze():
    with pytest.raises(ValueError):
        extraction_pages(320, 319)


def test_team_a_is_the_team_with_the_lower_id():
    match = {
        "id": 10,
        "status": "finished",
        "begin_at": "2026-09-01T12:00:00Z",
        "winner_id": 200,
        "opponents": [
            {"type": "Team", "opponent": {"id": 200, "name": "Segundo"}},
            {"type": "Team", "opponent": {"id": 100, "name": "Primero"}},
        ],
        "results": [
            {"team_id": 200, "score": 2},
            {"team_id": 100, "score": 1},
        ],
    }

    row, reason = normalize_match(match)

    assert reason is None
    assert row["team_a_id"] == 100
    assert row["team_b_id"] == 200
    assert row["team_a_win"] == 0


def test_match_without_winner_is_discarded():
    match = {
        "id": 10,
        "status": "finished",
        "begin_at": "2026-09-01T12:00:00Z",
        "winner_id": None,
        "opponents": [
            {"type": "Team", "opponent": {"id": 100, "name": "A"}},
            {"type": "Team", "opponent": {"id": 200, "name": "B"}},
        ],
    }

    row, reason = normalize_match(match)

    assert row is None
    assert reason == "winner_missing"


def test_quality_check_boolean_is_json_serializable():
    dataframe = pd.DataFrame({f"column_{number}": [1] for number in range(5)})

    check = bool((dataframe.notna().any()).sum() >= 5)

    assert json.dumps({"at_least_5_useful_columns": check})
