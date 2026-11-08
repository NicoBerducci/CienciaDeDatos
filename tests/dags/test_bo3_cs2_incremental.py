"""Pruebas offline de cuerpos de tareas, sin scheduler ni llamadas HTTP.

Se extraen las funciones del DAG mediante AST para ejecutar el codigo real sin
los decoradores de Airflow. Esto no reemplaza el test de importacion de DagBag.
"""

import ast
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory


def load_task_functions():
    source = Path(__file__).resolve().parents[2] / "dags" / "bo3_cs2_ingest.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name == "requests" for alias in node.names):
            # Las pruebas sustituyen fetch_json; nunca habilitar red aqui.
            continue
        if isinstance(node, ast.ImportFrom):
            if node.module in {"airflow.sdk", "pendulum"}:
                continue
            if node.module in {"bo3_cs2_features", "bo3_cs2_rebuild", "bo3_cs2_quality"}:
                node.module = "dags." + node.module
        if isinstance(node, ast.FunctionDef) and node.name == "bo3_cs2_ingest":
            for task in node.body:
                if isinstance(task, ast.FunctionDef):
                    task.decorator_list = []
                    nodes.append(task)
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            # No ejecutar bo3_cs2_ingest() al cargar la prueba.
            continue
        nodes.append(node)
    namespace = {"__file__": str(source)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    def network_disabled(*args, **kwargs):
        raise AssertionError("Una prueba offline intento consultar la API")
    namespace["fetch_json"] = network_disabled
    return namespace


def test_discovery_resumes_after_saved_cursor():
    ns = load_task_functions()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        ns.update(OUTPUT_DIR=root, CONTROL_DIR=root / "control",
                  STATE_PATH=root / "control" / "pipeline_state.json",
                  BRONZE_RUNS_DIR=root / "bronze" / "runs",
                  get_current_context=lambda: {},
                  utc_now=lambda: datetime(2026, 10, 15, tzinfo=timezone.utc),
                  extraction_identity=lambda: ("test", "manual_test"))
        ns["write_json_atomic"](ns["STATE_PATH"], {"last_discovered_date": "2026-10-03"})
        original = ns["STATE_PATH"].read_bytes()
        plan = ns["build_plan"]()
        assert plan["dates"][0] == "2026-10-04"
        assert plan["dates"][-1] == "2026-10-14"
        assert len(plan["dates"]) == 11
        assert not plan["bootstrap"]
        assert ns["STATE_PATH"].read_bytes() == original


def test_enrichment_reuses_terminal_matches_and_respects_retry_dates():
    ns = load_task_functions()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        ns.update(REGISTRY_PATH=root / "registry.json",
                  utc_now=lambda: datetime(2026, 10, 15, tzinfo=timezone.utc))
        matches = {
            str(i): {"match_id": i, "slug": f"match-{i}", "lifecycle_state": state}
            for i, state in enumerate(("ready", "final_partial", "quarantined", "excluded_forfeit"), 1)
        }
        matches.update({
            "5": {"match_id": 5, "slug": "pending-due", "lifecycle_state": "pending",
                  "next_retry_at_utc": "2026-10-14T00:00:00Z"},
            "6": {"match_id": 6, "slug": "pending-later", "lifecycle_state": "pending",
                  "next_retry_at_utc": "2026-10-16T00:00:00Z"},
        })
        ns["write_json_atomic"](ns["REGISTRY_PATH"], {"matches": matches})
        original = ns["REGISTRY_PATH"].read_bytes()
        discovery = root / "discovery.json"
        ns["write_json_atomic"](discovery, {"matches": [{"match_id": 7, "slug": "new-match"}]})
        queues = ns["prepare_enrichment"](
            {"run_directory": str(root / "run")}, [{"summary_path": str(discovery)}]
        )
        queued = [item["match_id"] for path in queues
                  for item in ns["read_json"](Path(path))["matches"]]
        assert set(queued) == {5, 7}
        assert ns["REGISTRY_PATH"].read_bytes() == original


def test_manual_run_uses_today_not_old_inferred_interval():
    from types import SimpleNamespace
    ns = load_task_functions()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        ns.update(OUTPUT_DIR=root, CONTROL_DIR=root / "control",
                  STATE_PATH=root / "control" / "pipeline_state.json",
                  BRONZE_RUNS_DIR=root / "bronze" / "runs",
                  get_current_context=lambda: {"dag_run": SimpleNamespace(run_type="manual"),
                                                "data_interval_end": object()},
                  utc_now=lambda: datetime(2026, 10, 16, tzinfo=timezone.utc),
                  extraction_identity=lambda: ("manual_test", "manual_test"))
        ns["write_json_atomic"](ns["STATE_PATH"], {"last_discovered_date": "2026-10-14"})
        assert ns["build_plan"]()["dates"] == ["2026-10-15"]
        ns["write_json_atomic"](ns["STATE_PATH"], {"last_discovered_date": "2026-10-15"})
        assert ns["build_plan"]()["dates"] == []


def test_ranking_download_only_fetches_missing_snapshot():
    ns = load_task_functions()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        ns.update(RANKING_INDEX_PATH=root / "ranking_index.json", STAGING_DIR=root / "staging")
        existing = root / "old_snapshot.json.gz"
        ns["write_gzip_json_atomic"](existing, {"payload": {"data": []}})
        ns["write_json_atomic"](ns["RANKING_INDEX_PATH"], {
            "snapshots": {"2026-09-01": {"pages": [str(existing)], "is_official": True}}
        })
        original = existing.read_bytes()
        calls = []

        def fetch(url, params=None, **kwargs):
            calls.append((url, params))
            if url == ns["RANKING_DATES_URL"]:
                return {"payload": {"data": ["2026-09-01", "2026-10-01"]}}
            assert url == ns["RANKINGS_URL"]
            assert params["filter[ranking_date][eq]"] == "2026-10-01"
            return {"payload": {"data": [], "meta": {"total_pages": 1, "is_official": True}}}

        ns["fetch_json"] = fetch
        result = ns["fetch_rankings"]({"run_directory": str(root / "run"), "extraction_id": "test"})
        assert result["ranking_snapshots_reused"] == 1
        assert result["ranking_snapshots_downloaded"] == 1
        assert len(calls) == 2
        assert existing.read_bytes() == original


def test_rebuild_and_publish_from_existing_bronze_without_network():
    ns = load_task_functions()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        ns.update(STAGING_DIR=root / "staging", MIN_SILVER_ROWS=1,
                  INTERMEDIATE_DIR=root / "intermediate", SILVER_DIR=root / "silver",
                  QUALITY_DIR=root / "quality")
        index = {}
        original_bronze = {}
        for match_id, version in ((1, 1), (2, None), (3, 2)):
            path = root / "bronze" / f"match-{match_id}.json.gz"
            detail = {
                "id": match_id, "slug": f"match-{match_id}", "game_version": version,
                "team1_id": 1, "team2_id": 2, "winner_team_id": 1,
                "bo_type": 3, "team1_score": 2, "team2_score": 0,
                "status": "finished", "start_date": f"2026-01-0{match_id}T12:00:00Z",
                "end_date": f"2026-01-0{match_id}T14:00:00Z",
            }
            ns["write_gzip_json_atomic"](path, {"ok": True, "payload": detail})
            index[str(match_id)] = {"detail": str(path)}
            original_bronze[path] = path.read_bytes()
        ns["write_json_atomic"](root / "index.json", {"matches": index})
        ns["write_json_atomic"](root / "registry.json", {"matches": {}})
        ns["write_json_atomic"](root / "rankings.json", {"snapshots": {}})
        control = {
            "extraction_id": "test", "airflow_run_id": "manual_test",
            "bronze_index_candidate_path": str(root / "index.json"),
            "registry_candidate_path": str(root / "registry.json"),
        }
        ranking = {"ranking_index_candidate_path": str(root / "rankings.json"),
                   "ranking_snapshots_downloaded": 0, "ranking_snapshots_reused": 0}
        candidate = ns["build_dataset_candidates"](control, ranking)
        assert candidate["facts_rows"] == 3
        assert candidate["rows"] == 1
        validation = ns["validate_and_report"](candidate)
        report = ns["read_json"](Path(validation["quality_report_path"]))
        assert report["valid"] and all(report["checks"].values())
        assert report["facts_excluded_from_silver"] == 2
        published = ns["publish_dataset"](validation)
        silver = ns["pd"].read_csv(published["latest_path"])
        assert list(silver["match_id"]) == [3]
        assert silver.iloc[0]["team_a_matches_prior"] == 0
        assert silver.iloc[0]["team_a_elo"] == 1500
        assert ns["pd"].isna(silver.iloc[0]["team_a_rest_days"])
        assert silver.iloc[0]["end_at_utc"] == "2026-01-03T14:00:00Z"
        metadata = ns["read_json"](root / "silver" / "metadata_latest.json")
        assert metadata["history_game_version"] == 2
        assert set(metadata["predictor_excluded_columns"]) == {
            "end_at_utc", "post_team_a_maps_won", "post_team_b_maps_won", "target_team_a_won"
        }
        assert all(path.read_bytes() == content for path, content in original_bronze.items())
