"""Integridad y reanudacion: un checkpoint nunca equivale a publicar Silver."""
from pathlib import Path
from tempfile import TemporaryDirectory
import runpy

from dags.bo3_cs2_rebuild import CheckpointStore, atomic_text_writer


def fixture(root):
    ns = runpy.run_path(str(Path(__file__).with_name("test_bo3_cs2_incremental.py")))["load_task_functions"]()
    ns.update(STAGING_DIR=root / "staging", MIN_SILVER_ROWS=1, REBUILD_BATCH_SIZE=1,
              INTERMEDIATE_DIR=root / "intermediate", SILVER_DIR=root / "silver",
              QUALITY_DIR=root / "quality")
    index, states = {}, {}
    for match_id in range(1, 4):
        path = root / "bronze" / f"{match_id}.json.gz"
        ns["write_gzip_json_atomic"](path, {"ok": True, "payload": {
            "id": match_id, "slug": f"match-{match_id}", "game_version": 2,
            "team1_id": 1, "team2_id": 2, "winner_team_id": 1,
            "bo_type": 3, "team1_score": 2, "team2_score": 0,
            "status": "finished", "start_date": f"2026-01-0{match_id}T12:00:00Z",
            "end_date": f"2026-01-0{match_id}T14:00:00Z",
        }})
        index[str(match_id)] = {"detail": str(path)}
        states[str(match_id)] = {"lifecycle_state": "final_partial"}
    ns["write_json_atomic"](root / "index.json", {"matches": index})
    ns["write_json_atomic"](root / "registry.json", {"matches": states})
    ns["write_json_atomic"](root / "rankings.json", {"snapshots": {}})
    control = {"extraction_id": "test", "airflow_run_id": "test",
               "bronze_index_candidate_path": str(root / "index.json"),
               "registry_candidate_path": str(root / "registry.json")}
    ranking = {"ranking_index_candidate_path": str(root / "rankings.json"),
               "ranking_snapshots_downloaded": 0, "ranking_snapshots_reused": 0}
    return ns, control, ranking


def test_failed_block_resumes_without_partial_rows_or_history():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        ns, control, ranking = fixture(root)
        original_normalize = ns["build_match_row"]

        def fail_third(detail, *args, **kwargs):
            if detail["id"] == 3:
                raise OSError(12, "Simulated allocation failure")
            return original_normalize(detail, *args, **kwargs)

        ns["build_match_row"] = fail_third
        try:
            ns["build_dataset_candidates"](control, ranking)
        except OSError as exc:
            assert exc.errno == 12
        else:
            raise AssertionError("No propago el fallo")
        stage = root / "staging" / "extraction=test"
        assert len(list((stage / "checkpoints").rglob("*.jsonl.gz"))) == 2
        assert not (stage / "rebuild_complete.json").exists()
        assert not (root / "silver" / "matches_latest.csv").exists()

        ns["build_match_row"] = original_normalize
        result = ns["build_dataset_candidates"](control, ranking)
        assert result["rebuild_blocks_reused"] == 2
        validation = ns["validate_and_report"](result)
        df = ns["pd"].read_csv(result["silver_candidate_path"])
        assert list(df["match_id"]) == [1, 2, 3]
        assert list(df["team_a_matches_prior"]) == [0, 1, 2]
        assert df["match_id"].is_unique
        assert ns["read_json"](Path(validation["quality_report_path"]))["valid"]
        assert len(Path(result["match_quality_candidate_path"]).read_text().splitlines()) == 3

        # Reintentar de nuevo no consulta ningun payload de partidos ni duplica.
        ns["read_gzip_json"] = lambda *args: (_ for _ in ()).throw(AssertionError("Releyo Bronze"))
        repeated = ns["build_dataset_candidates"](control, ranking)
        assert repeated["rebuild_blocks_reused"] == 3
        ns["pd"].testing.assert_frame_equal(df, ns["pd"].read_csv(repeated["silver_candidate_path"]))


def test_changed_payload_rebuilds_affected_block_and_recalculates_history():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        ns, control, ranking = fixture(root)
        first = ns["build_dataset_candidates"](control, ranking)
        before = ns["pd"].read_csv(first["silver_candidate_path"])
        path = root / "bronze" / "1.json.gz"
        envelope = ns["read_gzip_json"](path)
        envelope["payload"].update(winner_team_id=2, team1_score=0, team2_score=2, slug="changed-match-one")
        ns["write_gzip_json_atomic"](path, envelope)
        second = ns["build_dataset_candidates"](control, ranking)
        assert second["rebuild_blocks_reused"] == 2
        after = ns["pd"].read_csv(second["silver_candidate_path"])
        assert before.iloc[1]["team_a_elo"] > after.iloc[1]["team_a_elo"]
        assert list(after["team_a_matches_prior"]) == [0, 1, 2]


def test_control_change_invalidates_checkpoints():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        ns, control, ranking = fixture(root)
        first = ns["build_dataset_candidates"](control, ranking)
        document = ns["read_json"](root / "registry.json")
        document["matches"]["1"]["lifecycle_state"] = "ready"
        ns["write_json_atomic"](root / "registry.json", document)
        second = ns["build_dataset_candidates"](control, ranking)
        assert first["rebuild_fingerprint"] != second["rebuild_fingerprint"]
        assert second["rebuild_blocks_reused"] == 0


def test_reclassification_is_staged_and_bad_result_never_enters_history():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        ns, control, ranking = fixture(root)
        raw_path = root / "bronze" / "1.json.gz"
        payload = ns["read_gzip_json"](raw_path)
        payload["payload"].update(team1_score=1, team2_score=1)
        ns["write_gzip_json_atomic"](raw_path, payload)
        original_bronze = raw_path.read_bytes()
        original_control = (root / "registry.json").read_bytes()
        result = ns["build_dataset_candidates"](control, ranking)
        ns["validate_and_report"](result)
        df = ns["pd"].read_csv(result["silver_candidate_path"])
        assert list(df["match_id"]) == [2, 3]
        assert list(df["team_a_matches_prior"]) == [0, 1]
        registry = ns["read_json"](Path(result["registry_candidate_path"]))["matches"]
        assert registry["1"]["quality_state"] == "quarantined_result"
        assert registry["1"]["lifecycle_state"] == "quarantined"
        assert (root / "registry.json").read_bytes() == original_control
        assert raw_path.read_bytes() == original_bronze
        assert not (root / "silver" / "matches_latest.csv").exists()


def test_corrupt_checkpoint_is_recomputed():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        ns, control, ranking = fixture(root)
        ns["build_dataset_candidates"](control, ranking)
        path = next((root / "staging").rglob("block=00002.jsonl.gz"))
        path.write_bytes(path.read_bytes()[:40])
        result = ns["build_dataset_candidates"](control, ranking)
        assert result["rebuild_blocks_reused"] == 2
        assert result["rows"] == 3
        ns["validate_and_report"](result)


def test_code_fingerprint_and_ids_never_mix_checkpoints():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        store = CheckpointStore(root, "code-v1", ["match_id"])
        store.save(1, [1], [], [({"match_id": 1}, {"match_id": 1})])
        assert store.load(1, [1], []) is not None
        assert store.load(1, [2], []) is None
        assert CheckpointStore(root, "code-v2", ["match_id"]).load(1, [1], []) is None
        try:
            store.save(1, [1, 2], [], [({"match_id": 1}, {"match_id": 1})])
        except ValueError:
            pass
        else:
            raise AssertionError("Acepto un bloque incompleto")
        assert store.load(1, [1], []) is not None


def test_atomic_writer_does_not_replace_previous_file_on_failure():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "candidate.csv"
        path.write_text("previous complete file")
        try:
            with atomic_text_writer(path) as stream:
                stream.write("partial new file")
                raise RuntimeError("interrupted")
        except RuntimeError:
            pass
        assert path.read_text() == "previous complete file"
        assert list(path.parent.glob("*.tmp")) == []


def test_tampering_between_validation_and_publication_is_rejected():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        ns, control, ranking = fixture(root)
        result = ns["build_dataset_candidates"](control, ranking)
        validated = ns["validate_and_report"](result)
        Path(result["silver_candidate_path"]).write_text("partial")
        try:
            ns["publish_dataset"](validated)
        except ValueError:
            pass
        else:
            raise AssertionError("Publico un candidato modificado")
        assert not (root / "silver" / "matches_latest.csv").exists()
