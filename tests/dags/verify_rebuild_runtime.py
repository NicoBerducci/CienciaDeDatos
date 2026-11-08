"""Prueba manual en el runtime real: genera candidatos, nunca publica ni descarga.

Uso dentro de Docker:
python /usr/local/airflow/tests/dags/verify_rebuild_runtime.py EXTRACTION_ID
Los checkpoints quedan en staging de esa extraccion para un reintento del DAG.
No ejecutar en paralelo con ninguna corrida del mismo DAG.
"""
import argparse
import json
import logging
from pathlib import Path
import runpy
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("extraction_id")
    args = parser.parse_args()
    if not all(char.isalnum() or char in "_-" for char in args.extraction_id):
        raise ValueError("ID invalido")
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    ns = runpy.run_path(str(Path(__file__).with_name("test_bo3_cs2_incremental.py")))["load_task_functions"]()
    output = root / "include" / "output"
    controls = output / "staging" / f"extraction={args.extraction_id}" / "control"
    original_controls = {str(p): ns["file_sha256"](p) for p in (output / "control").glob("*.json")}
    published = {str(p): ns["file_sha256"](p) for folder in ("intermediate", "silver")
                 for p in (output / folder).glob("*latest*") if p.is_file()}
    control = {
        "extraction_id": args.extraction_id, "airflow_run_id": "offline_verification_only",
        "bronze_index_candidate_path": str(controls / "bronze_index.json"),
        "registry_candidate_path": str(controls / "match_registry.json"),
    }
    ranking = {"ranking_index_candidate_path": str(controls / "ranking_index.json"),
               "ranking_snapshots_downloaded": 0, "ranking_snapshots_reused": 0}
    started = time.monotonic()
    try:
        result = ns["build_dataset_candidates"](control, ranking)
        validation = ns["validate_and_report"](result)
        report = ns["read_json"](Path(validation["quality_report_path"]))
        assert report["valid"] and all(report["checks"].values())
        print(json.dumps({"result": "PASS", "elapsed_seconds": round(time.monotonic() - started, 1),
                          "rows": result["rows"], "columns": result["columns"],
                          "blocks": result["rebuild_blocks"], "reused": result["rebuild_blocks_reused"],
                          "report": validation["quality_report_path"], "published": False}), flush=True)
    finally:
        for path, digest in {**original_controls, **published}.items():
            assert ns["file_sha256"](Path(path)) == digest, f"Se modifico un archivo publicado: {path}"
        print("Controles confirmados y datasets publicados: sin cambios", flush=True)


if __name__ == "__main__":
    main()
