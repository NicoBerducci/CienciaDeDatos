"""Pipeline diario de partidos históricos de Counter-Strike 2.

La primera ejecución descarga el histórico completo de PandaScore. Las siguientes
reutilizan la capa bronze y descargan solamente la última página incompleta y las
páginas nuevas. Silver siempre se construye desde bronze, nunca desde la API.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from airflow.sdk import dag, get_current_context, task
from airflow.sdk.exceptions import AirflowSkipException
from pendulum import datetime, now


API_URL = "https://api.pandascore.co/csgo/matches/past"
PAGE_SIZE = 100
REQUEST_TIMEOUT_SECONDS = 45
MAX_HTTP_ATTEMPTS = 4

OUTPUT_DIR = Path("/usr/local/airflow/include/output")
BRONZE_DIR = OUTPUT_DIR / "bronze"
SNAPSHOT_DIR = BRONZE_DIR / "snapshot"
RUNS_DIR = BRONZE_DIR / "runs"
STATE_PATH = BRONZE_DIR / "bronze_state.json"
STAGING_DIR = OUTPUT_DIR / "staging"
QUALITY_DIR = OUTPUT_DIR / "quality"
SILVER_DIR = OUTPUT_DIR / "silver"

REQUEST_PARAMS = {
    "filter[videogame_title]": "cs-2",
    "filter[status]": "finished",
    "filter[opponents_filled]": "true",
    "filter[forfeit]": "false",
    "sort": "begin_at,id",
}

SILVER_COLUMNS = [
    "match_id",
    "begin_at",
    "scheduled_at",
    "end_at",
    "team_a_id",
    "team_a_name",
    "team_b_id",
    "team_b_name",
    "winner_id",
    "team_a_win",
    "score_a",
    "score_b",
    "match_type",
    "number_of_games",
    "rescheduled",
    "detailed_stats",
    "league_id",
    "league_name",
    "serie_id",
    "serie_name",
    "serie_year",
    "tournament_id",
    "tournament_name",
    "tournament_tier",
    "tournament_type",
    "tournament_region",
    "tournament_country",
    "tournament_prizepool",
]


def page_path(directory: Path, page: int) -> Path:
    return directory / f"page_{page:04d}.json.gz"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_gzip_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extraction_identity() -> tuple[str, str]:
    context = get_current_context()
    logical_date = context["logical_date"].in_timezone("UTC")
    return logical_date.format("YYYYMMDDTHHmmss[Z]"), context["run_id"]


def request_page(token: str, page: int, per_page: int = PAGE_SIZE) -> requests.Response:
    params = dict(REQUEST_PARAMS)
    params.update({"page": page, "per_page": per_page})
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}

    for attempt in range(1, MAX_HTTP_ATTEMPTS + 1):
        try:
            response = requests.get(
                API_URL,
                headers=headers,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except (requests.ConnectionError, requests.Timeout):
            if attempt == MAX_HTTP_ATTEMPTS:
                raise
            time.sleep(2**attempt)
            continue

        if response.status_code == 429 and attempt < MAX_HTTP_ATTEMPTS:
            retry_after = response.headers.get("Retry-After", "")
            wait_seconds = int(retry_after) if retry_after.isdigit() else 2**attempt
            time.sleep(min(wait_seconds, 60))
            continue

        if 500 <= response.status_code < 600 and attempt < MAX_HTTP_ATTEMPTS:
            time.sleep(2**attempt)
            continue

        if response.status_code != 200:
            body = response.text[:500]
            raise RuntimeError(
                f"PandaScore devolvió HTTP {response.status_code} en la página "
                f"{page}: {body}"
            )

        data = response.json()
        if not isinstance(data, list):
            raise RuntimeError("PandaScore respondió correctamente, pero el JSON no es una lista.")
        return response

    raise RuntimeError(f"No se pudo descargar la página {page}.")


def extraction_pages(old_total: int, remote_total: int) -> tuple[str, list[int]]:
    if old_total < 0 or remote_total < 0:
        raise ValueError("Las cantidades de registros no pueden ser negativas.")
    if remote_total < old_total:
        raise ValueError(
            "PandaScore informa menos partidos que el bronze existente. "
            "Se requiere revisar la fuente antes de modificar el snapshot."
        )
    if remote_total == old_total:
        return "no_changes", []

    last_remote_page = math.ceil(remote_total / PAGE_SIZE)
    if old_total == 0:
        return "bootstrap", list(range(1, last_remote_page + 1))

    # Si la última página estaba incompleta, esta fórmula la vuelve a pedir.
    first_page = old_total // PAGE_SIZE + 1
    return "incremental", list(range(first_page, last_remote_page + 1))


def normalize_match(match: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    match_id = match.get("id")
    if match_id is None:
        return None, "match_id_missing"
    if match.get("status") != "finished":
        return None, "status_not_finished"
    if not match.get("begin_at"):
        return None, "begin_at_missing"

    opponents = match.get("opponents") or []
    if len(opponents) != 2:
        return None, "opponents_not_two"

    teams: list[dict[str, Any]] = []
    for item in opponents:
        if item.get("type") != "Team":
            return None, "opponent_not_team"
        team = item.get("opponent") or {}
        if team.get("id") is None:
            return None, "team_id_missing"
        teams.append({"id": int(team["id"]), "name": team.get("name")})

    teams.sort(key=lambda team: team["id"])
    team_a, team_b = teams
    if team_a["id"] == team_b["id"]:
        return None, "same_team_twice"

    winner_id = match.get("winner_id")
    if winner_id is None:
        return None, "winner_missing"
    winner_id = int(winner_id)
    if winner_id not in {team_a["id"], team_b["id"]}:
        return None, "winner_not_opponent"

    score_by_team: dict[int, int] = {}
    for result in match.get("results") or []:
        team_id = result.get("team_id")
        score = result.get("score")
        if team_id is not None and score is not None:
            score_by_team[int(team_id)] = int(score)

    league = match.get("league") or {}
    serie = match.get("serie") or {}
    tournament = match.get("tournament") or {}

    row = {
        "match_id": int(match_id),
        "begin_at": match.get("begin_at"),
        "scheduled_at": match.get("scheduled_at"),
        "end_at": match.get("end_at"),
        "team_a_id": team_a["id"],
        "team_a_name": team_a["name"],
        "team_b_id": team_b["id"],
        "team_b_name": team_b["name"],
        "winner_id": winner_id,
        "team_a_win": int(winner_id == team_a["id"]),
        "score_a": score_by_team.get(team_a["id"]),
        "score_b": score_by_team.get(team_b["id"]),
        "match_type": match.get("match_type"),
        "number_of_games": match.get("number_of_games"),
        "rescheduled": match.get("rescheduled"),
        "detailed_stats": match.get("detailed_stats"),
        "league_id": league.get("id"),
        "league_name": league.get("name"),
        "serie_id": serie.get("id"),
        "serie_name": serie.get("name"),
        "serie_year": serie.get("year"),
        "tournament_id": tournament.get("id"),
        "tournament_name": tournament.get("name"),
        "tournament_tier": tournament.get("tier"),
        "tournament_type": tournament.get("type"),
        "tournament_region": tournament.get("region"),
        "tournament_country": tournament.get("country"),
        "tournament_prizepool": tournament.get("prizepool"),
    }
    return row, None


@dag(
    dag_id="pandascore_cs2_ingest",
    description="Ingesta incremental diaria de partidos históricos de CS2",
    start_date=datetime(2026, 9, 7, tz="America/Argentina/Buenos_Aires"),
    schedule="0 0 * * *",
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "grupo_5K10_07",
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
    },
    tags=["pandascore", "cs2", "incremental"],
)
def pandascore_cs2_ingest():
    @task
    def check_configuration() -> dict[str, Any]:
        token = os.getenv("PANDASCORE_TOKEN")
        if not token:
            raise ValueError(
                "Falta PANDASCORE_TOKEN. Configuralo en el archivo .env del proyecto Astro."
            )
        if len(token.strip()) < 10:
            raise ValueError("PANDASCORE_TOKEN parece tener un formato inválido.")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return {
            "configuration_valid": True,
            "api_url": API_URL,
            "page_size": PAGE_SIZE,
        }

    @task
    def inspect_bronze(_: dict[str, Any]) -> dict[str, Any]:
        if not STATE_PATH.exists():
            return {"exists": False, "total_records": 0, "total_pages": 0}

        state = read_json(STATE_PATH)
        old_total = int(state["total_records"])
        old_pages = int(state["total_pages"])
        missing = [
            str(page_path(SNAPSHOT_DIR, number))
            for number in range(1, old_pages + 1)
            if not page_path(SNAPSHOT_DIR, number).exists()
        ]
        if missing:
            raise RuntimeError(f"El estado de bronze referencia páginas inexistentes: {missing[:3]}")

        return {
            "exists": True,
            "total_records": old_total,
            "total_pages": old_pages,
        }

    @task
    def inspect_source(_: dict[str, Any]) -> dict[str, Any]:
        token = os.environ["PANDASCORE_TOKEN"]
        extraction_id, airflow_run_id = extraction_identity()
        run_directory = RUNS_DIR / f"extraction={extraction_id}"
        run_directory.mkdir(parents=True, exist_ok=True)

        response = request_page(token, page=1, per_page=1)
        data = response.json()
        probe_path = run_directory / "source_probe.json.gz"
        write_gzip_json_atomic(probe_path, data)

        remote_total = int(response.headers.get("X-Total", len(data)))
        if remote_total <= 0:
            raise RuntimeError("PandaScore no informó partidos históricos disponibles.")

        return {
            "extraction_id": extraction_id,
            "airflow_run_id": airflow_run_id,
            "checked_at": now("UTC").isoformat(),
            "run_directory": str(run_directory),
            "probe_path": str(probe_path),
            "remote_total": remote_total,
            "rate_limit_remaining": response.headers.get("X-Rate-Limit-Remaining"),
        }

    @task
    def build_extraction_plan(
        bronze: dict[str, Any], source: dict[str, Any]
    ) -> dict[str, Any]:
        old_total = int(bronze["total_records"])
        remote_total = int(source["remote_total"])
        strategy, pages = extraction_pages(old_total, remote_total)
        return {
            **source,
            "strategy": strategy,
            "old_total": old_total,
            "new_records_reported": remote_total - old_total,
            "pages": pages,
            "total_pages": math.ceil(remote_total / PAGE_SIZE),
        }

    @task
    def pages_to_download(plan: dict[str, Any]) -> list[int]:
        """Airflow mapea tareas sobre el valor completo de este XCom."""
        return plan["pages"]

    @task
    def land_bronze(page: int, plan: dict[str, Any]) -> dict[str, Any]:
        token = os.environ["PANDASCORE_TOKEN"]
        response = request_page(token, page=page)
        data = response.json()
        destination = page_path(Path(plan["run_directory"]), page)
        write_gzip_json_atomic(destination, data)
        return {
            "page": page,
            "rows": len(data),
            "path": str(destination),
            "sha256": file_sha256(destination),
        }

    @task(trigger_rule="none_failed")
    def build_bronze_manifest(
        plan: dict[str, Any], downloaded_pages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        pages = list(downloaded_pages or [])
        manifest = {
            "source": "PandaScore",
            "endpoint": "/csgo/matches/past",
            "extraction_id": plan["extraction_id"],
            "airflow_run_id": plan["airflow_run_id"],
            "checked_at": plan["checked_at"],
            "strategy": plan["strategy"],
            "previous_total": plan["old_total"],
            "remote_total": plan["remote_total"],
            "new_records_reported": plan["new_records_reported"],
            "page_size": PAGE_SIZE,
            "total_pages": plan["total_pages"],
            "pages_requested": len(plan["pages"]),
            "pages_saved": len(pages),
            "records_received": sum(int(item["rows"]) for item in pages),
            "requests_made": 1 + len(pages),
            "rate_limit_remaining_after_probe": plan["rate_limit_remaining"],
            "filters": REQUEST_PARAMS,
            "files": pages,
            "dataset_changed": bool(plan["pages"]),
        }
        if manifest["pages_requested"] != manifest["pages_saved"]:
            raise RuntimeError("No se guardaron todas las páginas incluidas en el plan.")

        manifest_path = Path(plan["run_directory"]) / "manifest.json"
        write_json_atomic(manifest_path, manifest)
        return {**manifest, "manifest_path": str(manifest_path)}

    @task
    def update_bronze_snapshot(manifest: dict[str, Any]) -> dict[str, Any]:
        remote_total = int(manifest["remote_total"])
        total_pages = int(manifest["total_pages"])
        run_directory = RUNS_DIR / f"extraction={manifest['extraction_id']}"
        refreshed_pages = {int(item["page"]) for item in manifest["files"]}

        # Validamos el snapshot que resultaría antes de reemplazar ningún archivo.
        record_count = 0
        match_ids: set[int] = set()
        for page_number in range(1, total_pages + 1):
            candidate = (
                page_path(run_directory, page_number)
                if page_number in refreshed_pages
                else page_path(SNAPSHOT_DIR, page_number)
            )
            if not candidate.exists():
                raise RuntimeError(f"Falta la página necesaria para el snapshot: {candidate}")
            records = read_gzip_json(candidate)
            if not isinstance(records, list):
                raise RuntimeError(f"La página {candidate} no contiene una lista JSON.")
            record_count += len(records)
            for record in records:
                match_id = record.get("id") if isinstance(record, dict) else None
                if match_id is None:
                    raise RuntimeError(f"Hay un partido sin ID en {candidate}.")
                match_ids.add(int(match_id))

        if record_count != remote_total or len(match_ids) != remote_total:
            raise RuntimeError(
                "El snapshot incremental no coincide con X-Total o contiene IDs duplicados. "
                "Esto puede indicar que PandaScore insertó datos en páginas antiguas."
            )

        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        for page_number in refreshed_pages:
            source = page_path(run_directory, page_number)
            destination = page_path(SNAPSHOT_DIR, page_number)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copy2(source, temporary)
            temporary.replace(destination)

        previous_state = read_json(STATE_PATH) if STATE_PATH.exists() else {}
        state = {
            "source": "PandaScore",
            "endpoint": "/csgo/matches/past",
            "page_size": PAGE_SIZE,
            "sort": REQUEST_PARAMS["sort"],
            "total_records": remote_total,
            "total_pages": total_pages,
            "last_page": total_pages,
            "last_page_records": remote_total % PAGE_SIZE or PAGE_SIZE,
            "last_successful_extraction": manifest["extraction_id"],
            "last_successful_run_id": manifest["airflow_run_id"],
            "updated_at": now("UTC").isoformat(),
            "last_data_extraction": (
                manifest["extraction_id"]
                if manifest["dataset_changed"]
                else previous_state.get("last_data_extraction")
            ),
        }
        write_json_atomic(STATE_PATH, state)
        return {
            **state,
            "dataset_changed": manifest["dataset_changed"],
            "manifest_path": manifest["manifest_path"],
        }

    @task
    def transform_to_silver(state: dict[str, Any]) -> dict[str, Any]:
        if not state["dataset_changed"]:
            raise AirflowSkipException("No hay partidos nuevos; silver no necesita actualizarse.")

        rows: list[dict[str, Any]] = []
        discard_reasons: Counter[str] = Counter()
        raw_records = 0
        for page_number in range(1, int(state["total_pages"]) + 1):
            records = read_gzip_json(page_path(SNAPSHOT_DIR, page_number))
            for match in records:
                raw_records += 1
                row, reason = normalize_match(match)
                if row is None:
                    discard_reasons[reason or "unknown"] += 1
                else:
                    rows.append(row)

        if not rows:
            raise RuntimeError("La transformación no produjo ningún partido válido.")

        dataframe = pd.DataFrame(rows, columns=SILVER_COLUMNS)
        dataframe.sort_values(["begin_at", "match_id"], inplace=True)

        extraction_id = state["last_successful_extraction"]
        candidate_directory = STAGING_DIR / f"extraction={extraction_id}"
        candidate_directory.mkdir(parents=True, exist_ok=True)
        candidate_path = candidate_directory / "matches_candidate.csv"
        dataframe.to_csv(candidate_path, index=False, encoding="utf-8-sig")

        return {
            "extraction_id": extraction_id,
            "airflow_run_id": state["last_successful_run_id"],
            "candidate_path": str(candidate_path),
            "manifest_path": state["manifest_path"],
            "raw_records": raw_records,
            "valid_rows": len(dataframe),
            "discarded_rows": raw_records - len(dataframe),
            "discard_reasons": dict(discard_reasons),
        }

    @task
    def validate_and_report(transformation: dict[str, Any]) -> dict[str, Any]:
        candidate_path = Path(transformation["candidate_path"])
        dataframe = pd.read_csv(candidate_path)
        errors: list[str] = []

        parsed_dates = {
            column: pd.to_datetime(dataframe[column], errors="coerce", utc=True)
            for column in ["begin_at", "scheduled_at", "end_at"]
        }
        all_null_columns = dataframe.columns[dataframe.isna().all()].tolist()
        duplicate_match_ids = int(dataframe["match_id"].duplicated().sum())
        target_values = set(dataframe["team_a_win"].dropna().astype(int).unique())

        checks = {
            "more_than_1000_rows": len(dataframe) > 1000,
            "at_least_5_useful_columns": bool(
                (dataframe.notna().any()).sum() >= 5
            ),
            "primary_key_not_null": not dataframe["match_id"].isna().any(),
            "primary_key_unique": duplicate_match_ids == 0,
            "no_all_null_columns": not all_null_columns,
            "target_is_binary": target_values <= {0, 1},
            "both_target_classes_present": target_values == {0, 1},
            "teams_are_different": bool(
                (dataframe["team_a_id"] != dataframe["team_b_id"]).all()
            ),
            "winner_is_participant": bool(
                (
                    (dataframe["winner_id"] == dataframe["team_a_id"])
                    | (dataframe["winner_id"] == dataframe["team_b_id"])
                ).all()
            ),
            "begin_at_is_parseable": not parsed_dates["begin_at"].isna().any(),
            "schema_is_complete": set(SILVER_COLUMNS) == set(dataframe.columns),
            "bronze_count_matches": transformation["raw_records"]
            == transformation["valid_rows"] + transformation["discarded_rows"],
        }
        for name, passed in checks.items():
            if not passed:
                errors.append(name)

        null_percentages = {
            column: round(float(value) * 100, 2)
            for column, value in dataframe.isna().mean().sort_values(ascending=False).items()
        }
        target_distribution = {
            str(int(key)): int(value)
            for key, value in dataframe["team_a_win"].value_counts().sort_index().items()
        }
        report = {
            "extraction_id": transformation["extraction_id"],
            "valid": not errors,
            "source_records": transformation["raw_records"],
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
            "primary_key": "match_id",
            "primary_key_nulls": int(dataframe["match_id"].isna().sum()),
            "primary_key_unique": duplicate_match_ids == 0,
            "duplicate_match_ids": duplicate_match_ids,
            "all_null_columns": all_null_columns,
            "date_min": parsed_dates["begin_at"].min().isoformat(),
            "date_max": parsed_dates["begin_at"].max().isoformat(),
            "target_column": "team_a_win",
            "target_distribution": target_distribution,
            "discarded_records": transformation["discarded_rows"],
            "discard_reasons": transformation["discard_reasons"],
            "null_percentages": null_percentages,
            "semantic_types": {
                "numeric": [
                    "match_id",
                    "team_a_id",
                    "team_b_id",
                    "winner_id",
                    "team_a_win",
                    "score_a",
                    "score_b",
                    "number_of_games",
                ],
                "categorical": [
                    "team_a_name",
                    "team_b_name",
                    "league_name",
                    "tournament_tier",
                    "tournament_region",
                ],
                "datetime": ["begin_at", "scheduled_at", "end_at"],
            },
            "checks": checks,
            "errors": errors,
        }

        report_directory = QUALITY_DIR / f"extraction={transformation['extraction_id']}"
        report_path = report_directory / "quality_report.json"
        write_json_atomic(report_path, report)
        if errors:
            raise ValueError("Fallaron validaciones de silver: " + ", ".join(errors))

        return {
            **transformation,
            "quality_report_path": str(report_path),
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
        }

    @task
    def publish_dataset(validation: dict[str, Any]) -> dict[str, Any]:
        extraction_id = validation["extraction_id"]
        version_directory = SILVER_DIR / f"extraction={extraction_id}"
        version_directory.mkdir(parents=True, exist_ok=True)

        candidate_path = Path(validation["candidate_path"])
        version_path = version_directory / "matches.csv"
        shutil.copy2(candidate_path, version_path)

        metadata = {
            "dataset": "PandaScore CS2 matches",
            "version": extraction_id,
            "unit_of_analysis": "Un partido finalizado de CS2 entre dos equipos",
            "primary_key": "match_id",
            "target": "team_a_win",
            "rows": validation["rows"],
            "columns": validation["columns"],
            "airflow_run_id": validation["airflow_run_id"],
            "published_at": now("UTC").isoformat(),
            "dataset_path": str(version_path),
            "source_manifest": validation["manifest_path"],
            "quality_report": validation["quality_report_path"],
            "modeling_warning": (
                "winner_id, score_a, score_b y end_at describen el resultado y no deben "
                "usarse como variables predictoras antes del partido."
            ),
        }
        metadata_path = version_directory / "metadata.json"
        write_json_atomic(metadata_path, metadata)

        SILVER_DIR.mkdir(parents=True, exist_ok=True)
        latest_path = SILVER_DIR / "matches_latest.csv"
        latest_temp = SILVER_DIR / "matches_latest.csv.tmp"
        shutil.copy2(version_path, latest_temp)
        latest_temp.replace(latest_path)

        latest_metadata = {
            "current_version": extraction_id,
            "dataset_path": str(version_path),
            "rows": validation["rows"],
            "columns": validation["columns"],
            "published_at": metadata["published_at"],
        }
        write_json_atomic(SILVER_DIR / "metadata_latest.json", latest_metadata)
        return {
            "dataset_path": str(version_path),
            "latest_path": str(latest_path),
            "metadata_path": str(metadata_path),
        }

    configuration = check_configuration()
    bronze = inspect_bronze(configuration)
    source = inspect_source(configuration)
    plan = build_extraction_plan(bronze, source)
    pages = pages_to_download(plan)
    downloaded = land_bronze.partial(plan=plan).expand(page=pages)
    manifest = build_bronze_manifest(plan, downloaded)
    state = update_bronze_snapshot(manifest)
    candidate = transform_to_silver(state)
    validation = validate_and_report(candidate)
    publish_dataset(validation)


pandascore_cs2_ingest()
