"""Pipeline diario de partidos históricos de Counter-Strike 2.

Migrado a la API de bo3.gg, con Feature Engineering avanzado para evitar Data Leakage.
Arquitectura adaptada a paginación basada en fechas (Date-Based) en lugar de offset/limit.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import time
import random
from collections import Counter
from datetime import timedelta, datetime as std_datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from airflow.sdk import dag, get_current_context, task
from airflow.sdk.exceptions import AirflowSkipException
from pendulum import datetime, now


API_URL = "https://api.bo3.gg/api/v2/matches/finished"
MIN_DATE = "2024-01-01"
REQUEST_TIMEOUT_SECONDS = 1
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
    "filter[discipline_id][eq]": "1",
    "utc_offset": "0",
}


def date_path(directory: Path, date_str: str) -> Path:
    return directory / f"date_{date_str}.json.gz"


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


def request_date(date_str: str, endpoint: str = API_URL, extra_params: dict = None) -> requests.Response:
    params = dict(REQUEST_PARAMS)
    if extra_params:
        params.update(extra_params)
    params["date"] = date_str
    
    headers = {"Accept": "application/json"}

    for attempt in range(1, MAX_HTTP_ATTEMPTS + 1):
        try:
            response = requests.get(
                endpoint,
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
                f"API devolvió HTTP {response.status_code} para la fecha "
                f"{date_str}: {body}"
            )

        return response

    raise RuntimeError(f"No se pudo descargar la fecha {date_str}.")


def request_secondary(endpoint: str) -> dict:
    headers = {"Accept": "application/json"}
    for attempt in range(1, MAX_HTTP_ATTEMPTS + 1):
        try:
            response = requests.get(endpoint, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == MAX_HTTP_ATTEMPTS:
                return {}
            time.sleep(2**attempt)
            continue
        
        if response.status_code == 429 and attempt < MAX_HTTP_ATTEMPTS:
            time.sleep(2**attempt)
            continue
            
        if response.status_code == 200:
            return response.json()
        return {}
    return {}


@dag(
    dag_id="bo3_cs2_ingest",
    description="Ingesta incremental diaria de partidos históricos de CS2",
    start_date=datetime(2026, 9, 7, tz="America/Argentina/Buenos_Aires"),
    schedule="0 0 * * *",
    catchup=False,
    max_active_runs=1,
    max_active_tasks=4,  # Limitado para no saturar la API oculta de bo3.gg
    default_args={
        "owner": "grupo_5K10_07",
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
    },
    tags=["bo3", "cs2", "incremental", "momentum"],
)
def bo3_cs2_ingest():
    @task
    def check_configuration() -> dict[str, Any]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return {
            "configuration_valid": True,
            "api_url": API_URL,
        }

    @task
    def inspect_bronze(_: dict[str, Any]) -> dict[str, Any]:
        if not STATE_PATH.exists():
            return {"exists": False, "last_extracted_date": None}

        state = read_json(STATE_PATH)
        return {
            "exists": True,
            "last_extracted_date": state.get("last_extracted_date"),
        }

    @task
    def inspect_source(_: dict[str, Any]) -> dict[str, Any]:
        extraction_id, airflow_run_id = extraction_identity()
        run_directory = RUNS_DIR / f"extraction={extraction_id}"
        run_directory.mkdir(parents=True, exist_ok=True)

        current_date_str = now("UTC").format("YYYY-MM-DD")

        return {
            "extraction_id": extraction_id,
            "airflow_run_id": airflow_run_id,
            "checked_at": now("UTC").isoformat(),
            "run_directory": str(run_directory),
            "current_date": current_date_str,
        }

    @task
    def build_extraction_plan(
        bronze: dict[str, Any], source: dict[str, Any]
    ) -> dict[str, Any]:
        last_extracted_date = bronze.get("last_extracted_date")
        current_date = source["current_date"]
        
        end_dt = std_datetime.strptime(current_date, "%Y-%m-%d")
        dates_to_fetch = []
        
        if not last_extracted_date:
            strategy = "bootstrap"
            start_dt = std_datetime.strptime(MIN_DATE, "%Y-%m-%d")
        else:
            strategy = "incremental"
            start_dt = std_datetime.strptime(last_extracted_date, "%Y-%m-%d")
            # Extra safety intraday: always include yesterday and today to overwrite previous partial pulls
            start_dt = start_dt - timedelta(days=1)
            min_dt = std_datetime.strptime(MIN_DATE, "%Y-%m-%d")
            if start_dt < min_dt:
                start_dt = min_dt
                
        curr = end_dt
        while curr >= start_dt:
            dates_to_fetch.append(curr.strftime("%Y-%m-%d"))
            curr -= timedelta(days=1)
            
        return {
            **source,
            "strategy": strategy,
            "dates": dates_to_fetch,
        }

    @task
    def dates_to_download(plan: dict[str, Any]) -> list[str]:
        return plan["dates"]

    @task(max_active_tis_per_dag=4)
    def land_bronze(date_str: str, plan: dict[str, Any]) -> dict[str, Any]:
        response = request_date(date_str)
        data = response.json()
        matches = []
        if isinstance(data, dict):
            d = data.get("data", {})
            if "tiers" in d:
                for tier_data in d["tiers"].values():
                    matches.extend(tier_data.get("matches", []))
            elif "matches" in d:
                matches = d.get("matches", [])
        elif isinstance(data, list):
            matches = data
        
        enriched_matches = []
        for match in matches:
            match_slug = match.get("slug") or match.get("id")
            if match_slug:
                time.sleep(random.uniform(1, 4))  # Protección Cloudflare
                stats = request_secondary(f"https://api.bo3.gg/api/v1/matches/{match_slug}/players_stats")
                match["players_stats"] = stats
            enriched_matches.append(match)
            
        if isinstance(data, dict):
            data["data"] = enriched_matches
        else:
            data = enriched_matches

        destination = date_path(Path(plan["run_directory"]), date_str)
        write_gzip_json_atomic(destination, data)
        return {
            "date": date_str,
            "rows": len(enriched_matches),
            "path": str(destination),
            "sha256": file_sha256(destination),
        }

    @task(trigger_rule="none_failed")
    def build_bronze_manifest(
        plan: dict[str, Any], downloaded_dates: list[dict[str, Any]]
    ) -> dict[str, Any]:
        dates_info = list(downloaded_dates or [])
        manifest = {
            "source": "Bo3",
            "endpoint": "/api/v2/matches/finished",
            "extraction_id": plan["extraction_id"],
            "airflow_run_id": plan["airflow_run_id"],
            "checked_at": plan["checked_at"],
            "strategy": plan["strategy"],
            "dates_requested": len(plan["dates"]),
            "dates_saved": len(dates_info),
            "records_received": sum(int(item["rows"]) for item in dates_info),
            "filters": REQUEST_PARAMS,
            "files": dates_info,
            "dataset_changed": bool(plan["dates"]),
        }
        if manifest["dates_requested"] != manifest["dates_saved"]:
            raise RuntimeError("No se guardaron todas las fechas incluidas en el plan.")

        manifest_path = Path(plan["run_directory"]) / "manifest.json"
        write_json_atomic(manifest_path, manifest)
        return {**manifest, "manifest_path": str(manifest_path)}

    @task
    def update_bronze_snapshot(manifest: dict[str, Any]) -> dict[str, Any]:
        run_directory = RUNS_DIR / f"extraction={manifest['extraction_id']}"
        refreshed_dates = {item["date"] for item in manifest["files"]}

        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        for d in refreshed_dates:
            source = date_path(run_directory, d)
            destination = date_path(SNAPSHOT_DIR, d)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copy2(source, temporary)
            temporary.replace(destination)

        previous_state = read_json(STATE_PATH) if STATE_PATH.exists() else {}
        
        all_snapshot_files = list(SNAPSHOT_DIR.glob("date_*.json.gz"))
        if all_snapshot_files:
            all_dates = [f.name.replace("date_", "").replace(".json.gz", "") for f in all_snapshot_files]
            last_extracted_date = max(all_dates)
        else:
            last_extracted_date = previous_state.get("last_extracted_date")

        state = {
            "source": "Bo3",
            "last_extracted_date": last_extracted_date,
            "last_successful_extraction": manifest["extraction_id"],
            "last_successful_run_id": manifest["airflow_run_id"],
            "updated_at": now("UTC").isoformat(),
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
        player_rows: list[dict[str, Any]] = []
        raw_records = 0
        
        all_snapshot_files = sorted(SNAPSHOT_DIR.glob("date_*.json.gz"))
        if not all_snapshot_files:
            raise RuntimeError("No hay archivos en el snapshot para procesar.")
            
        for date_file in all_snapshot_files:
            records_data = read_gzip_json(date_file)
            records = records_data.get("data", []) if isinstance(records_data, dict) else records_data
            included_teams = records_data.get("included", {}).get("teams", {}) if isinstance(records_data, dict) else {}
            for match in records:
                raw_records += 1
                
                if match.get("forfeit"):
                    continue
                if match.get("status") != "finished":
                    continue
                if not match.get("players_stats"):
                    continue
                    
                match_id = match.get("id")
                begin_at = match.get("start_date") or match.get("begin_at")
                winner_id = match.get("winner_team_id") or match.get("winner_id")
                
                team_a_id = match.get("team1_id")
                team_b_id = match.get("team2_id")
                
                team_a_name = included_teams.get(str(team_a_id), {}).get("name") if team_a_id else None
                team_b_name = included_teams.get(str(team_b_id), {}).get("name") if team_b_id else None
                
                bo_type = match.get("bo_type", 3)
                tier = match.get("tier", "unknown")
                stars = match.get("stars", 0)
                
                if not all([match_id, begin_at, winner_id, team_a_id, team_b_id]):
                    continue
                
                games = match.get("games", [])
                map_names = [g.get("map_name") for g in games if g.get("map_name")]
                
                # Extraer jugadores usando players_stats
                stats_data = match.get("players_stats", [])
                if isinstance(stats_data, list):
                    for p in stats_data:
                        team_clan = p.get("team_clan", {}) or {}
                        steam_profile = p.get("steam_profile", {}) or {}
                        
                        t_id = team_clan.get("team_id")
                        p_id = steam_profile.get("player_id")
                        if not t_id or not p_id:
                            continue
                        
                        game_profiles = steam_profile.get("game_steam_profiles", [])
                        game_ids = [gp.get("game_id") for gp in game_profiles if gp.get("game_id")]
                        game_ids_str = ",".join(map(str, game_ids)) if game_ids else ""
                        games_count = len(game_profiles)

                        player_dict = steam_profile.get("player", {}) or {}
                        team_dict = team_clan.get("team", {}) or {}

                        row_dict = {
                            "match_id": match_id,
                            "team_id": t_id,
                            "player_id": p_id,
                            "game_ids": game_ids_str,
                            "games_count": games_count,
                            "adr": p.get("adr"),
                            "kills": p.get("kills"),
                            "death": p.get("death"),
                            "assists": p.get("assists"),
                            "headshots": p.get("headshots"),
                            "first_kills": p.get("first_kills"),
                            "first_death": p.get("first_death"),
                            "trade_kills": p.get("trade_kills"),
                            "trade_death": p.get("trade_death"),
                            "kast": p.get("kast"),
                            "player_rating": p.get("player_rating"),
                            "hits": p.get("hits"),
                            "shots": p.get("shots"),
                            "got_damage": p.get("got_damage"),
                            "damage": p.get("damage"),
                            "utility_value": p.get("utility_value"),
                            "money_spent": p.get("money_spent"),
                            "money_save": p.get("money_save"),
                            "clutches": p.get("clutches"),
                            "total_equipment_value": p.get("total_equipment_value"),
                            "additional_value": p.get("additional_value"),
                            "pistols_value": p.get("pistols_value"),
                            "weapons_value": p.get("weapons_value"),
                            "player_rating_value": p.get("player_rating_value"),
                            "clan_name": p.get("clan_name"),
                            "team_name": team_dict,
                            "player_name": player_dict,
                        }
                        
                        multikills = p.get("multikills") or {}
                        if isinstance(multikills, dict):
                            for k, v in multikills.items():
                                row_dict[f"multikills_{k}"] = v
                        
                        player_rows.append(row_dict)
                
                rows.append({
                    "match_id": match_id,
                    "begin_at": begin_at,
                    "team_a_id": team_a_id,
                    "team_a_name": team_a_name,
                    "team_b_id": team_b_id,
                    "team_b_name": team_b_name,
                    "winner_id": winner_id,
                    "bo_type": bo_type,
                    "tier": tier,
                    "stars": stars,
                    "map_names": map_names,
                })

        if not rows:
            raise RuntimeError("La transformación no produjo ningún partido válido tras filtrar forfeits/ff y nulos.")

        df = pd.DataFrame(rows)
        df["begin_at"] = pd.to_datetime(df["begin_at"], utc=True)
        df.sort_values(by="begin_at", ascending=True, inplace=True)
        df = df.drop_duplicates(subset=["match_id"])
        
        if player_rows:
            players_df = pd.DataFrame(player_rows)
            # Ordenamos por player_id para darles un "rank" 1 al 5 determinista sin data leakage
            players_df.sort_values(["match_id", "team_id", "player_id"], inplace=True)
            players_df["rank"] = players_df.groupby(["match_id", "team_id"]).cumcount() + 1
            
            players_df = players_df.set_index(["match_id", "team_id", "rank"])
            p_unstacked = players_df.unstack("rank")
            p_unstacked.columns = [f"p{rank}_{col}" for col, rank in p_unstacked.columns]
            p_pivot = p_unstacked.reset_index()
            
            p_pivot_a = p_pivot.copy()
            p_pivot_a = p_pivot_a.rename(columns={"team_id": "team_a_id"})
            p_pivot_a = p_pivot_a.add_prefix("team_a_")
            p_pivot_a = p_pivot_a.rename(columns={"team_a_match_id": "match_id", "team_a_team_a_id": "team_a_id"})
            
            p_pivot_b = p_pivot.copy()
            p_pivot_b = p_pivot_b.rename(columns={"team_id": "team_b_id"})
            p_pivot_b = p_pivot_b.add_prefix("team_b_")
            p_pivot_b = p_pivot_b.rename(columns={"team_b_match_id": "match_id", "team_b_team_b_id": "team_b_id"})
            
            df = df.merge(p_pivot_a, on=["match_id", "team_a_id"], how="left")
            df = df.merge(p_pivot_b, on=["match_id", "team_b_id"], how="left")
        extraction_id = state["last_successful_extraction"]
        candidate_directory = STAGING_DIR / f"extraction={extraction_id}"
        candidate_directory.mkdir(parents=True, exist_ok=True)
        candidate_path = candidate_directory / "matches_candidate.csv"
        df.to_csv(candidate_path, index=False, encoding="utf-8-sig")

        return {
            "extraction_id": extraction_id,
            "airflow_run_id": state["last_successful_run_id"],
            "candidate_path": str(candidate_path),
            "manifest_path": state["manifest_path"],
            "raw_records": raw_records,
            "valid_rows": len(df),
            "discarded_rows": raw_records - len(df),
            "discard_reasons": {},
        }

    @task
    def validate_and_report(transformation: dict[str, Any]) -> dict[str, Any]:
        candidate_path = Path(transformation["candidate_path"])
        dataframe = pd.read_csv(candidate_path)
        errors: list[str] = []

        all_null_columns = dataframe.columns[dataframe.isna().all()].tolist()
        duplicate_match_ids = int(dataframe["match_id"].duplicated().sum())

        checks = {
            "at_least_5_useful_columns": bool(
                (dataframe.notna().any()).sum() >= 5
            ),
            "primary_key_not_null": not dataframe["match_id"].isna().any(),
            "primary_key_unique": duplicate_match_ids == 0,
            "no_all_null_columns": not all_null_columns,
            "winner_is_participant": bool(
                (
                    (dataframe["winner_id"] == dataframe["team_a_id"])
                    | (dataframe["winner_id"] == dataframe["team_b_id"])
                ).all()
            ),
        }
        for name, passed in checks.items():
            if not passed:
                errors.append(name)

        report = {
            "extraction_id": transformation["extraction_id"],
            "valid": not errors,
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
            "primary_key": "match_id",
            "primary_key_nulls": int(dataframe["match_id"].isna().sum()),
            "primary_key_unique": duplicate_match_ids == 0,
            "duplicate_match_ids": duplicate_match_ids,
            "all_null_columns": all_null_columns,
            "target_column": "winner_id",
            "checks": checks,
            "errors": errors,
        }

        report_directory = QUALITY_DIR / f"extraction={transformation['extraction_id']}"
        report_directory.mkdir(parents=True, exist_ok=True)
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
            "published_at": now("UTC").isoformat(),
        }
        write_json_atomic(SILVER_DIR / "metadata_latest.json", latest_metadata)
        return {
            "dataset_path": str(version_path),
            "latest_path": str(latest_path),
        }

    configuration = check_configuration()
    bronze = inspect_bronze(configuration)
    source = inspect_source(configuration)
    plan = build_extraction_plan(bronze, source)
    dates = dates_to_download(plan)
    downloaded = land_bronze.partial(plan=plan).expand(date_str=dates)
    manifest = build_bronze_manifest(plan, downloaded)
    state = update_bronze_snapshot(manifest)
    candidate = transform_to_silver(state)
    validation = validate_and_report(candidate)
    publish_dataset(validation)


bo3_cs2_ingest()
