from __future__ import annotations

import gzip
import time
from pathlib import Path

import pendulum
from airflow.sdk import Param, Variable, dag, task

from fifa import schema
from fifa.leagues import fetch_catalog
from fifa.sofifa import (PAGE_SIZE, EndOfLeague, fetch, page_url, parse_page,
                         snapshot_meta)
from fifa.transform import to_row

OUTPUT_DIR = Path("/usr/local/airflow/include/output")
BRONCE_DIR = OUTPUT_DIR / "bronze"
PLATA_DIR = OUTPUT_DIR / "silver"
PARTIAL_DIR = PLATA_DIR / "_parciales"


def bronze_path(roster, league_id, offset) -> Path:
    return (BRONCE_DIR / f"roster={roster}" / f"liga={league_id}"
            / f"pagina_{offset:05d}.html.gz")


def bronze_write(destino: Path, html: str) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destino, "wt", encoding="utf-8") as f:
        f.write(html)


def bronze_read(ruta: Path) -> str:
    with gzip.open(ruta, "rt", encoding="utf-8") as f:
        return f.read()


@dag(
    dag_id="tp1_5K10_07",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 8, 1, tz="America/Argentina/Buenos_Aires"),
    catchup=False,
    max_active_tasks=8,
    tags=["ciencia-de-datos", "unidad-1", "dataset-canonico"],
    params={
        "mode": Param("subset", enum=["subset", "full"]),
        "roster": Param(260046, type=["null", "integer"]),
        "engine": Param("auto", enum=["auto", "http", "browser"]),
        "force": Param(False, type="boolean"),
    },
)
def fifa_ingest():

    @task
    def discover_leagues(**context) -> list[dict]:
        params = context["params"]
        roster = params["roster"]
        catalogo = fetch_catalog(engine=params["engine"], roster=roster)
        tope = 1 if params["mode"] == "subset" else None

        tareas = []
        for lg in catalogo:
            if lg["league_id"] == 50:
                n = lg["n_pages"] if tope is None else min(tope, lg["n_pages"])
                tareas.append({**lg, "n_pages": n, "roster": roster,
                               "engine": params["engine"],
                               "force": params["force"]})
        return tareas

    @task(map_index_template="{{ task.op_kwargs['league']['league_name'] }}",
          retries=2, retry_delay=pendulum.duration(seconds=30))
    def land_bronze(league: dict) -> dict:
        paginas = []
        for i in range(league["n_pages"]):
            offset = i * PAGE_SIZE
            destino = bronze_path(league["roster"], league["league_id"], offset)

            if destino.exists() and not league["force"]:
                paginas.append(str(destino))
                continue

            url = page_url(league_id=league["league_id"], offset=offset,
                           roster=league["roster"])
            try:
                html = fetch(url, engine=league["engine"])
            except EndOfLeague:
                break
            bronze_write(destino, html)
            paginas.append(str(destino))
            time.sleep(0.25)
        return {"league": league, "pages": paginas}

    @task(map_index_template="{{ task.op_kwargs['lote']['league']['league_name'] }}")
    def refine_silver(lote: dict) -> str:
        import pandas as pd
        league = lote["league"]
        filas, meta = [], None

        for ruta in lote["pages"]:
            html = bronze_read(Path(ruta))
            if meta is None:
                meta = snapshot_meta(html)
            lote_filas = parse_page(html)
            if not lote_filas:
                break
            filas.extend(lote_filas)

        registros = [to_row(f, meta or {}, league) for f in filas]
        PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
        destino = PARTIAL_DIR / f"liga_{league['league_id']}.csv"
        pd.DataFrame(registros, columns=schema.COLUMNS).to_csv(destino, index=False)
        return str(destino)

    @task
    def consolidate(rutas: list[str]) -> str:
        import pandas as pd
        partes = [pd.read_csv(r, low_memory=False) for r in rutas if r]
        partes = [p for p in partes if len(p)]
        if not partes:
            raise ValueError("ninguna liga devolvió filas")

        df = pd.concat(partes, ignore_index=True)[schema.COLUMNS]
        df = df.drop_duplicates("player_id").reset_index(drop=True)

        for c in schema.ENTEROS:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        destino = OUTPUT_DIR / "_consolidado.csv"
        df.to_csv(destino, index=False)
        return str(destino)

    @task
    def validate(ruta: str) -> str:
        import pandas as pd
        df = pd.read_csv(ruta, low_memory=False)

        problemas = []
        if list(df.columns) != schema.COLUMNS:
            problemas.append("columnas distintas")
        if len(df) < 300: #EN ESTE CAMPO TUVIMOS QUE MODIFICAR EL NUMERO DE 500 A 300, YA QUE LA LIGA TENIA APROXIMADAMENTE 328 JUGADORES Y EL FILTRO ANTERIOR ERA DE 500, SIEMPRE IBA A FALLAR DE ESA MANERA, POR LO QUE TENIAMOS QUE PONER UN NUMERO MENOR A 328 PARA QUE NO FALLE SI NO TRAIA EL DATASET COMPLETO Y EVITAR SER TAN ESTRICTOS DE SI SE VA UN JUGADOR FALLE AUTOMATICAMENTE
            problemas.append(f"muy pocas filas: {len(df)}")
        if df["player_id"].duplicated().any():
            problemas.append("player_id repetidos")
        for c in schema.OBLIGATORIAS:
            if c in df.columns and df[c].isna().any():
                problemas.append(f"{c} tiene nulos")
        if "overall" in df.columns and not df["overall"].between(1, 99).all():
            problemas.append("overall fuera del rango 1-99")

        if problemas:
            raise ValueError("Validación fallida:\n  - " + "\n  - ".join(problemas))
        return ruta

    @task
    def save(ruta: str, **context) -> str:
        import shutil
        dag_run = context["dag_run"]
        momento = dag_run.logical_date or dag_run.run_after
        ds = momento.date().isoformat()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        destino = OUTPUT_DIR / f"fifa_{ds}.csv"
        shutil.copy(ruta, destino)
        return str(destino)

    @task
    def package_delivery(csv_path: str, **context) -> str:
        import json
        import zipfile
        from pathlib import Path

        import pandas as pd
        import pendulum

        df = pd.read_csv(csv_path)
        filas, columnas = df.shape

        manifiesto = {
            "grupo": "5K10-07",
            "integrantes": ["Berducci, Nicolás", "Genaulaz Martín", "Yanardi Tomás", "Yanardi Juan Cruz", "Gallardo Federico", "León Mario"],
            "league_id": 50,
            "league_name": "Premiership",
            "roster": context["params"]["roster"],
            "filas": filas,
            "columnas": columnas,
            "run_id": context["run_id"],
            "generado_en": pendulum.now().isoformat()
        }

        manifiesto_path = OUTPUT_DIR / "manifiesto.json"
        with open(manifiesto_path, "w", encoding="utf-8") as f:
            json.dump(manifiesto, f, indent=4, ensure_ascii=False)

        bronce_path_file = OUTPUT_DIR / "bronce.txt"
        with open(bronce_path_file, "w", encoding="utf-8") as f:
            for p in (OUTPUT_DIR / "bronze").rglob("*.html.gz"):
                if p.is_file():
                    f.write(str(p) + "\n")

        zip_path = OUTPUT_DIR / "tp1_5K10_07.zip"
        dag_id = context["dag"].dag_id
        logs_dir = Path(f"/usr/local/airflow/logs/dag_id={dag_id}")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv_path, arcname="dataset.csv")
            zf.write(manifiesto_path, arcname="manifiesto.json")
            zf.write(bronce_path_file, arcname="bronce.txt")

            dag_file = Path(__file__)
            zf.write(dag_file, arcname=dag_file.name)

            if logs_dir.exists():
                for p in logs_dir.rglob("*"):
                    if p.is_file() and context["run_id"] in str(p):
                        arcname = f"logs/{p.relative_to(logs_dir)}"
                        zf.write(p, arcname=str(arcname))

        return str(zip_path)

    ligas = discover_leagues()
    bronces = land_bronze.expand(league=ligas)
    parciales = refine_silver.expand(lote=bronces)
    consolidado = consolidate(parciales)
    package_delivery(save(validate(consolidado)))


fifa_ingest()
