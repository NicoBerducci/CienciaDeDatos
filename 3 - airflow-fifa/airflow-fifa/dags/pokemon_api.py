"""
### Ejemplo de clase — Ingesta desde una API REST

DAG chico y completo, pensado para leerse entero durante la clase teórica.
Toma datos de la [PokéAPI](https://pokeapi.co), que es abierta, rápida y
devuelve JSON limpio.

Es el **contraste** del DAG `fifa_ingest`, que ingesta por scraping. La misma
materia, dos fuentes de naturaleza distinta, y por eso dos diseños distintos:

| | `pokemon_api` | `fifa_ingest` |
|---|---|---|
| Fuente | API REST, JSON | HTML que hay que parsear |
| Acceso | `HttpOperator` + Connection | `urllib` a mano |
| Paralelismo | fijo, escrito a mano | dinámico, con `.expand()` |
| Datos entre tareas | XCom | archivos en disco |
| Ante fallos | reintentos | degradación en tres niveles |

Ninguna de esas diferencias es capricho: cada una responde a cómo es la fuente.
Ese es el punto de tener los dos ejemplos.

**Este DAG no se modifica en el TP.** Es material de lectura.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pendulum
from airflow.providers.http.hooks.http import HttpHook
from airflow.providers.http.operators.http import HttpOperator
from airflow.sdk import Param, dag, task

log = logging.getLogger(__name__)

OUTPUT_DIR = Path("/usr/local/airflow/include/output")

# La URL base NO está acá: vive en la conexión `pokeapi`, que Astro crea sola
# a partir de airflow_settings.yaml. Ver la sección de conexiones del apunte.
CONN_ID = "pokeapi"


@dag(
    dag_id="pokemon_api",
    schedule=None,
    start_date=pendulum.datetime(2026, 8, 1, tz="America/Argentina/Buenos_Aires"),
    catchup=False,
    tags=["ciencia-de-datos", "unidad-1", "ejemplo-de-clase"],
    doc_md=__doc__,
    params={
        "cantidad": Param(
            20, type="integer", minimum=1, maximum=100,
            title="Cuántos Pokémon traer",
            description="La API pagina de a 20 por defecto. Con 100 ya se nota la espera.",
        ),
    },
)
def pokemon_api():

    # ---------------------------------------------------------------- 1
    # Operador prefabricado: no escribimos el pedido HTTP, sólo lo configuramos.
    # `http_conn_id` toma el host de la conexión; acá sólo va el endpoint.
    # `response_filter` transforma la respuesta antes de guardarla en XCom:
    # sin esto guardaríamos el objeto Response entero, que no es serializable.
    listar = HttpOperator(
        task_id="listar_pokemon",
        http_conn_id=CONN_ID,
        endpoint="api/v2/pokemon",
        method="GET",
        data={"limit": "{{ params.cantidad }}", "offset": "0"},
        response_filter=lambda r: [x["url"] for x in r.json()["results"]],
        log_response=False,
    )

    # ---------------------------------------------------------------- 2 y 3
    # Dos tareas que consultan endpoints distintos y NO dependen entre sí, así
    # que Airflow las corre en paralelo. El paralelismo acá es fijo: son dos
    # porque nosotros escribimos dos. En fifa_ingest, en cambio, la cantidad de
    # tareas la decide la fuente en tiempo de ejecución.
    @task(retries=2, retry_delay=pendulum.duration(seconds=10))
    def traer_atributos(urls: list[str]) -> list[dict]:
        """Datos de combate de cada Pokémon: /api/v2/pokemon/{id}.

        Acá no alcanza un operador prefabricado: hay que recorrer una lista y
        quedarse con ciertos campos. Pero tampoco hace falta abrir la conexión
        a mano: el **hook** es la capa de abajo del operador, y expone la misma
        conexión para usarla desde nuestro propio código.

            HttpOperator  ->  usa un hook por dentro, vos sólo lo configurás
            HttpHook      ->  lo usás vos, cuando necesitás lógica propia
        """
        hook = HttpHook(method="GET", http_conn_id=CONN_ID)
        salida = []
        for url in urls:
            pid = url.rstrip("/").split("/")[-1]
            d = hook.run(endpoint=f"api/v2/pokemon/{pid}").json()
            salida.append({
                "id": d["id"],
                "nombre": d["name"],
                "altura_dm": d["height"],
                "peso_hg": d["weight"],
                "experiencia_base": d.get("base_experience"),
                "tipos": ", ".join(t["type"]["name"] for t in d["types"]),
                **{s["stat"]["name"].replace("-", "_"): s["base_stat"]
                   for s in d["stats"]},
            })
        log.info("atributos de %s Pokémon", len(salida))
        return salida

    @task(retries=2, retry_delay=pendulum.duration(seconds=10))
    def traer_especies(urls: list[str]) -> list[dict]:
        """Datos de especie: /api/v2/pokemon-species/{id}. Otro endpoint.

        Misma conexión, mismo hook, distinto endpoint. Y como no depende de
        traer_atributos, Airflow las corre en paralelo.
        """
        hook = HttpHook(method="GET", http_conn_id=CONN_ID)
        salida = []
        for url in urls:
            pid = url.rstrip("/").split("/")[-1]
            d = hook.run(endpoint=f"api/v2/pokemon-species/{pid}").json()
            salida.append({
                "id": int(pid),
                "color": d["color"]["name"],
                "habitat": (d.get("habitat") or {}).get("name"),
                "generacion": d["generation"]["name"],
                "legendario": d["is_legendary"],
                "mitico": d["is_mythical"],
                "tasa_captura": d["capture_rate"],
            })
        log.info("especies de %s Pokémon", len(salida))
        return salida

    # ---------------------------------------------------------------- 4
    @task
    def combinar_y_guardar(atributos: list[dict], especies: list[dict],
                           **context) -> str:
        """Une los dos endpoints por `id` y escribe el CSV.

        Acá los datos viajaron por XCom y está bien: son 20 filas. La regla
        práctica es que XCom sirve para lo que cabe cómodo en la base de
        metadatos de Airflow. Cuando el volumen crece hay que pasar a archivos,
        que es lo que hace fifa_ingest con sus 18.936 filas.
        """
        import pandas as pd

        df = (pd.DataFrame(atributos)
                .merge(pd.DataFrame(especies), on="id", how="left")
                .sort_values("id")
                .reset_index(drop=True))

        # Orden explícito de columnas: el que sale del merge depende del orden
        # de los diccionarios y es incómodo de leer.
        orden = ["id", "nombre", "tipos", "generacion", "color", "habitat",
                 "legendario", "mitico", "altura_dm", "peso_hg",
                 "experiencia_base", "tasa_captura",
                 "hp", "attack", "defense", "special_attack",
                 "special_defense", "speed"]
        df = df[[c for c in orden if c in df.columns]]

        dag_run = context["dag_run"]
        momento = dag_run.logical_date or dag_run.run_after
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        destino = OUTPUT_DIR / f"pokemon_{momento.date().isoformat()}.csv"
        df.to_csv(destino, index=False)

        log.info("%s filas x %s columnas -> %s", len(df), len(df.columns), destino)
        return str(destino)

    urls = listar.output
    combinar_y_guardar(traer_atributos(urls), traer_especies(urls))


pokemon_api()
