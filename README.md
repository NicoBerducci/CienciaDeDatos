# ¿Quién gana la serie? · Predicción de partidos de Counter-Strike 2

**Ciencia de Datos · UTN FRM · 2026 · Comisión 5K10 · Grupo 07**
Berducci Nicolás · Genaulaz Martin · Yanardi Tomás · Yanardi Juan Cruz · Gallardo Federico · León Mario

> Con lo que se sabe antes de empezar, el modelo elige al ganador del **65,1 %** de las series de CS2 que nunca vio, y cuando se equivoca, **8 de cada 10 veces** es porque ganó el equipo que llegaba con menos Elo.

**Aplicación:** https://nicoberducci.github.io/CienciaDeDatos/ (GitHub Pages, carpeta `docs/`)

| Entrega | Qué | Dónde |
|---|---|---|
| 1 y 2 | Pipeline de Airflow: Bo3.gg → Bronze → Intermediate → Silver (política de calidad v3, detallada abajo) | `dags/` |
| 2 | EDA, cinco hipótesis y redundancia | `entrega 2/` |
| 3 (corregida) | Modelado: cuatro familias con su grilla, logística final, dónde y por qué falla | `entrega 3/` |
| 4 | La aplicación y cómo se genera | `app/` (código) y `docs/` (página publicada) |

Cómo regenerar la aplicación y cómo llega el modelo al navegador: ver [`app/README.md`](app/README.md).

---

Overview
========

Este proyecto ingiere partidos de Counter-Strike 2 desde Bo3.gg con Airflow.

Pipeline de datos
=================

- `include/output/bronze`: respuestas originales append-only de partidos y rankings.
- `include/output/intermediate/match_facts_latest.csv`: tabla de apoyo con el resultado observado, campos `post_*`, estados y validez por familia.
- `include/output/silver/matches_latest.csv`: dataset analítico de CS2 con 113 columnas. Conserva contexto, features históricas, target y tres metadatos pospartido: `end_at_utc`, `post_team_a_maps_won` y `post_team_b_maps_won`. Estos tres campos y el target **no son entradas del modelo**; el contrato `predictor_excluded_columns` en `metadata_latest.json` los identifica explícitamente. Los mapas ganados son el marcador de la serie, no las rondas de cada mapa.

La selección `game_version = 2` se aplica **antes** de construir todos los
historiales (Elo, winrates, H2H, descanso, alineaciones y estadísticas). Los
partidos de CS:GO (`1`) y de versión desconocida siguen conservados en Bronze y
en la tabla de apoyo, pero no generan filas del Silver ni alimentan features.
No se infiere la versión solamente por la fecha. El reporte de calidad incluye
la distribución de versiones y la cantidad de hechos excluidos del Silver.

La primera ejecución posterior a la migración reutiliza `bronze_index.json` y los
payloads ya descargados. No vuelve a consultar todos los partidos: descubre sólo
las fechas posteriores a `pipeline_state.json`, procesa los reintentos pendientes
y descarga los snapshots oficiales de ranking todavía ausentes. La reconstrucción
de Intermediate y Silver sí recorre localmente todo Bronze para que las features
históricas queden consistentes ante backfills y correcciones.

Para aplicar esta corrección, ejecutar una nueva corrida normal de
`bo3_cs2_ingest`, sin borrar `include/output/bronze` ni `include/output/control`,
sin reiniciar el cursor y sin lanzar un backfill completo. Se publicará una
nueva extracción y se actualizará `matches_latest.csv`; las extracciones
anteriores se conservan. Los estados terminales (`ready`, `final_partial`,
`quarantined`, `excluded_forfeit`) no se reconsultan por cambiar el Silver.
Después hay que actualizar el archivo de Drive y volver a ejecutar/revisar la
Entrega 2: el nuevo historial puede modificar features, nulos e hipótesis.

Los rankings se vinculan por `team_id` usando el último snapshot oficial cuya
fecha sea estrictamente anterior al día del partido. Un equipo ausente del
snapshot queda con ranking nulo; no se le asigna el peor puesto ni se cruza por
nombre.

Una observación parcial puede alimentar las familias para las que tiene datos
válidos. Por ejemplo, un resultado válido alimenta winrate/Elo aunque falten las
estadísticas de jugadores. Las ventanas toman los últimos N partidos reales y no
buscan encuentros más antiguos para reemplazar mediciones ausentes.

Controles de resultados y cronología (v3)
======================================

- El marcador de serie proviene de `team1_score`/`team2_score` del detalle,
  asignado a A/B por ID. Debe ser entero, compatible con BO1/BO2/BO3/BO5 y
  coincidir con `winner_team_id`. No se inventa un ganador ante un empate,
  marcador ausente o contradicción. Los resultados no verificables quedan
  `quarantined_result` en apoyo, fuera de Silver y de todos los historiales.
- La serie puede alimentar mapas ganados, winrate y Elo aunque no tenga
  estadísticas detalladas, siempre que las fechas sean válidas. Para utilizar
  estadísticas de mapas se exigen IDs únicos, numeración consecutiva, cantidad
  de mapas acorde a la serie, ganadores coherentes, marcadores finales válidos
  (MR12 CS2, MR15 CS:GO, prórrogas MR3) y rondas declaradas compatibles.
  Formatos especiales no acreditados quedan sin estadísticas, no se adivinan.
- Si los mapas están incompletos o contradicen la serie, sus resúmenes de
  jugadores no se consideran una fuente independiente: se bloquean métricas
  mecánicas, rating, ADR, KAST y alineación derivadas de esos mapas. Si los mapas
  son válidos pero faltan rondas detalladas, siguen disponibles los fallbacks
  de jugadores con sus comprobaciones de cobertura habituales.
- Sin finalización fiable no se alimentan partidos posteriores, pero una fila
  puede permanecer si inicio y resultado son válidos. No se imputa el fin con
  una duración promedio ni con la fecha de ingesta/actualización.
- Final anterior al inicio, final anterior al comienzo de un mapa o mapas que
  empiezan antes del inicio declarado generan `quarantined_temporal`: esa fila
  tampoco entra al Silver. Bronze y las fechas originales en apoyo se conservan.
- `bo3_cs2_quality.py` registra la exclusión revisada del abandono 1win–M1X
  (`100556`) con slug, equipos y fuentes. No se infieren otros abandonos por nulos.
- `quality_state` distingue elegibilidad analítica de `lifecycle_state` y sus
  reintentos. Los controles terminales antiguos se reconcilian al reconstruir;
  los reintentos pendientes mantienen su política. El registro reconciliado sólo
  se confirma tras validación y publicación, en `finalize_run`.
- El reporte incluye conteos por estado y controles que impiden publicar
  resultados empatados/contradictorios o usar familias bloqueadas.

Para aplicar v3, con Airflow encendido, lanzar **una nueva ejecución única** de
`bo3_cs2_ingest` desde Trigger (no Backfill ni Clear de una corrida exitosa vieja).
En una ejecución manual se consultan sólo los días UTC completos aún pendientes
del cursor y los reintentos vencidos. Si no hay días nuevos, igualmente se
reconstruye desde el Bronze. No borrar datos ni reiniciar el cursor.
Los cambios en `dags/` se cargan desde el montaje de desarrollo; comprobar que
no hay errores de importación antes de lanzar. Si ya hay una corrida activa,
no iniciar otra. No es necesario reconstruir la imagen por estos cambios Python.

Reconstrucción recuperable y memoria
===================================

- Se mantienen **8 workers** para leer Bronze. No se serializan las lecturas.
- La normalización confirma bloques de 1.000 partidos en
  `staging/extraction=.../checkpoints/<fingerprint>/`. Solo retiene en memoria
  los resultados del bloque actual, no toda la tabla ni todos los informes.
- El fingerprint incluye SHA-256 de los índices candidatos, el registro de
  estados, el índice de rankings, los cuatro módulos de transformación, el esquema
  y el tamaño del bloque. Además, cada bloque comprueba rutas, tamaños y fechas
  de modificación de sus payloads. Bronze es append-only: no editar sus archivos
  manualmente ni restaurar timestamps después de modificarlos.
- Cada checkpoint tiene IDs ordenados, una entrada de calidad por partido
  (incluso si no produce una fila), checksum y cierre con cantidad. Se escribe
  en un temporal único y se renombra solo al terminar. Un bloque incompleto,
  corrupto o con entradas distintas se recalcula entero; nunca se consume a medias.
- Los checkpoints guardan **hechos por partido**, no Elo ni ventanas parciales.
  Cuando están todos, se reconstruye desde cero el historial temporal de CS2 y
  se escribe el Silver por filas, sin una lista y un DataFrame completos extra.
  La tabla de apoyo se ordena por `match_id`; el Silver, por inicio y `match_id`.
- La calidad se escribe progresivamente en un archivo temporal. Un sello
  `rebuild_complete.json` identifica los candidatos completos y los controles reconciliados por SHA-256
  y cantidades. Se verifica antes de validar y otra vez antes de publicar.
  Los checkpoints no actualizan `matches_latest.csv`, el registro ni el cursor.
- Los logs muestran bloques calculados/reutilizados, RSS y pico del proceso,
  memoria disponible de Linux y memoria del contenedor. No son la misma medida.

Después de un fallo, reintentar `build_dataset_candidates` de **la misma corrida**
y las tareas posteriores permite reutilizar los bloques compatibles. Una corrida
nueva con controles diferentes no hereda ciegamente esos bloques. No es necesario
volver a descargar los partidos para reintentar la transformación. El guardado y
la verificación agregan algo de I/O; no se promete costo cero ni se baja el
paralelismo. Los archivos definitivos solo se publican tras el cálculo completo y
las validaciones; los temporales abandonados no se usan como datos terminados.

Prueba completa opcional, sin consultas a la API ni publicación, dentro del
runtime Docker (no ejecutar simultáneamente con el DAG):

```text
python /usr/local/airflow/tests/dags/verify_rebuild_runtime.py EXTRACTION_ID
```

Esta prueba lee los controles candidatos de la extracción elegida, conserva sus
checkpoints en staging, calcula/valida los candidatos y verifica que los controles
confirmados y los datasets publicados no hayan cambiado. No equivale a una corrida
completa de Airflow ni marca sus tareas como exitosas.

Project Contents
================

Your Astro project contains the following files and folders:

- dags: This folder contains the Python files for your Airflow DAGs. By default, this directory includes one example DAG:
    - `example_astronauts`: This DAG shows a simple ETL pipeline example that queries the list of astronauts currently in space from the Open Notify API and prints a statement for each astronaut. The DAG uses the TaskFlow API to define tasks in Python, and dynamic task mapping to dynamically print a statement for each astronaut. For more on how this DAG works, see our [Getting started tutorial](https://www.astronomer.io/docs/learn/get-started-with-airflow).
- Dockerfile: This file contains a versioned Astro Runtime Docker image that provides a differentiated Airflow experience. If you want to execute other commands or overrides at runtime, specify them here.
- include: This folder contains any additional files that you want to include as part of your project. It is empty by default.
- packages.txt: Install OS-level packages needed for your project by adding them to this file. It is empty by default.
- requirements.txt: Install Python packages needed for your project by adding them to this file. It is empty by default.
- plugins: Add custom or community plugins for your project to this file. It is empty by default.
- airflow_settings.yaml: Use this local-only file to specify Airflow Connections, Variables, and Pools instead of entering them in the Airflow UI as you develop DAGs in this project.

Deploy Your Project Locally
===========================

Start Airflow on your local machine by running 'astro dev start'.

This command will spin up five Docker containers on your machine, each for a different Airflow component:

- Postgres: Airflow's Metadata Database
- Scheduler: The Airflow component responsible for monitoring and triggering tasks
- DAG Processor: The Airflow component responsible for parsing DAGs
- API Server: The Airflow component responsible for serving the Airflow UI and API
- Triggerer: The Airflow component responsible for triggering deferred tasks

When all five containers are ready the command will open the browser to the Airflow UI at http://localhost:8080/. You should also be able to access your Postgres Database at 'localhost:5432/postgres' with username 'postgres' and password 'postgres'.

Note: If you already have either of the above ports allocated, you can either [stop your existing Docker containers or change the port](https://www.astronomer.io/docs/astro/cli/troubleshoot-locally#ports-are-not-available-for-my-local-airflow-webserver).

Deploy Your Project to Astronomer
=================================

If you have an Astronomer account, pushing code to a Deployment on Astronomer is simple. For deploying instructions, refer to Astronomer documentation: https://www.astronomer.io/docs/astro/deploy-code/

Contact
=======

The Astronomer CLI is maintained with love by the Astronomer team. To report a bug or suggest a change, reach out to our support.
