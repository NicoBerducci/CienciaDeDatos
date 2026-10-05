Overview
========

Este proyecto ingiere partidos de Counter-Strike 2 desde Bo3.gg con Airflow.

Pipeline de datos
=================

- `include/output/bronze`: respuestas originales append-only de partidos y rankings.
- `include/output/intermediate/match_facts_latest.csv`: tabla de apoyo con el resultado observado, campos `post_*`, estados y validez por familia.
- `include/output/silver/matches_latest.csv`: dataset analítico point-in-time. Conserva contexto y target, pero sólo contiene predictores que estaban disponibles antes de comenzar cada partido.

La primera ejecución posterior a la migración reutiliza `bronze_index.json` y los
payloads ya descargados. No vuelve a consultar todos los partidos: descubre sólo
las fechas posteriores a `pipeline_state.json`, procesa los reintentos pendientes
y descarga los snapshots oficiales de ranking todavía ausentes. La reconstrucción
de Intermediate y Silver sí recorre localmente todo Bronze para que las features
históricas queden consistentes ante backfills y correcciones.

Los rankings se vinculan por `team_id` usando el último snapshot oficial cuya
fecha sea estrictamente anterior al día del partido. Un equipo ausente del
snapshot queda con ranking nulo; no se le asigna el peor puesto ni se cruza por
nombre.

Una observación parcial puede alimentar las familias para las que tiene datos
válidos. Por ejemplo, un resultado válido alimenta winrate/Elo aunque falten las
estadísticas de jugadores. Las ventanas toman los últimos N partidos reales y no
buscan encuentros más antiguos para reemplazar mediciones ausentes.

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
