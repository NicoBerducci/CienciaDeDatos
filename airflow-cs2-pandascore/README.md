# Pipeline de partidos de Counter-Strike 2

Proyecto Astro/Airflow que obtiene diariamente los partidos históricos de CS2
desde PandaScore. Una fila del dataset representa un partido finalizado entre
dos equipos y la variable objetivo es `team_a_win`.

## Ejecución

1. Configurar `PANDASCORE_TOKEN` en `.env`.
2. Iniciar Docker Desktop.
3. Ejecutar `astro dev start`.
4. Activar el DAG `pandascore_cs2_ingest` en Airflow.

El DAG se ejecuta todos los días a las 00:00 de Argentina. La primera corrida
descarga el histórico completo. Las siguientes consultan el total remoto y
reutilizan las páginas completas de bronze; solamente vuelven a descargar la
última página incompleta y las páginas nuevas.

## Salidas

Todos los directorios se crean automáticamente bajo `include/output/`:

```text
include/output/
├── bronze/
│   ├── snapshot/page_XXXX.json.gz
│   ├── runs/extraction=.../manifest.json
│   └── bronze_state.json
├── staging/extraction=.../matches_candidate.csv
├── quality/extraction=.../quality_report.json
└── silver/
    ├── matches_latest.csv
    ├── metadata_latest.json
    └── extraction=.../
        ├── matches.csv
        └── metadata.json
```

- `bronze/snapshot/`: última versión acumulada de las respuestas crudas.
- `bronze/runs/`: páginas recibidas y manifiesto de cada corrida.
- `staging/`: CSV candidato, todavía no publicado.
- `quality/`: métricas y validaciones de calidad.
- `silver/`: versiones validadas y último dataset disponible.

Si no hay partidos nuevos, la corrida termina correctamente sin regenerar
silver. Si alguna validación falla, `matches_latest.csv` no se reemplaza.

## Pruebas

```powershell
astro dev pytest
```

Las pruebas verifican la importación de los DAGs, la paginación incremental y
la normalización determinista de los equipos.
