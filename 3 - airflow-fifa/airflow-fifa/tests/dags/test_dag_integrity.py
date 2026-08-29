"""Chequeos mínimos sobre el DAG: que parsee, que tenga dueño y reintentos."""
from airflow.models import DagBag


def test_no_import_errors():
    dagbag = DagBag(include_examples=False)
    assert not dagbag.import_errors, f"errores de import: {dagbag.import_errors}"


def test_dag_fifa():
    dagbag = DagBag(include_examples=False)
    dag = dagbag.get_dag("fifa_ingest")
    assert dag is not None
    ids = set(dag.task_dict)
    esperadas = {"wait_for_source", "check_source", "has_new_snapshot",
                 "discover_leagues", "scrape_league", "consolidate",
                 "load_frozen", "validate", "save"}
    assert esperadas <= ids, f"faltan tareas: {esperadas - ids}"


def test_dag_pokemon():
    dagbag = DagBag(include_examples=False)
    dag = dagbag.get_dag("pokemon_api")
    assert dag is not None
    ids = set(dag.task_dict)
    esperadas = {"listar_pokemon", "traer_atributos", "traer_especies",
                 "combinar_y_guardar"}
    assert esperadas <= ids, f"faltan tareas: {esperadas - ids}"


def test_fifa_corre_programado():
    """El DAG canónico se programa; el de ejemplo se dispara a mano."""
    dagbag = DagBag(include_examples=False)
    assert dagbag.get_dag("fifa_ingest").schedule == "@daily"
    assert dagbag.get_dag("pokemon_api").schedule is None


def test_ambos_dags_registrados():
    """La cátedra publica exactamente estos dos DAGs, ni uno más."""
    dagbag = DagBag(include_examples=False)
    assert set(dagbag.dag_ids) == {"fifa_ingest", "pokemon_api"}
