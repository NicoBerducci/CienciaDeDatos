# airflow-fifa — Dataset canónico de la materia

**Ciencia de Datos · UTN FRM · Ingeniería en Sistemas de Información · 2026**

Pipeline de Airflow que construye el dataset de jugadores de FC 26 a partir
del listado público de [sofifa.com](https://sofifa.com/players).

Es el caso canónico de la materia: el mismo dataset atraviesa las cuatro
unidades. Acá se genera; en U2 se explora, en U3 se modela y en U4 se
visualiza.

## Los dos DAGs

El proyecto trae **dos** DAGs, a propósito. Las dos formas de conseguir datos
que se van a encontrar en su proyecto integrador, una al lado de la otra.

| | `pokemon_api` | `fifa_ingest` |
|---|---|---|
| **Rol** | Ejemplo de lectura para la clase teórica | El que se modifica en el TP1 |
| **Cuándo corre** | a demanda | **programado, todos los días** |
| **Fuente** | API REST, JSON limpio | HTML que hay que parsear |
| **Acceso** | `HttpOperator` + `HttpHook` sobre una Connection | `urllib` a mano, sin provider |
| **Paralelismo** | fijo, dos ramas escritas a mano | dinámico, `.expand()` sobre 52 ligas |
| **Datos entre tareas** | XCom, son 20 filas | archivos en disco, son 18.936 |
| **Ante fallos** | reintentos | degradación en tres niveles |
| **Tamaño** | ~60 líneas | ~290 |

Ninguna de esas diferencias es capricho: cada una responde a cómo es la fuente.
Comparar las dos columnas es medio programa de la unidad.

`pokemon_api` usa la conexión `pokeapi`, declarada en `airflow_settings.yaml`.
Astro la crea sola al arrancar; no hay que darla de alta a mano.

---

## Arrancar

```bash
astro dev start
```

El primer arranque tarda varios minutos porque la imagen instala Chromium
para la ruta de respaldo. Después queda cacheada.

Con Airflow arriba en `localhost:8080`, disparar `fifa_ingest` desde la UI.

## Modos de corrida

| Parámetro | Valores | Qué hace |
|---|---|---|
| `mode` | `subset` / `full` | `subset` baja la primera página de cada liga (~2.900 jugadores, ~40 s). `full` baja el catálogo entero (~18.900, ~75 s). |
| `roster` | vacío o `260046` | Fija el snapshot. Vacío usa el más reciente. Fijarlo hace la corrida **reproducible**. |

Al fijar `roster`, **el catálogo de ligas se pide para ese mismo snapshot**. No
es cosmético: la actualización del 07/08/2025 (`roster=260001`) tenía 51 ligas
y 18.035 jugadores, y la del 23/07/2026 (`260046`) tiene 52 y 18.936. Pedir el
catálogo nuevo para scrapear un snapshot viejo daría conteos equivocados y
ligas que en ese momento no existían.

De paso, esto habilita algo que no estaba buscado: **sofifa guarda 46
snapshots de FC 26**, de agosto de 2025 a julio de 2026. Cambiando `roster` el
mismo DAG genera cualquiera de ellos, así que si alguna vez querés una serie
temporal para U2 o U4, está ahí.
| `engine` | `auto` / `http` / `browser` | `auto` prueba HTTP y cae a navegador si recibe 403. |
| `force` | `false` / `true` | Baja los datos aunque el snapshot no haya cambiado. |

### Por qué corre todos los días si casi nunca hay nada nuevo

sofifa publica un parche cada una o dos semanas. Un DAG diario que bajara las
18.936 filas siempre estaría rehaciendo el mismo trabajo seis de cada siete
veces. Por eso `has_new_snapshot` corta la corrida cuando el snapshot no cambió.

**La mayoría de las corridas de un pipeline sano no hacen nada**, y eso está
bien. Lo que no está bien es no darse cuenta.

La Variable `fifa_ultimo_roster` guarda el último snapshot procesado, y sólo la
actualiza una corrida que efectivamente bajó datos de la fuente: si los datos
salieron del respaldo, no se toca.

El resultado queda en `include/output/fifa_AAAA-MM-DD.csv`.

---

## Cómo está armado

```
wait_for_source → check_source ──┬─→ has_new_snapshot → discover_leagues
                                 │        → scrape_league[52] → consolidate ─┐
                                 │                                            ├→ validate → save
                                 └─→ load_frozen ─────────────────────────────┘
```

| Tarea | Qué hace |
|---|---|
| `wait_for_source` | **Sensor.** Espera a que sofifa responda: sondea cada 5 min, hasta 30. Con `mode="reschedule"` libera el worker entre sondeos en vez de ocuparlo durmiendo. |
| `check_source` | Según lo que consiguió el sensor, sigue por la fuente o por el respaldo. Es un `@task.branch`, y corre con `ALL_DONE` porque su tarea de arriba puede haber quedado en `skipped`. |
| `has_new_snapshot` | **Cortocircuito.** Compara el snapshot actual contra el último procesado, guardado en una Variable. Si no cambió, todo lo de abajo queda en `skipped` y la corrida termina bien sin hacer nada. |
| `discover_leagues` | Lee `sofifa.com/leagues`, que publica **cuántos jugadores tiene cada liga**. Con eso sabemos de antemano cuántas páginas pedir: no hay que tantear. |
| `scrape_league` | Una tarea mapeada por liga. En la UI cada instancia aparece con el nombre de su liga. |
| `consolidate` | Junta los parciales, deduplica por `player_id` y castea tipos. |
| `load_frozen` | Rama de último recurso: el respaldo más fresco que haya en disco (ver abajo). |
| `validate` | Chequeos duros. Si alguno falla, el DAG falla: no se publica un dataset roto. |
| `save` | Escribe el entregable fechado. |

### Tres niveles de degradación

1. **HTTP directo** con `urllib` — el camino normal, ~0,5 s por página.
2. **Playwright + Chromium** — si Cloudflare devuelve 403.
3. **Respaldo en disco** — si sofifa está caído o cambió de estructura.

Los dos primeros viven dentro de `fetch()` y no aparecen en el grafo: son la
misma página por otra puerta, así que el reintento es transparente. El tercero
sí es una rama visible, porque ahí **el dato es otro** — más viejo — y eso hay
que comunicarlo.

Un pipeline de producción no tiene una fuente: tiene un plan para cuando la
fuente falla. Eso es lo que se ve en el grafo.

### El respaldo tiene dos capas

| Archivo | Quién lo escribe | Versionado |
|---|---|---|
| `include/frozen/fifa_snapshot.csv` | **La semilla.** Se genera una vez y no se toca. | Sí, viaja con el repo |
| `include/frozen/ultimo_ok.csv` | Cada corrida **`full`** exitosa lo reescribe. | No, está en `.gitignore` |

`load_frozen` prefiere `ultimo_ok.csv` y cae a la semilla si todavía no existe.
Así el **arranque en frío** lo cubre la semilla —el respaldo funciona desde el
primer `git clone`, sin depender de haber tenido suerte antes— y la **frescura**
la cubre el último éxito.

Dos detalles del refresco:

- Una corrida **`subset` no toca el respaldo**. Es una corrida de desarrollo, y
  pisar un respaldo completo con uno parcial sería un retroceso.
- Si los datos **vinieron del respaldo**, tampoco se reescribe. Sería copiarlo
  sobre sí mismo y falsear su fecha.

En cualquier caso, `load_frozen` deja en el log **de cuándo son los datos** y
cuántas filas trae, para que nadie confunda una corrida degradada con una normal.

---

## Tres cosas que conviene mirar del código

**1. Se usa `urllib`, no `requests`.** Cloudflare bloquea con 403 a los
clientes construidos sobre `urllib3` — `requests` entre ellos — por la huella
del handshake TLS. La biblioteca estándar pasa. Cambiar una por otra rompe el
pipeline; está comentado en `include/fifa/sofifa.py`.

**2. Por XCom viajan rutas, no filas.** `scrape_league` escribe su liga a
disco y devuelve la ruta. Pasar miles de filas por XCom satura la base de
metadatos de Airflow: es de los errores más comunes al empezar.

**3. No se usa `ds`.** Sólo existe cuando el DAG tiene `schedule` y por lo
tanto intervalo de datos. Este corre a demanda, así que la fecha sale del
`DagRun`. Es un tropiezo clásico al pasar de Airflow 2 a 3.

---

## El esquema

90 columnas, una fila por jugador, clave `player_id`. Definido en
`include/fifa/schema.py`.

**Qué quedó afuera y por qué:**

- **Los 27 ratings por posición** (`ls`, `st`, `cb`, …). sofifa los publica
  sólo en la ficha del jugador, que desde 2025 exige cuenta. Se podrían
  calcular a partir de los atributos base con ~98 % de acierto, pero eso es
  inferirlos, no observarlos. Si no está en la fuente, no está en el dataset.
- **`dob`.** El listado expone únicamente el año de nacimiento. Se usa `age`,
  que sí viene explícito.
- **Columnas siempre vacías** en la fuente, como `work_rate`.

**Qué se agregó, aprovechando que el listado lo publica:** `best_position`
(objetivo multiclase natural para U3), `growth`, `best_overall`, los totales
por grupo de atributos, los playstyles y `acceleration_type`.

---

## Probar

```bash
astro dev pytest              # integridad del DAG
astro dev run dags test fifa_ingest 2026-08-18 --conf '{"mode":"subset"}'
```

---

## Nota sobre la fuente

Los datos son de sofifa.com, sitio no oficial de la comunidad de FC/FIFA.
El uso es didáctico. El pipeline pausa entre pedidos y baja sólo el listado
público. Si vas a correrlo muchas veces seguidas, usá `mode=subset`.
