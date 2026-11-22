# La aplicación (Entrega 4) y el modelo que usa (Entrega 3)

**Ciencia de Datos · UTN FRM · 2026 · Comisión 5K10 · Grupo 07**

> Con lo que se sabe antes de empezar, el modelo elige al ganador del **65,1 %** de las series de CS2 que nunca vio, y cuando se equivoca, **8 de cada 10 veces** es porque ganó el equipo que llegaba con menos Elo.

**Aplicación publicada:** https://nicoberducci.github.io/CienciaDeDatos/ *(una vez activado GitHub Pages, ver abajo)*

## Qué hay en esta carpeta

| Archivo | Qué es |
|---|---|
| `cs2_modelo.py` | Todo lo que decide un número, con la misma configuración del notebook de la Entrega 3: semilla, corte del test, folds temporales, variantes de columnas, pipeline, grillas de las cuatro familias, modelo final, lectura de errores e hipótesis de la Entrega 2. |
| `graficos.py` | Los gráficos de Altair, con un solo juego de colores (azul CT = equipo A, dorado T = equipo B). |
| `build_app.py` | La construcción: lee el Silver, entrena el modelo con la misma semilla y genera `docs/index.html`. |
| `apuestas.py` | La sección «La apuesta»: une las cuotas con las predicciones del test, prueba las reglas de apuesta, elige la mejor en marzo–julio y la evalúa en agosto–octubre. |
| `bajar_cuotas_oddspapi.py` | Baja de OddsPapi el historial de cuotas de Pinnacle de las series de tier S y A del test y guarda la última cuota antes del inicio. |
| `plantilla.html` | El esqueleto de la página: textos, estilos y la lógica del Pick'em. `build_app.py` lo completa con números y gráficos. |
| `fuentes/` | Rajdhani y Barlow (licencia SIL OFL, incluida), embebidas en la página para que funcione sin conexión. |

El resto del repositorio: `dags/` es el pipeline (Entregas 1 y 2), `entrega 2/` el EDA y las hipótesis, `entrega 3/` el modelado corregido, y `docs/` la aplicación generada.

## Cómo se corre

```bash
pip install -r app/requirements.txt

# Los datos: la extracción fija de las Entregas 2 y 3 (20261016T021804Z__3779b1d9).
mkdir -p datos
gdown 1l3o2EA1VHUgKmjWHILXWLfbFlCsFraqG -O datos/matches_latest.csv

python app/build_app.py          # corre los 3.420 ajustes de la búsqueda, el modelo, los gráficos y la página
python app/build_app.py --cache  # reusa app/resultados/busqueda_v4.pkl si ya existe
```

`build_app.py` busca el Silver en `datos/matches_latest.csv` y, si no está, en `include/output/silver/matches_latest.csv`, que es donde lo deja el DAG. Escribe `docs/index.html` y `docs/numeros.json`. La página se abre con doble clic, sin servidor ni conexión.

## Las cuotas de la sección «La apuesta»

La sección decide **cuándo conviene apostar** siguiendo al modelo final: apostar $100 al equipo elegido solo si la confianza del modelo llega a un mínimo τ y el valor esperado (confianza × cuota − 1) llega a un mínimo m. τ y m se eligen con las series de marzo a julio de 2026 y se prueban con las de agosto a octubre.

Las cuotas son las de **Pinnacle**, tomadas del historial de OddsPapi: para cada serie de tier S y A del test, la última cuota publicada antes del inicio (el más temprano entre el horario de OddsPapi y el de Bo3.gg). Quedan en `datos/cuotas_oddspapi_pinnacle_20260301_20261015.csv` y `build_app.py` las lee de ahí. Para volver a bajarlas hace falta una API key de OddsPapi:

```bash
export ODDSPAPI_KEY=...        # en Windows: $env:ODDSPAPI_KEY="..."
python app/bajar_cuotas_oddspapi.py --desde 2026-03-01 --hasta 2026-10-15 --limite 1000
```

El script guarda lo que baja en `datos/oddspapi_cache/` (fuera del repo) y con `--solo-parsear` regenera el CSV sin consultar la API.

## Si cambian los datos

1. Correr el DAG `bo3_cs2_ingest`: deja una extracción nueva en `include/output/silver/matches_latest.csv`.
2. `python app/build_app.py --sin-huella --csv include/output/silver/matches_latest.csv`. La huella SHA-256 asegura que, por defecto, se use la extracción de las Entregas 2 y 3. Con datos nuevos hay que desactivarla a propósito.
3. Commit de `docs/` y GitHub Pages publica la versión nueva.

El script vuelve a correr las grillas, el modelo, las predicciones y todos los números de la página, incluidas las hipótesis de la Entrega 2. Si con los datos nuevos la regla ya no eligiera la logística, se detiene y avisa: es una decisión de modelado que hay que revisar en el notebook.

## Cómo se publica

En GitHub: **Settings → Pages → Build and deployment → Deploy from a branch**, la rama donde esté esta versión, carpeta `/docs`.

## Cómo llega el modelo a la aplicación

En el navegador no corre Python, así que:

- **Para explorar**, las predicciones se calculan antes y viajan con los datos. Las de validación cruzada las hace un modelo entrenado sólo con el pasado de cada fold; las del test, el modelo final.
- **Para probar el modelo**, `build_app.py` descompone la logística entrenada: medianas de imputación, marcas de dato faltante, medias y desvíos del escalado, `log1p`, one-hot y coeficientes. Con eso la escribe como una expresión de Vega adentro del gráfico. Antes comprueba que esa fórmula da las mismas probabilidades que `predict_proba` en las 8.819 series del test. En el navegador la diferencia máxima medida es de 1e-16.
- Los controles van del mínimo al máximo del entrenamiento, así que no se puede cargar un valor que el modelo nunca vio. Si alguno cae en el 1 % más extremo, el gráfico avisa. Si al elegir dos equipos una diferencia se sale del rango, se recorta y la página lo dice.

## Lo que muestra la aplicación coincide con el notebook

`cs2_modelo.py` repite la configuración de `entrega 3/entrega_3_grupo_5K10-07_v4.ipynb`: la misma semilla, el mismo corte del test, los mismos 5 folds temporales, las mismas nueve variantes de columnas y las mismas grillas para las cuatro familias. Por eso la aplicación da los mismos números que el notebook: 63,15 % ± 0,69 en validación y 65,14 % en el test para la logística elegida, y lo mismo para el árbol, el bosque y el boosting. `build_app.py` deja esos números en `docs/numeros.json` para poder compararlos.

## Versiones

Las de la Entrega 3: numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1 y scipy 1.14.1; Altair 5.5 para los gráficos. Están fijadas en `requirements.txt`. Son independientes de las del contenedor de Airflow (`requirements.txt` de la raíz).
