# La aplicación (Entrega 4) y el modelo que usa (Entrega 3)

**Ciencia de Datos · UTN FRM · 2026 · Comisión 5K10 · Grupo 07**

> Con lo que se sabe antes de empezar, el modelo elige al ganador del **65,1 %** de las series de CS2 que nunca vio, y cuando se equivoca, **8 de cada 10 veces** es porque ganó el equipo que llegaba con menos Elo.

**Aplicación publicada:** https://nicoberducci.github.io/CienciaDeDatos/ *(una vez activado GitHub Pages, ver abajo)*

## Qué hay en esta carpeta

| Archivo | Qué es |
|---|---|
| `cs2_modelo.py` | Todo lo que decide un número: semilla, corte del test, folds temporales, columnas, pipeline, grillas de las cuatro familias, modelo final, lectura de errores e hipótesis de la Entrega 2. Lo importan el notebook y `build_app.py`. |
| `graficos.py` | Los gráficos de Altair, con un solo juego de colores (azul CT = equipo A, dorado T = equipo B). El notebook usa los mismos. |
| `build_app.py` | La construcción: lee el Silver, entrena el modelo con la misma semilla y genera `docs/index.html`. |
| `plantilla.html` | El esqueleto de la página: textos, estilos y la lógica del Pick'em. `build_app.py` lo completa con números y gráficos. |
| `fuentes/` | Rajdhani y Barlow (licencia SIL OFL, incluida), embebidas en la página para que funcione sin conexión. |
| `GUIA_DEFENSA.md` | Lo que se pregunta en los 25 minutos, respondido con lo que hay en la app. |

El resto del repositorio: `dags/` es el pipeline (Entregas 1 y 2), `entrega 2/` el EDA y las hipótesis, `entrega 3/` el modelado corregido, y `docs/` la aplicación generada.

## Cómo se corre

```bash
pip install -r app/requirements.txt

# Los datos: la extracción fija de las Entregas 2 y 3 (20261016T021804Z__3779b1d9).
mkdir -p datos
gdown 1l3o2EA1VHUgKmjWHILXWLfbFlCsFraqG -O datos/matches_latest.csv

python app/build_app.py          # ~5 minutos en 2 núcleos: grillas, modelo, gráficos y página
python app/build_app.py --cache  # reusa app/resultados/grillas.pkl si ya existe
```

`build_app.py` busca el Silver en `datos/matches_latest.csv` y, si no está, en `include/output/silver/matches_latest.csv`, que es donde lo deja el DAG. Escribe `docs/index.html` y `docs/numeros.json`. La página se abre con doble clic, sin servidor ni conexión.

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

La última sección de `entrega 3/entrega_3_grupo_5K10-07_corregida.ipynb` lee `docs/numeros.json` y comprueba que la accuracy del test, la de validación y el intercepto son los mismos. Las dos piezas usan las mismas funciones de `cs2_modelo.py`.

## Versiones

Las de la Entrega 3: numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1 y scipy 1.14.1; Altair 5.5 para los gráficos. Están fijadas en `requirements.txt`. Son independientes de las del contenedor de Airflow (`requirements.txt` de la raíz).
