# Guía para los 25 minutos

Lo que la consigna dice que se pregunta, respondido con lo que hay en la app. Repártanla: cada uno tiene que poder contestar cualquier punto.

## 0–2 · Qué cambió desde la Entrega 3 (dos oraciones)

> Ahora la logística compite contra un árbol de decisión, un random forest y un gradient boosting, **cada uno con su propia grilla de hiperparámetros** sobre los mismos folds: el random forest saca 0,3 puntos más, menos que un desvío, así que la regla sigue eligiendo la logística. Y ampliamos la lectura de los errores: el 81 % son sorpresas (ganó el de menos Elo), los cuatro modelos fallan en las mismas series y las probabilidades están calibradas, lo que nos permitió usarlas para la sección de apuestas.

## Por qué cada título afirma lo que afirma

| Sección | Título | El número que lo sostiene |
|---|---|---|
| Portada | El modelo elige al ganador del 65,1 % de las series nuevas, y 8 de cada 10 errores son sorpresas | Accuracy en el test 0,6514 (8.819 series); 81,5 % de los errores tienen `sorpresa = True` |
| Los datos | 43.719 series en tres años, y 9 de cada 10 se juegan en tiers B y C | B + C = 91,2 % de las filas |
| Lo que encontramos | El de más Elo gana el 62 %, y con 200 puntos de ventaja, 8 de cada 10 | 62,0 % de 43.354 series; 80,2 % en el tramo 200–300 |
| El modelo | La logística acierta 65,1 % y ningún modelo complejo le gana por más que el ruido | RF 63,36 % ± 0,81 → límite 62,55 %; logística 63,04 % |
| El modelo (2.º gráfico) | Cuando duda acierta 54 %; cuando está seguro, 86 % | Tramos de confianza del test: 50–55 % → 54,0 %; ≥ 80 % → 85,5 % |

## Las hipótesis de la Entrega 2, como están en la app

«Lo que encontramos» usa los veredictos reales de la Entrega 2 (los recalcula `cm.hipotesis_e2` con la función `separacion` de la cátedra, en el tramo de entrenamiento):

- **H2, Elo:** separación 0,584, amarilla → **inconclusa**. Es la del gráfico: el Elo ordena (el favorito gana el 62 %) pero no alcanza solo. Ningún tier llega a verde; el A es el que menos separa (0,414), y es donde el modelo acierta menos.
- **H3, ventanas de winrate (la que se sostuvo):** Pearson 0,803, verde; la de 20 separa más que la de 10 y el promedio no mejora.
- **H1, descanso (no se sostuvo):** 0,080, rojo, y con el signo al revés.
- **H5, sorpresas en BO1 (no se sostuvo):** 38,5 % en BO1 y en el total. Conecta con el kill feed: las sorpresas aparecen igual en todos los formatos.

Si preguntan «¿por qué el título dice que el Elo funciona si H2 quedó inconclusa?»: el título describe un hecho (el favorito gana el 62 %), y el semáforo mide otra cosa (cuánto se separan las distribuciones). Las dos cosas son compatibles: el Elo ordena bien, pero hay mucha superposición; por eso el modelo toca techo cerca del 65 %.

## Cada gráfico, componente por componente

**Los datos (barras apiladas por mes)**
- Datos: el Silver agrupado en pandas por mes y tier (`tabla_meses`).
- Marca: `mark_bar`. Una regla (`mark_rule`) y un texto para el corte del test: son capas.
- Mapeo: x = mes (temporal, `yearmonth`), y = `sum(series)` (agregación en el canal), color = tier (nominal).
- Escala: color con dominio y rango fijos (`ESC_TIER`, de oscuro = élite a claro = tier bajo); y lineal desde 0 porque son barras.

**Lo que encontramos (línea del favorito por Elo)**
- Datos: cada serie con su favorito por Elo; en pandas se calcula si ganó y se agrupa en 7 tramos de ventaja (transformación: `pd.cut` + media).
- Marca: `mark_line` con puntos, más dos capas de texto (el porcentaje y la cantidad) y una regla en 50 %.
- Mapeo: x = tramo (ordinal, con el orden de los tramos), y = proporción que gana el favorito (cuantitativa).
- Escala: y de 0 a 100 %. La línea de 50 % es la moneda.

**El modelo (puntos con rango)**
- Datos: la mejor combinación de cada grilla (media y desvío de los 5 folds).
- Marca: `mark_point` para la media y `mark_rule` de media − desvío a media + desvío. Un `mark_rect` sombrea la zona «a menos de un desvío del mejor».
- Mapeo: x = accuracy, y = modelo (nominal), color = elegido / otros / piso.
- Escala: x cortada entre 50 % y 66 %. **Se puede cortar porque son puntos, no barras** (Práctica, 4.2).

**Dónde se equivoca (barras por confianza)**
- Datos: el test, agrupado por tramo de confianza (`tabla_confianza`).
- Marcas: barra (lo que acierta), `mark_tick` blanco (lo que el modelo dijo en promedio), textos (porcentaje, cuota mínima = 1 / accuracy, cantidad).
- Escala: y desde 0 porque son barras.

**Explorá (nube de 38.524 series)**
- Datos: las predicciones fuera de fold (validación) y del modelo final (test), embebidas como CSV.
- Marca: `mark_circle`. Mapeo: x = diferencia de Elo (recortada a ±700), y = probabilidad de A según el modelo, color = quién ganó (azul CT = A, dorado T = B; gris oscuro si queda afuera del filtro).
- Transformaciones en el gráfico: `transform_calculate` arma `en_filtro` con los seis controles; el panel lateral usa `transform_filter` y `transform_aggregate` sobre lo filtrado.

**Probá el modelo (barras de probabilidad y apuesta)**
- Datos: una tabla de dos filas (A y B). Todo lo demás son parámetros.
- `z` es un parámetro con `expr`: la fórmula de la logística escrita con los coeficientes. `pA = 1 / (1 + exp(−z))`.
- Las barras de la apuesta comparan la probabilidad del modelo con la implícita de la casa (`(1/cuota) / (1/cuota_A + 1/cuota_B)`, sin margen).

## Qué componente toca cada control de la copa

| Control | Qué toca |
|---|---|
| Tier, Formato, Período, Series, Equipo, Confianza mínima | La transformación: entran en la expresión `en_filtro` del `transform_calculate`. Esa columna decide el **color** (gris si no está), la **opacidad**, el **orden** de dibujo, el **texto** del total y los **datos** del panel de confianza (`transform_filter`). |
| Clic en un punto | Una selección (`selection_point` sobre `id`): cambia el **tamaño** y el **borde** del punto, y **filtra los datos** del panel de detalle y de las barras de probabilidad. Doble clic la borra. |
| Mouse encima | El canal **tooltip**, con lo que dijo el modelo. |
| Sliders y menús de «Armá el partido» | Son los **parámetros** que lee la expresión `z`. Cambian la longitud de las barras y los textos. |
| Cuotas | Parámetros que leen las transformaciones de la apuesta (valor esperado, Kelly, margen). |
| Elegir dos equipos | JavaScript fuera de Vega: calcula las diferencias A − B con la foto de cada equipo y **setea las señales** (`view.signal`), igual que el «Partir de un jugador real» del ejemplo. |

## Cómo llega el modelo a la aplicación, y qué pasa con un caso raro

1. `build_app.py` entrena el mismo modelo que el notebook (`cm.entrenar_final`: 18 columnas, C = 0,01, balanced, semilla 42).
2. Descompone el pipeline: mediana de imputación, indicador de dato faltante, media y desvío del escalado, `log1p` de los partidos previos, one-hot de formato y tier, coeficientes e intercepto.
3. Lo escribe como expresión de Vega y **comprueba en Python** que da lo mismo que `predict_proba` en las 8.819 series del test. Lo probamos también en el navegador: diferencia máxima 1e-16.
4. **Caso raro:** los sliders van del mínimo al máximo del entrenamiento, así que no se puede cargar algo nunca visto. Si un valor cae fuera del percentil 1–99, el gráfico muestra un aviso en rojo. Si al elegir dos equipos una diferencia se sale del rango, se recorta y la página lo dice arriba de los controles. Un formato o tier que no estaba en el entrenamiento no se puede elegir.

## Cómo se regenera si cambian los datos

`python app/build_app.py --sin-huella --csv include/output/silver/matches_latest.csv` después de una corrida nueva del DAG. Vuelve a correr las cuatro grillas, el modelo, las predicciones y reescribe los números de la página. Commit de `docs/` y GitHub Pages publica. Si la regla dejara de elegir la logística, el script se detiene.

## Decisiones de diseño que cambiamos

- **La identidad visual:** la primera versión copiaba el formato del ejemplo (fondo blanco, menú lateral). La rehicimos con la estética de CS2: fondo oscuro tipo HUD, el **azul de los CT para el equipo A y el dorado de los T para el B** en todos los gráficos y en el Pick'em, la historia contada como una serie al mejor de 3 (Briefing, Mapa 1/3, 2/3, 3/3) y la copa como el overtime. La paleta está definida una sola vez en `graficos.py`, por eso los colores significan lo mismo en toda la app.

- **Accuracy por tipo de partido:** la primera versión eran barras con el eje empezando en 50 %. Una barra codifica con su largo, así que cortada miente (tier A parecía la mitad de bueno). La pasamos a **puntos**, que codifican con la posición y admiten el eje cortado.
- **Los títulos:** al principio cada gráfico tenía su título adentro y la sección repetía otro arriba. Ahora el título que afirma es el de la sección y el gráfico se queda con el subtítulo que dice qué mirar.
- **La nube de Explorá:** el rótulo de la línea del 50 % usaba un desplazamiento que Vega descartaba en un eje continuo y los textos se encimaban; lo separamos en dos capas.
- **El gráfico de confianza:** las cantidades debajo de las barras no se leían (gris sobre azul); pasaron adentro de la barra, en blanco, y las barras se ensancharon.
