"""Genera la aplicación de la Entrega 4 (docs/index.html) a partir del CSV del pipeline.

    python build_app.py            # vuelve a correr todo: grillas, modelo final y gráficos
    python build_app.py --cache    # reusa las grillas de la sección 7 si ya están en resultados/

Qué hace, en orden:
  1. Carga el Silver que produce el pipeline (datos/matches_latest.csv) con cs2_modelo.cargar.
  2. Separa entrenamiento y test y arma los 5 folds temporales (las mismas funciones del notebook).
  3. Corre las grillas de las cuatro familias (sección 7 del notebook) y entrena el modelo final
     con cs2_modelo.entrenar_final: logística, 18 columnas, C = 0,01, balanced, semilla 42.
  4. Calcula las predicciones para explorar: fuera de fold para la validación cruzada y del modelo
     final para el test. Viajan con los datos, porque en el navegador no corre Python.
  5. Escribe la fórmula de la logística (medianas, medias, desvíos y coeficientes) adentro del
     gráfico de «Probá el modelo», para que calcule la probabilidad en el navegador.
  6. Arma los gráficos con Altair (graficos.py) y escribe docs/index.html y docs/numeros.json.

Si los datos se actualizan (una corrida nueva del DAG deja include/output/silver/matches_latest.csv),
alcanza con volver a correrlo con --sin-huella: la huella SHA-256 controla, por defecto, que sea la
extracción de las Entregas 2 y 3.
"""
import argparse
import json
import math
import pickle
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import vl_convert as vlc
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import cs2_modelo as cm
import graficos as g
from graficos import pct, num

RAIZ = Path(__file__).resolve().parent
SALIDA = cm.RAIZ_REPO / 'docs'          # GitHub Pages publica esta carpeta
RESULTADOS = RAIZ / 'resultados'

EXTRACCION = '20261016T021804Z__3779b1d9'
PUBLICADA = '16/10/2026 a las 02:58 UTC'
# Pasos del pipeline: carpeta dags/ del repositorio (bo3_cs2_ingest, bo3_cs2_quality,
# bo3_cs2_features y bo3_cs2_rebuild; política de calidad series_temporal_family_v3).
PIPELINE = [
    ('Descubrimiento', 'El DAG <code>bo3_cs2_ingest</code> (Airflow, una corrida diaria a las 00:00 UTC, incremental) recorre la API '
                       'pública de Bo3.gg día por día desde el 27/09/2023 y pide los partidos de Counter-Strike. '
                       'En la Entrega 1 usábamos PandaScore; lo cambiamos porque sólo daba el resultado.'),
    ('Bronze', 'Para cada partido terminado guarda, sin tocarlas, las respuestas de detalle, mapas, estadísticas de jugadores y '
               'alineaciones, y además los snapshots oficiales del ranking. Es append-only: nunca se pisa lo descargado. '
               'Los partidos recientes incompletos se reintentan a 1, 3 y 7 días.'),
    ('Controles de calidad', '<code>bo3_cs2_quality</code> revisa cada partido sin corregir ni inventar nada. Verifica que el marcador de la serie '
                             'sea compatible con el formato (BO1, BO2, BO3 o BO5) y coincida con el ganador, y que la cronología cierre '
                             '(el fin no puede ser anterior al inicio ni a un mapa). Valida los mapas (MR12 en CS2, prórrogas MR3) antes '
                             'de usar sus estadísticas. Si los mapas no cierran, se bloquean las estadísticas de jugadores que salen de '
                             'ellos, pero la serie sigue valiendo para el Elo y los winrates.'),
    ('Silver', '<code>bo3_cs2_features</code> se queda sólo con CS2 (<code>game_version = 2</code>) <b>antes</b> de armar los historiales: '
               'los partidos de CS:GO quedan en el Bronze pero no alimentan ninguna estadística. Después recorre los partidos en orden y, '
               'para cada uno, calcula las estadísticas de cada equipo con partidos <b>terminados antes de su inicio</b>: Elo (arranca en 1500, '
               'K = 32), winrates de 5, 10 y 20, rondas, KDA, rating, ADR, mismo formato, mismo tier, descanso, actividad, estabilidad de la '
               'alineación, cruces directos y el ranking del último snapshot oficial estrictamente anterior al día del partido. Una ventana '
               'con menos de 5 observaciones válidas queda nula. La reconstrucción se hace por bloques de 1.000 partidos con checkpoints '
               '(<code>bo3_cs2_rebuild</code>).'),
    ('Validación y publicación', 'Antes de publicar, 24 controles: clave única, sólo CS2, un Silver por hecho elegible, marcador coherente '
                                 'con el objetivo, ganador que sea uno de los dos equipos, sin partidos en cuarentena, fin no anterior al '
                                 'inicio, ranking estrictamente previo, mínimo de observaciones, estadísticas bloqueadas que no alimentan '
                                 'el historial, entre otros. Si alguno falla, no se publica.'),
]


# ---------------------------------------------------------------------------
# 1–3 · Datos, partición, grillas y modelo final
# ---------------------------------------------------------------------------
def correr_grillas(train, cv, usar_cache):
    archivo = RESULTADOS / 'grillas.pkl'
    if usar_cache and archivo.is_file():
        return pickle.loads(archivo.read_bytes())
    X, y = train[cm.COLUMNAS_E2 + [cm.EXTRA]], train.target_team_a_won
    salida = {}
    for nombre in cm.ORDEN_SIMPLICIDAD:
        print(f'  grilla: {nombre}…', flush=True)
        b = GridSearchCV(cm.armar(cm.FAMILIAS[nombre], cm.COLUMNAS_E2), cm.GRILLAS[nombre], scoring=cm.SCORING,
                         refit=False, cv=cv, n_jobs=cm.N_JOBS, return_train_score=True, error_score='raise').fit(X, y)
        r = pd.DataFrame(b.cv_results_)
        salida[nombre] = {'tabla': cm.tabla_de(b), 'mejor': r.loc[r.rank_test_accuracy.idxmin()],
                          'params': r.loc[r.rank_test_accuracy.idxmin(), 'params']}
    piso, _ = cm.medir(cm.armar(DummyClassifier(strategy='most_frequent'), cm.COLUMNAS_E2), X, y, cv)
    salida['piso'] = piso
    RESULTADOS.mkdir(exist_ok=True)
    archivo.write_bytes(pickle.dumps(salida))
    return salida


def tabla_comparacion(grillas):
    filas = {'piso': {**grillas['piso'], 'combinación': '—'}}
    for nombre in cm.ORDEN_SIMPLICIDAD:
        r = grillas[nombre]['mejor']
        folds = np.array([r[f'split{k}_test_accuracy'] for k in range(cm.N_SPLITS)])
        filas[nombre] = {'accuracy': folds.mean(), 'desvío': folds.std(), 'accuracy train': r.mean_train_accuracy,
                         'f1 macro': r.mean_test_f1_macro, 'recall A': r.mean_test_recall_A,
                         'recall B': r.mean_test_recall_B,
                         'combinación': ', '.join(f"{k.replace('clasificador__', '')} = {str(v).replace('.', ',')}"
                                                  for k, v in grillas[nombre]['params'].items())}
    t = pd.DataFrame(filas).T
    for c in ['accuracy', 'desvío', 'accuracy train', 'f1 macro', 'recall A', 'recall B']:
        t[c] = t[c].astype(float)
    return t


# ---------------------------------------------------------------------------
# 5 · La logística, escrita como fórmula
# ---------------------------------------------------------------------------
# Nombre corto de cada columna del modelo (es el nombre del parámetro en el gráfico) y cómo se muestra.
ENTRADAS = {
    'elo_advantage_a': ('elo', 'Elo (A − B)', 'Fuerza', 1),
    'ranking_score_advantage_a': ('rank', 'Puntaje de ranking (A − B)', 'Fuerza', 1),
    'round_winrate_last20_advantage_a': ('rondas', '% de rondas ganadas, últimos 20 (A − B)', 'Fuerza', 0.01),
    'form_last5_advantage_a': ('forma', '% de victorias, últimos 5 (A − B)', 'Forma', 0.2),
    'team_a_form_trend_5v5': ('tend_a', 'Tendencia de A (últimos 5 − 5 anteriores)', 'Forma', 0.2),
    'team_b_form_trend_5v5': ('tend_b', 'Tendencia de B (últimos 5 − 5 anteriores)', 'Forma', 0.2),
    'same_bo_winrate_last20_advantage_a': ('mismo_bo', '% de victorias en este formato (A − B)', 'Forma', 0.01),
    'same_tier_winrate_last20_advantage_a': ('mismo_tier', '% de victorias en este tier (A − B)', 'Forma', 0.01),
    'lineup_stability_last5_advantage_a': ('plantel', 'Estabilidad de la alineación (A − B)', 'Plantel y cruces', 0.01),
    'h2h_team_a_winrate_last5': ('h2h', '% de victorias de A en los últimos 5 cruces', 'Plantel y cruces', 0.2),
    'team_a_matches_prior': ('previos_a', 'Partidos previos de A', 'Experiencia y actividad', 1),
    'team_b_matches_prior': ('previos_b', 'Partidos previos de B', 'Experiencia y actividad', 1),
    'team_a_matches_last7d': ('sem_a', 'Partidos de A en los últimos 7 días', 'Experiencia y actividad', 1),
    'team_b_matches_last7d': ('sem_b', 'Partidos de B en los últimos 7 días', 'Experiencia y actividad', 1),
    'team_a_matches_last30d': ('mes_a', 'Partidos de A en los últimos 30 días', 'Experiencia y actividad', 1),
    'team_b_matches_last30d': ('mes_b', 'Partidos de B en los últimos 30 días', 'Experiencia y actividad', 1),
}
GRUPOS = ['Fuerza', 'Forma', 'Experiencia y actividad', 'Plantel y cruces']


def formula(modelo):
    """Descompone el pipeline entrenado en términos que se pueden escribir como expresión."""
    pre, clf = modelo.named_steps['preparar'], modelo.named_steps['clasificador']
    nombres = list(pre.get_feature_names_out())
    w = dict(zip(nombres, clf.coef_[0]))
    num_cols = pre.transformers_[0][2]
    imp, esc = pre.named_transformers_['numericas']['imputar'], pre.named_transformers_['numericas']['escalar']
    con_indicador = [num_cols[i] for i in imp.indicator_.features_]
    salida_imp = list(imp.get_feature_names_out(num_cols))
    medias, desvios = dict(zip(salida_imp, esc.mean_)), dict(zip(salida_imp, esc.scale_))
    terminos = []
    for c, mediana in zip(num_cols, imp.statistics_):
        terminos.append({'tipo': 'num', 'col': c, 'mediana': float(mediana), 'media': float(medias[c]),
                         'desvio': float(desvios[c]), 'w': float(w[f'numericas__{c}'])})
    for c in con_indicador:
        k = f'missingindicator_{c}'
        terminos.append({'tipo': 'ind', 'col': c, 'media': float(medias[k]), 'desvio': float(desvios[k]),
                         'w': float(w[f'numericas__{k}'])})
    he = pre.named_transformers_['historia']['escalar']
    for c, m, s in zip(pre.transformers_[1][2], he.mean_, he.scale_):
        terminos.append({'tipo': 'log', 'col': c, 'media': float(m), 'desvio': float(s), 'w': float(w[f'historia__{c}'])})
    oh = pre.named_transformers_['categoricas']
    for c, cats in zip(pre.transformers_[2][2], oh.categories_):
        for v in cats:
            terminos.append({'tipo': 'cat', 'col': c, 'valor': v.item() if hasattr(v, 'item') else v,
                             'w': float(w[f'categoricas__{c}_{v}'])})
    return {'intercepto': float(clf.intercept_[0]), 'terminos': terminos, 'con_indicador': con_indicador}


def z_python(f, fila):
    """La misma fórmula en Python, para comprobar que da lo mismo que predict_proba."""
    z = f['intercepto']
    for t in f['terminos']:
        x = fila[t['col']]
        falta = x is None or (isinstance(x, float) and math.isnan(x))
        if t['tipo'] == 'num':
            z += t['w'] * (((t['mediana'] if falta else x)) - t['media']) / t['desvio']
        elif t['tipo'] == 'ind':
            z += t['w'] * ((1.0 if falta else 0.0) - t['media']) / t['desvio']
        elif t['tipo'] == 'log':
            z += t['w'] * (math.log1p(x) - t['media']) / t['desvio']
        else:
            z += t['w'] * (1.0 if x == t['valor'] else 0.0)
    return z


def z_vega(f):
    """La fórmula como expresión de Vega: los nombres son los parámetros de los controles."""
    partes = [repr(f['intercepto'])]
    for t in f['terminos']:
        if t['tipo'] == 'cat':
            p = 'bo' if t['col'] == 'bo_type' else 'tier'
            valor = f"'{t['valor']}'" if isinstance(t['valor'], str) else str(t['valor'])
            partes.append(f"({p} == {valor} ? {t['w']!r} : 0)")
            continue
        p = ENTRADAS[t['col']][0]
        if t['tipo'] == 'num':
            x = f"(sd_{p} ? {t['mediana']!r} : {p})" if t['col'] in f['con_indicador'] else p
            partes.append(f"{t['w']!r} * ({x} - {t['media']!r}) / {t['desvio']!r}")
        elif t['tipo'] == 'ind':
            partes.append(f"{t['w']!r} * ((sd_{p} ? 1 : 0) - {t['media']!r}) / {t['desvio']!r}")
        else:
            partes.append(f"{t['w']!r} * (log(1 + {p}) - {t['media']!r}) / {t['desvio']!r}")
    return ' + '.join(partes)


# ---------------------------------------------------------------------------
# Equipos para «Probá el modelo»: el estado de cada uno después de su último partido
# ---------------------------------------------------------------------------
def foto_de_equipos(df, minimo=20, activos_desde='2026-06-01'):
    """Para cada equipo activo, los valores que tendría si jugara al día siguiente del último dato.

    Lo que sale de los resultados (Elo, partidos previos, forma, tendencia y actividad) se recalcula
    con todos los partidos terminados, como en la auditoría de la sección 3. Lo que necesita el
    detalle del Bronze (rondas, mismo formato, mismo tier, alineación, ranking) es el último valor
    registrado del equipo.
    """
    hoy = df.start_at_utc.max().normalize() + pd.Timedelta(days=1)
    completos = df[df.end_at_utc.notna() & df.end_at_utc.lt(hoy)].sort_values(['end_at_utc', 'match_id'])
    elo, hist = defaultdict(lambda: 1500.0), defaultdict(list)
    h2h = defaultdict(list)
    for a, b, y, fin in completos[['team_a_id', 'team_b_id', 'target_team_a_won', 'end_at_utc']].itertuples(index=False):
        delta = 32 * (y - 1 / (1 + 10 ** ((elo[b] - elo[a]) / 400)))
        elo[a] += delta
        elo[b] -= delta
        hist[a].append((fin, y))
        hist[b].append((fin, 1 - y))
        h2h[(a, b)].append(y)       # a < b siempre: y = 1 si ganó el de menor ID
    filas = []
    for lado in ('a', 'b'):
        t = df[['start_at_utc', f'team_{lado}_id', f'team_{lado}_name', f'team_{lado}_ranking_score',
                f'team_{lado}_round_winrate_last20', f'team_{lado}_same_bo_winrate_last20',
                f'team_{lado}_same_tier_winrate_last20', f'team_{lado}_lineup_stability_last5']].copy()
        t.columns = ['fecha', 'id', 'nombre', 'rank', 'rondas', 'mismo_bo', 'mismo_tier', 'plantel']
        filas.append(t)
    todas = pd.concat(filas).sort_values('fecha')
    ultimo = todas.groupby('id').last()          # último valor no nulo de cada columna
    nombre = todas.groupby('id').nombre.last()
    cuantos = todas.groupby('id').size()
    ultima_fecha = todas.groupby('id').fecha.max()
    elegidos = cuantos[(cuantos >= minimo) & (ultima_fecha >= pd.Timestamp(activos_desde, tz='UTC'))].index
    equipos = []
    for i in elegidos:
        h = hist[i]
        res = [v for _, v in h]
        equipos.append({
            'id': int(i), 'n': nombre[i], 'elo': round(elo[i], 1), 'previos': len(h),
            'forma': float(np.mean(res[-5:])) if len(res) >= 5 else None,
            'tend': float(np.mean(res[-5:]) - np.mean(res[-10:-5])) if len(res) >= 10 else None,
            'sem': sum(1 for f, _ in h if f >= hoy - pd.Timedelta(days=7)),
            'mes': sum(1 for f, _ in h if f >= hoy - pd.Timedelta(days=30)),
            **{c: (None if pd.isna(ultimo.loc[i, c]) else round(float(ultimo.loc[i, c]), 4))
               for c in ['rank', 'rondas', 'mismo_bo', 'mismo_tier', 'plantel']},
        })
    equipos.sort(key=lambda e: -e['elo'])
    ids = set(int(i) for i in elegidos)
    cruces = {f'{a}-{b}': [int(sum(v[-5:])), len(v[-5:])] for (a, b), v in h2h.items() if a in ids and b in ids}
    return equipos, cruces, hoy


# ---------------------------------------------------------------------------
# 6 · Gráficos de la copa
# ---------------------------------------------------------------------------
def datos_explorar(train, test, p_val, fold_val, p_test):
    filas = []
    for datos, p, periodo in [(train[fold_val > 0], p_val[fold_val > 0], 'Validación cruzada'),
                              (test, pd.Series(p_test, index=test.index), 'Test')]:
        d = cm.detalle_predicciones(datos, p.to_numpy())
        filas.append(pd.DataFrame({
            'id': d.match_id, 'fecha': d.start_at_utc.dt.strftime('%Y-%m-%d'),
            'a': d.team_a_name, 'b': d.team_b_name, 'torneo': d.tournament_name.str.strip(),
            'bo': 'BO' + d.bo_type.astype(str), 'tier': d.tier.map(g.TIER_NOMBRE),
            'elo': d.elo_advantage_a.round(0).astype(int), 'p': d.prob_a.round(3),
            'ganador': np.where(d.real == 1, 'Ganó A', 'Ganó B'), 'ok': d.acierto.astype(int),
            'mapas': d.post_team_a_maps_won.astype(str) + '-' + d.post_team_b_maps_won.astype(str),
            'periodo': periodo, 'sorpresa': d.sorpresa.astype(int)}))
    return pd.concat(filas, ignore_index=True)


def g_explora(ex):
    csv = ex.to_csv(index=False)
    datos = alt.InlineData(values=csv, format=alt.DataFormat(type='csv', parse={
        'id': 'number', 'elo': 'number', 'p': 'number', 'ok': 'number', 'sorpresa': 'number'}))
    E = '#c-explora'
    tier_sel = alt.param(name='tier_sel', value='Todos', bind=alt.binding_select(
        options=['Todos'] + g.TIER_ORDEN[:4], name='Tier ', element=E))
    bo_sel = alt.param(name='bo_sel', value='Todos', bind=alt.binding_select(
        options=['Todos', 'BO1', 'BO3', 'BO5'], name='Formato ', element=E))
    periodo = alt.param(name='periodo', value='Todo', bind=alt.binding_radio(
        options=['Todo', 'Validación cruzada', 'Test'], name='Período ', element=E))
    cuales = alt.param(name='cuales', value='Todas', bind=alt.binding_radio(
        options=['Todas', 'Sólo errores', 'Sólo sorpresas'], name='Series ', element=E))
    equipo = alt.param(name='equipo', value='', bind=alt.binding(
        input='search', name='Equipo ', placeholder='p. ej. Vitality', element=E))
    conf_min = alt.param(name='conf_min', value=50, bind=alt.binding_range(
        min=50, max=95, step=5, name='Confianza mínima del modelo (%) ', element=E))
    serie = alt.selection_point(name='serie', fields=['id'], on='click', empty=False, clear='dblclick')

    filtro = ("(tier_sel == 'Todos' || datum.tier == tier_sel) && (bo_sel == 'Todos' || datum.bo == bo_sel)"
              " && (periodo == 'Todo' || datum.periodo == periodo)"
              " && (cuales == 'Todas' || (cuales == 'Sólo errores' ? !datum.ok : datum.sorpresa))"
              " && (equipo == '' || indexof(lower(datum.a + ' ' + datum.b), lower(equipo)) >= 0)"
              " && max(datum.p, 1 - datum.p) * 100 >= conf_min")
    base = alt.Chart(datos).transform_calculate(en_filtro=filtro, conf='max(datum.p, 1 - datum.p)',
                                                elo_c='clamp(datum.elo, -700, 700)')

    titulo = base.transform_filter('datum.en_filtro').transform_aggregate(
        n='count()', aciertos='sum(ok)').transform_calculate(
        t="datum.n > 0 ? replace(format(datum.n, ','), ',', '.') + ' series en el filtro · el modelo acierta el ' + "
          "format(datum.aciertos / datum.n * 100, '.1f') + ' %' : 'Ninguna serie cumple el filtro'"
    ).mark_text(align='left', fontSize=15, fontWeight='bold', color=g.TINTA).encode(
        text='t:N', x=alt.value(0), y=alt.value(10)).properties(width=760, height=20)

    esc_x = alt.Scale(domain=[-700, 700], nice=False)
    puntos = base.mark_circle().encode(
        x=alt.X('elo_c:Q', title='Diferencia de Elo antes de la serie (A − B)', scale=esc_x),
        y=alt.Y('p:Q', title='Probabilidad de que gane A, según el modelo', scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format='.0%')),
        color=alt.condition('datum.en_filtro', alt.Color('ganador:N', scale=g.ESC_GANADOR, title='Resultado real',
                                                         legend=alt.Legend(orient='bottom-right')), alt.value(g.APAGADO)),
        opacity=alt.condition('datum.en_filtro', alt.value(0.8), alt.value(0.6)),
        size=alt.condition(serie, alt.value(170), alt.value(14)),
        stroke=alt.condition(serie, alt.value('#ffffff'), alt.value(None)),
        order=alt.Order('en_filtro_n:Q'),
        tooltip=[alt.Tooltip('fecha:N', title='Fecha'), alt.Tooltip('a:N', title='Equipo A'),
                 alt.Tooltip('b:N', title='Equipo B'), alt.Tooltip('torneo:N', title='Torneo'),
                 alt.Tooltip('bo:N', title='Formato'), alt.Tooltip('tier:N', title='Tier'),
                 alt.Tooltip('elo:Q', title='Elo A − B'),
                 alt.Tooltip('p:Q', title='Modelo: prob. de que gane A', format='.1%'),
                 alt.Tooltip('ganador:N', title='Resultado'), alt.Tooltip('mapas:N', title='Mapas A-B'),
                 alt.Tooltip('periodo:N', title='Predicho en')],
    ).transform_calculate(en_filtro_n='datum.en_filtro ? 1 : 0').add_params(serie)
    medio = alt.Chart(pd.DataFrame({'y': [0.5]})).mark_rule(color=g.GRIS, strokeDash=[4, 3]).encode(y='y:Q')
    rot = (alt.Chart(pd.DataFrame({'y': [0.5], 't': ['el modelo dice «gana A» ↑']})).mark_text(
               align='right', x=496, dy=-8, color=g.GRIS, fontSize=12).encode(y='y:Q', text='t:N')
           + alt.Chart(pd.DataFrame({'y': [0.5], 't': ['el modelo dice «gana B» ↓']})).mark_text(
               align='right', x=496, dy=12, color=g.GRIS, fontSize=12).encode(y='y:Q', text='t:N'))
    nube = (puntos + medio + rot).properties(width=500, height=430)

    sel = base.transform_filter(serie)
    lineas = sel.transform_calculate(
        l1="datum.a + ' vs. ' + datum.b",
        l2="datum.fecha + ' · ' + datum.bo + ' · tier ' + datum.tier",
        l3="datum.torneo",
        l4="'Resultado: ' + (datum.ganador == 'Ganó A' ? datum.a : datum.b) + ' ' + datum.mapas",
        l5="'Modelo: ' + (datum.p >= 0.5 ? datum.a : datum.b) + ' con ' + format(max(datum.p, 1 - datum.p) * 100, '.0f') + ' %' + (datum.ok ? '  ✓' : '  ✗')",
    ).transform_fold(['l1', 'l2', 'l3', 'l4', 'l5'], as_=['k', 'linea']).transform_calculate(
        fila="indexof(['l1', 'l2', 'l3', 'l4', 'l5'], datum.k)")
    detalle = (lineas.transform_filter('datum.fila == 0').mark_text(align='left', baseline='top', fontSize=14,
                                                                     fontWeight='bold', color=g.TINTA, limit=260)
               + lineas.transform_filter('datum.fila > 0').mark_text(align='left', baseline='top', fontSize=12.5,
                                                                      color=g.TINTA, limit=260)
               ).encode(text='linea:N', x=alt.value(0), y=alt.Y('fila:O', axis=None, scale=alt.Scale(domain=[0, 1, 2, 3, 4])))
    vacio = alt.Chart(pd.DataFrame({'t': ['Hacé clic en una serie para ver', 'qué dijo el modelo (doble clic borra).']})
                      ).transform_window(n='row_number()').transform_calculate(fila='datum.n - 1').mark_text(align='left', baseline='top', fontSize=13,
                                                                        color=g.GRIS).encode(
        text='t:N', x=alt.value(0), y=alt.Y('fila:O', axis=None, scale=alt.Scale(domain=[0, 1, 2, 3, 4]))
    ).transform_filter("!length(data('serie_store'))")
    panel_detalle = (detalle + vacio).properties(width=260, height=110)

    barras_p = sel.transform_calculate(pA='datum.p', pB='1 - datum.p').transform_fold(
        ['pA', 'pB'], as_=['lado', 'prob']).transform_calculate(
        quien="datum.lado == 'pA' ? 'Ganó A' : 'Ganó B'",
        equipo="datum.lado == 'pA' ? datum.a : datum.b")
    prob_sel = (barras_p.mark_bar(cornerRadiusEnd=3, height=18).encode(
        x=alt.X('prob:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='.0%', tickCount=3),
                title='Probabilidad según el modelo'),
        y=alt.Y('equipo:N', title=None, sort=None, axis=alt.Axis(labelLimit=120)),
        color=alt.Color('quien:N', scale=g.ESC_GANADOR, title='Resultado real', legend=alt.Legend(orient='bottom-right')))
        + barras_p.mark_text(align='left', dx=4, fontSize=12, color=g.TINTA).encode(
            x='prob:Q', y=alt.Y('equipo:N', sort=None), text=alt.Text('prob:Q', format='.0%'))
    ).properties(width=150, height=alt.Step(26))

    por_conf = base.transform_filter('datum.en_filtro').transform_calculate(
        tramo="datum.conf < 0.6 ? '50–60 %' : datum.conf < 0.7 ? '60–70 %' : datum.conf < 0.8 ? '70–80 %' : '80 % o más'"
    ).transform_aggregate(n='count()', ac='mean(ok)', groupby=['tramo'])
    tramos = ['50–60 %', '60–70 %', '70–80 %', '80 % o más']
    barras_conf = (por_conf.mark_bar(color=g.ACENTO, cornerRadiusEnd=3, height=16).encode(
        x=alt.X('ac:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='.0%', tickCount=3), title='Acierta (en el filtro)'),
        y=alt.Y('tramo:N', sort=tramos, scale=alt.Scale(domain=tramos), title='Confianza'),
        tooltip=[alt.Tooltip('tramo:N', title='Confianza'), alt.Tooltip('n:Q', title='Series', format=','),
                 alt.Tooltip('ac:Q', title='Acierta', format='.1%')])
        + por_conf.mark_text(align='left', dx=4, fontSize=12, color=g.TINTA).encode(
            x='ac:Q', y=alt.Y('tramo:N', sort=tramos), text=alt.Text('ac:Q', format='.0%'))
    ).properties(width=150, height=alt.Step(24))

    lado = alt.vconcat(panel_detalle, prob_sel, barras_conf, spacing=22)
    return alt.vconcat(titulo, alt.hconcat(nube, lado, spacing=30).resolve_scale(color='shared'), spacing=12).add_params(
        tier_sel, bo_sel, periodo, cuales, equipo, conf_min).resolve_scale(color='independent').configure_view(stroke=None)


def g_proba(f, train, equipo_ejemplo):
    """Probá el modelo: la logística calculada en el navegador, y la apuesta."""
    params = []
    X = train[cm.COLUMNAS_FINAL]
    rangos = {}
    for col, (p, etiqueta, grupo, paso) in ENTRADAS.items():
        lo, hi = float(X[col].min()), float(X[col].max())
        if paso >= 1:
            lo, hi = math.floor(lo), math.ceil(hi)
        mediana = float(X[col].median())
        valor = round(round(mediana / paso) * paso, 4)
        rangos[p] = {'min': lo, 'max': hi, 'paso': paso, 'mediana': valor, 'col': col, 'etiqueta': etiqueta,
                     'grupo': grupo, 'sin_dato': col in f['con_indicador'],
                     'p01': float(X[col].quantile(0.01)), 'p99': float(X[col].quantile(0.99))}
        params.append(alt.param(name=p, value=valor, bind=alt.binding_range(
            min=lo, max=hi, step=paso, name=etiqueta, element=f'#d-{p}')))
        if col in f['con_indicador']:
            params.append(alt.param(name=f'sd_{p}', value=False, bind=alt.binding_checkbox(
                name='sin dato', element=f'#d-{p}')))
    bos = sorted(int(v) for v in train.bo_type.unique())
    params.append(alt.param(name='bo', value=3, bind=alt.binding_select(
        options=bos, labels=[f'BO{b}' for b in bos], name='Formato ', element='#d-contexto')))
    params.append(alt.param(name='tier', value='b', bind=alt.binding_select(
        options=cm_tiers(train), labels=[g.TIER_NOMBRE[t] for t in cm_tiers(train)], name='Tier ', element='#d-contexto')))
    params += [alt.param(name='nombre_a', value='Equipo A'), alt.param(name='nombre_b', value='Equipo B')]
    params += [alt.param(name='cuota_a', value=1.9, bind=alt.binding_range(
                   min=1.01, max=8, step=0.01, name='Cuota que paga la casa por A', element='#d-cuotas')),
               alt.param(name='cuota_b', value=1.9, bind=alt.binding_range(
                   min=1.01, max=8, step=0.01, name='Cuota que paga la casa por B', element='#d-cuotas'))]
    params += [alt.param(name='z', expr=z_vega(f)), alt.param(name='pA', expr='1 / (1 + exp(-z))')]
    rara = ' || '.join((f"(!sd_{p} && " if r['sin_dato'] else "(") + f"({p} < {r['p01']!r} || {p} > {r['p99']!r}))"
                       for p, r in rangos.items())
    params.append(alt.param(name='raro', expr=f"({rara})"))

    lados = pd.DataFrame({'lado': ['A', 'B']})
    base = alt.Chart(lados).transform_calculate(
        quien="datum.lado == 'A' ? 'Ganó A' : 'Ganó B'",
        equipo="datum.lado == 'A' ? nombre_a : nombre_b",
        p="datum.lado == 'A' ? pA : 1 - pA",
        cuota="datum.lado == 'A' ? cuota_a : cuota_b",
        casa="(1 / (datum.lado == 'A' ? cuota_a : cuota_b)) / (1 / cuota_a + 1 / cuota_b)",
        ev="(datum.lado == 'A' ? pA : 1 - pA) * (datum.lado == 'A' ? cuota_a : cuota_b) - 1",
        texto="format((datum.lado == 'A' ? pA : 1 - pA) * 100, '.1f') + ' %'",
    )
    y = alt.Y('equipo:N', title=None, sort=None, axis=alt.Axis(labelLimit=200, labelFontSize=14, labelFontWeight='bold'))
    barras = (base.mark_bar(cornerRadiusEnd=4, height=34).encode(
        x=alt.X('p:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='.0%'), title='Probabilidad de ganar la serie, según el modelo'),
        y=y, color=alt.Color('quien:N', scale=g.ESC_GANADOR, legend=None),
        tooltip=[alt.Tooltip('equipo:N', title='Equipo'), alt.Tooltip('p:Q', title='Probabilidad', format='.1%')])
        + base.mark_text(align='left', dx=6, fontSize=15, fontWeight='bold', color=g.TINTA).encode(
            x='p:Q', y=y, text='texto:N')
    ).properties(width=330, height=alt.Step(48))
    cabeza = alt.Chart(pd.DataFrame({'x': [0]})).mark_text(align='left', fontSize=17, fontWeight='bold', color=g.TINTA,
                                                          text=alt.expr("'El modelo dice: ' + (pA >= 0.5 ? nombre_a : nombre_b)")
                                                          ).encode(x=alt.value(0), y=alt.value(8)).properties(width=330, height=18)
    aviso = alt.Chart(pd.DataFrame({'x': [0]})).mark_text(
        align='left', fontSize=12.5, color=g.COLOR_ERR, text=alt.expr(
            "raro ? '⚠ Algún valor está en el 1 % más extremo del entrenamiento: el modelo casi no vio casos así.' : ''")
    ).encode(x=alt.value(0), y=alt.value(6)).properties(width=330, height=14)

    # La apuesta: lo que dice el modelo contra lo que dice la casa, y el valor esperado.
    comp = base.transform_fold(['p', 'casa'], as_=['fuente', 'prob']).transform_calculate(
        fuente="datum.fuente == 'p' ? 'Modelo' : 'Casa (sin margen)'")
    y2 = alt.Y('equipo:N', title=None, sort=None, axis=alt.Axis(labelLimit=200))
    comparar = (comp.mark_bar(height=14).encode(
        x=alt.X('prob:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='.0%'), title='Probabilidad de ganar'),
        y=y2, yOffset=alt.YOffset('fuente:N', sort=['Modelo', 'Casa (sin margen)']),
        color=alt.Color('fuente:N', scale=alt.Scale(domain=['Modelo', 'Casa (sin margen)'], range=[g.ACENTO, g.SUAVE]),
                        legend=alt.Legend(orient='top', title=None)),
        tooltip=[alt.Tooltip('equipo:N', title='Equipo'), alt.Tooltip('fuente:N', title='Según'),
                 alt.Tooltip('prob:Q', title='Probabilidad', format='.1%')])
    ).properties(width=330, height=alt.Step(40))
    filas_ev = base.transform_calculate(
        linea="'$100 a ' + datum.equipo + ' (cuota ' + replace(format(datum.cuota, '.2f'), '.', ',') + '): ' + "
              "(datum.ev >= 0 ? '+' : '−') + '$' + format(abs(datum.ev) * 100, '.0f') + ' de valor esperado'",
        orden="datum.lado == 'A' ? 0 : 1")
    ev = filas_ev.mark_text(align='left', fontSize=13, limit=330).encode(
        x=alt.value(0), y=alt.Y('orden:O', axis=None), text='linea:N',
        color=alt.condition('datum.ev > 0', alt.value(g.COLOR_OK), alt.value(g.GRIS))
    ).properties(width=330, height=40)
    veredicto = alt.Chart(pd.DataFrame({'x': [0]})).mark_text(
        align='left', baseline='top', fontSize=15, fontWeight='bold', color=g.TINTA, lineBreak='\n', lineHeight=20, text=alt.expr(
            "pA * cuota_a - 1 > 0 ? '→ Hay valor en apostar a ' + nombre_a + '\\nKelly: ' + format(100 * (pA * cuota_a - 1) / (cuota_a - 1), '.0f') + ' % de la banca' : "
            "(1 - pA) * cuota_b - 1 > 0 ? '→ Hay valor en apostar a ' + nombre_b + '\\nKelly: ' + format(100 * ((1 - pA) * cuota_b - 1) / (cuota_b - 1), '.0f') + ' % de la banca' : "
            "'→ Sin valor en ninguno de los dos:\\nla casa paga menos de lo que el modelo cree'")
    ).encode(x=alt.value(0), y=alt.value(0)).properties(width=330, height=40)
    margen = alt.Chart(pd.DataFrame({'x': [0]})).mark_text(
        align='left', baseline='top', fontSize=12, color=g.GRIS, lineBreak='\n', lineHeight=16, text=alt.expr(
            "'Margen de la casa: ' + replace(format((1 / cuota_a + 1 / cuota_b - 1) * 100, '.1f'), '.', ',') + ' %.' + "
            "'\\nValor esperado = probabilidad del modelo × cuota − 1.'")
    ).encode(x=alt.value(0), y=alt.value(0)).properties(width=330, height=32)

    graf = alt.vconcat(cabeza, barras, aviso, comparar, ev, veredicto, margen, spacing=10).add_params(*params).resolve_scale(color='independent')
    return graf.configure_view(stroke=None), rangos


def cm_tiers(train):
    return [t for t in g.TIERS if t in set(train.tier)]


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def tabla_html(df, columnas=None, formatos=None):
    formatos = formatos or {}
    columnas = columnas or list(df.columns)
    cab = ''.join(f'<th>{c}</th>' for c in columnas)
    filas = []
    for _, r in df.iterrows():
        celdas = []
        for c in columnas:
            v = r[c]
            celdas.append(f'<td>{formatos[c](v) if c in formatos else v}</td>')
        filas.append('<tr>' + ''.join(celdas) + '</tr>')
    return f'<div class="tabla"><table><thead><tr>{cab}</tr></thead><tbody>{"".join(filas)}</tbody></table></div>'


def pm(m, s, d=1):
    return f"{pct(m, d)} ± {pct(s, d).replace(' %', '')}"


LOCALE_TIEMPO = {"dateTime": "%A, %e de %B de %Y, %X", "date": "%d/%m/%Y", "time": "%H:%M:%S",
                 "periods": ["AM", "PM"],
                 "days": ["domingo", "lunes", "martes", "miércoles", "jueves", "viernes", "sábado"],
                 "shortDays": ["dom", "lun", "mar", "mié", "jue", "vie", "sáb"],
                 "months": ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
                            "septiembre", "octubre", "noviembre", "diciembre"],
                 "shortMonths": ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', action='store_true', help='reusar resultados/grillas.pkl si existe')
    ap.add_argument('--sin-huella', action='store_true', help='no exigir la huella del CSV de la Entrega 2')
    ap.add_argument('--csv', default=None, help='por defecto: datos/matches_latest.csv o la salida del DAG')
    args = ap.parse_args()

    print('1 · Cargando datos…')
    df = cm.cargar(args.csv, verificar_huella=not args.sin_huella)
    print(f'  {len(df):,} series desde {args.csv or cm.ruta_csv()}')
    train, test, purgados = cm.separar(df)
    cv, tabla_folds = cm.construir_cv_temporal(train)
    X_train, y_train = train[cm.COLUMNAS_E2 + [cm.EXTRA]], train.target_team_a_won
    X_test, y_test = test[cm.COLUMNAS_E2 + [cm.EXTRA]], test.target_team_a_won

    print('2 · Grillas de las cuatro familias…')
    grillas = correr_grillas(train, cv, args.cache)
    comp = tabla_comparacion(grillas)
    mejor, limite, familia = cm.elegir_simple(comp, cm.ORDEN_SIMPLICIDAD)
    assert familia == 'regresión logística', 'La regla ya no elige la logística: revisar el notebook.'

    print('3 · Modelo final…')
    final_pipe = cm.armar(LogisticRegression(max_iter=5000, random_state=cm.SEMILLA, **cm.PARAMS_FINAL), cm.COLUMNAS_FINAL)
    res_cv, cv_raw = cm.medir(final_pipe, X_train, y_train, cv)
    grilla_final = GridSearchCV(cm.armar(cm.FAMILIAS['regresión logística'], cm.COLUMNAS_FINAL),
                                cm.GRILLAS['regresión logística'], scoring=cm.SCORING, refit=False, cv=cv,
                                n_jobs=cm.N_JOBS, return_train_score=True).fit(X_train, y_train)
    tabla_grilla = cm.tabla_de(grilla_final)
    modelo = cm.entrenar_final(train)
    p_test = modelo.predict_proba(test[cm.COLUMNAS_FINAL])[:, 1]
    detalle = cm.detalle_predicciones(test, p_test)
    acc_test = detalle.acierto.mean()
    piso_test = (y_test == y_train.mode()[0]).mean()

    # La fórmula en Python tiene que dar lo mismo que scikit-learn.
    f = formula(modelo)
    z = np.array([z_python(f, r) for r in test[cm.COLUMNAS_FINAL].to_dict('records')])
    assert np.allclose(1 / (1 + np.exp(-z)), p_test, atol=1e-10), 'La fórmula no reproduce al modelo.'

    print('4 · Predicciones para explorar…')
    p_val, fold_val = cm.predicciones_fuera_de_fold(final_pipe, train[cm.COLUMNAS_FINAL], y_train, cv)
    ex = datos_explorar(train, test, p_val, fold_val, p_test)
    acc_val_oof = ex[ex.periodo == 'Validación cruzada'].ok.mean()

    print('5 · Otros modelos en el test (contexto)…')
    otros = {}
    for nombre in cm.ORDEN_SIMPLICIDAD:
        if nombre == 'regresión logística':
            otros[nombre] = p_test
            continue
        clf = clone(cm.FAMILIAS[nombre]).set_params(**{k.replace('clasificador__', ''): v
                                                       for k, v in grillas[nombre]['params'].items()})
        otros[nombre] = cm.armar(clf, cm.COLUMNAS_FINAL).fit(train[cm.COLUMNAS_FINAL], y_train).predict_proba(
            test[cm.COLUMNAS_FINAL])[:, 1]
    acierta = pd.DataFrame({n: ((p >= 0.5).astype(int) == y_test.to_numpy()) for n, p in otros.items()})
    falla_todos = (~acierta).all(axis=1)
    test_otros = acierta.mean()

    print('6 · Gráficos y página…')
    equipos, cruces, hoy = foto_de_equipos(df)
    graficos = {
        'g-datos': g.g_datos(df), 'g-hallazgo': g.g_hallazgo(df), 'g-modelos': g.g_modelos(comp),
        'g-confianza': g.g_confianza(detalle), 'g-explora': g_explora(ex),
    }
    gp, rangos = g_proba(f, train, None)
    graficos['g-proba'] = gp
    # Sin «$schema»: la página trae Vega-Lite 6 y Vega 6, y así vega-embed compila cada gráfico con
    # esas versiones en lugar de avisar que fue escrito para Vega-Lite 5.
    specs = {k: {c: x for c, x in v.to_dict().items() if c != '$schema'} for k, v in graficos.items()}

    # ---- números que cita el texto
    errores = detalle[~detalle.acierto]
    sorpresa_err = errores.sorpresa.mean()
    tc = g.tabla_confianza(detalle)
    alta = detalle[detalle.confianza >= 0.7]
    t_elo, gana_fav, n_elo = g.tabla_elo(df)
    por_tier_elo = df[df.elo_advantage_a != 0].assign(
        gana=lambda d: np.where(d.elo_advantage_a > 0, d.target_team_a_won, 1 - d.target_team_a_won)
    ).groupby('tier').gana.agg(['mean', 'size'])
    r = df.rest_days_advantage_a
    m_rest = r.notna() & (r.abs() > 1)
    gana_desc = np.where(r[m_rest] > 0, df.target_team_a_won[m_rest], 1 - df.target_team_a_won[m_rest]).mean()
    calib_media, calib_real = detalle.prob_a.mean(), detalle.real.mean()
    h = cm.hipotesis_e2(df)
    dec = lambda x, d=3: f'{x:.{d}f}'.replace('.', ',')
    mensual = detalle.groupby('mes').acierto.agg(['mean', 'size'])
    peor_mes = mensual['mean'].idxmin()
    coefs = pd.Series(modelo['clasificador'].coef_[0], index=modelo['preparar'].get_feature_names_out())
    coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index)
    tipos = []
    for variable, nombre_v in [('tier', 'Tier'), ('bo_type', 'Formato'), ('historia', 'Historia previa')]:
        for valor, gr in detalle.groupby(variable):
            tipos.append({'Corte': nombre_v, 'Grupo': g.TIER_NOMBRE.get(valor, f'BO{valor}' if variable == 'bo_type' else valor),
                          'Series': num(len(gr)), 'Sorpresas': pct(gr.sorpresa.mean(), 0), 'Accuracy': pct(gr.acierto.mean())})
    caso = errores.sort_values('confianza', ascending=False).iloc[0]

    numeros = {'accuracy_test': float(acc_test), 'accuracy_cv': float(res_cv['accuracy']),
               'desvio_cv': float(res_cv['desvío']), 'series_test': int(len(test)),
               'intercepto': f['intercepto'], 'piso_test': float(piso_test),
               'errores_sorpresa': float(sorpresa_err), 'generado': datetime.now().isoformat(timespec='seconds')}
    assert abs(numeros['accuracy_cv'] - tabla_grilla.iloc[0].mean_test_accuracy) < 1e-12

    # ---- textos y tablas de los «Cómo lo sabemos»
    filas_comp = comp.reset_index().rename(columns={'index': 'familia'})
    filas_comp['Modelo'] = filas_comp.familia.map(g.NOMBRE_MODELO)
    # (la columna accuracy se completa con media ± desvío)
    filas_comp['Accuracy (media ± desvío)'] = [pm(a, s) for a, s in zip(filas_comp.accuracy, filas_comp['desvío'])]
    filas_comp = filas_comp.rename(columns={'combinación': 'Mejor combinación de su grilla', 'accuracy train': 'Accuracy en entrenamiento',
                                            'f1 macro': 'f1 macro', 'recall A': 'Recall A', 'recall B': 'Recall B'})
    tabla_comp = tabla_html(filas_comp, ['Modelo', 'Mejor combinación de su grilla', 'Accuracy (media ± desvío)',
                                         'Accuracy en entrenamiento', 'f1 macro', 'Recall A', 'Recall B'],
                            {'Accuracy en entrenamiento': pct, 'f1 macro': lambda v: f'{v:.3f}'.replace('.', ','),
                             'Recall A': pct, 'Recall B': pct})
    grillas_cortas = []
    for nombre in cm.ORDEN_SIMPLICIDAD:
        t = grillas[nombre]['tabla'].head(3)
        for _, fila in t.iterrows():
            params = ', '.join(f"{c} = {str(fila[c]).replace('.', ',')}" for c in t.columns if not c.startswith(('mean_', 'std_', 'rank_')))
            grillas_cortas.append({'Familia': g.NOMBRE_MODELO[nombre], 'Combinación': params,
                                   'Accuracy': pm(fila.mean_test_accuracy, fila.std_test_accuracy),
                                   'En entrenamiento': pct(fila.mean_train_accuracy),
                                   'Puesto': int(fila.rank_test_accuracy),
                                   'Probadas': len(grillas[nombre]['tabla'])})
    tabla_grillas = tabla_html(pd.DataFrame(grillas_cortas))
    tg = tabla_grilla.copy()
    tg['Accuracy'] = [pm(a, s, 2) for a, s in zip(tg.mean_test_accuracy, tg.std_test_accuracy)]
    tg['Recall A'], tg['Recall B'] = tg.mean_test_recall_A.map(pct), tg.mean_test_recall_B.map(pct)
    tg['class_weight'] = tg.class_weight.map(lambda v: 'balanced' if v == 'balanced' else 'None')
    tabla_ajuste = tabla_html(tg, ['C', 'class_weight', 'Accuracy', 'Recall A', 'Recall B'],
                              {'C': lambda v: str(v).replace('.', ',')})
    tabla_tipos = tabla_html(pd.DataFrame(tipos))
    tabla_otros = tabla_html(pd.DataFrame({'Modelo': [g.NOMBRE_MODELO[n] for n in test_otros.index],
                                           'Accuracy en el test': [pct(v) for v in test_otros.values]}))
    tabla_coefs = tabla_html(pd.DataFrame({'Variable (escalada)': [c.split('__')[1] for c in coefs.index[:8]],
                                           'Coeficiente': [f'{v:+.3f}'.replace('.', ',') for v in coefs.values[:8]]}))
    pasos_filas = pd.DataFrame([
        ('Partidos descubiertos', '—', 'Todos los partidos de Counter-Strike que lista Bo3.gg desde el 27/09/2023.'),
        ('Sale: no terminados', '—', 'Partidos sin estado «finished»: no tienen resultado.'),
        ('Sale: abandono', '—', 'Forfeit, walkover o «defwin» según la fuente, y un abandono revisado a mano (1win–M1X, 25/09/2025): '
         'no se jugaron, no dicen nada de la fuerza de los equipos.'),
        ('Sale: resultado no verificable', '—', 'Empates, marcadores incompatibles con el formato o que contradicen al ganador '
         '(<code>quarantined_result</code>). No se inventa un ganador.'),
        ('Sale: cronología contradictoria', '—', 'Fin anterior al inicio o mapas que empiezan antes del partido (<code>quarantined_temporal</code>).'),
        ('Sale: no es CS2', '—', 'Partidos de CS:GO o de versión desconocida: quedan en el Bronze, fuera del Silver y de los historiales.'),
        ('Silver del pipeline', num(len(df)), 'Una fila por serie de CS2 (game_version = 2), del '
         f'{df.start_at_utc.min():%d/%m/%Y} al {df.start_at_utc.max():%d/%m/%Y}, con 113 columnas. '
         f'{int(df.end_at_utc.isna().sum())} no tienen hora de finalización: no se inventa, y no alimentan el historial de partidos posteriores.'),
        ('Entrenamiento', num(len(train)), f'Series que empezaron y terminaron antes del {cm.CORTE_TEST:%d/%m/%Y}. Acá se elige todo.'),
        ('Purgadas', num(len(purgados)), 'Empezaron antes del corte pero no consta que terminaran antes: su resultado no se conocía.'),
        ('Test', num(len(test)), f'Las que empiezan desde el {cm.CORTE_TEST:%d/%m/%Y}. Se miran una sola vez, al final.'),
        ('Para explorar', num(len(ex)), f'Las {num((ex.periodo == "Validación cruzada").sum())} de validación (predichas fuera de fold) '
         f'más las {num(len(test))} del test. Las {num((fold_val == 0).sum())} del primer bloque de entrenamiento no tienen predicción: '
         'ningún fold las valida.'),
    ], columns=['Paso', 'Series', 'Qué se hizo'])
    tabla_pasos = tabla_html(pasos_filas)
    folds = tabla_folds.assign(**{'Entrena con': tabla_folds['n train'].map(num), 'Valida con': tabla_folds['n validación'].map(num),
                                  'Valida desde': tabla_folds['validación desde'].dt.strftime('%d/%m/%Y'),
                                  'hasta': (tabla_folds['validación hasta (excl.)'] - pd.Timedelta(days=1)).dt.strftime('%d/%m/%Y'),
                                  'Accuracy del modelo final': [pct(v) for v in cv_raw['test_accuracy']]})
    tabla_folds_html = tabla_html(folds, ['fold', 'Entrena con', 'Valida desde', 'hasta', 'Valida con', 'Accuracy del modelo final'])

    ctx = dict(
        n=num(len(df)), desde=f'{df.start_at_utc.min():%d/%m/%Y}', hasta=f'{df.start_at_utc.max():%d/%m/%Y}',
        acc_test=pct(acc_test), piso_test=pct(piso_test), dif_piso=f'{100 * (acc_test - piso_test):.1f}'.replace('.', ','),
        acc_cv=pm(res_cv['accuracy'], res_cv['desvío']), n_test=num(len(test)), n_train=num(len(train)),
        sorpresa_err=pct(sorpresa_err, 0), sorpresa_err_n=round(10 * sorpresa_err),
        sorpresas_test=pct(detalle.sorpresa.mean(), 0),
        acc_no_sorpresa=pct(detalle[~detalle.sorpresa].acierto.mean(), 0), acc_sorpresa=pct(detalle[detalle.sorpresa].acierto.mean(), 0),
        conf80=pct(tc.iloc[-1].accuracy, 0), conf_baja=pct(tc.iloc[0].accuracy, 0), n80=num(tc.iloc[-1].series),
        alta_acc=pct(alta.acierto.mean(), 0), alta_n=num(len(alta)),
        gana_fav=pct(gana_fav, 0), n_elo=num(n_elo), empates_elo=num(len(df) - n_elo),
        tier_a=pct(por_tier_elo.loc['a', 'mean']), tier_min=pct(por_tier_elo.drop(index=['a', 'd'])['mean'].min()),
        tier_max=pct(por_tier_elo.drop(index=['a', 'd'])['mean'].max()),
        gana_desc=pct(gana_desc), n_desc=num(m_rest.sum()),
        mejor=g.NOMBRE_MODELO[mejor].lower(), mejor_acc=pm(comp.loc[mejor, 'accuracy'], comp.loc[mejor, 'desvío']),
        log_acc=pm(comp.loc['regresión logística', 'accuracy'], comp.loc['regresión logística', 'desvío']),
        limite=pct(limite, 2), arbol_acc=pm(comp.loc['árbol de decisión', 'accuracy'], comp.loc['árbol de decisión', 'desvío']),
        rf_train=pct(comp.loc['random forest', 'accuracy train']), rf_acc=pm(comp.loc['random forest', 'accuracy'], comp.loc['random forest', 'desvío']),
        arbol_prof=grillas['árbol de decisión']['params']['clasificador__max_depth'], piso_cv=pct(comp.loc['piso', 'accuracy']),
        falla_todos=pct(falla_todos.mean(), 0), falla_todos_err=pct(falla_todos[~acierta['regresión logística']].mean(), 0),
        calib_media=pct(calib_media), calib_real=pct(calib_real),
        peor_mes=['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre',
                  'noviembre', 'diciembre'][peor_mes.month - 1], peor_mes_acc=pct(mensual.loc[peor_mes, 'mean']),
        caso=caso, hoy=f'{hoy:%d/%m/%Y}', n_equipos=len(equipos),
        h_tramo=num(h['tramo']),
        h1_n=num(h['H1']['n']), h1_sep=dec(h['H1']['sep']), h1_zona=h['H1']['zona'],
        h1_a=dec(abs(h['H1']['media_a']), 2), h1_b=dec(h['H1']['media_b'], 2),
        h2_n=num(h['H2']['n']), h2_sep=dec(h['H2']['sep']), h2_zona=h['H2']['zona'],
        h2_a=dec(h['H2']['media_a'], 1), h2_b=dec(abs(h['H2']['media_b']), 1),
        h2_tiers=', '.join(f"{g.TIER_NOMBRE[t].split(' ')[0]} {dec(v['sep'])}" for t, v in h['H2_tiers'].items() if t != 'd'),
        h2_d=dec(h['H2_tiers']['d']['sep']), h2_d_n=num(h['H2_tiers']['d']['n']),
        h3_n=num(h['H3']['n']), h3_r=dec(h['H3']['pearson']), h3_10=dec(h['H3']['sep10']),
        h3_20=dec(h['H3']['sep20']), h3_prom=dec(h['H3']['sep_prom']),
        h5_n=num(h['H5']['n']), h5_bo1=pct(h['H5']['bo1']), h5_global=pct(h['H5']['global']), h5_sep=dec(h['H5']['sep']),
        extraccion=EXTRACCION, publicada=PUBLICADA,
        acc_val_oof=pct(acc_val_oof), n_ex=num(len(ex)), piso_test_num=float(piso_test),
    )
    html = pagina(ctx, specs, rangos, equipos, cruces, tabla_pasos, tabla_comp, tabla_grillas, tabla_ajuste,
                  tabla_tipos, tabla_otros, tabla_coefs, tabla_folds_html, alta, detalle)
    SALIDA.mkdir(exist_ok=True)
    (SALIDA / 'index.html').write_text(html, encoding='utf-8')
    (SALIDA / 'numeros.json').write_text(json.dumps(numeros, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Listo: docs/index.html ({len(html) / 1e6:.1f} MB) · accuracy test {acc_test:.4f} · validación {res_cv["accuracy"]:.4f}')


def fuentes_css():
    """Las tipografías (OFL) van adentro del HTML, así la página funciona sin conexión."""
    import base64
    reglas = []
    for archivo, familia, peso, estilo in [('Rajdhani-SemiBold', 'Rajdhani', 600, 'normal'),
                                           ('Rajdhani-Bold', 'Rajdhani', 700, 'normal'),
                                           ('Barlow-Regular', 'Barlow', 400, 'normal'),
                                           ('Barlow-SemiBold', 'Barlow', 600, 'normal'),
                                           ('Barlow-Italic', 'Barlow', 400, 'italic')]:
        datos = base64.b64encode((RAIZ / 'fuentes' / f'{archivo}.woff2').read_bytes()).decode()
        reglas.append(f"@font-face{{font-family:{familia};font-weight:{peso};font-style:{estilo};font-display:block;"
                      f"src:url(data:font/woff2;base64,{datos}) format('woff2')}}")
    return '\n'.join(reglas)


def killfeed(detalle, n=6):
    """Los errores con más confianza, con el formato del kill feed de CS2."""
    filas = []
    for _, r in detalle[~detalle.acierto].sort_values('confianza', ascending=False).head(n).iterrows():
        gano_a = r.real == 1
        ganador, cae = (r.team_a_name, r.team_b_name) if gano_a else (r.team_b_name, r.team_a_name)
        lado_g, lado_c = ('a', 'b') if gano_a else ('b', 'a')
        mapas = f'{max(r.post_team_a_maps_won, r.post_team_b_maps_won)}-{min(r.post_team_a_maps_won, r.post_team_b_maps_won)}'
        filas.append(f'<li><span class="{lado_g}">{ganador}</span><svg style="color:var(--err)"><use href="#mira"/></svg>'
                     f'<span class="{lado_c} cae">{cae}</span><span class="dato"><span class="prob">el modelo le daba {pct(r.confianza, 0)} a {cae}</span>'
                     f' · {mapas} · {r.start_at_utc:%d/%m/%Y} · {r.tournament_name.strip()}</span></li>')
    return ''.join(filas)


def sliders_html(rangos):
    bloques = ['<fieldset><legend>El partido</legend><div class="slot" id="d-contexto"></div></fieldset>']
    for grupo in GRUPOS:
        slots = ''.join(f'<div class="slot" id="d-{p}"></div>' for p, r in rangos.items() if r['grupo'] == grupo)
        bloques.append(f'<fieldset><legend>{grupo}</legend>{slots}</fieldset>')
    return ''.join(bloques)


def pagina(c, specs, rangos, equipos, cruces, tabla_pasos, tabla_comp, tabla_grillas, tabla_ajuste, tabla_tipos,
           tabla_otros, tabla_coefs, tabla_folds, alta, detalle):
    caso = c['caso']
    # El título que afirma va en el <h2> de la sección; el gráfico se queda con su subtítulo.
    titulos = {}
    for k in ['g-datos', 'g-hallazgo', 'g-confianza']:
        titulos[k] = specs[k]['title']['text']
        specs[k]['title'] = {'text': specs[k]['title'].get('subtitle', ''), 'fontSize': 13, 'fontWeight': 'normal',
                             'color': g.GRIS, 'anchor': 'start', 'offset': 14}
    inactivos = ('ninguno de los dos había jugado en los 30 días anteriores'
                 if (caso.team_a_matches_last30d == 0 and caso.team_b_matches_last30d == 0)
                 else f'{caso.favorito} venía de jugar {int(caso.team_a_matches_last30d if caso.predicho == 1 else caso.team_b_matches_last30d)} series en los 30 días anteriores')
    valores = {k: v for k, v in c.items() if k not in ('caso', 'piso_test_num')}
    valores.update(
        t_datos=titulos['g-datos'], t_hallazgo=titulos['g-hallazgo'], t_confianza=titulos['g-confianza'],
        tabla_pasos=tabla_pasos, tabla_comp=tabla_comp, tabla_grillas=tabla_grillas, tabla_ajuste=tabla_ajuste,
        tabla_tipos=tabla_tipos, tabla_otros=tabla_otros, tabla_coefs=tabla_coefs, tabla_folds=tabla_folds,
        sliders=sliders_html(rangos),
        opciones=''.join(f'<option value="{e["id"]}">{e["n"]} · Elo {e["elo"]:.0f}</option>' for e in equipos),
        pipeline=''.join(f'<li><b>{a}.</b> {b}</li>' for a, b in PIPELINE),
        caso_fecha=f"{caso.start_at_utc:%d/%m/%Y}", caso_a=caso.team_a_name, caso_b=caso.team_b_name,
        caso_torneo=caso.tournament_name.strip(), caso_fav=caso.favorito, caso_ganador=caso.ganador,
        caso_conf=pct(caso.confianza), caso_elo=f'{abs(caso.elo_advantage_a):.0f}',
        caso_mapas=f'{max(caso.post_team_a_maps_won, caso.post_team_b_maps_won)}-{min(caso.post_team_a_maps_won, caso.post_team_b_maps_won)}',
        caso_inactivos_cap=inactivos[0].upper() + inactivos[1:],
        js_vega=vlc.javascript_bundle(vl_version='6.4'),
        fuentes_css=fuentes_css(), killfeed=killfeed(detalle),
        n_aciertos=num(int(detalle.acierto.sum())), n_errores=num(int((~detalle.acierto).sum())),
        acc_test_css=f'{100 * detalle.acierto.mean():.2f}%', piso_test_css=f"{100 * c['piso_test_num']:.2f}%",
        specs=json.dumps(specs, ensure_ascii=False, separators=(',', ':')),
        locale_tiempo=json.dumps(LOCALE_TIEMPO, ensure_ascii=False),
        equipos_json=json.dumps(equipos, ensure_ascii=False, separators=(',', ':')),
        cruces_json=json.dumps(cruces, separators=(',', ':')),
        rangos_json=json.dumps({p: {k: r[k] for k in ['min', 'max', 'paso', 'mediana', 'sin_dato', 'etiqueta']}
                                for p, r in rangos.items()}, ensure_ascii=False),
    )
    plantilla = (RAIZ / 'plantilla.html').read_text(encoding='utf-8')
    faltan = sorted(set(re.findall(r'\[\[(\w+)\]\]', plantilla)) - set(valores))
    assert not faltan, f'La plantilla pide valores que no se calcularon: {faltan}'
    # Una sola pasada: lo que se inserta (por ejemplo, el JavaScript de Vega) no se vuelve a revisar.
    html = re.sub(r'\[\[(\w+)\]\]', lambda m: str(valores[m.group(1)]), plantilla)
    return html


if __name__ == '__main__':
    main()
