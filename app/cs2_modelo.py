"""Funciones compartidas del proyecto: las usan el notebook de modelado y build_app.py.

Todo lo que decide un número (semilla, corte del test, folds, columnas, pipeline,
grillas) vive acá, en un solo lugar. Así lo que muestra la aplicación es, por
construcción, lo mismo que da el notebook.

Grupo 07 · Comisión 5K10 · Ciencia de Datos · UTN FRM · 2026
"""
from collections import defaultdict
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer, recall_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------------------------------
# Configuración fija (la misma de la Entrega 3)
# ----------------------------------------------------------------------------
SEMILLA = 42
N_SPLITS = 5
N_JOBS = 2
CORTE_TEST = pd.Timestamp('2026-03-01', tz='UTC')
UMBRAL = 0.50
ARCHIVO = Path('datos/matches_latest.csv')
# Funciona suelto o dentro del repositorio del grupo (app/ al lado de dags/).
AQUI = Path(__file__).resolve().parent
RAIZ_REPO = AQUI.parent if (AQUI.parent / 'dags').is_dir() else AQUI
CANDIDATOS_CSV = [RAIZ_REPO / 'datos' / 'matches_latest.csv',                         # la extracción fija
                  RAIZ_REPO / 'include' / 'output' / 'silver' / 'matches_latest.csv',  # la salida del DAG
                  AQUI / 'datos' / 'matches_latest.csv']


def ruta_csv():
    return next((p for p in CANDIDATOS_CSV if p.is_file()), CANDIDATOS_CSV[0])
DRIVE_ID = '1l3o2EA1VHUgKmjWHILXWLfbFlCsFraqG'
SHA256_ESPERADO = '7bb158fa8f7f27f8318270a772d45fca7ed930dc187907507ea84b0584136fa5'

CATEGORICAS = ['bo_type', 'tier']
HISTORIA = ['team_a_matches_prior', 'team_b_matches_prior']

SCORING = {
    'accuracy': 'accuracy',
    'f1_macro': make_scorer(f1_score, average='macro'),
    'recall_A': make_scorer(recall_score, pos_label=1),
    'recall_B': make_scorer(recall_score, pos_label=0),
}

# Las 27 columnas con las que cerró la Entrega 2.
COLUMNAS_E2 = ['bo_type', 'tier', 'elo_advantage_a', 'ranking_score_advantage_a',
               'round_winrate_last20_advantage_a', 'form_last5_advantage_a',
               'same_bo_winrate_last20_advantage_a', 'same_tier_winrate_last20_advantage_a',
               'lineup_stability_last5_advantage_a', 'h2h_team_a_winrate_last5', 'h2h_last5_count',
               'team_a_matches_prior', 'team_b_matches_prior',
               'team_a_form_trend_5v5', 'team_b_form_trend_5v5',
               'team_a_matches_last7d', 'team_b_matches_last7d',
               'team_a_matches_last30d', 'team_b_matches_last30d',
               'team_a_kda_last20_count', 'team_b_kda_last20_count',
               'team_a_same_bo_winrate_last20_count', 'team_b_same_bo_winrate_last20_count',
               'team_a_same_tier_winrate_last20_count', 'team_b_same_tier_winrate_last20_count',
               'team_a_lineup_stability_last5_count', 'team_b_lineup_stability_last5_count']
EXTRA = 'winrate_last20_advantage_a'
CONTADORES = [c for c in COLUMNAS_E2 if c.endswith('_count')]
# Variante elegida en la sección 8 de la Entrega 3: «sin contadores» (18 columnas).
COLUMNAS_FINAL = [c for c in COLUMNAS_E2 if c not in CONTADORES]

# ----------------------------------------------------------------------------
# Las cuatro familias y sus grillas (corrección de la Entrega 3)
# ----------------------------------------------------------------------------
FAMILIAS = {
    'regresión logística': LogisticRegression(max_iter=5000, random_state=SEMILLA),
    'árbol de decisión': DecisionTreeClassifier(random_state=SEMILLA),
    'random forest': RandomForestClassifier(n_estimators=200, max_features='sqrt', n_jobs=1,
                                            random_state=SEMILLA),
    'gradient boosting': HistGradientBoostingClassifier(early_stopping=False, random_state=SEMILLA),
}
GRILLAS = {
    'regresión logística': {'clasificador__C': [0.01, 0.1, 1.0, 10.0],
                            'clasificador__class_weight': [None, 'balanced']},
    'árbol de decisión': {'clasificador__max_depth': [3, 4, 6, 8, 10, 14],
                          'clasificador__min_samples_leaf': [1, 50, 200],
                          'clasificador__class_weight': [None, 'balanced']},
    'random forest': {'clasificador__max_depth': [6, 10, 14],
                      'clasificador__min_samples_leaf': [10, 50],
                      'clasificador__class_weight': [None, 'balanced']},
    'gradient boosting': {'clasificador__learning_rate': [0.03, 0.1],
                          'clasificador__max_leaf_nodes': [7, 15, 31],
                          'clasificador__max_iter': [100, 300],
                          'clasificador__class_weight': [None, 'balanced']},
}
ORDEN_SIMPLICIDAD = ['regresión logística', 'árbol de decisión', 'random forest', 'gradient boosting']

# Hiperparámetros del modelo final, congelados en la sección 10 del notebook.
PARAMS_FINAL = {'C': 0.01, 'class_weight': 'balanced'}


# ----------------------------------------------------------------------------
# Pipeline y medición
# ----------------------------------------------------------------------------
def armar(clasificador, columnas, usar_log=True):
    """Pipeline completo: elegir columnas -> preparar -> clasificar (armar de la Práctica 2)."""
    historia = [c for c in columnas if c in HISTORIA]
    categoricas = [c for c in columnas if c in CATEGORICAS]
    numericas = [c for c in columnas if c not in HISTORIA + CATEGORICAS]
    log = [('log1p', FunctionTransformer(np.log1p, feature_names_out='one-to-one'))] if usar_log else []
    preparar = ColumnTransformer([
        ('numericas', Pipeline([
            ('imputar', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
            ('escalar', StandardScaler())]), numericas),
        ('historia', Pipeline(log + [('escalar', StandardScaler())]), historia),
        ('categoricas', OneHotEncoder(handle_unknown='ignore'), categoricas),
    ])
    return Pipeline([('preparar', preparar), ('clasificador', clone(clasificador))])


def medir(pipeline, X, y, cv):
    """cross_validate con media y desvío, como en la Práctica 2."""
    r = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=N_JOBS,
                       return_train_score=True, error_score='raise')
    resumen = {'accuracy': r['test_accuracy'].mean(), 'desvío': r['test_accuracy'].std(),
               'accuracy train': r['train_accuracy'].mean(), 'f1 macro': r['test_f1_macro'].mean(),
               'recall A': r['test_recall_A'].mean(), 'recall B': r['test_recall_B'].mean()}
    return resumen, r


def tabla_de(busqueda):
    """Tabla completa de la grilla: todas las combinaciones, no solo la mejor."""
    r = pd.DataFrame(busqueda.cv_results_)
    cols = [c for c in r if c.startswith('param_')]
    cols += ['mean_test_accuracy', 'std_test_accuracy', 'mean_train_accuracy', 'mean_test_f1_macro',
             'mean_test_recall_A', 'mean_test_recall_B', 'rank_test_accuracy']
    t = r[cols].sort_values('rank_test_accuracy')
    t.columns = [c.replace('param_clasificador__', '') for c in t.columns]
    return t


def metricas(y, pred):
    return {'accuracy': accuracy_score(y, pred), 'f1 macro': f1_score(y, pred, average='macro')}


def elegir_simple(tabla, orden, col='accuracy', desvio='desvío'):
    """Regla de la sección 4: el más simple que esté a menos de un desvío del mejor."""
    mejor = tabla.loc[orden, col].idxmax()
    limite = tabla.loc[mejor, col] - tabla.loc[mejor, desvio]
    elegido = next(n for n in orden if tabla.loc[n, col] >= limite)
    return mejor, limite, elegido


# ----------------------------------------------------------------------------
# Datos, partición temporal y folds
# ----------------------------------------------------------------------------
def huella_archivo(ruta):
    h = hashlib.sha256()
    with Path(ruta).open('rb') as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b''):
            h.update(bloque)
    return h.hexdigest()


def cargar(ruta=None, verificar_huella=True):
    """Lee el Silver, controla forma, clave y fechas, y lo ordena en el tiempo."""
    ruta = Path(ruta) if ruta else ruta_csv()
    assert ruta.is_file(), f'Falta {ruta}: es el CSV que produce el pipeline (ver README).'
    if verificar_huella:
        assert huella_archivo(ruta) == SHA256_ESPERADO, (
            f'{ruta} no es la extracción de las Entregas 2 y 3. Para usar datos nuevos: build_app.py --sin-huella.')
    df = pd.read_csv(ruta, low_memory=False)
    assert df.match_id.is_unique
    assert df.target_team_a_won.isin([0, 1]).all()
    assert df.team_a_id.lt(df.team_b_id).all()
    for c in ['start_at_utc', 'end_at_utc']:
        df[c] = pd.to_datetime(df[c], format='mixed', utc=True)
    assert df.start_at_utc.notna().all()
    return df.sort_values(['start_at_utc', 'match_id'], kind='stable').reset_index(drop=True)


def separar(df):
    """Test desde el corte; entrenamiento lo que empezó y terminó antes; el resto se purga."""
    antes = df.start_at_utc.lt(CORTE_TEST)
    termino_antes = df.end_at_utc.notna() & df.end_at_utc.lt(CORTE_TEST)
    train = df.loc[antes & termino_antes].reset_index(drop=True)
    test = df.loc[~antes].reset_index(drop=True)
    purgados = df.loc[antes & ~termino_antes, ['match_id', 'start_at_utc', 'end_at_utc']]
    assert len(train) + len(test) + len(purgados) == len(df)
    assert train.end_at_utc.max() < CORTE_TEST <= test.start_at_utc.min()
    return train, test, purgados


def construir_cv_temporal(datos, n_splits=N_SPLITS):
    """TimeSeriesSplit sobre calendario diario, mapeado a partidos, con purga."""
    inicio = datos['start_at_utc']
    fin = datos['end_at_utc']
    calendario = pd.date_range(inicio.min().floor('D'), inicio.max().floor('D'), freq='D')
    cv, filas = [], []
    for k, (dias_e, dias_v) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(calendario), 1):
        limite = calendario[dias_v[0]]
        hasta = calendario[dias_v[-1]] + pd.Timedelta(days=1)
        anteriores = inicio.lt(limite)
        disponibles = fin.notna() & fin.lt(limite)
        e = np.flatnonzero((anteriores & disponibles).to_numpy())
        v = np.flatnonzero((inicio.ge(limite) & inicio.lt(hasta)).to_numpy())
        assert len(e) and len(v)
        assert inicio.iloc[e].max() < inicio.iloc[v].min()
        assert fin.iloc[e].max() < limite
        cv.append((e, v))
        filas.append({'fold': k, 'train desde': inicio.iloc[e].min(),
                      'train hasta': inicio.iloc[e].max(), 'validación desde': limite,
                      'validación hasta (excl.)': hasta, 'n train': len(e), 'n validación': len(v),
                      'purgados': int((anteriores & ~disponibles).sum()),
                      '% A train': 100 * datos.iloc[e].target_team_a_won.mean(),
                      '% A validación': 100 * datos.iloc[v].target_team_a_won.mean()})
    return cv, pd.DataFrame(filas)


def predicciones_fuera_de_fold(pipeline, X, y, cv):
    """Probabilidad de «gana A» para cada serie de validación, con un modelo que no la vio."""
    p = pd.Series(np.nan, index=X.index)
    fold = pd.Series(0, index=X.index)
    for k, (e, v) in enumerate(cv, 1):
        m = clone(pipeline).fit(X.iloc[e], y.iloc[e])
        p.iloc[v] = m.predict_proba(X.iloc[v])[:, 1]
        fold.iloc[v] = k
    return p, fold


def entrenar_final(train):
    """El modelo congelado: logística, 18 columnas, C = 0,01, balanced, con todo el entrenamiento."""
    modelo = armar(LogisticRegression(max_iter=5000, random_state=SEMILLA, **PARAMS_FINAL),
                   COLUMNAS_FINAL)
    return modelo.fit(train[COLUMNAS_FINAL], train.target_team_a_won)


# ----------------------------------------------------------------------------
# Auditorías de la sección 3 (sin cambios desde la Entrega 3)
# ----------------------------------------------------------------------------
SUFIJOS = ('elo', 'winrate_last10', 'winrate_last20', 'form_last5', 'kda_last20', 'kd_last20',
           'rating_last20', 'adr_last20', 'map_winrate_last20', 'round_winrate_last20',
           'same_bo_winrate_last20', 'same_tier_winrate_last20', 'rest_days', 'lineup_stability_last5')
PARES = {s + '_advantage_a': ('team_a_' + s, 'team_b_' + s) for s in SUFIJOS}
PARES.update({'rank_advantage_a': ('team_b_rank', 'team_a_rank'),
              'ranking_score_advantage_a': ('team_a_ranking_score', 'team_b_ranking_score'),
              'round_diff_last20_advantage_a': ('team_a_round_diff_per_match_last20',
                                                'team_b_round_diff_per_match_last20')})


def auditar_derivadas(datos):
    filas = []
    for nombre, (a, b) in PARES.items():
        esperado = datos[a] - datos[b]
        correcto = np.isclose(datos[nombre], esperado, rtol=1e-8, atol=1e-8, equal_nan=True)
        filas.append({'derivada': nombre, 'fórmula': f'{a} - {b}',
                      'filas verificadas': len(datos), 'diferencias': int((~correcto).sum())})
    return pd.DataFrame(filas)


def auditar_historia_resultados(datos):
    """Reconstrucción independiente de Elo, winrates y H2H desde los resultados."""
    filas = datos.sort_values(['start_at_utc', 'match_id']).to_dict('records')
    completos = sorted((r for r in filas if pd.notna(r['end_at_utc'])),
                       key=lambda r: (r['end_at_utc'], r['match_id']))
    historial, elo, h2h = defaultdict(list), defaultdict(lambda: 1500.0), defaultdict(list)
    errores, maximos = defaultdict(int), defaultdict(float)
    j = 0

    def comparar(c, real, esperado):
        if not np.isclose(real, esperado, rtol=1e-7, atol=1e-7, equal_nan=True):
            errores[c] += 1
        else:
            errores.setdefault(c, 0)
        if pd.notna(real) and pd.notna(esperado):
            maximos[c] = max(maximos[c], abs(float(real) - float(esperado)))

    for r in filas:
        while j < len(completos) and completos[j]['end_at_utc'] < r['start_at_utc']:
            p = completos[j]
            a, b, y = p['team_a_id'], p['team_b_id'], p['target_team_a_won']
            delta = 32 * (y - 1 / (1 + 10 ** ((elo[b] - elo[a]) / 400)))
            elo[a] += delta
            elo[b] -= delta
            historial[a].append((p['end_at_utc'], y))
            historial[b].append((p['end_at_utc'], 1 - y))
            h2h[(min(a, b), max(a, b))].append(a if y else b)
            j += 1
        for lado in ('a', 'b'):
            equipo = r[f'team_{lado}_id']
            h = historial[equipo]
            pref = f'team_{lado}_'
            comparar(pref + 'matches_prior', r[pref + 'matches_prior'], len(h))
            comparar(pref + 'elo', r[pref + 'elo'], elo[equipo])
            for n, suf in [(5, 'form_last5'), (10, 'winrate_last10'), (20, 'winrate_last20')]:
                ventana = h[-n:]
                esperado = np.mean([v[1] for v in ventana]) if len(ventana) >= 5 else np.nan
                comparar(pref + suf, r[pref + suf], esperado)
                comparar(pref + suf + '_count', r[pref + suf + '_count'], len(ventana))
            tendencia = (np.mean([v[1] for v in h[-5:]]) - np.mean([v[1] for v in h[-10:-5]])) if len(h) >= 10 else np.nan
            comparar(pref + 'form_trend_5v5', r[pref + 'form_trend_5v5'], tendencia)
            descanso = (r['start_at_utc'] - h[-1][0]).total_seconds() / 86400 if h else np.nan
            comparar(pref + 'rest_days', r[pref + 'rest_days'], descanso)
        a, b = r['team_a_id'], r['team_b_id']
        previos = h2h[(min(a, b), max(a, b))][-5:]
        comparar('h2h_last5_count', r['h2h_last5_count'], len(previos))
        esperado = np.mean([v == a for v in previos]) if previos else np.nan
        comparar('h2h_team_a_winrate_last5', r['h2h_team_a_winrate_last5'], esperado)
    return pd.DataFrame([{'columna': c, 'diferencias': n, 'máximo error absoluto': maximos[c]}
                         for c, n in errores.items()])


def elo_actual(datos):
    """Elo de cada equipo después de su último partido terminado (para «Probá el modelo»)."""
    completos = datos[datos.end_at_utc.notna()].sort_values(['end_at_utc', 'match_id'])
    elo = defaultdict(lambda: 1500.0)
    for a, b, y in completos[['team_a_id', 'team_b_id', 'target_team_a_won']].itertuples(index=False):
        delta = 32 * (y - 1 / (1 + 10 ** ((elo[b] - elo[a]) / 400)))
        elo[a] += delta
        elo[b] -= delta
    return dict(elo)


# ----------------------------------------------------------------------------
# Lectura de los errores (sección 11 del notebook, y la app)
# ----------------------------------------------------------------------------
TRAMOS_CONFIANZA = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0001]
ETIQUETAS_CONFIANZA = ['50–55 %', '55–60 %', '60–65 %', '65–70 %', '70–75 %', '75–80 %', '80 % o más']
TRAMOS_ELO = [0, 50, 100, 150, 200, 300, 400, np.inf]
ETIQUETAS_ELO = ['0–50', '50–100', '100–150', '150–200', '200–300', '300–400', '400 o más']


def detalle_predicciones(datos, proba):
    """Una fila por serie con la predicción y todo lo que sirve para leer el error."""
    d = datos[['match_id', 'start_at_utc', 'team_a_id', 'team_b_id', 'team_a_name', 'team_b_name',
               'tournament_name', 'bo_type', 'tier', 'elo_advantage_a', 'ranking_score_advantage_a',
               'team_a_matches_prior', 'team_b_matches_prior', 'team_a_matches_last30d',
               'team_b_matches_last30d', 'lineup_stability_last5_advantage_a',
               'team_a_lineup_stability_last5', 'team_b_lineup_stability_last5',
               'h2h_last5_count', 'post_team_a_maps_won', 'post_team_b_maps_won',
               'target_team_a_won']].copy()
    d['prob_a'] = np.asarray(proba)
    d['predicho'] = (d.prob_a >= UMBRAL).astype(int)
    d['real'] = d.target_team_a_won
    d['acierto'] = d.predicho.eq(d.real)
    d['confianza'] = np.maximum(d.prob_a, 1 - d.prob_a)
    d['tramo_confianza'] = pd.cut(d.confianza, TRAMOS_CONFIANZA, labels=ETIQUETAS_CONFIANZA, right=False)
    d['dif_elo_abs'] = d.elo_advantage_a.abs()
    d['tramo_elo'] = pd.cut(d.dif_elo_abs, TRAMOS_ELO, labels=ETIQUETAS_ELO, right=False)
    # Sorpresa: ganó el equipo que llegaba con menos Elo.
    d['sorpresa'] = np.where(d.elo_advantage_a >= 0, d.real == 0, d.real == 1)
    fav_a = d.predicho == 1
    d['favorito'] = np.where(fav_a, d.team_a_name, d.team_b_name)
    d['no_favorito'] = np.where(fav_a, d.team_b_name, d.team_a_name)
    d['ganador'] = np.where(d.real == 1, d.team_a_name, d.team_b_name)
    d['fav_inactivo_30d'] = np.where(fav_a, d.team_a_matches_last30d, d.team_b_matches_last30d) == 0
    estab_fav = np.where(fav_a, d.team_a_lineup_stability_last5, d.team_b_lineup_stability_last5)
    d['fav_cambio_alineacion'] = pd.Series(estab_fav, index=d.index).lt(1)  # NaN → False
    d['historia'] = np.where(d[['team_a_matches_prior', 'team_b_matches_prior']].min(axis=1) < 5,
                             'historia corta', 'los dos con 5 o más')
    d['sin_ranking'] = d.ranking_score_advantage_a.isna()
    d['mes'] = d.start_at_utc.dt.tz_convert(None).dt.to_period('M').dt.to_timestamp()
    return d


def tabla_por(d, columna, orden=None):
    g = d.groupby(columna, observed=True).agg(series=('acierto', 'size'), accuracy=('acierto', 'mean'),
                                              prob_media=('confianza', 'mean'))
    if orden is not None:
        g = g.reindex(orden)
    return g


# ----------------------------------------------------------------------------
# Hipótesis de la Entrega 2 (mismas funciones de la cátedra y mismo tramo)
# ----------------------------------------------------------------------------
CORTES_SEMAFORO = {'separacion': (0.2, 0.8), 'correlacion': (0.2, 0.6)}


def separacion(x, y):
    """Distancia entre dos muestras, en desviaciones estándar (función de la cátedra)."""
    x, y = x.dropna(), y.dropna()
    return abs(x.mean() - y.mean()) / np.sqrt((x.std() ** 2 + y.std() ** 2) / 2)


def semaforo(valor, medida):
    bajo, alto = CORTES_SEMAFORO[medida]
    v = abs(valor)
    return 'rojo' if v < bajo else 'verde' if v > alto else 'amarillo'


def hipotesis_e2(df):
    """H1, H2, H3 y H5 de la Entrega 2, medidas en el tramo de entrenamiento (inicio < corte)."""
    tramo = df[df.start_at_utc < CORTE_TEST]
    y = tramo.target_team_a_won

    def comparar(col, datos=tramo):
        m = datos[[col, 'target_team_a_won']].dropna()
        a = m.loc[m.target_team_a_won.eq(1), col]
        b = m.loc[m.target_team_a_won.eq(0), col]
        s = separacion(a, b)
        return {'n': len(m), 'sep': s, 'zona': semaforo(s, 'separacion'), 'media_a': a.mean(), 'media_b': b.mean()}

    h = {'tramo': len(tramo), 'H1': comparar('rest_days_advantage_a'), 'H2': comparar('elo_advantage_a')}
    h['H2_tiers'] = {t: comparar('elo_advantage_a', g) for t, g in tramo.groupby('tier')}
    v = ['winrate_last10_advantage_a', 'winrate_last20_advantage_a']
    par = tramo[v + ['target_team_a_won']].dropna()
    gana = par.target_team_a_won.eq(1)
    prom = par[v].mean(axis=1)
    h['H3'] = {'n': len(par), 'pearson': par[v[0]].corr(par[v[1]]),
               'sep10': separacion(par.loc[gana, v[0]], par.loc[~gana, v[0]]),
               'sep20': separacion(par.loc[gana, v[1]], par.loc[~gana, v[1]]),
               'sep_prom': separacion(prom[gana], prom[~gana])}
    m = tramo.loc[tramo.elo_advantage_a.ne(0), ['elo_advantage_a', 'target_team_a_won', 'bo_type']].copy()
    m['sorpresa'] = (m.elo_advantage_a.gt(0) != m.target_team_a_won.eq(1)).astype(int)
    bo1 = m.bo_type.eq(1)
    s5 = separacion(m.loc[bo1, 'sorpresa'], m.loc[~bo1, 'sorpresa'])
    h['H5'] = {'n': len(m), 'global': m.sorpresa.mean(), 'bo1': m.loc[bo1, 'sorpresa'].mean(), 'sep': s5,
               'zona': semaforo(s5, 'separacion')}
    return h
