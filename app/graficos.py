"""Los gráficos de la aplicación, hechos con Altair.

Los usan el notebook (para verlos ahí mismo) y build_app.py (para publicarlos).
Un solo lugar define los colores, las categorías y las escalas, así significan
lo mismo en toda la aplicación: azul es siempre el equipo A, naranja el B,
verde un acierto, rojo un error, y la escala de tiers es siempre la misma.
"""
import altair as alt
import numpy as np
import pandas as pd

import cs2_modelo as cm

# ---------------------------------------------------------------------------
# Un solo diccionario visual para toda la aplicación
# ---------------------------------------------------------------------------
# Paleta de la aplicación, inspirada en la interfaz de CS2: fondo oscuro, azul de los
# antiterroristas (CT) para el equipo A y dorado de los terroristas (T) para el B.
FONDO, PANEL, LINEA = '#0f1318', '#161c24', '#2b3440'
TINTA, GRIS, SUAVE = '#e8ecf1', '#9aa5b4', '#566273'
COLOR_A, COLOR_B = '#5b9cf0', '#f0b23e'          # equipo A (CT) · equipo B (T)
COLOR_OK, COLOR_ERR = '#3ddc97', '#ff5a5f'       # acierto · error
ACENTO = '#3fd2d8'                               # lo que dice el modelo
APAGADO = '#232a34'                              # lo que queda afuera de un filtro
TIERS = ['s', 'a', 'b', 'c', 'd']
TIER_NOMBRE = {'s': 'S (élite)', 'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D'}
TIER_ORDEN = [TIER_NOMBRE[t] for t in TIERS]
TIER_COLORES = ['#eef2f7', '#b3c2d6', '#7d93b2', '#4d607d', '#313d50']
ESC_TIER = alt.Scale(domain=TIER_ORDEN, range=TIER_COLORES)
GANADOR = ['Ganó A', 'Ganó B']
ESC_GANADOR = alt.Scale(domain=GANADOR, range=[COLOR_A, COLOR_B])
RESULTADO = ['Acierto', 'Error']
ESC_RESULTADO = alt.Scale(domain=RESULTADO, range=[COLOR_OK, COLOR_ERR])
FUENTE = 'Barlow, "Segoe UI", Helvetica, Arial, sans-serif'
FUENTE_HUD = 'Rajdhani, "Arial Narrow", "Segoe UI", sans-serif'


def tema():
    return {'config': {
        'background': FONDO, 'font': FUENTE, 'padding': 10,
        'view': {'continuousWidth': 300, 'continuousHeight': 300, 'stroke': None},
        'axis': {'domainColor': LINEA, 'gridColor': '#1f262f', 'labelColor': GRIS, 'labelFont': FUENTE,
                 'labelFontSize': 13, 'tickColor': LINEA, 'titleColor': GRIS, 'titleFont': FUENTE,
                 'titleFontSize': 13, 'titleFontWeight': 'normal'},
        'legend': {'labelColor': TINTA, 'labelFontSize': 13, 'titleColor': GRIS, 'titleFontSize': 13,
                   'titleFontWeight': 'normal'},
        'header': {'labelColor': TINTA, 'labelFontSize': 13, 'titleColor': GRIS, 'titleFontSize': 13},
        'text': {'color': TINTA, 'font': FUENTE},
        'title': {'anchor': 'start', 'fontSize': 19, 'font': FUENTE_HUD, 'fontWeight': 700, 'color': TINTA,
                  'subtitleColor': GRIS, 'subtitleFontSize': 13, 'subtitleFont': FUENTE, 'offset': 12},
    }}


alt.theme.register('cs2', enable=True)(tema)
alt.data_transformers.disable_max_rows()


def pct(x, d=1):
    """0.6514 -> '65,1 %' (formato de la aplicación: coma decimal)."""
    return f'{100 * x:.{d}f} %'.replace('.', ',')


def num(x, d=0):
    s = f'{x:,.{d}f}'
    return s.replace(',', 'X').replace('.', ',').replace('X', '.')


# ---------------------------------------------------------------------------
# Tallo 1 · Los datos
# ---------------------------------------------------------------------------
def tabla_meses(df):
    m = df.assign(mes=df.start_at_utc.dt.tz_convert(None).dt.to_period('M').dt.to_timestamp(),
                  tier_n=df.tier.map(TIER_NOMBRE))
    return m.groupby(['mes', 'tier_n'], as_index=False).size().rename(columns={'size': 'series'})


def g_datos(df):
    """Series por mes, apiladas por tier, con el entrenamiento y la prueba marcados como dos zonas."""
    t = tabla_meses(df)
    corte = cm.CORTE_TEST.tz_convert(None)
    inicio = t.mes.min() - pd.Timedelta(days=16)
    fin = t.mes.max() + pd.Timedelta(days=46)
    tr, te, _ = cm.separar(df)
    n_train, n_test = len(tr), len(te)
    zonas = pd.DataFrame({'x': [inicio, corte], 'x2': [corte, fin],
                          'zona': ['Entrenamiento', 'Prueba'],
                          't1': ['ENTRENAMIENTO', 'PRUEBA'],
                          't2': [f'el modelo aprende de estas {num(n_train)} series',
                                 f'{num(n_test)} series que el modelo nunca vio']})
    fondo = alt.Chart(zonas).mark_rect(opacity=0.09).encode(
        x=alt.X('x:T'), x2='x2:T',
        color=alt.Color('zona:N', scale=alt.Scale(domain=['Entrenamiento', 'Prueba'], range=[ACENTO, COLOR_ERR]), legend=None))
    rot1 = alt.Chart(zonas).mark_text(align='left', dx=8, y=14, fontSize=13, fontWeight='bold', font=FUENTE_HUD).encode(
        x='x:T', text='t1:N',
        color=alt.Color('zona:N', scale=alt.Scale(domain=['Entrenamiento', 'Prueba'], range=[ACENTO, COLOR_ERR]), legend=None))
    rot2 = alt.Chart(zonas).mark_text(align='left', dx=8, y=30, fontSize=11.5, color=GRIS).encode(x='x:T', text='t2:N')
    base = alt.Chart(t)
    barras = base.mark_bar().encode(
        x=alt.X('yearmonth(mes):T', title='Mes y año en que se jugó la serie',
                scale=alt.Scale(domain=[inicio.isoformat(), fin.isoformat()]),
                axis=alt.Axis(format='%b %Y', labelAngle=0, tickCount=7)),
        y=alt.Y('sum(series):Q', title='Cantidad de series jugadas en el mes', scale=alt.Scale(domainMax=float(t.groupby('mes').series.sum().max() * 1.22))),
        color=alt.Color('tier_n:N', title='Tier (nivel) del torneo', scale=ESC_TIER, sort=TIER_ORDEN,
                        legend=alt.Legend(orient='top', direction='horizontal')),
        order=alt.Order('orden:Q'),
        tooltip=[alt.Tooltip('yearmonth(mes):T', title='Mes', format='%B %Y'),
                 alt.Tooltip('tier_n:N', title='Tier'), alt.Tooltip('series:Q', title='Series', format=',')],
    ).transform_calculate(orden=f"indexof({TIER_ORDEN}, datum.tier_n)")
    regla = alt.Chart(pd.DataFrame({'mes': [corte]})).mark_rule(color=COLOR_ERR, strokeDash=[5, 3], strokeWidth=2).encode(x='mes:T')
    n = len(df)
    fuera = df.tier.isin(['b', 'c']).mean()
    return alt.layer(fondo, barras, regla, rot1, rot2).resolve_scale(color='independent').properties(
        width=680, height=300, title=alt.Title(
            f'{num(n)} series en tres años, y {round(10 * fuera)} de cada 10 se juegan en tiers B y C',
            subtitle=[f'Cada barra es un mes y su altura, cuántas series profesionales se jugaron; los colores dicen en qué tier (nivel de torneo).',
                      f'La élite (S) y el tier A suman el {pct(df.tier.isin(["s", "a"]).mean())}. '
                      f'La línea roja ({cm.CORTE_TEST:%d/%m/%Y}) separa las series de entrenamiento de las de prueba.']))


# ---------------------------------------------------------------------------
# Tallo 2 · Lo que encontramos
# ---------------------------------------------------------------------------
def tabla_elo(df):
    d = df[df.elo_advantage_a != 0]
    gana_fav = np.where(d.elo_advantage_a > 0, d.target_team_a_won, 1 - d.target_team_a_won)
    tramo = pd.cut(d.elo_advantage_a.abs(), cm.TRAMOS_ELO, labels=cm.ETIQUETAS_ELO, right=False)
    t = (pd.DataFrame({'tramo': tramo.values, 'gana': gana_fav})
         .groupby('tramo', observed=True).gana.agg(['mean', 'size']).reset_index())
    t.columns = ['tramo', 'gana_favorito', 'series']
    t['tramo'] = t.tramo.astype(str)
    t['texto'] = t.gana_favorito.map(lambda x: pct(x, 0))
    return t, gana_fav.mean(), len(d)


def g_hallazgo(df):
    t, total, n = tabla_elo(df)
    x = alt.X('tramo:O', sort=cm.ETIQUETAS_ELO, title='Ventaja de Elo del favorito (puntos)',
              axis=alt.Axis(labelAngle=0))
    y = alt.Y('gana_favorito:Q', title='Series que gana el de más Elo',
              scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='.0%'))
    puntos = alt.Chart(t).mark_line(color=ACENTO, strokeWidth=3, point=alt.OverlayMarkDef(
        filled=True, size=90, color=ACENTO)).encode(
        x=x, y=y, tooltip=[alt.Tooltip('tramo:O', title='Ventaja de Elo'),
                           alt.Tooltip('gana_favorito:Q', title='Gana el favorito', format='.1%'),
                           alt.Tooltip('series:Q', title='Series', format=',')])
    textos = alt.Chart(t).mark_text(dy=-16, fontSize=13, color=TINTA, fontWeight='bold').encode(
        x=x, y='gana_favorito:Q', text='texto:N')
    cantidades = alt.Chart(t).mark_text(dy=18, fontSize=11, color=GRIS).encode(
        x=x, y='gana_favorito:Q', text=alt.Text('series:Q', format=','))
    moneda = alt.Chart(pd.DataFrame({'y': [0.5], 't': ['50 %: tirar una moneda']}))
    regla = moneda.mark_rule(color=SUAVE, strokeDash=[4, 3]).encode(y='y:Q')
    rot = moneda.mark_text(align='right', x=676, dy=-7, color=GRIS, fontSize=12).encode(y='y:Q', text='t:N')
    ocho = t.loc[t.tramo == '200–300', 'gana_favorito'].iloc[0]
    return (regla + rot + puntos + textos + cantidades).properties(width=680, height=300, title=alt.Title(
        f'El que llega con más Elo gana el {pct(total, 0)} de las series, y con 200 puntos de ventaja, '
        f'{round(10 * ocho)} de cada 10',
        subtitle=[f'Las {num(n)} series del dataset con alguna diferencia de Elo, agrupadas por el tamaño de esa diferencia.',
                  'El número chico, debajo de cada punto, es cuántas series hay en el tramo.']))


# ---------------------------------------------------------------------------
# Tallo 3 · El modelo
# ---------------------------------------------------------------------------
NOMBRE_MODELO = {'piso': 'Piso (siempre «gana A»)', 'regresión logística': 'Regresión logística',
                 'árbol de decisión': 'Árbol de decisión', 'random forest': 'Random forest',
                 'gradient boosting': 'Gradient boosting'}


def g_modelos(tabla, elegido='regresión logística'):
    """tabla: índice = familia; columnas accuracy y desvío (de la mejor combinación de su grilla)."""
    t = tabla.reset_index().rename(columns={'index': 'familia'})
    t['modelo'] = t.familia.map(NOMBRE_MODELO)
    t['bajo'], t['alto'] = t.accuracy - t['desvío'], t.accuracy + t['desvío']
    t['texto'] = t.apply(lambda r: f"{pct(r.accuracy)} ± {pct(r['desvío']).replace(' %', '')}", axis=1)
    t['tipo'] = np.where(t.familia == elegido, 'Elegido', np.where(t.familia == 'piso', 'Piso', 'Otros'))
    orden = [NOMBRE_MODELO[k] for k in ['piso', 'regresión logística', 'árbol de decisión',
                                        'random forest', 'gradient boosting'] if k in tabla.index]
    esc = alt.Scale(domain=['Elegido', 'Otros', 'Piso'], range=[ACENTO, '#8b96a5', SUAVE])
    ex = alt.Scale(domain=[0.5, 0.66])
    y = alt.Y('modelo:N', sort=orden, title=None, axis=alt.Axis(labelLimit=240))
    x = alt.X('accuracy:Q', title='Accuracy en validación cruzada (media ± desvío, 5 folds temporales)',
              scale=ex, axis=alt.Axis(format='.0%'))
    base = alt.Chart(t)
    rango = base.mark_rule(strokeWidth=3).encode(x=alt.X('bajo:Q', scale=ex),
                                                 x2='alto:Q', y=y, color=alt.Color('tipo:N', scale=esc, legend=None))
    punto = base.mark_point(filled=True, size=120, opacity=1).encode(
        x=x, y=y, color=alt.Color('tipo:N', scale=esc, legend=None),
        tooltip=[alt.Tooltip('modelo:N', title='Modelo'),
                 alt.Tooltip('accuracy:Q', title='Accuracy (media)', format='.2%'),
                 alt.Tooltip('desvío:Q', title='Desvío entre folds', format='.2%'),
                 alt.Tooltip('accuracy train:Q', title='Accuracy en entrenamiento', format='.2%'),
                 alt.Tooltip('combinación:N', title='Mejor combinación de su grilla')])
    texto = base.mark_text(align='left', dx=10, fontSize=13, color=TINTA).encode(x='alto:Q', y=y, text='texto:N')
    mejor = tabla.drop(index='piso')['accuracy'].idxmax()
    limite = tabla.loc[mejor, 'accuracy'] - tabla.loc[mejor, 'desvío']
    banda = alt.Chart(pd.DataFrame({'x': [limite], 'x2': [tabla.loc[mejor, 'accuracy'] + tabla.loc[mejor, 'desvío']],
                                    't': ['a menos de un desvío del mejor']}))
    zona = banda.mark_rect(color=ACENTO, opacity=0.13).encode(x=alt.X('x:Q', scale=ex), x2='x2:Q')
    rot = banda.mark_text(align='right', baseline='top', dx=-4, y=2, color=ACENTO, fontSize=11).encode(
        x=alt.X('x:Q', scale=ex), text='t:N')
    dif = tabla.loc[mejor, 'accuracy'] - tabla.loc[elegido, 'accuracy']
    return (zona + rot + rango + punto + texto).properties(width=560, height=alt.Step(42), title=alt.Title(
        'Las cuatro familias, cada una con su mejor ajuste: la logística queda a menos de un desvío de la mejor',
        subtitle=[f'El {NOMBRE_MODELO[mejor].lower()} da {pct(dif, 2).replace(" %", "")} puntos más que la logística, '
                  f'menos que su desvío ({pct(tabla.loc[mejor, "desvío"], 2).replace(" %", "")}).',
                  texto_variantes(tabla)]))


def texto_variantes(tabla):
    if 'variante' not in tabla:
        return 'Mismas columnas y mismos 5 folds temporales, cada familia con su grilla.'
    elegidas = tabla.drop(index='piso')
    if elegidas.variante.nunique() == 1:
        return (f'Cada familia con su mejor variante de columnas (las cuatro eligieron «{elegidas.variante.iloc[0]}», '
                f'{int(elegidas.columnas.iloc[0])} columnas) y su grilla, en los mismos 5 folds temporales.')
    return 'Cada familia con su mejor variante de columnas y su grilla, en los mismos 5 folds temporales.'


def tabla_confianza(detalle):
    t = cm.tabla_por(detalle, 'tramo_confianza', cm.ETIQUETAS_CONFIANZA).reset_index()
    t.columns = ['tramo', 'series', 'accuracy', 'confianza']
    t['tramo'] = t.tramo.astype(str)
    t['cuota'] = 1 / t.accuracy
    t['texto'] = t.accuracy.map(lambda x: pct(x, 0))
    t['texto_cuota'] = t.cuota.map(lambda c: f'cuota {c:.2f}'.replace('.', ','))
    t['texto_n'] = t.series.map(lambda n: f'n = {num(n)}')
    return t


def g_confianza(detalle):
    """Accuracy en el test por tramo de confianza, con lo que dijo el modelo y la cuota mínima."""
    t = tabla_confianza(detalle)
    x = alt.X('tramo:O', sort=cm.ETIQUETAS_CONFIANZA, title='Probabilidad que el modelo le dio a su favorito',
              axis=alt.Axis(labelAngle=0))
    y = alt.Y('accuracy:Q', title='Series acertadas en el test', scale=alt.Scale(domain=[0, 1]),
              axis=alt.Axis(format='.0%'))
    base = alt.Chart(t)
    barras = base.mark_bar(color=ACENTO, cornerRadiusEnd=3, size=72).encode(
        x=x, y=y, tooltip=[alt.Tooltip('tramo:O', title='Confianza'),
                           alt.Tooltip('series:Q', title='Series', format=','),
                           alt.Tooltip('accuracy:Q', title='Acierta', format='.1%'),
                           alt.Tooltip('confianza:Q', title='Confianza media del modelo', format='.1%'),
                           alt.Tooltip('cuota:Q', title='Cuota mínima para no perder', format='.2f')])
    marcas = base.mark_tick(color=TINTA, thickness=3, size=72).encode(x=x, y='confianza:Q')
    texto = base.mark_text(dy=-9, fontSize=13, fontWeight='bold', color=TINTA).encode(x=x, y='accuracy:Q', text='texto:N')
    cuota = base.mark_text(dy=14, fontSize=11, color=FONDO, fontWeight='bold').encode(x=x, y='accuracy:Q', text='texto_cuota:N')
    n = base.mark_text(y=288, fontSize=11, color=FONDO).encode(x=x, text='texto_n:N')
    duda = t.iloc[0]
    alta = t.iloc[-1]
    return (barras + marcas + texto + cuota + n).properties(width=680, height=300, title=alt.Title(
        f'Cuando el modelo duda acierta el {pct(duda.accuracy, 0)}; cuando está seguro, el {pct(alta.accuracy, 0)}',
        subtitle=[f'Las {num(len(detalle))} series del test, agrupadas por la probabilidad que el modelo le dio a su favorito.',
                  'La rayita blanca es esa probabilidad media: casi toca el borde de cada barra, así que cuando dice 70 % acierta 7 de 10.',
                  'Adentro de cada barra, la cuota mínima a la que apostar siempre al favorito deja de perder plata.']))


# ---------------------------------------------------------------------------
# Contra los modelos que no piensan, y la matriz de confusión
# ---------------------------------------------------------------------------
NOMBRE_TONTO = {'B': 'Siempre «gana B»', 'A': 'Siempre «gana A»', 'modelo': 'Nuestro modelo'}


def g_tontos(t):
    """t: columnas modelo, periodo ('Validación cruzada' | 'Prueba'), accuracy."""
    orden = [NOMBRE_TONTO[k] for k in ['B', 'A', 'modelo']]
    periodos = ['Validación cruzada', 'Prueba']
    esc = alt.Scale(domain=periodos, range=['#8b96a5', ACENTO])
    y = alt.Y('nombre:N', sort=orden, title=None, axis=alt.Axis(labelLimit=200, labelFontSize=13))
    base = alt.Chart(t).transform_calculate(texto="replace(format(datum.accuracy * 100, '.1f'), '.', ',') + ' %'")
    barras = base.mark_bar(cornerRadiusEnd=3, height=14).encode(
        x=alt.X('accuracy:Q', title='Series acertadas de cada 100', scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format='.0%', values=[0, 0.25, 0.5, 0.75, 1])),
        y=y, yOffset=alt.YOffset('periodo:N', sort=periodos, scale=alt.Scale(domain=periodos)),
        color=alt.Color('periodo:N', scale=esc, sort=periodos, title=None, legend=alt.Legend(orient='bottom')),
        tooltip=[alt.Tooltip('nombre:N', title='Modelo'), alt.Tooltip('periodo:N', title='Medido en'),
                 alt.Tooltip('accuracy:Q', title='Acierta', format='.1%')])
    texto = base.mark_text(align='left', dx=6, fontSize=12, color=TINTA).encode(
        x='accuracy:Q', y=y, yOffset=alt.YOffset('periodo:N', sort=periodos, scale=alt.Scale(domain=periodos)), text='texto:N')
    moneda = alt.Chart(pd.DataFrame({'x': [0.5], 't': ['tirar una moneda: 50 %']}))
    regla = moneda.mark_rule(color=GRIS, strokeDash=[4, 3]).encode(x='x:Q')
    rot = moneda.mark_text(align='left', dx=5, y=-8, color=GRIS, fontSize=11).encode(x='x:Q', text='t:N')
    m = t[(t.modelo == 'modelo') & (t.periodo == 'Prueba')].accuracy.iloc[0]
    a = t[(t.modelo == 'A') & (t.periodo == 'Prueba')].accuracy.iloc[0]
    return alt.layer(barras, texto, regla, rot).properties(width=600, height=alt.Step(46), title=alt.Title(
        f'El modelo acierta {round(100 * (m - a))} series más de cada 100 que el mejor de los modelos que no piensan',
        subtitle=['Un modelo «tonto» (dummy) da siempre la misma respuesta, sin mirar ningún dato. Es la vara mínima:',
                  'un modelo que no le gana a eso no sirve.']))


def g_matriz_lecturas(real, pred):
    """La matriz del modelo leída de tres formas, como en el apartado 11.1 del notebook:
    cantidades, por fila (la diagonal es el recall) y por columna (la diagonal es la precisión)."""
    reales, preds = ['Ganó A', 'Ganó B'], ['Dijo «gana A»', 'Dijo «gana B»']
    filas = []
    for r, nr in [(1, 'Ganó A'), (0, 'Ganó B')]:
        for p_, npred in [(1, 'Dijo «gana A»'), (0, 'Dijo «gana B»')]:
            n = int(((real == r) & (pred == p_)).sum())
            filas.append({'real': nr, 'pred': npred, 'n': n, 'ok': 'Acierto' if r == p_ else 'Error',
                          'fila': n / int((real == r).sum()), 'col': n / int((pred == p_).sum())})
    t = pd.DataFrame(filas)
    t['equipo'] = t.real.str[-1]
    lecturas = [
        ('Cantidades', 'n', lambda d: num(d.n),
         'Series de prueba en cada casilla.'),
        ('Por fila → recall', 'fila', lambda d: pct(d.fila),
         'De las que ganó cada equipo, cuántas detectó.'),
        ('Por columna → precisión', 'col', lambda d: pct(d.col),
         'De las veces que dijo cada equipo, cuántas acertó.')]
    paneles = []
    for i, (titulo, campo, fmt, sub) in enumerate(lecturas):
        d = t.assign(texto=t.apply(fmt, axis=1),
                     detalle=t.apply(lambda r: (f'recall de {r.equipo}' if campo == 'fila' else f'precisión de {r.equipo}')
                                     if (campo != 'n' and r.ok == 'Acierto') else '', axis=1))
        base = alt.Chart(d)
        x = alt.X('pred:N', sort=preds, title=None, axis=alt.Axis(orient='top', labelAngle=0, labelFontSize=12))
        y = alt.Y('real:N', sort=reales, title='Lo que pasó' if i == 0 else None,
                  axis=alt.Axis(labels=i == 0, ticks=False, labelFontSize=12))
        intensidad = 'fila' if campo == 'n' else campo
        celda = base.mark_rect(stroke=FONDO, strokeWidth=3).encode(
            x=x, y=y, color=alt.Color('ok:N', scale=ESC_RESULTADO, legend=None),
            opacity=alt.Opacity(f'{intensidad}:Q', scale=alt.Scale(domain=[0, 1], range=[0.15, 0.95]), legend=None),
            tooltip=[alt.Tooltip('real:N', title='Lo que pasó'), alt.Tooltip('pred:N', title='Lo que dijo el modelo'),
                     alt.Tooltip('n:Q', title='Series', format=','),
                     alt.Tooltip('fila:Q', title='Parte de su fila', format='.1%'),
                     alt.Tooltip('col:Q', title='Parte de su columna', format='.1%')])
        valor = base.mark_text(fontSize=21, fontWeight='bold', dy=-7, color=TINTA, font=FUENTE_HUD).encode(x=x, y=y, text='texto:N')
        det = base.mark_text(fontSize=11, dy=13, color=TINTA).encode(x=x, y=y, text='detalle:N')
        paneles.append(alt.layer(celda, valor, det).properties(
            width=230, height=170, title=alt.Title(titulo, subtitle=sub, fontSize=15, anchor='middle')))
    return alt.hconcat(*paneles, spacing=18).resolve_scale(color='shared', opacity='independent').properties(
        title=alt.Title('Lo que dijo el modelo', anchor='middle', fontSize=13, fontWeight='normal', color=GRIS))
