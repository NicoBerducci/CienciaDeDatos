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
    """Series por mes, apiladas por tier, con el corte del test marcado."""
    t = tabla_meses(df)
    base = alt.Chart(t)
    barras = base.mark_bar().encode(
        x=alt.X('yearmonth(mes):T', title=None, axis=alt.Axis(format='%b %Y', labelAngle=0, tickCount=7)),
        y=alt.Y('sum(series):Q', title='Series por mes'),
        color=alt.Color('tier_n:N', title='Tier del torneo', scale=ESC_TIER, sort=TIER_ORDEN,
                        legend=alt.Legend(orient='top', direction='horizontal')),
        order=alt.Order('orden:Q'),
        tooltip=[alt.Tooltip('yearmonth(mes):T', title='Mes', format='%B %Y'),
                 alt.Tooltip('tier_n:N', title='Tier'), alt.Tooltip('series:Q', title='Series', format=',')],
    ).transform_calculate(orden=f"indexof({TIER_ORDEN}, datum.tier_n)")
    corte = pd.DataFrame({'mes': [cm.CORTE_TEST.tz_convert(None)], 't': ['empieza el test →']})
    regla = alt.Chart(corte).mark_rule(color=COLOR_ERR, strokeDash=[5, 3], strokeWidth=2).encode(x='mes:T')
    texto = alt.Chart(corte).mark_text(align='left', dx=6, dy=-4, color=COLOR_ERR, fontSize=13).encode(
        x='mes:T', y=alt.value(8), text='t:N')
    n = len(df)
    fuera = df.tier.isin(['b', 'c']).mean()
    return (barras + regla + texto).properties(width=680, height=280, title=alt.Title(
        f'{num(n)} series en tres años, y {round(10 * fuera)} de cada 10 se juegan en tiers B y C',
        subtitle=[f'Cada barra es un mes; el color, el nivel del torneo según Bo3.gg. La élite (S) y el tier A '
                  f'suman el {pct(df.tier.isin(["s", "a"]).mean())}.',
                  'Todo lo anterior a la línea roja es entrenamiento; lo posterior, el test que el modelo nunca vio.']))


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
                  'Mismas 27 columnas, mismos 5 folds temporales, cada familia con su grilla completa.']))


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
