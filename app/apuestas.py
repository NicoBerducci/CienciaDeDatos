"""La apuesta: cuándo conviene apostar siguiendo al modelo final, con cuotas pre-partido de Pinnacle.

La regla tiene dos números y se elige con datos, como el modelo:

    apostar $100 al equipo que elige el modelo  si  confianza ≥ τ  y  confianza × cuota − 1 ≥ m

- τ (confianza mínima) evita apostar en series parejas, donde el modelo acierta poco más que una moneda.
- m (valor esperado mínimo) evita apostar cuando la casa paga demasiado poco por lo que el modelo cree.

τ y m se eligen en el período de elección (marzo a julio de 2026) y se prueban, sin tocarlos,
en el período de prueba (agosto a mediados de octubre). Si ninguna regla gana plata en la elección,
la recomendación es no apostar.
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

import graficos as g
from graficos import num, pct

APUESTA = 100
CORTE_ELECCION = pd.Timestamp('2026-08-01', tz='UTC')
TAUS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
EVS = [None, -0.10, -0.05, 0.0, 0.05, 0.10]
MIN_APUESTAS = 30
PERIODOS = ['Elección (mar–jul)', 'Prueba (ago–oct)']
SEMILLA = 42
MESES_CORTOS = {'03': 'mar', '04': 'abr', '05': 'may', '06': 'jun', '07': 'jul', '08': 'ago', '09': 'sep', '10': 'oct'}


def etiqueta_ev(m):
    return 'sin filtro' if m is None else f"{'+' if m > 0 else ('−' if m < 0 else '')}{abs(m) * 100:.0f} %"


def ruta_cuotas(raiz):
    hay = sorted((Path(raiz) / 'datos').glob('cuotas_oddspapi_pinnacle_*.csv'))
    assert hay, 'Faltan las cuotas de Pinnacle: python app/bajar_cuotas_oddspapi.py (ver README).'
    return hay[-1]


def tabla(detalle, test, cuotas):
    """Una fila por serie del test con cuota: a quién elige el modelo, qué paga la casa y qué deja apostarle $100."""
    d = detalle.assign(match_slug=test.match_slug.values)
    c = cuotas[['match_slug', 'team_a_id', 'team_b_id', 'cuota_a', 'cuota_b', 'margen_casa', 'ts_cuota']].rename(
        columns={'team_a_id': 'cuota_team_a', 'team_b_id': 'cuota_team_b'})
    t = d.merge(c, on='match_slug', how='inner', validate='one_to_one')
    assert (t.cuota_team_a == t.team_a_id).all() and (t.cuota_team_b == t.team_b_id).all()
    assert (pd.to_datetime(t.ts_cuota, utc=True, format='mixed') < t.start_at_utc + pd.Timedelta(minutes=1)).all()
    elige_a = t.predicho == 1
    t['pick'] = np.where(elige_a, t.team_a_name, t.team_b_name)
    t['rival'] = np.where(elige_a, t.team_b_name, t.team_a_name)
    t['cuota_pick'] = np.where(elige_a, t.cuota_a, t.cuota_b)
    t['cuota_rival'] = np.where(elige_a, t.cuota_b, t.cuota_a)
    t['casa_pick'] = (1 / t.cuota_pick) / (1 / t.cuota_a + 1 / t.cuota_b)   # probabilidad de la casa, sin margen
    t['ev'] = t.confianza * t.cuota_pick - 1
    t['resultado'] = np.where(t.acierto, APUESTA * (t.cuota_pick - 1), -APUESTA)
    t['periodo'] = np.where(t.start_at_utc < CORTE_ELECCION, PERIODOS[0], PERIODOS[1])
    return t.sort_values('start_at_utc').reset_index(drop=True)


def aplicar(t, tau, m):
    s = t.confianza >= tau - 1e-12
    if m is not None:
        s &= t.ev >= m - 1e-12
    return t[s]


def resumen(s):
    n = len(s)
    return {'n': n, 'ganadas': int(s.acierto.sum()), 'acierto': float(s.acierto.mean()) if n else np.nan,
            'neto': float(s.resultado.sum()), 'roi': float(s.resultado.sum() / (APUESTA * n)) if n else np.nan,
            'cuota_med': float(s.cuota_pick.median()) if n else np.nan}


def grilla(t):
    filas = []
    for periodo in PERIODOS:
        tp = t[t.periodo == periodo]
        for tau in TAUS:
            for m in EVS:
                filas.append({'periodo': periodo, 'tau': tau, 'm': m, 'ev_txt': etiqueta_ev(m),
                              **resumen(aplicar(tp, tau, m))})
    return pd.DataFrame(filas)


def elegir(gr):
    """La regla que más plata deja en la elección, con al menos MIN_APUESTAS apuestas. Si ninguna gana, no apostar."""
    e = gr[(gr.periodo == PERIODOS[0]) & (gr.n >= MIN_APUESTAS)]
    if e.empty:
        return None
    mejor = e.sort_values(['neto', 'tau'], ascending=[False, False], kind='stable').iloc[0]
    if mejor.neto <= 0:
        return None
    return float(mejor.tau), (None if pd.isna(mejor.m) else float(mejor.m))


def intervalo(s, n_boot=5000, nivel=0.90):
    """Intervalo bootstrap del rendimiento (neto / apostado), remuestreando apuestas."""
    if len(s) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(SEMILLA)
    r = s.resultado.to_numpy()
    rois = rng.choice(r, size=(n_boot, len(r)), replace=True).mean(axis=1) / APUESTA
    a = (1 - nivel) / 2
    return float(np.quantile(rois, a)), float(np.quantile(rois, 1 - a))


# ---------------------------------------------------------------------------- textos
def plata(x, signo=False):
    s = num(abs(x))
    return (('+' if x >= 0 else '−') if signo else ('−' if x < 0 else '')) + '$' + s


def por_ciento(x, signo=True):
    s = f'{abs(x) * 100:.1f}'.replace('.', ',') + ' %'
    return (('+' if x >= 0 else '−') + s) if signo else s


def textos(t, gr, regla, cuotas, n_sa, ruta):
    el, pr = (t[t.periodo == p] for p in PERIODOS)
    tx = {'ap_archivo': ruta.name, 'ap_n_sa': num(n_sa), 'ap_n': num(len(t)), 'ap_cobertura': pct(len(t) / n_sa, 0),
          'ap_n_el': num(len(el)), 'ap_n_pr': num(len(pr)),
          'ap_margen': por_ciento(t.margen_casa.median(), False),
          'ap_desde': f'{t.start_at_utc.min():%d/%m/%Y}', 'ap_hasta': f'{t.start_at_utc.max():%d/%m/%Y}',
          'ap_min': num(MIN_APUESTAS), 'ap_n_reglas': num(len(TAUS) * len(EVS))}
    for clave, s in [('siempre_el', el), ('siempre_pr', pr)]:
        r = resumen(s)
        tx.update({f'ap_{clave}_n': num(r['n']), f'ap_{clave}_neto': plata(r['neto'], True),
                   f'ap_{clave}_roi': por_ciento(r['roi']), f'ap_{clave}_acc': pct(r['acierto']),
                   f'ap_{clave}_cuota': f"{r['cuota_med']:.2f}".replace('.', ','),
                   f'ap_{clave}_equilibrio': pct(1 / r['cuota_med'])})
    sp = resumen(pr)
    if regla is None:
        tx.update({
            'ap_titulo': 'En marzo–julio ninguna regla le ganó a la casa: con este modelo, la recomendación es no apostar',
            'ap_regla_frase': 'en el período de elección ninguna combinación de τ y m dejó ganancia, así que la regla es no apostar.',
            'ap_regla_roi': '$0', 'ap_regla_detalle': 'La regla no apuesta en ninguna serie.',
            'ap_mirar_reglas': 'Ninguna casilla de la izquierda es verde con suficientes apuestas: por eso la regla es no apostar.',
            'ap_conclusion': f"No apostar deja $0; apostar a todas las series de prueba habría dejado {plata(sp['neto'], True)} ({por_ciento(sp['roi'])}).",
            'ap_limite': 'Con estas cuotas, el modelo no encuentra series en las que la casa pague de más.'})
        return tx
    tau, m = regla
    re, rp = aplicar(el, tau, m), aplicar(pr, tau, m)
    a, b = resumen(re), resumen(rp)
    lo, hi = intervalo(rp)
    tx.update({
        'ap_hay_regla': 'sí', 'ap_tau': pct(tau, 0), 'ap_m': etiqueta_ev(m),
        'ap_regla_txt': (f'confianza del modelo de {pct(tau, 0)} o más' +
                         ('' if m is None else f' y valor esperado de {etiqueta_ev(m)} o más')),
        'ap_el_n': num(a['n']), 'ap_el_neto': plata(a['neto'], True), 'ap_el_roi': por_ciento(a['roi']), 'ap_el_acc': pct(a['acierto']),
        'ap_pr_n': num(b['n']), 'ap_pr_neto': plata(b['neto'], True), 'ap_pr_roi': por_ciento(b['roi']), 'ap_pr_acc': pct(b['acierto']),
        'ap_pr_cuota': f"{b['cuota_med']:.2f}".replace('.', ','), 'ap_pr_cob': pct(b['n'] / max(len(pr), 1), 0),
        'ap_pr_gano': 'ganó' if b['neto'] > 0 else 'perdió',
        'ap_ic_lo': por_ciento(lo), 'ap_ic_hi': por_ciento(hi),
        'ap_ic_cero': 'incluye el cero' if lo <= 0 <= hi else ('queda todo por encima de cero' if lo > 0 else 'queda todo por debajo de cero'),
        'ap_mejora_pr': plata(b['neto'] - resumen(pr)['neto'], True),
    })
    cond = f'la confianza llega a {pct(tau, 0)}' + ('' if m is None else f' y el valor esperado a {etiqueta_ev(m)}')
    verbo = 'gana' if b['neto'] > 0 else 'pierde'
    tx['ap_titulo'] = (f"Apostando solo cuando {cond}, el modelo {verbo} {plata(abs(b['neto']))} en las series de prueba; "
                       f"apostando a todas, {'gana' if sp['neto'] > 0 else 'pierde'} {plata(abs(sp['neto']))}")
    tx['ap_regla_frase'] = f'apostar solo si la confianza es de {pct(tau, 0)} o más' + ('' if m is None else f' y el valor esperado de {etiqueta_ev(m)} o más') + '.'
    tx['ap_regla_roi'] = por_ciento(b['roi']) if b['n'] else '$0'
    tx['ap_regla_detalle'] = (f"{plata(b['neto'], True)} en {num(b['n'])} apuestas ({pct(b['n'] / max(len(pr), 1), 0)} de las series): "
                              f"acierta {pct(b['acierto'])} con cuota mediana {tx['ap_pr_cuota']}.")
    validas = gr[(gr.periodo == PERIODOS[1]) & (gr.n >= MIN_APUESTAS)].sort_values('neto', ascending=False).reset_index(drop=True)
    pos = validas.index[(validas.tau.round(2) == round(tau, 2)) & (validas.m.isna() if m is None else validas.m.round(2) == round(m, 2))]
    puesto = f'{int(pos[0]) + 1}.ª de {len(validas)}' if len(pos) else 'fuera del ranking (apuesta en menos de 30 series)'
    tx['ap_mirar_reglas'] = (f'En la prueba, la regla elegida queda {puesto} entre las reglas que apuestan en al menos {MIN_APUESTAS} series. '
                             'Elegirla mirando la prueba sería hacer trampa: lo que importa es que se eligió antes.')
    tx['ap_conclusion'] = (
        f"En el período de elección la regla dejó {tx['ap_el_neto']} ({tx['ap_el_roi']}) en {tx['ap_el_n']} apuestas. "
        f"En el de prueba {'ganó' if b['neto'] > 0 else 'perdió'} {plata(abs(b['neto']))} ({tx['ap_pr_roi']}) en {tx['ap_pr_n']} apuestas, "
        f"{'ganándole' if b['neto'] > sp['neto'] else 'perdiendo contra'} a apostar a todas ({plata(sp['neto'], True)}) "
        f"{'y' if b['neto'] > 0 else 'pero no'} a no apostar ($0). "
        f"Si se repitiera con otras series parecidas, el rendimiento estaría con 90 % de probabilidad entre {tx['ap_ic_lo']} y {tx['ap_ic_hi']} "
        f"(intervalo bootstrap): {tx['ap_ic_cero']}.")
    if b['neto'] <= 0:
        tx['ap_conclusion'] += (
            f" <b>Conclusión:</b> la regla no alcanza para ganarle a la casa con este modelo. Lo que sí hace es evitar las peores apuestas: "
            f"en la prueba recorta la pérdida de apostar a todas en {plata(b['neto'] - sp['neto'])}. "
            "La recomendación honesta es usarla para saber en qué series <b>no</b> apostar, no como una forma de ganar plata.")
    else:
        tx['ap_conclusion'] += (" <b>Conclusión:</b> la regla gana en las dos mitades, pero con pocas apuestas; "
                                "es una señal a favor, no una garantía.")
    tx['ap_limite'] = (f"Son {num(len(pr))} series de prueba y {num(b['n'])} apuestas: alcanza para descartar las peores reglas, no para asegurar "
                       "una ganancia. El modelo no ve el veto de mapas ni las novedades del día, que la casa sí incorpora; por eso la regla es "
                       "una guía para decidir cuándo no apostar, no una garantía de ganar.")
    return tx


# ---------------------------------------------------------------------------- gráficos
def g_reglas(gr, regla):
    """Las 42 reglas en los dos períodos: el color es el rendimiento; el recuadro, la regla elegida en marzo–julio."""
    d = gr.copy()
    d['m_orden'] = d.m.map(lambda m: -1 if pd.isna(m) else m)
    d['tau_txt'] = d.tau.map(lambda x: pct(x, 0))
    d['elegida'] = False
    if regla is not None:
        tau, m = regla
        d['elegida'] = (d.tau.round(2) == round(tau, 2)) & (d.m.isna() if m is None else d.m.round(2) == round(m, 2))
    d['roi_c'] = d.roi.clip(-0.25, 0.25)
    d['t1'] = d.apply(lambda r: '—' if r.n == 0 else plata(r.neto, True), axis=1)
    d['t2'] = d.apply(lambda r: f'{r.n} apuestas' if r.n else 'ninguna', axis=1)
    d['pocas'] = d.n < MIN_APUESTAS
    orden_ev = [etiqueta_ev(m) for m in EVS]
    orden_tau = [pct(x, 0) for x in TAUS]
    paneles = []
    for i, periodo in enumerate(PERIODOS):
        base = alt.Chart(d[d.periodo == periodo])
        x = alt.X('ev_txt:N', sort=orden_ev, title='Valor esperado mínimo (m)', axis=alt.Axis(labelAngle=0, orient='top'))
        y = alt.Y('tau_txt:N', sort=orden_tau, title='Confianza mínima (τ)' if i == 0 else None, axis=alt.Axis(labels=i == 0))
        celda = base.mark_rect(stroke=g.FONDO, strokeWidth=2).encode(
            x=x, y=y,
            color=alt.condition('datum.n == 0', alt.value('#1a2029'), alt.Color('roi_c:Q', title='Rendimiento', scale=alt.Scale(
                domain=[-0.25, 0, 0.25], range=[g.COLOR_ERR, '#1f262f', g.COLOR_OK]), legend=alt.Legend(format='+.0%', orient='bottom', gradientLength=220))),
            opacity=alt.condition('datum.pocas', alt.value(0.45), alt.value(1)),
            tooltip=[alt.Tooltip('tau_txt:N', title='Confianza mínima'), alt.Tooltip('ev_txt:N', title='Valor esperado mínimo'),
                     alt.Tooltip('n:Q', title='Apuestas'), alt.Tooltip('acierto:Q', title='Acierto', format='.1%'),
                     alt.Tooltip('neto:Q', title='Neto ($)', format='+,.0f'), alt.Tooltip('roi:Q', title='Rendimiento', format='+.1%')])
        marco = base.transform_filter('datum.elegida').mark_rect(fill=None, stroke=g.TINTA, strokeWidth=3).encode(x=x, y=y)
        t1 = base.mark_text(fontSize=12, fontWeight='bold', dy=-6, color=g.TINTA).encode(x=x, y=y, text='t1:N')
        t2 = base.mark_text(fontSize=9.5, dy=8, color=g.GRIS).encode(x=x, y=y, text='t2:N')
        paneles.append(alt.layer(celda, marco, t1, t2).properties(
            width=420, height=alt.Step(40), title=alt.Title(periodo, fontSize=15, anchor='middle')))
    return alt.hconcat(*paneles, spacing=24).resolve_scale(color='shared').configure_view(stroke=None)


def datos_simulador(t):
    mapas = lambda r: f'{int(r.post_team_a_maps_won)}-{int(r.post_team_b_maps_won)}' if pd.notna(r.post_team_a_maps_won) else ''
    return pd.DataFrame({
        't': t.start_at_utc.dt.strftime('%Y-%m-%dT%H:%M:00Z'), 'fecha': t.start_at_utc.dt.strftime('%d/%m %H:%M'),
        'mes': t.start_at_utc.dt.strftime('%Y-%m'), 'mes_txt': t.start_at_utc.dt.strftime('%m').map(MESES_CORTOS) + ' ' + t.start_at_utc.dt.strftime('%y'),
        'periodo': t.periodo,
        'a': t.team_a_name, 'b': t.team_b_name, 'torneo': t.tournament_name.str.strip(),
        'tier': t.tier.map(g.TIER_NOMBRE), 'bo': 'BO' + t.bo_type.astype(str),
        'pick': t.pick, 'conf': t.confianza.round(4), 'cuota': t.cuota_pick, 'cuota_r': t.cuota_rival,
        'casa': t.casa_pick.round(4), 'ev': t.ev.round(4),
        'ok': t.acierto.astype(int), 'res': t.resultado.round(2), 'ganador': t.ganador, 'mapas': t.apply(mapas, axis=1)})



def g_simulador(ap, regla):
    tau0, m0 = (regla if regla is not None else (0.5, None))
    datos = alt.InlineData(values=ap.to_csv(index=False), format=alt.DataFormat(type='csv', parse={
        'conf': 'number', 'cuota': 'number', 'cuota_r': 'number', 'casa': 'number', 'ev': 'number', 'ok': 'number',
        'res': 'number', 't': 'date'}))
    E = '#c-apuesta'
    periodo = alt.param(name='periodo_ap', value=PERIODOS[1], bind=alt.binding_select(
        options=PERIODOS + ['Todo el test'], name='Período ', element=E))
    tier_ap = alt.param(name='tier_ap', value='Todos', bind=alt.binding_select(
        options=['Todos'] + g.TIER_ORDEN[:2], name='Tier ', element=E))
    bo_ap = alt.param(name='bo_ap', value='Todos', bind=alt.binding_select(
        options=['Todos', 'BO1', 'BO3', 'BO5'], name='Formato ', element=E))
    equipo_ap = alt.param(name='equipo_ap', value='', bind=alt.binding(
        input='search', name='Equipo ', placeholder='p. ej. Vitality', element=E))
    conf_ap = alt.param(name='conf_ap', value=round(tau0 * 100), bind=alt.binding_range(
        min=50, max=80, step=5, name='Confianza mínima τ (%) ', element=E))
    opciones_ev = [-999] + [round(m * 100) for m in EVS if m is not None]
    ev_ap = alt.param(name='ev_ap', value=-999 if m0 is None else round(m0 * 100), bind=alt.binding_select(
        options=opciones_ev, labels=[etiqueta_ev(None)] + [etiqueta_ev(m / 100) for m in opciones_ev[1:]],
        name='Valor esperado mínimo m ', element=E))
    mes_sel = alt.selection_point(name='mes_ap', fields=['mes'], on='click', clear='dblclick', empty=True)

    f_base = ("(periodo_ap == 'Todo el test' || datum.periodo == periodo_ap) && (tier_ap == 'Todos' || datum.tier == tier_ap)"
              " && (bo_ap == 'Todos' || datum.bo == bo_ap)"
              " && (equipo_ap == '' || indexof(lower(datum.a + ' ' + datum.b), lower(equipo_ap)) >= 0)")
    f_regla = "datum.conf * 100 >= conf_ap - 1e-9 && (ev_ap == -999 || datum.ev * 100 >= ev_ap - 1e-9)"
    f_mes = "(!length(data('mes_ap_store')) || vlSelectionTest('mes_ap_store', datum))"
    base = alt.Chart(datos).transform_calculate(
        en_base=f'{f_base} && {f_mes}', apuesta=f'{f_base} && {f_mes} && {f_regla}', sin_mes=f'{f_base} && {f_regla}')

    base_ap = base.transform_calculate(ap='datum.apuesta ? 1 : 0', r_ap='datum.apuesta ? datum.res : 0', ok_ap='datum.apuesta ? datum.ok : 0')
    agg = base_ap.transform_filter('datum.en_base').transform_aggregate(
        n_base='count()', neto_base='sum(res)', n='sum(ap)', neto='sum(r_ap)', ok='sum(ok_ap)').transform_calculate(
        gano='datum.neto >= 0',
        l1=("datum.n == 0 ? 'Con esta regla no apostás ninguna serie: $0' : (datum.neto >= 0 ? 'GANÁS $' : 'PERDÉS $') + "
            "format(abs(datum.neto) + 1e-6, ',.0f') + ' siguiendo la regla'"),
        l2=("datum.n_base == 0 ? '' : 'La regla apuesta en ' + format(datum.n, ',') + ' de ' + format(datum.n_base, ',') + ' series' + "
            "(datum.n > 0 ? ': acierta ' + format(datum.ok / datum.n * 100, '.0f') + ' % y rinde ' + (datum.neto >= 0 ? '+' : '−') + "
            "format(abs(datum.neto) / (datum.n * " + str(APUESTA) + ") * 100, '.1f') + ' % de lo apostado.' : '.') + "
            "' Apostando a todas: ' + (datum.neto_base >= 0 ? '+' : '−') + '$' + format(abs(datum.neto_base) + 1e-6, ',.0f') + '.'"))
    concl = (agg.mark_text(align='left', baseline='top', fontSize=28, fontWeight='bold', font=g.FUENTE_HUD).encode(
                 text='l1:N', x=alt.value(0), y=alt.value(0),
                 color=alt.condition('datum.gano', alt.value(g.COLOR_OK), alt.value(g.COLOR_ERR)))
             + agg.mark_text(align='left', baseline='top', fontSize=14, color=g.TINTA, limit=900).encode(
                 text='l2:N', x=alt.value(0), y=alt.value(36))).properties(width=900, height=56)

    # ---- saldo acumulado: la regla contra apostar siempre
    def saldo(filtro, nombre):
        return base.transform_filter(filtro).transform_window(
            saldo='sum(res)', n_acum='count()', sort=[alt.SortField('t')], frame=[None, 0]).transform_calculate(
            serie=f"'{nombre}'",
            frase=("'" + nombre + ": después de ' + datum.a + ' vs. ' + datum.b + ' llevás ' + (datum.saldo >= 0 ? '+' : '−') + '$' + "
                   "format(abs(datum.saldo) + 1e-6, ',.0f') + ' en ' + datum.n_acum + ' apuestas'"))
    esc_serie = alt.Scale(domain=['Siguiendo la regla', 'Apostando a todas'], range=[g.ACENTO, g.GRIS])
    enc = dict(x=alt.X('t:T', title=None, scale=alt.Scale(type='utc'), axis=alt.Axis(format='%d/%m', labelOverlap=True)),
               y=alt.Y('saldo:Q', title='Saldo acumulado ($)', axis=alt.Axis(format=',.0f')),
               color=alt.Color('serie:N', scale=esc_serie, title=None, legend=alt.Legend(orient='top-left')),
               tooltip=[alt.Tooltip('frase:N', title='Saldo'), alt.Tooltip('fecha:N', title='Fecha (UTC)')])
    cero = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color=g.GRIS, strokeDash=[4, 3]).encode(y='y:Q')
    linea = (saldo('datum.en_base', 'Apostando a todas').mark_line(interpolate='step-after', strokeWidth=1.5, strokeDash=[5, 3]).encode(**enc)
             + saldo('datum.apuesta', 'Siguiendo la regla').mark_line(interpolate='step-after', strokeWidth=2.5).encode(**enc)
             + cero).properties(width=900, height=200, title=alt.Title(
                 'Tu saldo, apuesta por apuesta', subtitle='Arranca en $0; cada apuesta de $100 suma lo que gana o resta los $100.',
                 anchor='start', fontSize=14, subtitleColor=g.GRIS))

    # ---- un punto por serie
    frase = ("datum.apuesta ? ((datum.ok ? 'GANÁS $' + format(datum.res, ',.2f') : 'PERDÉS $" + str(APUESTA) + "') + "
             "' apostando $" + str(APUESTA) + " a ' + datum.pick + ' (cuota ' + format(datum.cuota, '.2f') + ')') : "
             "'La regla NO apuesta: ' + (datum.conf * 100 < conf_ap ? 'confianza ' + format(datum.conf * 100, '.0f') + ' % < ' + conf_ap + ' %' : "
             "'valor esperado ' + format(datum.ev * 100, '+.0f') + ' %')")
    puntos = base.transform_filter('datum.en_base').transform_calculate(
        frase=frase, quien="datum.apuesta ? (datum.ok ? 'Apuesta ganada' : 'Apuesta perdida') : 'No se apuesta'",
        y_res='datum.apuesta ? datum.res : 0', orden='datum.apuesta ? 1 : 0').mark_circle(size=40).encode(
        x=alt.X('t:T', title='Inicio de la serie (UTC)', scale=alt.Scale(type='utc'), axis=alt.Axis(format='%d/%m', labelOverlap=True)),
        y=alt.Y('y_res:Q', title='Resultado de la apuesta ($)', scale=alt.Scale(type='symlog', constant=10, domain=[-110, 1000]),
                axis=alt.Axis(values=[-100, -10, 0, 10, 100, 1000], format=',.0f')),
        color=alt.Color('quien:N', title=None, scale=alt.Scale(domain=['Apuesta ganada', 'Apuesta perdida', 'No se apuesta'],
                                                             range=[g.COLOR_OK, g.COLOR_ERR, g.APAGADO]),
                        legend=alt.Legend(orient='top-right')),
        order=alt.Order('orden:Q'),
        tooltip=[alt.Tooltip('frase:N', title='Con la regla'), alt.Tooltip('fecha:N', title='Fecha (UTC)'),
                 alt.Tooltip('a:N', title='Equipo A'), alt.Tooltip('b:N', title='Equipo B'), alt.Tooltip('torneo:N', title='Torneo'),
                 alt.Tooltip('tier:N', title='Tier'), alt.Tooltip('bo:N', title='Formato'),
                 alt.Tooltip('conf:Q', title='Confianza del modelo', format='.0%'), alt.Tooltip('casa:Q', title='Según la casa', format='.0%'),
                 alt.Tooltip('ev:Q', title='Valor esperado', format='+.0%'), alt.Tooltip('cuota_r:Q', title='Cuota del rival', format='.2f'),
                 alt.Tooltip('ganador:N', title='Ganó'), alt.Tooltip('mapas:N', title='Mapas A-B')])
    nube = (puntos + cero).properties(width=900, height=240, title=alt.Title(
        'Cada serie: si la regla apuesta y cómo le va', subtitle='En gris, las series en las que la regla no apuesta. Pasá el mouse por un punto.',
        anchor='start', fontSize=14, subtitleColor=g.GRIS))

    # ---- neto por mes (clic para quedarse con uno)
    barras_mes = base.transform_filter('datum.sin_mes').transform_aggregate(
        neto='sum(res)', n='count()', groupby=['mes', 'mes_txt']).transform_calculate(
        signo="datum.neto >= 0 ? 'g' : 'p'",
        frase="'En ' + datum.mes_txt + ' la regla habría ' + (datum.neto >= 0 ? 'ganado' : 'perdido') + ' $' + format(abs(datum.neto) + 1e-6, ',.0f') + ' en ' + datum.n + ' apuestas'"
    ).mark_bar(cornerRadius=2).encode(
        x=alt.X('mes_txt:N', title='Mes (clic para elegir uno, doble clic para volver)', sort=alt.EncodingSortField('mes'),
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y('neto:Q', title='Neto del mes ($)', axis=alt.Axis(format=',.0f')),
        color=alt.Color('signo:N', scale=alt.Scale(domain=['g', 'p'], range=[g.COLOR_OK, g.COLOR_ERR]), legend=None),
        opacity=alt.condition(mes_sel, alt.value(0.95), alt.value(0.3)),
        tooltip=[alt.Tooltip('frase:N', title='Ese mes')]).add_params(mes_sel).properties(width=900, height=140, title=alt.Title(
            'Mes por mes, con la regla', anchor='start', fontSize=14))

    return alt.vconcat(concl, linea, nube, barras_mes, spacing=24).add_params(
        periodo, tier_ap, bo_ap, equipo_ap, conf_ap, ev_ap).resolve_scale(color='independent').configure_view(stroke=None)
