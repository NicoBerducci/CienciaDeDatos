"""Baja de Bo3.gg las cuotas de las series terminadas, día por día, para el backtest de apuestas.

Bo3.gg arma su página de resultados (https://bo3.gg/matches/finished) con una llamada por día a

    https://api.bo3.gg/api/v2/matches/finished?date=AAAA-MM-DD&utc_offset=0&filter[discipline_id][eq]=1

Cada serie de esa respuesta trae un bloque ``bet_updates`` con la cuota decimal de cada equipo en la
casa asociada (Thunderpick). Es el número que la página muestra en violeta como «lo que te devuelve la
casa por $100»: cuota 1,84 → $184. Este script guarda una fila por serie con esas dos cuotas.

Uso (desde la raíz del repositorio):
    python app/bajar_cuotas.py                         # el mes que termina en la última serie del Silver
    python app/bajar_cuotas.py --desde 2026-09-14 --hasta 2026-10-16

Sale en datos/cuotas_bo3_<desde>_<hasta>.csv. Se pide un día de más a cada lado porque Bo3.gg agrupa
por fecha de inicio y la medianoche UTC parte algunos torneos.
"""
import argparse
import time
from datetime import date, timedelta

import pandas as pd
import requests

import cs2_modelo as cm

API = 'https://api.bo3.gg/api/v2/matches/finished'
CABECERAS = {'User-Agent': 'Mozilla/5.0 (UTN FRM - Ciencia de Datos, trabajo académico)',
             'Origin': 'https://bo3.gg', 'Referer': 'https://bo3.gg/matches/finished'}


def series_del_dia(dia):
    r = requests.get(API, params={'date': dia.isoformat(), 'utc_offset': 0, 'filter[discipline_id][eq]': 1},
                     headers=CABECERAS, timeout=30)
    r.raise_for_status()
    for grupo in r.json()['data']['tiers'].values():
        for m in grupo['matches']:
            b = m.get('bet_updates') or {}
            t1, t2 = b.get('team_1') or {}, b.get('team_2') or {}
            yield {'proveedor': b.get('provider', ''), 'match_slug': m['slug'], 'start_utc': m['start_date'][:16],
                   'estado': m['status'], 'tier': m['tier'], 'bo_type': m['bo_type'],
                   'team1_id': m['team1_id'], 'team2_id': m['team2_id'], 'winner_team_id': m['winner_team_id'],
                   'team1_score': m['team1_score'], 'team2_score': m['team2_score'],
                   'cuota1_team_id': t1.get('team_id'), 'cuota1': t1.get('coeff'), 'cuota1_max': t1.get('max_coeff'),
                   'cuota1_agreement': t1.get('aggrement_score'),
                   'cuota2_team_id': t2.get('team_id'), 'cuota2': t2.get('coeff'), 'cuota2_max': t2.get('max_coeff'),
                   'cuota2_agreement': t2.get('aggrement_score')}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--desde')
    ap.add_argument('--hasta')
    a = ap.parse_args()
    if a.hasta:
        hasta = date.fromisoformat(a.hasta)
    else:
        ultima = pd.read_csv(cm.ruta_csv(), usecols=['start_at_utc']).start_at_utc
        hasta = pd.to_datetime(ultima, format='mixed', utc=True).max().date() + timedelta(days=1)
    desde = date.fromisoformat(a.desde) if a.desde else hasta - timedelta(days=32)
    filas, dia = [], desde
    while dia <= hasta:
        filas += list(series_del_dia(dia))
        print(f'{dia} · {len(filas)} series acumuladas')
        dia += timedelta(days=1)
        time.sleep(1)
    salida = cm.RAIZ_REPO / 'datos' / f'cuotas_bo3_{desde:%Y%m%d}_{hasta:%Y%m%d}.csv'
    pd.DataFrame(filas).drop_duplicates('match_slug').to_csv(salida, index=False)
    print(f'Listo: {salida}')


if __name__ == '__main__':
    main()
