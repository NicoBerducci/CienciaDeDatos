"""Baja cuotas históricas pre-partido (y pre-veto) de Pinnacle desde OddsPapi v4
para las series CS2 de Tier S y A del Test (datos/matches_latest.csv).

Guarda caché local en datos/oddspapi_cache/ para no repetir requests.

Uso (desde la raíz del repositorio, con $env:ODDSPAPI_KEY seteada):
    python app/bajar_cuotas_oddspapi.py --limite 1 --debug   # verifica 1 partido (gasta pocos requests)
    python app/bajar_cuotas_oddspapi.py --limite 200         # descarga hasta 200 partidos Tier S/A
    python app/bajar_cuotas_oddspapi.py --solo-parsear       # regenera el CSV solo desde la caché local
"""
import argparse
import json
import os
import re
import time
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from rapidfuzz import fuzz

import cs2_modelo as cm

BASE = "https://api.oddspapi.io/v4"
SPORT_ID = 17  # ESport Counter-Strike
BOOKMAKER = "pinnacle"

ALIAS = {
    "navi": "natus vincere",
    "nip": "ninjas in pyjamas",
    "vp": "virtus pro",
    "virtuspro": "virtus pro",
    "bb": "betboom",
    "mousesports": "mouz",
    "the mongolz": "mongolz",
    "col": "complexity",
    "the huns": "huns",
}
SUFIJOS = (" esports", " esport", " gaming", " team", " clan", " club", " cs2", " cs", " gg")
TAGS_FILIAL = {"academy", "junior", "jr", "nxt", "fe", "female", "ares", "rising", "talent", "women"}


def normalizar(nombre: str) -> str:
    if not isinstance(nombre, str):
        return ""
    s = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower()).strip()
    s = re.sub(r"\s+", " ", s)
    for suf in SUFIJOS:
        if s.endswith(suf) and len(s) > len(suf) + 1:
            s = s[: -len(suf)].strip()
    if s.startswith("team ") and len(s) > 6:
        s = s[5:].strip()
    return ALIAS.get(s, s)


def similitud(a: str, b: str) -> float:
    na, nb = normalizar(a), normalizar(b)
    if not na or not nb:
        return 0.0
    if (set(na.split()) & TAGS_FILIAL) != (set(nb.split()) & TAGS_FILIAL):
        return 0.0
    return 100.0 if na == nb else float(fuzz.token_sort_ratio(na, nb))


def get_json_cache(cache_file: Path, url: str, params: dict, solo_cache: bool = False):
    if cache_file.is_file():
        return json.loads(cache_file.read_text(encoding="utf-8")), False
    if solo_cache:
        return None, False
    for intento in range(4):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 429:
            espera = 2.0
            try:
                espera = r.json().get("error", {}).get("retryMs", 1500) / 1000.0 + 0.5
            except Exception:
                pass
            time.sleep(espera)
            continue
        if not r.ok:
            print("Error API:", r.status_code, r.text)
        r.raise_for_status()
        data = r.json()
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        time.sleep(1.3)
        return data, True
    raise RuntimeError(f"Demasiados 429 en {url}")


def bajar_fixtures(cache_dir: Path, api_key: str, desde: date, hasta: date, solo_cache: bool = False):
    """Baja fixtures de CS en bloques de 15 días (4-5 requests para 2 meses)."""
    todos = []
    cur = desde
    usados = 0
    while cur <= hasta:
        fin = min(cur + timedelta(days=6), hasta)
        f_cache = cache_dir / f"fixtures_{cur:%Y%m%d}_{fin:%Y%m%d}.json"
        data, hizo_req = get_json_cache(
            f_cache,
            f"{BASE}/fixtures",
            {
                "apiKey": api_key,
                "sportId": SPORT_ID,
                "from": f"{cur.isoformat()}T00:00:00Z",
                "to": f"{fin.isoformat()}T23:59:59Z",
            },
            solo_cache=solo_cache,
        )
        if hizo_req:
            usados += 1
            time.sleep(0.3)
        if isinstance(data, list):
            todos.extend(data)
        cur = fin + timedelta(days=1)

    unicos = {f["fixtureId"]: f for f in todos if isinstance(f, dict) and "fixtureId" in f}
    print(f"Fixtures cargados: {len(unicos)} (requests usados en fixtures: {usados})")
    return list(unicos.values())


def cruzar_fixtures(test_sub: pd.DataFrame, fixtures: list[dict]):
    """Empareja cada serie del Test con su fixtureId en OddsPapi."""
    por_fecha = {}
    for f in fixtures:
        st = f.get("startTime")
        if not st:
            continue
        dt = pd.to_datetime(st, utc=True)
        f["_dt"] = dt
        por_fecha.setdefault(dt.date(), []).append(f)

    usados = set()
    cruces = []
    for row in test_sub.itertuples(index=False):
        d0 = row.start_at_utc.date()
        cands = []
        for delta in (0, -1, 1):
            cands.extend(por_fecha.get(d0 + timedelta(days=delta), []))

        mejor_sc, mejor = -1.0, None
        for f in cands:
            fid = f["fixtureId"]
            if fid in usados:
                continue
            diff_h = abs((f["_dt"] - row.start_at_utc).total_seconds()) / 3600.0
            if diff_h > 30:
                continue

            p1, p2 = f.get("participant1Name", ""), f.get("participant2Name", "")
            for inv in (False, True):
                ta, tb = (p2, p1) if inv else (p1, p2)
                sa, sb = similitud(row.team_a_name, ta), similitud(row.team_b_name, tb)
                if min(sa, sb) < 84.0:
                    continue
                sc = (sa + sb) / 2.0 - diff_h * 0.5
                if sc > mejor_sc:
                    mejor_sc = sc
                    mejor = (fid, f, inv, round(sa, 1), round(sb, 1))

        if mejor:
            fid, f, inv, sa, sb = mejor
            usados.add(fid)
            cruces.append((row, f, inv, sa, sb))
    return cruces


def ultimo_precio_antes(entries: list[dict], limite_utc: pd.Timestamp):
    """Devuelve (precio, createdAt) del último registro activo estrictamente anterior a limite_utc."""
    validos = []
    for e in entries:
        if not isinstance(e, dict) or e.get("price") is None:
            continue
        if e.get("active") is False:
            continue
        ts = pd.to_datetime(e.get("createdAt"), utc=True, errors="coerce")
        if pd.notna(ts) and ts < limite_utc:
            validos.append((ts, float(e["price"]), e.get("createdAt")))
    if not validos:
        return None, None
    validos.sort(key=lambda x: x[0])
    _, precio, ts_str = validos[-1]
    return precio, ts_str


def extraer_cuotas_prepartido(data: dict, start_utc: pd.Timestamp, debug: bool = False):
    """Extrae cuota de cierre pre-partido (< start_utc) y pre-veto (< start_utc - 2h)
    del mercado Moneyline (ganador de serie, 2 outcomes).
    """
    bms = data.get("bookmakers") or {}
    bm_data = bms.get(BOOKMAKER) or (next(iter(bms.values())) if bms else {})
    markets = bm_data.get("markets") or {}

    if debug:
        print("\n[DEBUG] Markets disponibles:", list(markets.keys()))
        for mid, mval in list(markets.items())[:5]:
            outs = mval.get("outcomes") or {}
            print(f"  marketId={mid} -> outcomes={list(outs.keys())}")
            for oid, oval in outs.items():
                players = oval.get("players") or {}
                for pid, plist in players.items():
                    if isinstance(plist, list) and plist:
                        print(f"    outcome={oid}, player={pid}, n={len(plist)}, primero={plist[0]}, ultimo={plist[-1]}")

    # En OddsPapi v4 para CS (sportId=17), el Match Winner suele ser el mercado 171 o 101 o 1 con 2 outcomes
    orden_preferido = ["171", "101", "1"]
    candidatos = [m for m in orden_preferido if m in markets] + [m for m in markets if m not in orden_preferido]

    for mid in candidatos:
        outcomes = markets[mid].get("outcomes") or {}
        if len(outcomes) != 2:
            continue
        oids = sorted(outcomes.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
        listas = []
        for oid in oids:
            players = outcomes[oid].get("players") or {}
            # Tomar player "0" (mercado de equipo) si existe, si no el primero
            entries = players.get("0") or (next(iter(players.values())) if players else [])
            if isinstance(entries, list):
                listas.append(entries)
        if len(listas) != 2:
            continue

        c1_close, ts1 = ultimo_precio_antes(listas[0], start_utc)
        c2_close, ts2 = ultimo_precio_antes(listas[1], start_utc)
        if c1_close is None or c2_close is None:
            continue

        c1_preveto, _ = ultimo_precio_antes(listas[0], start_utc - pd.Timedelta(hours=2))
        c2_preveto, _ = ultimo_precio_antes(listas[1], start_utc - pd.Timedelta(hours=2))
        return {
            "market_id": mid,
            "cuota1_close": c1_close,
            "cuota2_close": c2_close,
            "cuota1_preveto": c1_preveto,
            "cuota2_preveto": c2_preveto,
            "ts_cuota": ts1 or ts2,
        }
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--desde", default="2026-08-15", help="Fecha inicial AAAA-MM-DD (default: últimos 2 meses)")
    ap.add_argument("--hasta", default="2026-10-16", help="Fecha final AAAA-MM-DD")
    ap.add_argument("--tiers", default="s,a", help="Tiers a incluir separados por coma (default: s,a)")
    ap.add_argument("--limite", type=int, default=200, help="Máximo de partidos a pedir a /historical-odds")
    ap.add_argument("--debug", action="store_true", help="Imprime estructura de markets del primer partido")
    ap.add_argument("--solo-parsear", action="store_true", help="No hace requests; usa solo la caché local")
    args = ap.parse_args()

    api_key = os.environ.get("ODDSPAPI_KEY", "")
    if not args.solo_parsear and not api_key:
        raise SystemExit("Falta la variable de entorno ODDSPAPI_KEY.")

    desde = date.fromisoformat(args.desde)
    hasta = date.fromisoformat(args.hasta)
    tiers = [t.strip().lower() for t in args.tiers.split(",") if t.strip()]

    df = cm.cargar()
    _, test, _ = cm.separar(df)
    test_sub = test[
        (test.start_at_utc >= pd.Timestamp(desde, tz="UTC"))
        & (test.start_at_utc <= pd.Timestamp(hasta + timedelta(days=1), tz="UTC"))
        & (test.tier.isin(tiers))
    ].copy()
    print(f"Series objetivo en Test ({desde} a {hasta}, tiers={tiers}): {len(test_sub)}")

    cache_dir = cm.RAIZ_REPO / "datos" / "oddspapi_cache"
    fixtures = bajar_fixtures(cache_dir, api_key, desde - timedelta(days=1), hasta + timedelta(days=1), args.solo_parsear)
    cruces = cruzar_fixtures(test_sub, fixtures)
    print(f"Series cruzadas con fixtureId de OddsPapi: {len(cruces)} / {len(test_sub)} ({len(cruces)/max(len(test_sub),1):.1%})")

    filas = []
    reqs_odds = 0
    for idx, (row, fix, inv, sa, sb) in enumerate(cruces[: args.limite], 1):
        fid = fix["fixtureId"]
        f_odds = cache_dir / f"odds_{BOOKMAKER}_{fid}.json"
        try:
            data, hizo_req = get_json_cache(
                f_odds,
                f"{BASE}/historical-odds",
                {"apiKey": api_key, "fixtureId": fid, "bookmaker": BOOKMAKER},
                solo_cache=args.solo_parsear,
            )
            if hizo_req:
                reqs_odds += 1
                time.sleep(0.25)
            if not data:
                continue

            start_utc = pd.to_datetime(fix["startTime"], utc=True)
            res = extraer_cuotas_prepartido(data, start_utc, debug=(args.debug and idx == 1))
            if not res:
                continue

            c_a = res["cuota2_close"] if inv else res["cuota1_close"]
            c_b = res["cuota1_close"] if inv else res["cuota2_close"]
            pv_a = res["cuota2_preveto"] if inv else res["cuota1_preveto"]
            pv_b = res["cuota1_preveto"] if inv else res["cuota2_preveto"]
            margen = (1.0 / c_a) + (1.0 / c_b) - 1.0

            filas.append({
                "proveedor": f"oddspapi_{BOOKMAKER}",
                "fixture_id": fid,
                "market_id": res["market_id"],
                "match_id": int(row.match_id),
                "match_slug": row.match_slug,
                "start_utc": row.start_at_utc.strftime("%Y-%m-%dT%H:%M"),
                "ts_cuota": res["ts_cuota"],
                "tier": row.tier,
                "bo_type": int(row.bo_type),
                "team_a_id": int(row.team_a_id),
                "team_b_id": int(row.team_b_id),
                "team_a_name": row.team_a_name,
                "team_b_name": row.team_b_name,
                "cuota_a": c_a,
                "cuota_b": c_b,
                "cuota_a_preveto": pv_a,
                "cuota_b_preveto": pv_b,
                "margen_casa": round(margen, 4),
                "sim_a": sa,
                "sim_b": sb,
            })
            if idx % 20 == 0 or idx == len(cruces[: args.limite]):
                print(f"  Procesados {idx}/{len(cruces[:args.limite])} · con cuota válida: {len(filas)} · requests nuevos: {reqs_odds}", flush=True)
        except Exception as e:
            print(f"  Error en {fid}: {e}")

    out = cm.RAIZ_REPO / "datos" / f"cuotas_oddspapi_{BOOKMAKER}_{desde:%Y%m%d}_{hasta:%Y%m%d}.csv"
    df_out = pd.DataFrame(filas)
    df_out.to_csv(out, index=False)
    print(f"\nGuardado: {out} ({len(df_out)} series con cuota pre-partido)")
    if len(df_out):
        min_c = df_out[["cuota_a", "cuota_b"]].min(axis=1)
        print(f"Cuotas extremas (<= 1.06): {(min_c <= 1.06).mean():.1%} (antes en Bo3.gg era 56%)")
        print(f"Margen implícito mediano : {df_out.margen_casa.median():.2%}")
        print(f"Con cuota pre-veto (-2h) : {df_out.cuota_a_preveto.notna().mean():.1%}")


if __name__ == "__main__":
    main()