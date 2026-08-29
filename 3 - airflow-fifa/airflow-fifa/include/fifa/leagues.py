"""
Catálogo de ligas, leído de sofifa.com/leagues.

Esa página es pública y trae, por liga: id, nombre, país y -- lo más útil --
cuántos jugadores tiene. Con ese número el DAG sabe exactamente cuántas
páginas pedir por liga, en vez de tantear hasta recibir un 404.

Que el catálogo salga de la fuente y no de una tabla prestada es lo que
sostiene la afirmación de que el dataset se reconstruye desde sofifa.

El catálogo se pide para el MISMO snapshot que el scraping. No es un detalle:
las ligas cambian entre actualizaciones. La del 07/08/2025 tenía 51 ligas y
18.035 jugadores; la del 23/07/2026 tiene 52 y 18.936. Pedir el catálogo del
snapshot más nuevo para scrapear uno viejo daría conteos equivocados, y hasta
ligas que en ese momento no existían.
"""
import re

from bs4 import BeautifulSoup

from fifa.sofifa import PAGE_SIZE, fetch

LEAGUES_URL = "https://sofifa.com/leagues"


def catalog_url(roster=None):
    """URL del catálogo, opcionalmente fijada a un snapshot."""
    if roster is None:
        return LEAGUES_URL
    return f"{LEAGUES_URL}?r={roster}&set=true"


def fetch_catalog(engine="http", roster=None):
    """[{league_id, league_name, league_country, n_teams, n_players, n_pages}]

    roster fija el snapshot, igual que en el listado de jugadores. Tiene que
    ser el mismo que se le pasa a page_url(), o el catálogo y los datos
    quedan desalineados.
    """
    soup = BeautifulSoup(fetch(catalog_url(roster), engine=engine), "lxml")
    table = soup.find("table")
    if table is None:
        raise RuntimeError("sofifa/leagues cambió de estructura: no hay tabla")

    catalogo = []
    for tr in table.find("tbody").find_all("tr"):
        a = tr.find("a", href=re.compile(r"/league/\d+"))
        if a is None:
            continue
        tds = tr.find_all("td")
        bandera = tr.find("img", class_="flag")
        n_teams = _entero(tds[2]) if len(tds) > 2 else None
        n_players = _entero(tds[3]) if len(tds) > 3 else None
        if not n_players:
            continue                      # liga vacía: no hay nada que bajar
        catalogo.append({
            "league_id": int(re.search(r"/league/(\d+)", a["href"]).group(1)),
            "league_name": a.get_text(strip=True),
            "league_country": bandera.get("title") if bandera else None,
            "n_teams": n_teams,
            "n_players": n_players,
            "n_pages": -(-n_players // PAGE_SIZE),   # división entera hacia arriba
        })
    if not catalogo:
        raise RuntimeError("sofifa/leagues no devolvió ninguna liga")
    return catalogo


def _entero(td):
    m = re.search(r"\d+", td.get_text(" ", strip=True))
    return int(m.group(0)) if m else None
