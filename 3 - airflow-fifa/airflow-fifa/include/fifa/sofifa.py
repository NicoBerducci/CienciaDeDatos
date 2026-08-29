"""
Cliente de sofifa.com para el dataset canónico de la materia.

Ciencia de Datos - UTN FRM - 2026.

Dos rutas de descarga, en orden de preferencia:

  1. HTTP con urllib (rápida, ~0,5 s por página).
  2. Playwright + Chromium (lenta, pero pasa si Cloudflare endurece el filtro).

Por qué urllib y no requests: Cloudflare bloquea con 403 a los clientes
construidos sobre urllib3 -- requests entre ellos -- por la huella del
handshake TLS. La biblioteca estándar pasa. Verificado el 18/08/2026 desde
dos redes distintas. No es un capricho: cambiar urllib por requests rompe
el pipeline.
"""
import re
import time
import urllib.error
import urllib.request

from bs4 import BeautifulSoup

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
BASE = "https://sofifa.com/players"
PAGE_SIZE = 60

# Los 80 códigos que sofifa acepta en showCol[]. Salen del <select> de la
# propia página, así que la lista es exhaustiva.
COL_CODES = [
    "pi", "ae", "by", "hi", "wi", "pf", "oa", "pt", "bo", "bp", "gu", "jt",
    "le", "vl", "wg", "rc",
    "ta", "cr", "fi", "he", "sh", "vo",
    "ts", "dr", "cu", "fr", "lo", "bl",
    "to", "ac", "sp", "ag", "re", "ba",
    "tp", "so", "ju", "st", "sr", "ln",
    "te", "ar", "in", "po", "vi", "pe", "cm",
    "td", "ma", "sa", "sl",
    "tg", "gd", "gh", "gc", "gp", "gr",
    "tt", "bs", "wk", "sk", "ir", "bt", "hc",
    "pac", "sho", "pas", "dri", "def", "phy",
    "ps1", "ps2", "tc", "at", "cp", "cj",
]
SHOW_COLS = "&".join("showCol%5B%5D=" + c for c in COL_CODES)


def page_url(league_id=None, offset=0, roster=None):
    """URL de una página del listado.

    roster fija el snapshot (ej. 260046 = FC 26, actualización 46). Pasarlo
    hace que la corrida sea reproducible: el mismo DAG en noviembre produce
    el mismo dataset que en agosto.
    """
    parts = [f"{BASE}?type=all", "col=oa", "sort=desc", SHOW_COLS,
             f"offset={offset}"]
    if league_id is not None:
        parts.append(f"lg%5B%5D={league_id}")
    if roster is not None:
        parts.append(f"r={roster}&set=true")
    return "&".join(parts)


# --------------------------------------------------------------- descarga

class EndOfLeague(Exception):
    """El offset pasó el final de la liga. No es un error."""


def fetch_http(url, timeout=30, retries=3):
    for intento in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if "signIn" in r.geturl():
                    raise RuntimeError("sofifa exige login: la vista pública cambió")
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise EndOfLeague()
            if e.code == 403:
                raise PermissionError("sofifa devolvió 403 (Cloudflare)")
            if intento == retries - 1:
                raise
            time.sleep(2 * (intento + 1))
        except Exception:
            if intento == retries - 1:
                raise
            time.sleep(2 * (intento + 1))


def fetch_browser(url, timeout=45000):
    """Ruta de respaldo: navegador real. Sólo se usa si la HTTP da 403."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage",
                  "--disable-blink-features=AutomationControlled"])
        ctx = browser.new_context(user_agent=UA,
                                  viewport={"width": 1920, "height": 1080})
        page = ctx.new_page()
        # no bajamos imágenes ni tipografías: sólo necesitamos la tabla
        page.route("**/*", lambda r: r.abort()
                   if r.request.resource_type in ("image", "stylesheet", "font", "media")
                   else r.continue_())
        resp = page.goto(url, wait_until="domcontentloaded", timeout=timeout)
        page.wait_for_timeout(1500)
        html = page.content()
        status = resp.status if resp else None
        browser.close()
    if status == 404:
        raise EndOfLeague()
    if any(k in html for k in ("Just a moment", "Checking your browser",
                               "cf-browser-verification")):
        raise RuntimeError("Cloudflare frenó también al navegador")
    return html


def fetch(url, engine="http"):
    """engine: 'http' | 'browser' | 'auto' (http y si da 403, navegador)."""
    if engine == "http":
        return fetch_http(url)
    if engine == "browser":
        return fetch_browser(url)
    try:
        return fetch_http(url)
    except PermissionError:
        return fetch_browser(url)


# ----------------------------------------------------------------- parseo

def snapshot_meta(html):
    """Versión, número de actualización y fecha del snapshot."""
    meta = {"fifa_version": None, "fifa_update": None, "fifa_update_date": None,
            "roster": None}
    m = re.search(r"<title>[^<]*?FC\s*(\d+)\s*-\s*([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})",
                  html)
    if m:
        meta["fifa_version"] = int(m.group(1))
        meta["fifa_update_date"] = iso_date(m.group(2))
    r = re.search(r"[?&]r=(\d{2})(\d{4})&set=true", html)
    if r:
        meta["fifa_version"] = meta["fifa_version"] or int(r.group(1))
        meta["fifa_update"] = int(r.group(2))
        meta["roster"] = int(r.group(1) + r.group(2))
    return meta


def parse_page(html):
    """Filas crudas: los <td data-col> más lo que vive en el marcado."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if table is None or table.find("tbody") is None:
        return []
    filas = []
    for tr in table.find("tbody").find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
        celda_id = tds[1]
        a = celda_id.find("a", href=re.compile(r"/player/\d+"))
        if a is None:
            continue
        rec = {
            "player_id": int(re.search(r"/player/(\d+)", a["href"]).group(1)),
            "player_url": "https://sofifa.com" + a["href"],
            "short_name": a.get_text(strip=True),
            "long_name": a.get("data-tippy-content") or a.get_text(strip=True),
        }
        bandera = celda_id.find("img", class_="flag")
        rec["nationality_name"] = bandera.get("title") if bandera else None
        na = celda_id.find("a", href=re.compile(r"na=\d+"))
        rec["nationality_id"] = int(re.search(r"na=(\d+)", na["href"]).group(1)) if na else None
        # sólo las posiciones de esta celda: la columna 'Best position' repite el span
        rec["player_positions"] = ", ".join(s.get_text(strip=True)
                                            for s in celda_id.select("span.pos")) or None

        team = tr.find("a", href=re.compile(r"/team/\d+"))
        if team is not None:
            rec["club_name"] = team.get_text(strip=True)
            rec["club_team_id"] = int(re.search(r"/team/(\d+)", team["href"]).group(1))
            sub = team.find_parent("td").find("div", class_="sub")
            rec["_contrato"] = sub.get_text(" ", strip=True) if sub else ""
        else:
            rec["club_name"] = rec["club_team_id"] = None
            rec["_contrato"] = ""

        for td in tr.find_all("td", attrs={"data-col": True}):
            code = td["data-col"]
            if code in ("ps1", "ps2"):
                rec["_" + code] = [sp.get_text(strip=True)
                                   for sp in td.find_all("span") if sp.get_text(strip=True)]
            rec[code] = td.get_text(" ", strip=True)
        filas.append(rec)
    return filas


# ------------------------------------------------------------ conversiones

_MESES = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def iso_date(txt):
    """'Jul 1, 2022' -> '2022-07-01'."""
    if not txt or txt in ("N/A", "-"):
        return None
    m = re.match(r"([A-Z][a-z]{2})\s+(\d{1,2}),\s+(\d{4})", txt.strip())
    return f"{m.group(3)}-{_MESES[m.group(1)]:02d}-{int(m.group(2)):02d}" if m else None


def money(txt):
    """'€172.5M' -> 172500000 ; '€390K' -> 390000."""
    if not txt or txt in ("N/A", "-"):
        return None
    m = re.match(r"[^\d.]*([\d.]+)\s*([MK]?)", txt.strip())
    if not m:
        return None
    return int(float(m.group(1)) * {"M": 1_000_000, "K": 1_000, "": 1}[m.group(2)])


def num(txt):
    if txt is None:
        return None
    m = re.match(r"\s*(-?\d+)", str(txt))
    return int(m.group(1)) if m else None


def cm(txt):
    m = re.match(r"\s*(\d+)cm", str(txt or ""))
    return int(m.group(1)) if m else None


def kg(txt):
    m = re.match(r"\s*(\d+)kg", str(txt or ""))
    return int(m.group(1)) if m else None


def texto(txt):
    t = (txt or "").strip()
    return None if t in ("", "N/A", "-") else t
