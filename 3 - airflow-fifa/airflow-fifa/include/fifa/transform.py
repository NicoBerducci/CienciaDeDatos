"""Del HTML crudo de sofifa al esquema canónico de la cátedra."""
from fifa import schema
from fifa.sofifa import cm, iso_date, kg, money, num, texto

# data-col de sofifa -> columna del esquema, para los enteros directos.
ENTEROS_DIRECTOS = {
    "ae": "age", "oa": "overall", "pt": "potential", "gu": "growth",
    "bo": "best_overall", "wk": "weak_foot", "sk": "skill_moves",
    "ir": "international_reputation", "cj": "club_jersey_number",
    "tc": "playstyles_count",
    "pac": "pace", "sho": "shooting", "pas": "passing", "dri": "dribbling",
    "def": "defending", "phy": "physic",
    "ta": "total_attacking", "ts": "total_skill", "to": "total_movement",
    "tp": "total_power", "te": "total_mentality", "td": "total_defending",
    "tg": "total_goalkeeping", "tt": "total_stats", "bs": "base_stats",
    "cr": "attacking_crossing", "fi": "attacking_finishing",
    "he": "attacking_heading_accuracy", "sh": "attacking_short_passing",
    "vo": "attacking_volleys",
    "dr": "skill_dribbling", "cu": "skill_curve", "fr": "skill_fk_accuracy",
    "lo": "skill_long_passing", "bl": "skill_ball_control",
    "ac": "movement_acceleration", "sp": "movement_sprint_speed",
    "ag": "movement_agility", "re": "movement_reactions", "ba": "movement_balance",
    "so": "power_shot_power", "ju": "power_jumping", "st": "power_stamina",
    "sr": "power_strength", "ln": "power_long_shots",
    "ar": "mentality_aggression", "in": "mentality_interceptions",
    "po": "mentality_positioning", "vi": "mentality_vision",
    "pe": "mentality_penalties", "cm": "mentality_composure",
    "ma": "defending_marking_awareness", "sa": "defending_standing_tackle",
    "sl": "defending_sliding_tackle",
    "gd": "goalkeeping_diving", "gh": "goalkeeping_handling",
    "gc": "goalkeeping_kicking", "gp": "goalkeeping_positioning",
    "gr": "goalkeeping_reflexes",
}

TEXTOS_DIRECTOS = {
    "pf": "preferred_foot", "bt": "body_type", "hc": "real_face",
    "bp": "best_position", "cp": "club_position", "at": "acceleration_type",
}


def face_url(player_id):
    """El CDN sirve la foto con el id partido en dos grupos de tres."""
    s = f"{int(player_id):06d}"
    return f"https://cdn.sofifa.net/players/{s[:3]}/{s[3:6]}/26_120.png"


def playstyles(raw):
    """Los dorados primero, con ' +', como los muestra sofifa."""
    dorados = raw.get("_ps2") or []
    normales = raw.get("_ps1") or []
    out = [t[:-1].strip().title() + " +" if t.endswith("+") else t.title()
           for t in list(dorados) + list(normales)]
    return ", ".join(out) if out else None


def contract_year(contrato):
    """'2022 ~ 2034' -> 2034 ; 'Jun 30, 2027 On Loan' -> 2027."""
    import re
    años = re.findall(r"((?:19|20)\d{2})", contrato or "")
    return int(años[-1]) if años else None


def to_row(raw, meta, league):
    """raw: fila cruda del parser. meta: snapshot. league: dict de la liga."""
    r = {c: None for c in schema.COLUMNS}

    r["player_id"] = raw["player_id"]
    r["short_name"] = raw["short_name"]
    r["long_name"] = raw["long_name"]
    r["player_url"] = raw["player_url"]
    r["nationality_id"] = raw.get("nationality_id")
    r["nationality_name"] = raw.get("nationality_name")
    r["player_positions"] = raw.get("player_positions")

    r["fifa_version"] = meta.get("fifa_version")
    r["fifa_update"] = meta.get("fifa_update")
    r["fifa_update_date"] = meta.get("fifa_update_date")

    # La liga NO se infiere: es el filtro con el que pedimos la página.
    r["league_id"] = league["league_id"]
    r["league_name"] = league["league_name"]
    r["league_country"] = league.get("league_country")

    r["club_team_id"] = raw.get("club_team_id")
    r["club_name"] = raw.get("club_name")
    r["club_joined_date"] = iso_date(raw.get("jt"))
    r["club_loan_end_date"] = iso_date(raw.get("le"))
    r["club_contract_valid_until_year"] = contract_year(raw.get("_contrato"))

    for code, col in ENTEROS_DIRECTOS.items():
        r[col] = num(raw.get(code))
    for code, col in TEXTOS_DIRECTOS.items():
        r[col] = texto(raw.get(code))

    r["height_cm"] = cm(raw.get("hi"))
    r["weight_kg"] = kg(raw.get("wi"))
    r["value_eur"] = money(raw.get("vl"))
    r["wage_eur"] = money(raw.get("wg"))
    r["release_clause_eur"] = money(raw.get("rc"))

    r["player_traits"] = playstyles(raw)
    r["player_face_url"] = face_url(raw["player_id"])
    return r
