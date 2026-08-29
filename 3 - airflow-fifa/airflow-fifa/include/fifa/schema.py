"""
Esquema canónico del dataset FIFA de la cátedra.

Ciencia de Datos - UTN FRM - 2026.

El esquema se define a partir de lo que la fuente publica realmente, no
copiando el de un dataset de terceros. Decisiones tomadas:

  * NO se incluyen los 27 ratings por posición (ls, st, cb, ...). sofifa los
    publica sólo en la ficha del jugador, que exige cuenta. Calcularlos a
    partir de los atributos base sería inferirlos, no observarlos.
  * NO se incluye dob: el listado sólo expone el año de nacimiento. Se usa
    age, que sí viene explícito.
  * NO se incluyen columnas que la fuente devuelve siempre vacías
    (work_rate, datos de selección nacional).
  * SÍ se incluyen columnas que el listado publica y que son útiles para las
    unidades siguientes: best_position (objetivo multiclase para U3), growth,
    los totales por grupo de atributos y los playstyles.

Unidad de observación: un jugador en un snapshot. Clave: player_id.
"""

# --- bloques del esquema, en el orden en que salen al CSV -------------------

IDENTIDAD = [
    "player_id", "short_name", "long_name", "player_url",
    "nationality_id", "nationality_name",
]

SNAPSHOT = [
    "fifa_version", "fifa_update", "fifa_update_date",
]

CLUB = [
    "league_id", "league_name", "league_country",
    "club_team_id", "club_name", "club_position", "club_jersey_number",
    "club_joined_date", "club_contract_valid_until_year", "club_loan_end_date",
]

PERFIL = [
    "age", "height_cm", "weight_kg", "preferred_foot",
    "weak_foot", "skill_moves", "international_reputation",
    "body_type", "real_face",
]

VALORACION = [
    "overall", "potential", "growth", "best_overall",
    "player_positions", "best_position",
]

ECONOMICO = [
    "value_eur", "wage_eur", "release_clause_eur",
]

AGREGADOS = [
    "pace", "shooting", "passing", "dribbling", "defending", "physic",
]

TOTALES = [
    "total_attacking", "total_skill", "total_movement", "total_power",
    "total_mentality", "total_defending", "total_goalkeeping",
    "total_stats", "base_stats",
]

ATRIBUTOS = [
    "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
    "attacking_short_passing", "attacking_volleys",
    "skill_dribbling", "skill_curve", "skill_fk_accuracy",
    "skill_long_passing", "skill_ball_control",
    "movement_acceleration", "movement_sprint_speed", "movement_agility",
    "movement_reactions", "movement_balance",
    "power_shot_power", "power_jumping", "power_stamina",
    "power_strength", "power_long_shots",
    "mentality_aggression", "mentality_interceptions", "mentality_positioning",
    "mentality_vision", "mentality_penalties", "mentality_composure",
    "defending_marking_awareness", "defending_standing_tackle",
    "defending_sliding_tackle",
    "goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking",
    "goalkeeping_positioning", "goalkeeping_reflexes",
]

ESTILO = [
    "player_traits", "playstyles_count", "acceleration_type", "player_face_url",
]

COLUMNS = (IDENTIDAD + SNAPSHOT + CLUB + PERFIL + VALORACION + ECONOMICO
           + AGREGADOS + TOTALES + ATRIBUTOS + ESTILO)

# --- tipos, para validar y para castear al final ---------------------------

ENTEROS = (["player_id", "nationality_id", "fifa_version", "fifa_update",
            "league_id", "club_team_id", "club_jersey_number",
            "club_contract_valid_until_year", "age", "height_cm", "weight_kg",
            "weak_foot", "skill_moves", "international_reputation",
            "overall", "potential", "growth", "best_overall",
            "value_eur", "wage_eur", "release_clause_eur", "playstyles_count"]
           + AGREGADOS + TOTALES + ATRIBUTOS)

TEXTO = ["short_name", "long_name", "player_url", "nationality_name",
         "fifa_update_date", "league_name", "league_country",
         "club_name", "club_position",
         "club_joined_date", "club_loan_end_date", "preferred_foot",
         "body_type", "real_face", "player_positions", "best_position",
         "player_traits", "acceleration_type", "player_face_url"]

# Columnas sin las cuales la fila no sirve.
OBLIGATORIAS = ["player_id", "short_name", "overall", "potential", "age",
                "best_position", "league_id"]
