"""Proceso de calibración, búsqueda de umbral óptimo y simulación de Kelly fraccionado.

Usa las funciones y particiones de app/cs2_modelo.py (sin fuga temporal) y las cuotas
pre-partido de Pinnacle (datos/cuotas_oddspapi_pinnacle_*.csv).

Uso:
    python app/evaluar_umbral.py
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

import cs2_modelo as cm


def ece_score(y_true, y_prob, n_bins=10):
    """Expected Calibration Error (ECE) en 10 tramos uniformes."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        if i == n_bins - 1:
            m = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        else:
            m = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if m.any():
            ece += (m.sum() / n) * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def calibrar_oof_y_test(train, test, cv):
    """Genera probabilidades OOF (validación) y Test para 4 variantes sin fuga temporal:
    1. Logística actual (C=0.01, balanced) sin calibrar
    2. Logística sin balanceo (C=0.01, class_weight=None)
    3. Logística actual + Platt Scaling (calibrada en sub-split temporal interno)
    4. Logística actual + Isotónica (calibrada en sub-split temporal interno)
    """
    X_tr, y_tr = train[cm.COLUMNAS_FINAL], train.target_team_a_won
    X_te = test[cm.COLUMNAS_FINAL]

    pipe_bal = cm.armar(LogisticRegression(max_iter=5000, random_state=cm.SEMILLA, C=0.01, class_weight='balanced'),
                        cm.COLUMNAS_FINAL)
    pipe_none = cm.armar(LogisticRegression(max_iter=5000, random_state=cm.SEMILLA, C=0.01, class_weight=None),
                         cm.COLUMNAS_FINAL)

    oof = {k: pd.Series(np.nan, index=X_tr.index) for k in ('bal', 'none', 'platt', 'iso')}
    fold_s = pd.Series(0, index=X_tr.index)

    for k, (e, v) in enumerate(cv, 1):
        fold_s.iloc[v] = k
        # 1 y 2: entrenamiento directo en e, predicción en v
        m_bal = clone(pipe_bal).fit(X_tr.iloc[e], y_tr.iloc[e])
        m_none = clone(pipe_none).fit(X_tr.iloc[e], y_tr.iloc[e])
        oof['bal'].iloc[v] = m_bal.predict_proba(X_tr.iloc[v])[:, 1]
        oof['none'].iloc[v] = m_none.predict_proba(X_tr.iloc[v])[:, 1]

        # 3 y 4: calibración respetando el orden temporal (80% base, 20% calibración)
        corte = int(len(e) * 0.80)
        e_base, e_cal = e[:corte], e[corte:]
        m_sub = clone(pipe_bal).fit(X_tr.iloc[e_base], y_tr.iloc[e_base])
        p_cal = m_sub.predict_proba(X_tr.iloc[e_cal])[:, 1]
        p_val_sub = m_sub.predict_proba(X_tr.iloc[v])[:, 1]

        platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p_cal).reshape(-1, 1), y_tr.iloc[e_cal])
        oof['platt'].iloc[v] = platt.predict_proba(logit(p_val_sub).reshape(-1, 1))[:, 1]

        iso = IsotonicRegression(out_of_bounds='clip').fit(p_cal, y_tr.iloc[e_cal])
        oof['iso'].iloc[v] = iso.predict(p_val_sub)

    # Modelos finales para el Test (Platt e Isotónica ajustados sobre las predicciones OOF de validación)
    m_bal_f = clone(pipe_bal).fit(X_tr, y_tr)
    m_none_f = clone(pipe_none).fit(X_tr, y_tr)
    p_te_bal = m_bal_f.predict_proba(X_te)[:, 1]
    p_te_none = m_none_f.predict_proba(X_te)[:, 1]

    val_mask = oof['bal'].notna()
    platt_f = LogisticRegression(C=1e6, max_iter=1000).fit(
        logit(oof['bal'][val_mask]).reshape(-1, 1), y_tr[val_mask]
    )
    iso_f = IsotonicRegression(out_of_bounds='clip').fit(oof['bal'][val_mask], y_tr[val_mask])

    p_test = {
        'bal': p_te_bal,
        'none': p_te_none,
        'platt': platt_f.predict_proba(logit(p_te_bal).reshape(-1, 1))[:, 1],
        'iso': iso_f.predict(p_te_bal),
    }
    return oof, fold_s, p_test, platt_f


def simular_estrategia(df_pin, col_prob, tau=0.50, delta_ev=-9.0, kelly_frac=None,
                       banca_ini=10000.0, tope_banca=0.05, apuesta_fija=100.0):
    """Simula apuestas sobre df_pin ordenado cronológicamente."""
    d = df_pin.sort_values('start_at_utc')
    banca = banca_ini
    pico = banca_ini
    max_dd = 0.0
    apostado = 0.0
    neto = 0.0
    n_ap = 0
    aciertos = 0
    cuotas_usadas = []

    for r in d.itertuples(index=False):
        pa = getattr(r, col_prob)
        pb = 1.0 - pa
        # Elegir el lado que favorece el modelo (si hay filtro EV, evaluar ambos lados o solo favorito)
        if pa >= 0.5:
            p_pick, c_pick, gano = pa, r.cuota_a, (r.target_team_a_won == 1)
        else:
            p_pick, c_pick, gano = pb, r.cuota_b, (r.target_team_a_won == 0)

        ev = p_pick * c_pick - 1.0
        if p_pick < tau or ev < delta_ev:
            continue

        if kelly_frac is not None:
            if ev <= 0 or c_pick <= 1.0:
                continue
            f_kelly = (p_pick * c_pick - 1.0) / (c_pick - 1.0)
            frac = min(max(0.0, kelly_frac * f_kelly), tope_banca)
            stake = banca * frac
            if stake < 1.0:
                continue
        else:
            stake = apuesta_fija

        res = stake * (c_pick - 1.0) if gano else -stake
        banca += res
        pico = max(pico, banca)
        max_dd = max(max_dd, (pico - banca) / pico)
        apostado += stake
        neto += res
        n_ap += 1
        aciertos += int(gano)
        cuotas_usadas.append(c_pick)

    roi = (neto / apostado) if apostado > 0 else np.nan
    acc = (aciertos / n_ap) if n_ap > 0 else np.nan
    return {
        'n': n_ap,
        'cobertura': n_ap / len(d) if len(d) else 0.0,
        'acierto': acc,
        'cuota_med': float(np.median(cuotas_usadas)) if cuotas_usadas else np.nan,
        'apostado': apostado,
        'neto': neto,
        'roi': roi,
        'max_dd': max_dd,
        'banca_fin': banca,
    }


def main():
    df = cm.cargar()
    train, test, _ = cm.separar(df)
    cv, _ = cm.construir_cv_temporal(train)

    oof, fold_s, p_test, platt_f = calibrar_oof_y_test(train, test, cv)
    val_m = oof['bal'].notna()
    y_val = train.loc[val_m, 'target_team_a_won']
    y_te = test['target_team_a_won']

    print("================================================================================")
    print("1. PROCESO DE CALIBRACIÓN (VALIDACIÓN TEMPORAL OOF: 29.705 series · 5 folds)")
    print("================================================================================")
    nombres = {
        'bal': 'Logística actual (balanced)',
        'none': 'Logística sin balanceo (None)',
        'platt': 'Logística + Platt Scaling',
        'iso': 'Logística + Isotónica',
    }
    print(f"Frecuencia real de A en validación: {y_val.mean():.4f} | en test: {y_te.mean():.4f}\n")
    rows_cal = []
    for k, nom in nombres.items():
        pv = oof[k][val_m]
        pt = p_test[k]
        rows_cal.append({
            'modelo': nom,
            'acc_val': accuracy_score(y_val, pv >= 0.5),
            'brier_val': brier_score_loss(y_val, pv),
            'logloss_val': log_loss(y_val, pv),
            'ece_val': ece_score(y_val, pv),
            'mediaA_val': pv.mean(),
            'brier_test': brier_score_loss(y_te, pt),
            'ece_test': ece_score(y_te, pt),
        })
    df_cal = pd.DataFrame(rows_cal)
    print(df_cal.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nParámetros Platt final sobre OOF: a = {platt_f.coef_[0][0]:.4f}, b = {platt_f.intercept_[0]:+.4f}")

    # Elegir mejor calibración por Log-Loss en validación
    mejor_k = min(nombres.keys(), key=lambda k: log_loss(y_val, oof[k][val_m]))
    print(f"--> Mejor calibración en validación (menor Log-Loss): {nombres[mejor_k]} ({mejor_k})\n")

    print("================================================================================")
    print(f"2. UMBRAL DE CONFIANZA EN VALIDACIÓN OOF (29.705 series) · usando [{mejor_k}]")
    print("================================================================================")
    p_val_best = oof[mejor_k][val_m].to_numpy()
    y_val_arr = y_val.to_numpy()
    folds_arr = fold_s[val_m].to_numpy()
    conf_val = np.maximum(p_val_best, 1.0 - p_val_best)
    acierto_val = ((p_val_best >= 0.5).astype(int) == y_val_arr)

    rows_u = []
    for tau in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        m = conf_val >= tau
        accs_f = [acierto_val[m & (folds_arr == f)].mean() for f in range(1, cm.N_SPLITS + 1)]
        acc_m = acierto_val[m].mean()
        rows_u.append({
            'umbral_tau': f">= {tau:.0%}",
            'n_series': int(m.sum()),
            'cobertura': m.mean(),
            'acierto_oof': acc_m,
            'desvio_folds': float(np.std(accs_f)),
            'cuota_equil': 1.0 / acc_m,
        })
    print(pd.DataFrame(rows_u).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Cargar cuotas pre-partido de Pinnacle
    rutas = sorted((cm.RAIZ_REPO / 'datos').glob('cuotas_oddspapi_pinnacle_*.csv'))
    assert rutas, "Falta el CSV de Pinnacle en datos/"
    cuotas = pd.read_csv(rutas[-1])

    df_pin = test.copy()
    for k in nombres:
        df_pin[f'p_{k}'] = p_test[k]
    df_pin = df_pin.merge(cuotas[['match_slug', 'cuota_a', 'cuota_b', 'margen_casa']], on='match_slug', how='inner')
    df_pin = df_pin.sort_values('start_at_utc').reset_index(drop=True)
    col_p = f'p_{mejor_k}'

    print("\n================================================================================")
    print(f"3. BACKTEST CON CUOTAS PRE-PARTIDO DE PINNACLE ({len(df_pin)} series Tier S/A) · [{mejor_k}]")
    print("================================================================================")

    print("\n3A) Solo umbral de confianza (apuesta fija $100):")
    r_tau = []
    for tau in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        s = simular_estrategia(df_pin, col_p, tau=tau)
        r_tau.append({'umbral_tau': f">= {tau:.0%}", **s})
    print(pd.DataFrame(r_tau)[['umbral_tau', 'n', 'cobertura', 'acierto', 'cuota_med', 'neto', 'roi']].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n3B) Umbral de Valor Esperado EV = p*cuota - 1 (apuesta fija $100):")
    r_ev = []
    for delta in [-0.05, 0.00, 0.02, 0.05, 0.08, 0.10, 0.15]:
        s = simular_estrategia(df_pin, col_p, tau=0.50, delta_ev=delta)
        r_ev.append({'ev_min': f">= {delta:+.0%}", **s})
    print(pd.DataFrame(r_ev)[['ev_min', 'n', 'cobertura', 'acierto', 'cuota_med', 'neto', 'roi']].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n3C) Doble umbral (Confianza >= tau AND EV >= delta) + Kelly Fraccionado (0.25x, banca $10.000, tope 5%):")
    r_combo = []
    for tau in [0.50, 0.55, 0.60, 0.65, 0.70]:
        for delta in [0.00, 0.03, 0.05, 0.08]:
            sf = simular_estrategia(df_pin, col_p, tau=tau, delta_ev=delta)
            sk = simular_estrategia(df_pin, col_p, tau=tau, delta_ev=delta, kelly_frac=0.25)
            r_combo.append({
                'regla': f"p>={tau:.0%} & EV>={delta:.0%}",
                'n': sf['n'],
                'acc': sf['acierto'],
                'neto_fijo100': sf['neto'],
                'roi_fijo': sf['roi'],
                'neto_kelly25': sk['neto'],
                'roi_kelly': sk['roi'],
                'max_dd_kelly': sk['max_dd'],
            })
    print(pd.DataFrame(r_combo).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # 4. Corte temporal interno en Pinnacle: Mes 1 (validación de regla) vs Mes 2 (prueba de regla)
    corte_pin = pd.Timestamp('2026-09-16', tz='UTC')
    m1 = df_pin[df_pin.start_at_utc < corte_pin]
    m2 = df_pin[df_pin.start_at_utc >= corte_pin]
    print("\n================================================================================")
    print(f"4. VALIDACIÓN TEMPORAL DE UMBRAL EN PINNACLE: Mes 1 ({len(m1)} series) vs Mes 2 ({len(m2)} series)")
    print("================================================================================")
    r_split = []
    for label, tau, delta, kf in [
        ("Siempre ($100)", 0.50, -9.0, None),
        ("p>=60% ($100)", 0.60, -9.0, None),
        ("p>=65% ($100)", 0.65, -9.0, None),
        ("p>=70% ($100)", 0.70, -9.0, None),
        ("EV>=0% ($100)", 0.50, 0.00, None),
        ("EV>=5% ($100)", 0.50, 0.05, None),
        ("p>=60% & EV>=0% ($100)", 0.60, 0.00, None),
        ("p>=60% & EV>=5% ($100)", 0.60, 0.05, None),
        ("p>=60% & EV>=0% (Kelly 0.25x)", 0.60, 0.00, 0.25),
        ("EV>=0% (Kelly 0.25x)", 0.50, 0.00, 0.25),
        ("EV>=5% (Kelly 0.25x)", 0.50, 0.05, 0.25),
    ]:
        s1 = simular_estrategia(m1, col_p, tau=tau, delta_ev=delta, kelly_frac=kf)
        s2 = simular_estrategia(m2, col_p, tau=tau, delta_ev=delta, kelly_frac=kf)
        r_split.append({
            'estrategia': label,
            'M1_n': s1['n'], 'M1_acc': s1['acierto'], 'M1_roi': s1['roi'],
            'M2_n': s2['n'], 'M2_acc': s2['acierto'], 'M2_roi': s2['roi'],
        })
    print(pd.DataFrame(r_split).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()