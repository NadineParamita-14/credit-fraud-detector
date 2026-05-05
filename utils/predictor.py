# utils/predictor.py

import joblib
import numpy as np
import pandas as pd

# ── Load artifacts ───────────────────────────────────────────
MODEL     = joblib.load('models/fraud_detection_model.pkl')
SCALER    = joblib.load('models/robust_scaler.pkl')
THRESHOLD = joblib.load('models/optimal_threshold.pkl')
FEATURES  = joblib.load('models/feature_columns.pkl')

SCALE_FEATURES = [
    'log_amount', 'amount_zscore',
    'hour_of_day', 'time_diff_log', 'v_fraud_composite'
]

# Nilai asli dari training set
AMOUNT_MEAN = 88.34961925093133
AMOUNT_STD  = 250.1201092402221

V_CORR_WEIGHTS = {
    'V17': -0.32648106724371434,
    'V14': -0.30254369580440440,
    'V12': -0.26059292487721686,
    'V10': -0.21688294364102725,
    'V16': -0.19653894030401792,
    'V3':  -0.19296082706741322,
    'V7':  -0.18725659151429797,
    'V11':  0.15487564474394730,
    'V4':   0.13344748623900432,
    'V18': -0.11148525388904092,
    'V1':  -0.10134729859508507,
    'V9':  -0.09773268607407870,
    'V5':  -0.09497429899144809,
    'V2':   0.09128865034461915,
}


def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Amount features
    df['log_amount']      = np.log1p(df['Amount'])
    df['amount_zscore']   = (df['Amount'] - AMOUNT_MEAN) / AMOUNT_STD
    df['is_small_amount'] = (df['Amount'] <= 10).astype(int)
    df['is_large_amount'] = (df['Amount'] >= 1000).astype(int)

    # Temporal features
    df['hour_of_day']   = (df['Time'] / 3600) % 24
    df['hour_sin']      = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos']      = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['is_night']      = ((df['hour_of_day'] >= 0) &
                           (df['hour_of_day'] < 6)).astype(int)
    df['day_of_week']   = (df['Time'] // (3600 * 24)) % 7
    df['is_weekend']    = (df['day_of_week'] >= 5).astype(int)
    df['time_diff']     = df['Time'].diff().fillna(0)
    df['time_diff_log'] = np.log1p(df['time_diff'])

    # Interaction features
    df['large_amount_at_night'] = (
        (df['is_large_amount'] == 1) & (df['is_night'] == 1)
    ).astype(int)
    df['small_amount_at_night'] = (
        (df['is_small_amount'] == 1) & (df['is_night'] == 1)
    ).astype(int)

    # V composite score
    df['v_fraud_composite'] = sum(
        df[feat] * weight
        for feat, weight in V_CORR_WEIGHTS.items()
        if feat in df.columns
    )

    return df


def scale_features(df_engineered: pd.DataFrame) -> pd.DataFrame:
    df = df_engineered.copy()
    cols = [c for c in SCALE_FEATURES if c in df.columns]
    df[cols] = SCALER.transform(df[cols])
    return df


def predict_single(raw_input: dict) -> dict:
    df_raw    = pd.DataFrame([raw_input])
    df_eng    = engineer_features(df_raw)
    df_scaled = scale_features(df_eng)

    X    = df_scaled[FEATURES]
    prob = float(MODEL.predict_proba(X)[:, 1][0])
    pred = int(prob >= THRESHOLD)

    return {
        'prediction'      : pred,
        'label'           : 'FRAUD' if pred == 1 else 'NORMAL',
        'probability'     : round(prob, 4),
        'risk_score'      : int(prob * 100),
        'threshold_used'  : float(THRESHOLD),
    }


def predict_batch(records: list[dict]) -> list[dict]:
    df_raw    = pd.DataFrame(records)
    df_eng    = engineer_features(df_raw)
    df_scaled = scale_features(df_eng)

    X     = df_scaled[FEATURES]
    probs = MODEL.predict_proba(X)[:, 1]
    preds = (probs >= THRESHOLD).astype(int)

    results = []
    for i, (prob, pred) in enumerate(zip(probs, preds)):
        results.append({
            'index'       : i,
            'prediction'  : int(pred),
            'label'       : 'FRAUD' if pred == 1 else 'NORMAL',
            'probability' : round(float(prob), 4),
            'risk_score'  : int(prob * 100),
        })

    return results