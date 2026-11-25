"""Shared preprocessing so training and the Streamlit app use identical logic."""
import numpy as np, pandas as pd

FEATURES = ['age_years','gender','height','weight','bmi','ap_hi','ap_lo','pulse_pressure',
            'cholesterol','gluc','smoke','alco','active']

def load_and_clean(path='data/cardio_train.csv'):
    raw = pd.read_csv(path, sep=';')
    df = raw.drop(columns='id').drop_duplicates().copy()
    df['age_years'] = (df['age'] / 365.25).round(1)
    df = df.drop(columns='age')
    ok = (df['height'].between(120, 220) & df['weight'].between(30, 200) &
          df['ap_hi'].between(70, 250) & df['ap_lo'].between(40, 150) & (df['ap_hi'] > df['ap_lo']))
    df = df[ok].copy()
    df['bmi'] = (df['weight'] / (df['height'] / 100) ** 2).round(2)
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    df = df[df['bmi'].between(12, 60)].copy()
    df['gender'] = (df['gender'] == 2).astype(int)   # 0 female, 1 male
    return df.reset_index(drop=True), len(raw)

def build_row(age_years, gender, height_cm, weight_kg, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active):
    if not (70 <= ap_hi <= 250 and 40 <= ap_lo <= 150 and ap_hi > ap_lo):
        raise ValueError('Implausible blood pressure (systolic must exceed diastolic).')
    if not (120 <= height_cm <= 220 and 30 <= weight_kg <= 200):
        raise ValueError('Implausible height/weight.')
    return pd.DataFrame([{
        'age_years': age_years, 'gender': int(gender == 'Male'), 'height': height_cm, 'weight': weight_kg,
        'bmi': weight_kg / (height_cm / 100) ** 2, 'ap_hi': ap_hi, 'ap_lo': ap_lo, 'pulse_pressure': ap_hi - ap_lo,
        'cholesterol': cholesterol, 'gluc': gluc, 'smoke': smoke, 'alco': alco, 'active': active}])[FEATURES]
