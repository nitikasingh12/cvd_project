"""Streamlit app: CVD risk prediction + model comparison dashboard.
Run:  streamlit run app.py   (after `python train.py`)
"""
import json, os, joblib
import numpy as np, pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from src_common import FEATURES, build_row

st.set_page_config(page_title='CVD Risk Predictor', page_icon='🫀', layout='wide')

@st.cache_resource
def load_all():
    if not os.path.exists('models/metrics.json'):
        return None, None, None
    metrics = json.load(open('models/metrics.json'))
    files = {'Decision Tree': 'decision.joblib', 'Naive Bayes': 'naive.joblib', 'KNN': 'knn.joblib', 'ANN (MLP)': 'ann.joblib'}
    models = {n: joblib.load(f'models/{f}') for n, f in files.items()}
    return metrics, models, metrics['meta']['best_model']

metrics, models, best_name = load_all()
if metrics is None:
    st.error('Models not found. Run `python train.py` first.'); st.stop()

st.title('🫀 AI-Based Cardiovascular Disease Prediction')
st.caption('Decision Tree · Naive Bayes · KNN · Neural Network — educational decision-support prototype, **not a medical device**.')
tab1, tab2, tab3, tab4 = st.tabs(['🩺 Predict', '📊 Model comparison', '🔍 Feature importance', '🏗️ System design'])

# ---------------------------------------------------------------- Predict
with tab1:
    st.subheader('Patient details')
    c1, c2, c3 = st.columns(3)
    age = c1.slider('Age (years)', 30, 70, 50)
    gender = c1.radio('Gender', ['Female', 'Male'], horizontal=True)
    height = c1.number_input('Height (cm)', 120, 220, 168)
    weight = c1.number_input('Weight (kg)', 30.0, 200.0, 75.0)
    ap_hi = c2.number_input('Systolic BP (mmHg)', 70, 250, 120)
    ap_lo = c2.number_input('Diastolic BP (mmHg)', 40, 150, 80)
    chol = c2.selectbox('Cholesterol', [1, 2, 3], format_func=lambda v: {1: 'Normal', 2: 'Above normal', 3: 'Well above normal'}[v])
    gluc = c3.selectbox('Glucose', [1, 2, 3], format_func=lambda v: {1: 'Normal', 2: 'Above normal', 3: 'Well above normal'}[v])
    smoke = c3.checkbox('Smoker'); alco = c3.checkbox('Alcohol intake'); active = c3.checkbox('Physically active', value=True)
    model_choice = st.selectbox('Model', [f'{best_name} (best)'] + [m for m in models if m != best_name])
    chosen = best_name if model_choice.endswith('(best)') else model_choice

    if st.button('Predict risk', type='primary'):
        try:
            row = build_row(age, gender, height, weight, ap_hi, ap_lo, chol, gluc, int(smoke), int(alco), int(active))
        except ValueError as e:
            st.error(str(e)); st.stop()
        p = float(models[chosen].predict_proba(row)[0, 1])
        tier = 'LOW' if p < 0.35 else 'MODERATE' if p < 0.65 else 'HIGH'
        color = {'LOW': 'green', 'MODERATE': 'orange', 'HIGH': 'red'}[tier]
        a, b = st.columns([1, 2])
        a.metric('Probability of CVD', f'{p:.1%}')
        a.markdown(f'### :{color}[{tier} RISK]')
        a.write({'LOW': 'Routine check-up and healthy-lifestyle advice.',
                 'MODERATE': 'Follow-up blood-pressure / lipid screening recommended.',
                 'HIGH': 'Refer for clinical assessment.'}[tier])
        b.write(f'**BMI:** {row.bmi.iloc[0]:.1f} &nbsp; **Pulse pressure:** {row.pulse_pressure.iloc[0]} mmHg')
        agree = pd.DataFrame({'Model': list(models), 'P(CVD)': [float(m.predict_proba(row)[0, 1]) for m in models.values()]})
        agree['Prediction'] = np.where(agree['P(CVD)'] >= 0.5, 'CVD', 'No CVD')
        b.dataframe(agree.style.format({'P(CVD)': '{:.1%}'}), hide_index=True, width='stretch')
        st.caption('Risk tiers are illustrative. Real thresholds must be set with clinicians and validated prospectively.')

# ---------------------------------------------------------------- Comparison
with tab2:
    m = metrics['models']
    rows = [{'Model': n, 'Accuracy': v['test']['accuracy'], 'Precision': v['test']['precision'], 'Recall': v['test']['recall'],
             'F1-Score': v['test']['f1'], 'ROC-AUC': v['test']['roc_auc'],
             'CV F1 (mean±std)': f"{v['cv']['f1'][0]:.4f} ± {v['cv']['f1'][1]:.4f}",
             'Train (s)': v['train_time_s'], 'Predict (s)': v['predict_time_s']} for n, v in m.items()]
    tbl = pd.DataFrame(rows).set_index('Model')
    st.subheader('Held-out test-set performance')
    st.dataframe(tbl.style.format({c: '{:.4f}' for c in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']} |
                                  {'Train (s)': '{:.2f}', 'Predict (s)': '{:.2f}'})
                 .background_gradient(cmap='Greens', subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']),
                 width='stretch')
    st.success(f"Best model by cross-validated F1: **{best_name}**")
    st.bar_chart(tbl[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**Confusion matrices**')
        fig, ax = plt.subplots(1, 4, figsize=(14, 3.3))
        for a, (n, v) in zip(ax, m.items()):
            c = v['confusion']; mat = np.array([[c['tn'], c['fp']], [c['fn'], c['tp']]])
            a.imshow(mat, cmap='Blues'); a.set_title(n, fontsize=9)
            for i in range(2):
                for j in range(2): a.text(j, i, mat[i, j], ha='center', va='center')
            a.set_xticks([0, 1], ['No', 'CVD']); a.set_yticks([0, 1], ['No', 'CVD'])
        plt.tight_layout(); st.pyplot(fig)
    with c2:
        st.markdown('**ROC curves**')
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for n, v in m.items(): ax.plot(v['roc']['fpr'], v['roc']['tpr'], label=f"{n} ({v['test']['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], 'k--'); ax.legend(fontsize=8); ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        st.pyplot(fig)
    st.markdown('**Best hyper-parameters**'); st.json({n: v['best_params'] for n, v in m.items()})

# ---------------------------------------------------------------- Importance
with tab3:
    st.subheader('Permutation importance (drop in test F1 when a feature is shuffled)')
    perm = pd.DataFrame(metrics['permutation_importance'])
    perm = perm.loc[perm.mean(axis=1).sort_values().index]
    st.bar_chart(perm, horizontal=True)
    st.subheader('Decision-tree (impurity) importance')
    st.bar_chart(pd.Series(metrics['meta']['tree_importance']).sort_values(), horizontal=True)
    st.info('Systolic blood pressure is by far the strongest predictor, followed by cholesterol, diastolic BP / pulse pressure and age. Lifestyle flags (smoking, alcohol, activity) and BMI add little once blood pressure is known. Importance shows association, not causation.')

# ---------------------------------------------------------------- Design
with tab4:
    st.subheader('Conceptual intelligent healthcare prediction system')
    st.graphviz_chart('''digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fb"];
      "Data input\\n(EHR / form / devices)" -> "Validation &\\nplausibility checks" -> "Preprocessing\\n(BMI, pulse pressure, scaling)"
      -> "Best ML model\\n(P(CVD))" -> "Risk tier +\\nexplanation" -> "Clinician review";
      "Clinician review" -> "Monitoring, audit &\\nretraining" [style=dashed]; "Monitoring, audit &\\nretraining" -> "Best ML model\\n(P(CVD))" [style=dashed]; }''')
    st.markdown('''
* **Human-in-the-loop:** output is decision support; a clinician makes the final call.
* **Screening bias:** thresholds should favour recall (missed cases are costlier than false alarms).
* **Governance:** audit log, data privacy (consent / de-identification), drift monitoring and periodic retraining.
* **Limits:** self-reported lifestyle fields, single dataset, no external validation.''')
