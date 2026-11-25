"""Train + evaluate Decision Tree, Naive Bayes, KNN and ANN. Saves models, metrics and figures.
Run:  python train.py
"""
import json, time, warnings, joblib
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve)
from sklearn.inspection import permutation_importance
from src_common import FEATURES, load_and_clean

RS = 42
df, n_raw = load_and_clean()
print(f'Raw rows {n_raw:,} -> clean rows {len(df):,}')
X, y = df[FEATURES], df['cardio']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
cv = StratifiedKFold(5, shuffle=True, random_state=RS)

def pipe(clf): return Pipeline([('sc', StandardScaler()), ('clf', clf)])
models = {
 'Decision Tree': (pipe(DecisionTreeClassifier(random_state=RS)),
    {'clf__max_depth':[3,5,7,10], 'clf__min_samples_leaf':[20,50,100], 'clf__criterion':['gini','entropy']}),
 'Naive Bayes': (pipe(GaussianNB()), {'clf__var_smoothing':[1e-9,1e-7,1e-5,1e-3]}),
 'KNN': (pipe(KNeighborsClassifier(n_jobs=-1)), {'clf__n_neighbors':[15,31,51,75], 'clf__weights':['uniform','distance']}),
 'ANN (MLP)': (pipe(MLPClassifier(early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
                                  max_iter=300, random_state=RS)),
    {'clf__hidden_layer_sizes':[(32,),(64,32)], 'clf__alpha':[1e-4,1e-2]}),
}
scoring = {'accuracy':'accuracy','precision':'precision','recall':'recall','f1':'f1','roc_auc':'roc_auc'}
best, out = {}, {}
for name, (p, grid) in models.items():
    t0 = time.time()
    gs = GridSearchCV(p, grid, cv=cv, scoring='f1', refit=True).fit(X_tr, y_tr)
    est = gs.best_estimator_
    cvr = cross_validate(est, X_tr, y_tr, cv=cv, scoring=scoring)
    t1 = time.time(); est.fit(X_tr, y_tr); train_t = time.time() - t1
    t1 = time.time(); pred = est.predict(X_te); pred_t = time.time() - t1
    proba = est.predict_proba(X_te)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    best[name] = est
    out[name] = {
        'best_params': {k.replace('clf__',''): (list(v) if isinstance(v, tuple) else v) for k, v in gs.best_params_.items()},
        'cv': {m: [float(cvr['test_'+m].mean()), float(cvr['test_'+m].std())] for m in scoring},
        'test': {'accuracy': accuracy_score(y_te, pred), 'precision': precision_score(y_te, pred),
                 'recall': recall_score(y_te, pred), 'f1': f1_score(y_te, pred), 'roc_auc': roc_auc_score(y_te, proba)},
        'confusion': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'train_time_s': train_t, 'predict_time_s': pred_t,
        'roc': {k: v.tolist() for k, v in zip(['fpr','tpr'], roc_curve(y_te, proba)[:2])}}
    out[name]['roc'] = {k: v[::max(1, len(v)//200)] for k, v in out[name]['roc'].items()}
    print(f"{name:<14} CV-F1={out[name]['cv']['f1'][0]:.4f}  test: acc={out[name]['test']['accuracy']:.4f} "
          f"f1={out[name]['test']['f1']:.4f} auc={out[name]['test']['roc_auc']:.4f}  ({time.time()-t0:.0f}s)")
    joblib.dump(est, f"models/{name.split()[0].lower()}.joblib")

# permutation importance (model-agnostic, test F1 drop)
perm = {}
for name, est in best.items():
    r = permutation_importance(est, X_te, y_te, scoring='f1', n_repeats=5, random_state=RS)
    perm[name] = dict(zip(FEATURES, map(float, r.importances_mean)))
out_meta = {'n_raw': n_raw, 'n_clean': len(df), 'n_train': len(X_tr), 'n_test': len(X_te),
            'best_model': max(out, key=lambda k: out[k]['cv']['f1'][0]), 'features': FEATURES,
            'tree_importance': dict(zip(FEATURES, map(float, best['Decision Tree'].named_steps['clf'].feature_importances_)))}
json.dump({'models': out, 'permutation_importance': perm, 'meta': out_meta}, open('models/metrics.json', 'w'), indent=2)
joblib.dump(best[out_meta['best_model']], 'models/best_model.joblib')
print('Best model:', out_meta['best_model'])

# figures
names = list(out)
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for a, n in zip(ax, names):
    c = out[n]['confusion']; m = np.array([[c['tn'], c['fp']], [c['fn'], c['tp']]])
    a.imshow(m, cmap='Blues')
    for i in range(2):
        for j in range(2): a.text(j, i, m[i, j], ha='center', va='center', fontsize=13)
    a.set_xticks([0,1], ['No CVD','CVD']); a.set_yticks([0,1], ['No CVD','CVD']); a.set_title(n)
    a.set_xlabel('Predicted'); a.set_ylabel('Actual')
plt.tight_layout(); plt.savefig('reports/figures/confusion_matrices.png', dpi=130); plt.close()
plt.figure(figsize=(6, 5))
for n in names: plt.plot(out[n]['roc']['fpr'], out[n]['roc']['tpr'], label=f"{n} (AUC={out[n]['test']['roc_auc']:.3f})")
plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC curves')
plt.tight_layout(); plt.savefig('reports/figures/roc_curves.png', dpi=130); plt.close()
pd.DataFrame(perm).sort_values('Decision Tree').plot(kind='barh', figsize=(9, 6), width=.8)
plt.xlabel('Drop in F1 when shuffled'); plt.title('Permutation importance'); plt.tight_layout()
plt.savefig('reports/figures/permutation_importance.png', dpi=130); plt.close()
print('Saved models/, metrics.json and figures.')
