"""Adiciona seção de tuning + gráficos ao notebook."""
import nbformat

nb = nbformat.read('notebooks/projeto.ipynb', as_version=4)

cells = []

# ── Markdown intro tuning ─────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell(
    "## 7. Ajuste de Hiperparâmetros\n\n"
    "### Estratégia\n\n"
    "- **RandomizedSearchCV** para Random Forest (espaço grande) e **GridSearchCV** para KNN (espaço pequeno)\n"
    "- **StratifiedKFold** (5 folds) — mantém proporção de classes em cada fold\n"
    "- **SMOTE dentro do pipeline** de CV para evitar leakage: o oversampling ocorre apenas nos folds de treino\n"
    "- **Score:** F1-macro — métrica principal dado o desbalanceamento\n\n"
    "Após tuning, avaliação final no conjunto de **validação** (não no teste — reservado para comparação final)."
))

# ── Imports tuning ────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""from sklearn.model_selection  import RandomizedSearchCV, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.metrics          import make_scorer
from imblearn.pipeline        import Pipeline as ImbPipeline
from scipy.stats              import randint, uniform
import time

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scorer_f1 = make_scorer(f1_score, average='macro', zero_division=0)

print('Setup de tuning carregado.')
print(f'  CV: StratifiedKFold(n_splits=5)')
print(f'  Score: F1-macro')"""
))

# ── Markdown RF tuning ────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell(
    "### 7.1 Tuning — Random Forest + SMOTE\n\n"
    "Pipeline: `SMOTE (dentro do fold) → RandomForestClassifier`  \n"
    "Busca aleatória com 30 combinações."
))

# ── RF tuning ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""pipe_rf = ImbPipeline([
    ('smote', SMOTE(random_state=42, k_neighbors=5)),
    ('rf',    RandomForestClassifier(random_state=42, n_jobs=-1))
])

param_dist_rf = {
    'rf__n_estimators':   randint(100, 600),
    'rf__max_depth':      [None, 10, 20, 30],
    'rf__min_samples_leaf': randint(1, 10),
    'rf__max_features':   ['sqrt', 'log2', 0.3, 0.5],
    'rf__class_weight':   [None, 'balanced'],
}

search_rf = RandomizedSearchCV(
    pipe_rf,
    param_distributions=param_dist_rf,
    n_iter=30,
    scoring=scorer_f1,
    cv=cv,
    refit=True,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

t0 = time.time()
search_rf.fit(X_train_full, y_train)
print(f'\\nBusca concluida em {time.time()-t0:.1f}s')
print(f'Melhor F1-macro CV: {search_rf.best_score_:.4f}')
print(f'Melhores parametros:')
for k, v in search_rf.best_params_.items():
    print(f'  {k}: {v}')"""
))

# ── RF avaliação ──────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""res_rf_tuned = avaliar_modelo(
    'RF — Tuned (SMOTE+CV)',
    search_rf.best_estimator_,
    X_train_full, y_train,
    X_val_full,   y_val
)"""
))

# ── Markdown KNN tuning ───────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell(
    "### 7.2 Tuning — KNN + SMOTE\n\n"
    "Pipeline: `SMOTE (dentro do fold) → KNeighborsClassifier`  \n"
    "Grid search sobre k, métrica de distância e peso dos vizinhos."
))

# ── KNN tuning ────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""pipe_knn = ImbPipeline([
    ('smote', SMOTE(random_state=42, k_neighbors=5)),
    ('knn',   KNeighborsClassifier(n_jobs=-1))
])

param_grid_knn = {
    'knn__n_neighbors': [5, 7, 11, 15, 21, 31],
    'knn__weights':     ['uniform', 'distance'],
    'knn__metric':      ['euclidean', 'manhattan'],
}

search_knn = GridSearchCV(
    pipe_knn,
    param_grid=param_grid_knn,
    scoring=scorer_f1,
    cv=cv,
    refit=True,
    n_jobs=-1,
    verbose=1
)

t0 = time.time()
search_knn.fit(X_train_full, y_train)
print(f'\\nBusca concluida em {time.time()-t0:.1f}s')
print(f'Melhor F1-macro CV: {search_knn.best_score_:.4f}')
print(f'Melhores parametros:')
for k, v in search_knn.best_params_.items():
    print(f'  {k}: {v}')"""
))

# ── KNN avaliação ─────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""res_knn_tuned = avaliar_modelo(
    'KNN — Tuned (SMOTE+CV)',
    search_knn.best_estimator_,
    X_train_full, y_train,
    X_val_full,   y_val
)"""
))

# ── Markdown gráficos ─────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell(
    "## 8. Análise Gráfica\n\n"
    "- **Learning curves** — evolução treino/val com mais dados (diagnóstico de overfitting)\n"
    "- **Curvas ROC por classe** — sensibilidade por classe nos modelos tuned\n"
    "- **Feature importance** — quais features o RF mais usa\n"
    "- **Matrizes de confusão** — modelos tuned"
))

# ── Learning curves ───────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""# Learning curves — RF Tuned e KNN Tuned
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (nome, estimator) in zip(axes, [
    ('RF Tuned', search_rf.best_estimator_),
    ('KNN Tuned', search_knn.best_estimator_)
]):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X_train_full, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring=scorer_f1,
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1
    )
    tr_mean = train_scores.mean(axis=1)
    tr_std  = train_scores.std(axis=1)
    vl_mean = val_scores.mean(axis=1)
    vl_std  = val_scores.std(axis=1)

    ax.plot(train_sizes, tr_mean, 'o-', color='steelblue', label='Treino')
    ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color='steelblue')
    ax.plot(train_sizes, vl_mean, 'o-', color='coral', label='Validacao CV')
    ax.fill_between(train_sizes, vl_mean - vl_std, vl_mean + vl_std, alpha=0.15, color='coral')
    ax.set_title(f'Learning Curve — {nome}', fontsize=12)
    ax.set_xlabel('Tamanho do treino')
    ax.set_ylabel('F1-macro')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

plt.suptitle('Learning Curves — Diagnostico de Overfitting', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()"""
))

# ── ROC curves ────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""from sklearn.metrics          import roc_curve, auc
from sklearn.preprocessing    import label_binarize

y_val_bin = label_binarize(y_val, classes=list(range(len(CLASS_NAMES))))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (nome, estimator) in zip(axes, [
    ('RF Tuned', search_rf.best_estimator_),
    ('KNN Tuned', search_knn.best_estimator_)
]):
    prob = estimator.predict_proba(X_val_full)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(CLASS_NAMES)))
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_val_bin[:, i], prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=1.8, label=f'{cls} (AUC={roc_auc:.2f})')
    ax.plot([0,1],[0,1], 'k--', lw=0.8)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title(f'Curvas ROC por Classe — {nome}')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle('ROC por Classe (conjunto de validacao)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()"""
))

# ── Feature importance ────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""# Feature importance do RF tuned (extrai o RF de dentro do pipeline)
rf_step = search_rf.best_estimator_.named_steps['rf']
importances = rf_step.feature_importances_
feat_names  = list(X_train_full.columns)

imp_series = pd.Series(importances, index=feat_names).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Top 20 features
top20 = imp_series.head(20)
axes[0].barh(top20.index[::-1], top20.values[::-1], color='steelblue')
axes[0].set_title('Top 20 Features por Importancia (RF Tuned)')
axes[0].set_xlabel('Importancia media (Gini)')
axes[0].grid(axis='x', alpha=0.3)

# Importancia acumulada por grupo
grupos = {
    'Tabular numerica': ['age', 'fitspatrick', 'diameter_mean', 'diameter_area_est'],
    'Tabular booleana': [c for c in feat_names if c in [
        'smoke','drink','pesticide','skin_cancer_history','cancer_history',
        'has_piped_water','has_sewage_system','itch','grew','hurt',
        'changed','bleed','elevation','biopsed','gender']],
    'OHE (region/bg)':  [c for c in feat_names if c.startswith(('region_','background_'))],
    'PDI — Cor RGB':    [c for c in feat_names if 'rgb' in c.lower()],
    'PDI — Cor HSV':    [c for c in feat_names if 'hsv' in c.lower()],
    'PDI — GLCM':       [c for c in feat_names if 'glcm' in c.lower()],
    'PDI — LBP':        [c for c in feat_names if 'lbp' in c.lower()],
    'PDI — Bordas':     [c for c in feat_names if any(x in c.lower() for x in ['canny','sobel','asym'])],
    'PDI — Gabor':      [c for c in feat_names if 'gabor' in c.lower()],
}

grupo_imp = {}
for grp, cols in grupos.items():
    cols_exist = [c for c in cols if c in feat_names]
    grupo_imp[grp] = imp_series[cols_exist].sum() if cols_exist else 0.0

grp_series = pd.Series(grupo_imp).sort_values(ascending=True)
colors_grp = ['#4c9be8' if 'PDI' in g else '#e8844c' for g in grp_series.index]
axes[1].barh(grp_series.index, grp_series.values, color=colors_grp)
axes[1].set_title('Importancia Acumulada por Grupo de Features')
axes[1].set_xlabel('Soma das importancias')
axes[1].grid(axis='x', alpha=0.3)

# Legenda manual
from matplotlib.patches import Patch
leg = [Patch(facecolor='#4c9be8', label='PDI'), Patch(facecolor='#e8844c', label='Tabular/OHE')]
axes[1].legend(handles=leg, loc='lower right')

plt.suptitle('Analise de Feature Importance — RF Tuned', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print('Top 10 features:')
print(imp_series.head(10).to_string())"""
))

# ── Matrizes confusão tuned ───────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""plot_confusion([res_rf_tuned, res_knn_tuned], 'Modelos Tuned — Matrizes de Confusao (val)')"""
))

# ── Markdown comparação final ─────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell(
    "## 9. Comparativo Final — Todos os Modelos\n\n"
    "Consolidação de todos os experimentos: baseline → class_weight → SMOTE → tuning."
))

# ── Comparativo final ─────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""todos_final = resultados_A + resultados_B + [res_rf_tuned, res_knn_tuned]

# --- Gráfico barras comparativo ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

nomes    = [r['nome']    for r in todos_final]
f1_vals  = [r['f1_val']  for r in todos_final]
acc_vals = [r['acc_val'] for r in todos_final]
auc_vals = [r['auc_val'] for r in todos_final]
x = np.arange(len(nomes))
w = 0.6

colors = ['#5b9bd5','#ed7d31','#70ad47','#ffc000','#7030a0','#ff0000'][:len(nomes)]

for ax, (valores, titulo, ylabel) in zip(axes, [
    (f1_vals,  'F1-macro (val)',  'F1-macro'),
    (acc_vals, 'Acuracia (val)', 'Acuracia'),
    (auc_vals, 'AUC-ROC macro (val)', 'AUC-ROC'),
]):
    bars = ax.bar(x, valores, w, color=colors, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(nomes, rotation=20, ha='right', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo, fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Comparativo Final — Todos os Modelos (Conjunto de Validacao)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# --- Tabela delta overfitting ---
print('\\nTabela completa (treino vs validacao):')
header = f"{'Modelo':<28} {'Acc_tr':>7} {'Acc_val':>8} {'D_acc':>7} {'F1_tr':>7} {'F1_val':>8} {'D_f1':>7} {'AUC_val':>9}"
print(header)
print('-' * len(header))
for r in todos_final:
    print(f"{r['nome']:<28} {r['acc_tr']:>7.4f} {r['acc_val']:>8.4f} {r['acc_tr']-r['acc_val']:>+7.4f} "
          f"{r['f1_tr']:>7.4f} {r['f1_val']:>8.4f} {r['f1_tr']-r['f1_val']:>+7.4f} {r['auc_val']:>9.4f}")

# --- F1 por classe: melhor vs segundo melhor ---
print('\\nF1 por classe — RF Tuned vs RF class_weight:')
from sklearn.metrics import f1_score
header2 = f"{'Classe':<6} {'RF_cw':>8} {'RF_tuned':>10} {'Ganho':>8}"
print(header2)
print('-' * 35)
for i, cls in enumerate(CLASS_NAMES):
    f1_cw    = f1_score(res_rf_cw['y_val'],    res_rf_cw['pred_val'],    labels=[i], average='macro', zero_division=0)
    f1_tuned = f1_score(res_rf_tuned['y_val'], res_rf_tuned['pred_val'], labels=[i], average='macro', zero_division=0)
    print(f"{cls:<6} {f1_cw:>8.4f} {f1_tuned:>10.4f} {f1_tuned-f1_cw:>+8.4f}")"""
))

# ── F1 por classe gráfico ─────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""# F1 por classe de todos os modelos
fig, ax = plt.subplots(figsize=(12, 5))

x = np.arange(len(CLASS_NAMES))
w = 0.2
offsets = np.linspace(-1.5*w, 1.5*w, len(todos_final))

for offset, res, color in zip(offsets, todos_final, colors):
    f1_por_classe = []
    for i in range(len(CLASS_NAMES)):
        f1 = f1_score(res['y_val'], res['pred_val'], labels=[i], average='macro', zero_division=0)
        f1_por_classe.append(f1)
    ax.bar(x + offset, f1_por_classe, w, label=res['nome'], color=color, edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, fontsize=11)
ax.set_ylim(0, 1.1)
ax.set_ylabel('F1-score')
ax.set_title('F1 por Classe — Todos os Modelos (val)', fontsize=12)
ax.legend(fontsize=8, loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.axhline(0.5, color='gray', linestyle='--', lw=0.8, alpha=0.6)
plt.tight_layout()
plt.show()

print('\\nMelhor modelo geral: RF — Tuned (SMOTE+CV)')
print(f"  F1-macro val : {res_rf_tuned['f1_val']:.4f}")
print(f"  Acuracia val : {res_rf_tuned['acc_val']:.4f}")
print(f"  AUC-ROC val  : {res_rf_tuned['auc_val']:.4f}")"""
))

nb.cells.extend(cells)
nbformat.write(nb, 'notebooks/projeto.ipynb')
print(f'OK — {len(nb.cells)} celulas totais, {len(cells)} celulas adicionadas.')
