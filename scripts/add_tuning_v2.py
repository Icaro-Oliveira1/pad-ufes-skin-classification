"""Seção de tuning v2: RF regularizado + KNN uniforme + Decision Tree + gráficos + comparativo."""
import nbformat

nb = nbformat.read('notebooks/projeto.ipynb', as_version=4)
cells = []

# ── Markdown intro tuning ─────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell(
    "## 7. Ajuste de Hiperparâmetros\n\n"
    "### Estratégia\n\n"
    "- **Pipeline com SMOTE dentro do CV** — oversampling só nos folds de treino, sem leakage\n"
    "- **StratifiedKFold 5 folds** — mantém proporção de classes\n"
    "- **Score:** F1-macro\n\n"
    "**Modelos:**\n\n"
    "| Modelo | Foco da regularização |\n"
    "|--------|----------------------|\n"
    "| Random Forest | `min_samples_leaf >= 2`, `max_depth` limitado |\n"
    "| KNN | `weights='uniform'`, k maior — evita memorização de pontos sintéticos |\n"
    "| Decision Tree | `max_depth` controlado — modelo simples como referência |\n"
))

# ── Imports ───────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""from sklearn.model_selection  import RandomizedSearchCV, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.tree             import DecisionTreeClassifier
from sklearn.metrics          import make_scorer, roc_curve, auc
from sklearn.preprocessing    import label_binarize
from imblearn.pipeline        import Pipeline as ImbPipeline
from scipy.stats              import randint
import time

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scorer_f1 = make_scorer(f1_score, average='macro', zero_division=0)

print('Setup carregado — StratifiedKFold(5), score=F1-macro')"""
))

# ── Markdown RF ───────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell(
    "### 7.1 Random Forest — Busca com Regularização\n\n"
    "Restrições aplicadas para reduzir overfitting:\n"
    "- `min_samples_leaf` >= 2 (sem folhas com amostra única)\n"
    "- `max_depth` <= 25 (profundidade limitada)\n"
    "- `max_features` em range reduzido (evita memorização)"
))

cells.append(nbformat.v4.new_code_cell(
"""pipe_rf = ImbPipeline([
    ('smote', SMOTE(random_state=42, k_neighbors=5)),
    ('rf',    RandomForestClassifier(random_state=42, n_jobs=-1))
])

param_dist_rf = {
    'rf__n_estimators':     randint(200, 600),
    'rf__max_depth':        [10, 15, 20, 25],
    'rf__min_samples_leaf': randint(2, 12),
    'rf__max_features':     ['sqrt', 'log2', 0.3],
    'rf__class_weight':     [None, 'balanced'],
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
print(f'\\nConcluido em {time.time()-t0:.1f}s')
print(f'Melhor F1-macro CV : {search_rf.best_score_:.4f}')
print('Melhores parametros:')
for k, v in search_rf.best_params_.items():
    print(f'  {k}: {v}')"""
))

cells.append(nbformat.v4.new_code_cell(
"""res_rf_tuned = avaliar_modelo(
    'RF Tuned',
    search_rf.best_estimator_,
    X_train_full, y_train,
    X_val_full,   y_val
)"""
))

# ── Markdown KNN ──────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell(
    "### 7.2 KNN — Regularização por k e peso\n\n"
    "`weights='distance'` com k pequeno memoriza pontos sintéticos do SMOTE.  \n"
    "Aqui forçamos `weights='uniform'` e buscamos k maior para uma fronteira mais suave."
))

cells.append(nbformat.v4.new_code_cell(
"""pipe_knn = ImbPipeline([
    ('smote', SMOTE(random_state=42, k_neighbors=5)),
    ('knn',   KNeighborsClassifier(n_jobs=-1))
])

param_grid_knn = {
    'knn__n_neighbors': [7, 11, 15, 21, 31, 41],
    'knn__weights':     ['uniform'],
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
print(f'\\nConcluido em {time.time()-t0:.1f}s')
print(f'Melhor F1-macro CV : {search_knn.best_score_:.4f}')
print('Melhores parametros:')
for k, v in search_knn.best_params_.items():
    print(f'  {k}: {v}')"""
))

cells.append(nbformat.v4.new_code_cell(
"""res_knn_tuned = avaliar_modelo(
    'KNN Tuned',
    search_knn.best_estimator_,
    X_train_full, y_train,
    X_val_full,   y_val
)"""
))

# ── Markdown DT ───────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell(
    "### 7.3 Decision Tree — Modelo Simples\n\n"
    "Árvore de decisão com `max_depth` controlado: modelo interpretável e menos propenso a overfitting.  \n"
    "Serve como referência — se performar próximo ao RF, o problema não exige alta complexidade."
))

cells.append(nbformat.v4.new_code_cell(
"""pipe_dt = ImbPipeline([
    ('smote', SMOTE(random_state=42, k_neighbors=5)),
    ('dt',    DecisionTreeClassifier(random_state=42))
])

param_grid_dt = {
    'dt__max_depth':        [3, 5, 7, 10, 15],
    'dt__min_samples_leaf': [2, 5, 10, 20],
    'dt__criterion':        ['gini', 'entropy'],
    'dt__class_weight':     [None, 'balanced'],
}

search_dt = GridSearchCV(
    pipe_dt,
    param_grid=param_grid_dt,
    scoring=scorer_f1,
    cv=cv,
    refit=True,
    n_jobs=-1,
    verbose=1
)

t0 = time.time()
search_dt.fit(X_train_full, y_train)
print(f'\\nConcluido em {time.time()-t0:.1f}s')
print(f'Melhor F1-macro CV : {search_dt.best_score_:.4f}')
print('Melhores parametros:')
for k, v in search_dt.best_params_.items():
    print(f'  {k}: {v}')"""
))

cells.append(nbformat.v4.new_code_cell(
"""res_dt_tuned = avaliar_modelo(
    'Decision Tree Tuned',
    search_dt.best_estimator_,
    X_train_full, y_train,
    X_val_full,   y_val
)"""
))

# ── Markdown gráficos ─────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell(
    "## 8. Análise Gráfica\n\n"
    "- **Learning curves** — diagnóstico de overfitting por modelo\n"
    "- **Curvas ROC por classe** — sensibilidade por classe nos melhores modelos\n"
    "- **Feature importance** — RF e DT\n"
    "- **Matrizes de confusão** — modelos tuned"
))

# ── Learning curves ───────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cv_lc = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for ax, (nome, est) in zip(axes, [
    ('RF Tuned',            search_rf.best_estimator_),
    ('KNN Tuned',           search_knn.best_estimator_),
    ('Decision Tree Tuned', search_dt.best_estimator_),
]):
    train_sizes, tr_scores, vl_scores = learning_curve(
        est, X_train_full, y_train,
        cv=cv_lc,
        scoring=scorer_f1,
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1
    )
    tr_m, tr_s = tr_scores.mean(1), tr_scores.std(1)
    vl_m, vl_s = vl_scores.mean(1), vl_scores.std(1)

    ax.plot(train_sizes, tr_m, 'o-', color='steelblue', label='Treino')
    ax.fill_between(train_sizes, tr_m-tr_s, tr_m+tr_s, alpha=0.15, color='steelblue')
    ax.plot(train_sizes, vl_m, 'o-', color='coral',    label='Val CV')
    ax.fill_between(train_sizes, vl_m-vl_s, vl_m+vl_s, alpha=0.15, color='coral')
    gap = tr_m[-1] - vl_m[-1]
    ax.set_title(f'{nome}\\nGap final: {gap:+.3f}', fontsize=11)
    ax.set_xlabel('Amostras de treino')
    ax.set_ylabel('F1-macro')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle('Learning Curves — Diagnostico de Overfitting', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()"""
))

# ── ROC curves ────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""y_val_bin = label_binarize(y_val, classes=list(range(len(CLASS_NAMES))))
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors_cls = plt.cm.tab10(np.linspace(0, 0.9, len(CLASS_NAMES)))

for ax, (nome, est) in zip(axes, [
    ('RF Tuned',            search_rf.best_estimator_),
    ('KNN Tuned',           search_knn.best_estimator_),
    ('Decision Tree Tuned', search_dt.best_estimator_),
]):
    prob = est.predict_proba(X_val_full)
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors_cls)):
        fpr, tpr, _ = roc_curve(y_val_bin[:, i], prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=1.8, label=f'{cls} (AUC={roc_auc:.2f})')
    ax.plot([0,1],[0,1], 'k--', lw=0.8)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title(f'ROC — {nome}', fontsize=11)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle('Curvas ROC por Classe (validacao)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()"""
))

# ── Feature importance ────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""feat_names = list(X_train_full.columns)

grupos = {
    'Tabular numerica':  ['age','fitspatrick','diameter_mean','diameter_area_est'],
    'Tabular booleana':  [c for c in feat_names if c in [
        'smoke','drink','pesticide','skin_cancer_history','cancer_history',
        'has_piped_water','has_sewage_system','itch','grew','hurt',
        'changed','bleed','elevation','biopsed','gender']],
    'OHE (region/bg)':   [c for c in feat_names if c.startswith(('region_','background_'))],
    'PDI — Cor RGB/HSV': [c for c in feat_names if any(x in c.lower() for x in ['rgb','hsv'])],
    'PDI — GLCM':        [c for c in feat_names if 'glcm' in c.lower()],
    'PDI — LBP':         [c for c in feat_names if 'lbp'  in c.lower()],
    'PDI — Bordas':      [c for c in feat_names if any(x in c.lower() for x in ['canny','sobel','asym'])],
    'PDI — Gabor':       [c for c in feat_names if 'gabor' in c.lower()],
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# RF — top 20 features
rf_step = search_rf.best_estimator_.named_steps['rf']
imp_rf = pd.Series(rf_step.feature_importances_, index=feat_names).sort_values(ascending=False)
top20_rf = imp_rf.head(20)
axes[0,0].barh(top20_rf.index[::-1], top20_rf.values[::-1], color='steelblue')
axes[0,0].set_title('RF Tuned — Top 20 Features', fontsize=11)
axes[0,0].set_xlabel('Importancia (Gini)')
axes[0,0].grid(axis='x', alpha=0.3)

# DT — top 20 features
dt_step = search_dt.best_estimator_.named_steps['dt']
imp_dt = pd.Series(dt_step.feature_importances_, index=feat_names).sort_values(ascending=False)
top20_dt = imp_dt.head(20)
axes[0,1].barh(top20_dt.index[::-1], top20_dt.values[::-1], color='#e8844c')
axes[0,1].set_title('Decision Tree Tuned — Top 20 Features', fontsize=11)
axes[0,1].set_xlabel('Importancia (Gini)')
axes[0,1].grid(axis='x', alpha=0.3)

# RF — importancia por grupo
grp_rf = pd.Series({g: imp_rf[[c for c in cols if c in feat_names]].sum()
                    for g, cols in grupos.items()}).sort_values()
colors_grp = ['#4c9be8' if 'PDI' in g else '#e8844c' for g in grp_rf.index]
axes[1,0].barh(grp_rf.index, grp_rf.values, color=colors_grp)
axes[1,0].set_title('RF Tuned — Importancia por Grupo', fontsize=11)
axes[1,0].set_xlabel('Soma das importancias')
axes[1,0].grid(axis='x', alpha=0.3)
from matplotlib.patches import Patch
axes[1,0].legend(handles=[Patch(facecolor='#4c9be8', label='PDI'),
                           Patch(facecolor='#e8844c', label='Tabular/OHE')], loc='lower right')

# DT — importancia por grupo
grp_dt = pd.Series({g: imp_dt[[c for c in cols if c in feat_names]].sum()
                    for g, cols in grupos.items()}).sort_values()
axes[1,1].barh(grp_dt.index, grp_dt.values, color=colors_grp)
axes[1,1].set_title('Decision Tree Tuned — Importancia por Grupo', fontsize=11)
axes[1,1].set_xlabel('Soma das importancias')
axes[1,1].grid(axis='x', alpha=0.3)
axes[1,1].legend(handles=[Patch(facecolor='#4c9be8', label='PDI'),
                           Patch(facecolor='#e8844c', label='Tabular/OHE')], loc='lower right')

plt.suptitle('Feature Importance — RF vs Decision Tree', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print('Top 5 RF:', imp_rf.head(5).to_string())
print()
print('Top 5 DT:', imp_dt.head(5).to_string())"""
))

# ── Matrizes confusão ─────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""plot_confusion(
    [res_rf_tuned, res_knn_tuned, res_dt_tuned],
    'Modelos Tuned — Matrizes de Confusao (val)'
)"""
))

# ── Markdown comparativo final ────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell(
    "## 9. Comparativo Final\n\n"
    "Consolidação de todos os experimentos realizados."
))

# ── Comparativo gráfico ───────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""todos_final = resultados_A + resultados_B + [res_rf_tuned, res_knn_tuned, res_dt_tuned]
cores = ['#5b9bd5','#ed7d31','#70ad47','#ffc000','#7030a0','#c00000','#00b0f0']

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
nomes    = [r['nome']    for r in todos_final]
f1_vals  = [r['f1_val']  for r in todos_final]
acc_vals = [r['acc_val'] for r in todos_final]
auc_vals = [r['auc_val'] for r in todos_final]
x = np.arange(len(nomes))
w = 0.6

for ax, (valores, titulo, ylabel) in zip(axes, [
    (f1_vals,  'F1-macro (val)',     'F1-macro'),
    (acc_vals, 'Acuracia (val)',    'Acuracia'),
    (auc_vals, 'AUC-ROC macro (val)', 'AUC-ROC'),
]):
    bars = ax.bar(x, valores, w, color=cores[:len(nomes)], edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(nomes, rotation=22, ha='right', fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo, fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Comparativo Final — Todos os Modelos (Conjunto de Validacao)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()"""
))

# ── Tabela delta overfitting ──────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""print('Tabela completa — treino vs validacao:')
header = f"{'Modelo':<28} {'Acc_tr':>7} {'Acc_val':>8} {'D_acc':>7} {'F1_tr':>7} {'F1_val':>8} {'D_f1':>7} {'AUC_val':>9}"
print(header)
print('-' * len(header))
for r in todos_final:
    d_acc = r['acc_tr'] - r['acc_val']
    d_f1  = r['f1_tr']  - r['f1_val']
    flag_acc = ' !' if d_acc > 0.25 else ''
    flag_f1  = ' !' if d_f1  > 0.25 else ''
    print(f"{r['nome']:<28} {r['acc_tr']:>7.4f} {r['acc_val']:>8.4f} {d_acc:>+7.4f}{flag_acc:<2} "
          f"{r['f1_tr']:>7.4f} {r['f1_val']:>8.4f} {d_f1:>+7.4f}{flag_f1}")
print('\\n(!) delta > 0.25 indica overfitting relevante')"""
))

# ── F1 por classe ─────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(CLASS_NAMES))
w = 1.0 / (len(todos_final) + 1)
offsets = np.linspace(-(len(todos_final)-1)/2*w, (len(todos_final)-1)/2*w, len(todos_final))

for offset, res, color in zip(offsets, todos_final, cores):
    f1_cls = [f1_score(res['y_val'], res['pred_val'], labels=[i],
                       average='macro', zero_division=0) for i in range(len(CLASS_NAMES))]
    ax.bar(x + offset, f1_cls, w, label=res['nome'], color=color, edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_ylabel('F1-score')
ax.set_title('F1 por Classe — Todos os Modelos (val)', fontsize=12)
ax.legend(fontsize=7, loc='upper right', ncol=2)
ax.axhline(0.5, color='gray', linestyle='--', lw=0.8, alpha=0.5)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Destaque melhor modelo
best = max(todos_final, key=lambda r: r['f1_val'])
print(f'Melhor modelo: {best[\"nome\"]}')
print(f'  F1-macro val : {best[\"f1_val\"]:.4f}')
print(f'  Acuracia val : {best[\"acc_val\"]:.4f}')
print(f'  AUC-ROC val  : {best[\"auc_val\"]:.4f}')
print(f'  Overfitting  : delta F1 = {best[\"f1_tr\"]-best[\"f1_val\"]:+.4f}')"""
))

nb.cells.extend(cells)
nbformat.write(nb, 'notebooks/projeto.ipynb')
print(f'OK — {len(nb.cells)} celulas totais, {len(cells)} adicionadas.')
