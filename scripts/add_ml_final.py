"""Seção de ML reorganizada: RF (complexo) → DT (simples) → Ato 1 → Ato 2 → Comparativo."""
import nbformat

nb = nbformat.read('notebooks/projeto.ipynb', as_version=4)
cells = []

# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 6 — RF: MODELO COMPLEXO
# ═══════════════════════════════════════════════════════════════════════════════

cells.append(nbformat.v4.new_markdown_cell(
    "## 6. ML Clássico\n\n"
    "### Estrutura da análise\n\n"
    "| Etapa | Objetivo |\n"
    "|-------|----------|\n"
    "| **6. RF — Modelo Complexo** | Avaliar até onde um modelo poderoso chega e identificar overfitting |\n"
    "| **7. DT — Modelo Simples** | Mostrar que modelo simples é competitivo com muito menos complexidade |\n"
    "| **8. Ato 1 — Evidência** | Quantificar o ganho real de complexidade adicional |\n"
    "| **9. Ato 2 — Ablação** | Avaliar o impacto das features PDI sobre os metadados clínicos |\n"
    "| **10. Comparativo Final** | RF vs DT — síntese dos resultados |\n\n"
    "**Features:** `X_train_full` — 95 features (40 tabular + 55 PDI)  \n"
    "**Métricas:** F1-macro, Acurácia, AUC-ROC, F1 por classe  \n"
    "**Desbalanceamento:** SMOTE dentro do pipeline de CV (sem leakage)"
))

# ── Imports e helpers ─────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""from sklearn.neighbors       import KNeighborsClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.model_selection import (RandomizedSearchCV, GridSearchCV,
                                     StratifiedKFold, learning_curve,
                                     validation_curve)
from sklearn.metrics         import (accuracy_score, f1_score,
                                     classification_report, confusion_matrix,
                                     roc_auc_score, roc_curve, auc,
                                     make_scorer)
from sklearn.preprocessing   import label_binarize
from imblearn.pipeline       import Pipeline as ImbPipeline
from imblearn.over_sampling  import SMOTE
from matplotlib.patches      import Patch
from scipy.stats             import randint
import time, warnings
warnings.filterwarnings('ignore')

CLASS_NAMES = list(le_target.classes_)
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scorer_f1 = make_scorer(f1_score, average='macro', zero_division=0)

def avaliar_modelo(nome, modelo, X_tr, y_tr, X_vl, y_vl):
    pred_tr  = modelo.predict(X_tr)
    pred_vl  = modelo.predict(X_vl)
    prob_vl  = modelo.predict_proba(X_vl) if hasattr(modelo, 'predict_proba') else None
    acc_tr   = accuracy_score(y_tr, pred_tr)
    acc_vl   = accuracy_score(y_vl, pred_vl)
    f1_tr    = f1_score(y_tr, pred_tr, average='macro', zero_division=0)
    f1_vl    = f1_score(y_vl, pred_vl, average='macro', zero_division=0)
    auc_vl   = (roc_auc_score(y_vl, prob_vl, multi_class='ovr', average='macro')
                if prob_vl is not None else float('nan'))
    sep = '=' * 55
    print(sep)
    print(nome)
    print(sep)
    print(f'  Acuracia  treino={acc_tr:.4f}  val={acc_vl:.4f}  delta={acc_tr-acc_vl:+.4f}')
    print(f'  F1-macro  treino={f1_tr:.4f}  val={f1_vl:.4f}  delta={f1_tr-f1_vl:+.4f}')
    print(f'  AUC-ROC val: {auc_vl:.4f}')
    print()
    print(classification_report(y_vl, pred_vl, target_names=CLASS_NAMES, zero_division=0))
    return dict(nome=nome, acc_tr=acc_tr, acc_vl=acc_vl,
                f1_tr=f1_tr, f1_vl=f1_vl, auc_vl=auc_vl,
                pred_vl=pred_vl, y_vl=y_vl)

def plot_confusion(resultados, titulo):
    n = len(resultados)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 5))
    if n == 1: axes = [axes]
    for ax, res in zip(axes, resultados):
        cm = confusion_matrix(res['y_vl'], res['pred_vl'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_title(res['nome'], fontsize=11)
        ax.set_xlabel('Predito'); ax.set_ylabel('Real')
    fig.suptitle(titulo, fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.show()

print('Helpers carregados.')"""
))

# ── Markdown RF ───────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell(
    "### 6.1 Random Forest — Modelo Complexo\n\n"
    "Três configurações progressivas:\n\n"
    "1. **class_weight** — sem alterar o dataset, apenas penaliza erros nas classes raras\n"
    "2. **SMOTE** — oversampling das classes minoritárias no treino\n"
    "3. **Tuned** — busca de hiperparâmetros com regularização (`min_samples_leaf >= 2`, `max_depth <= 25`)"
))

# ── RF class_weight ───────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""rf_cw = RandomForestClassifier(
    n_estimators=300, min_samples_leaf=2,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf_cw.fit(X_train_full, y_train)
res_rf_cw = avaliar_modelo('RF — class_weight', rf_cw, X_train_full, y_train, X_val_full, y_val)"""
))

# ── RF SMOTE ──────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train_full, y_train)
print(f'Treino original: {X_train_full.shape[0]}  |  SMOTE: {X_train_sm.shape[0]}')

rf_sm = RandomForestClassifier(
    n_estimators=300, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)
rf_sm.fit(X_train_sm, y_train_sm)
res_rf_sm = avaliar_modelo('RF — SMOTE', rf_sm, X_train_sm, y_train_sm, X_val_full, y_val)"""
))

# ── RF Tuned ──────────────────────────────────────────────────────────────────
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
    pipe_rf, param_distributions=param_dist_rf,
    n_iter=30, scoring=scorer_f1, cv=cv5,
    refit=True, n_jobs=-1, random_state=42, verbose=1
)
t0 = time.time()
search_rf.fit(X_train_full, y_train)
print(f'\\nConcluido em {time.time()-t0:.1f}s  |  Melhor F1-macro CV: {search_rf.best_score_:.4f}')
print('Melhores parametros:')
for k, v in search_rf.best_params_.items():
    print(f'  {k}: {v}')

res_rf_tuned = avaliar_modelo(
    'RF Tuned (SMOTE+CV)', search_rf.best_estimator_,
    X_train_full, y_train, X_val_full, y_val
)"""
))

# ── RF: curva de aprendizado + ROC + feature importance ──────────────────────
cells.append(nbformat.v4.new_markdown_cell(
    "#### Análise do RF Tuned"
))

cells.append(nbformat.v4.new_code_cell(
"""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Learning curve ---
tr_sz, tr_sc, vl_sc = learning_curve(
    search_rf.best_estimator_, X_train_full, y_train,
    cv=cv5, scoring=scorer_f1,
    train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1
)
tr_m, tr_s = tr_sc.mean(1), tr_sc.std(1)
vl_m, vl_s = vl_sc.mean(1), vl_sc.std(1)
axes[0].plot(tr_sz, tr_m, 'o-', color='steelblue', label='Treino')
axes[0].fill_between(tr_sz, tr_m-tr_s, tr_m+tr_s, alpha=0.15, color='steelblue')
axes[0].plot(tr_sz, vl_m, 'o-', color='coral', label='Val CV')
axes[0].fill_between(tr_sz, vl_m-vl_s, vl_m+vl_s, alpha=0.15, color='coral')
axes[0].set_title(f'Learning Curve\\nGap final: {tr_m[-1]-vl_m[-1]:+.3f}', fontsize=11)
axes[0].set_xlabel('Amostras de treino'); axes[0].set_ylabel('F1-macro')
axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_ylim(0, 1.05)

# --- ROC por classe ---
y_vl_bin = label_binarize(y_val, classes=list(range(len(CLASS_NAMES))))
prob_rf   = search_rf.best_estimator_.predict_proba(X_val_full)
colors_cls = plt.cm.tab10(np.linspace(0, 0.9, len(CLASS_NAMES)))
for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors_cls)):
    fpr, tpr, _ = roc_curve(y_vl_bin[:, i], prob_rf[:, i])
    axes[1].plot(fpr, tpr, color=color, lw=1.8, label=f'{cls} (AUC={auc(fpr,tpr):.2f})')
axes[1].plot([0,1],[0,1],'k--',lw=0.8)
axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
axes[1].set_title('Curvas ROC por Classe', fontsize=11)
axes[1].legend(loc='lower right', fontsize=8); axes[1].grid(alpha=0.3)

# --- Feature importance top 15 ---
rf_step  = search_rf.best_estimator_.named_steps['rf']
imp_rf   = pd.Series(rf_step.feature_importances_, index=X_train_full.columns)
top15    = imp_rf.nlargest(15)
axes[2].barh(top15.index[::-1], top15.values[::-1], color='steelblue')
axes[2].set_title('Top 15 Features (RF Tuned)', fontsize=11)
axes[2].set_xlabel('Importancia (Gini)'); axes[2].grid(axis='x', alpha=0.3)

plt.suptitle('RF Tuned — Analise Completa', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()"""
))

# ── RF matrizes de confusão ───────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""plot_confusion([res_rf_cw, res_rf_sm, res_rf_tuned],
               'RF — Evolucao: class_weight → SMOTE → Tuned (val)')"""
))

# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 7 — DT: MODELO SIMPLES
# ═══════════════════════════════════════════════════════════════════════════════

cells.append(nbformat.v4.new_markdown_cell(
    "## 7. Decision Tree — Modelo Simples\n\n"
    "Árvore de decisão com `max_depth` controlado: interpretável, muito menos complexa que o RF.  \n"
    "Se os resultados forem competitivos, o problema não exige alta complexidade."
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
    pipe_dt, param_grid=param_grid_dt,
    scoring=scorer_f1, cv=cv5,
    refit=True, n_jobs=-1, verbose=1
)
t0 = time.time()
search_dt.fit(X_train_full, y_train)
print(f'\\nConcluido em {time.time()-t0:.1f}s  |  Melhor F1-macro CV: {search_dt.best_score_:.4f}')
print('Melhores parametros:')
for k, v in search_dt.best_params_.items():
    print(f'  {k}: {v}')

res_dt_tuned = avaliar_modelo(
    'DT Tuned (SMOTE+CV)', search_dt.best_estimator_,
    X_train_full, y_train, X_val_full, y_val
)"""
))

# ── DT: curva de aprendizado + ROC + feature importance ──────────────────────
cells.append(nbformat.v4.new_code_cell(
"""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Learning curve ---
tr_sz, tr_sc, vl_sc = learning_curve(
    search_dt.best_estimator_, X_train_full, y_train,
    cv=cv5, scoring=scorer_f1,
    train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1
)
tr_m, tr_s = tr_sc.mean(1), tr_sc.std(1)
vl_m, vl_s = vl_sc.mean(1), vl_sc.std(1)
axes[0].plot(tr_sz, tr_m, 'o-', color='steelblue', label='Treino')
axes[0].fill_between(tr_sz, tr_m-tr_s, tr_m+tr_s, alpha=0.15, color='steelblue')
axes[0].plot(tr_sz, vl_m, 'o-', color='coral', label='Val CV')
axes[0].fill_between(tr_sz, vl_m-vl_s, vl_m+vl_s, alpha=0.15, color='coral')
axes[0].set_title(f'Learning Curve\\nGap final: {tr_m[-1]-vl_m[-1]:+.3f}', fontsize=11)
axes[0].set_xlabel('Amostras de treino'); axes[0].set_ylabel('F1-macro')
axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_ylim(0, 1.05)

# --- ROC por classe ---
prob_dt = search_dt.best_estimator_.predict_proba(X_val_full)
for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors_cls)):
    fpr, tpr, _ = roc_curve(y_vl_bin[:, i], prob_dt[:, i])
    axes[1].plot(fpr, tpr, color=color, lw=1.8, label=f'{cls} (AUC={auc(fpr,tpr):.2f})')
axes[1].plot([0,1],[0,1],'k--',lw=0.8)
axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
axes[1].set_title('Curvas ROC por Classe', fontsize=11)
axes[1].legend(loc='lower right', fontsize=8); axes[1].grid(alpha=0.3)

# --- Feature importance top 15 ---
dt_step = search_dt.best_estimator_.named_steps['dt']
imp_dt  = pd.Series(dt_step.feature_importances_, index=X_train_full.columns)
top15dt = imp_dt.nlargest(15)
axes[2].barh(top15dt.index[::-1], top15dt.values[::-1], color='#e8844c')
axes[2].set_title('Top 15 Features (DT Tuned)', fontsize=11)
axes[2].set_xlabel('Importancia (Gini)'); axes[2].grid(axis='x', alpha=0.3)

plt.suptitle('DT Tuned — Analise Completa', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()"""
))

cells.append(nbformat.v4.new_code_cell(
"""plot_confusion([res_dt_tuned], 'DT Tuned — Matriz de Confusao (val)')"""
))

# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 8 — ATO 1: EVIDÊNCIA DE PROBLEMA SIMPLES
# ═══════════════════════════════════════════════════════════════════════════════

cells.append(nbformat.v4.new_markdown_cell(
    "## 8. Ato 1 — Evidência de Problema Simples\n\n"
    "Duas análises que quantificam se o problema exige alta complexidade:\n\n"
    "1. **Curva de validação** — a partir de qual `max_depth` o DT satura?\n"
    "2. **Complexidade vs Ganho** — quanto de F1 o RF ganha sobre o DT pelo custo de centenas de árvores?"
))

# ── Curva de validação ────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""depths = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]

pipe_dt_vc = ImbPipeline([
    ('smote', SMOTE(random_state=42, k_neighbors=5)),
    ('dt',    DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10, random_state=42))
])
tr_sc, vl_sc = validation_curve(
    pipe_dt_vc, X_train_full, y_train,
    param_name='dt__max_depth', param_range=depths,
    scoring=scorer_f1,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1
)
tr_m, tr_s = tr_sc.mean(1), tr_sc.std(1)
vl_m, vl_s = vl_sc.mean(1), vl_sc.std(1)
best_idx   = vl_m.argmax()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Curva de validação
axes[0].plot(depths, tr_m, 'o-', color='steelblue', label='Treino (CV)')
axes[0].fill_between(depths, tr_m-tr_s, tr_m+tr_s, alpha=0.15, color='steelblue')
axes[0].plot(depths, vl_m, 'o-', color='coral', label='Validacao (CV)')
axes[0].fill_between(depths, vl_m-vl_s, vl_m+vl_s, alpha=0.15, color='coral')
axes[0].axvline(depths[best_idx], color='green', linestyle='--', lw=1.5,
                label=f'Ótimo: depth={depths[best_idx]} (F1={vl_m[best_idx]:.3f})')
axes[0].set_xlabel('max_depth'); axes[0].set_ylabel('F1-macro')
axes[0].set_title('Curva de Validacao — DT (max_depth)', fontsize=11)
axes[0].legend(); axes[0].grid(alpha=0.3)

# Complexidade vs ganho
modelos_c = [
    ('DT  Tuned',        res_dt_tuned['f1_vl'], res_dt_tuned['auc_vl'],  1),
    ('RF  class_weight', res_rf_cw['f1_vl'],    res_rf_cw['auc_vl'],    300),
    ('RF  SMOTE',        res_rf_sm['f1_vl'],    res_rf_sm['auc_vl'],    300),
    ('RF  Tuned',        res_rf_tuned['f1_vl'], res_rf_tuned['auc_vl'], search_rf.best_params_['rf__n_estimators']),
]
cores_c = ['#2ecc71', '#e67e22', '#e74c3c', '#9b59b6']
for i, (nome, f1, _, ne) in enumerate(modelos_c):
    axes[1].scatter(ne, f1, s=150+ne*0.3, color=cores_c[i],
                    zorder=3, label=nome, edgecolors='white', lw=1.5)
axes[1].set_xlabel('Complexidade (numero de arvores)')
axes[1].set_ylabel('F1-macro (val)')
axes[1].set_title('Complexidade vs F1-macro', fontsize=11)
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3); axes[1].set_xlim(-20, 650)

f1_dt = res_dt_tuned['f1_vl']
f1_rf = res_rf_tuned['f1_vl']
n_rf  = modelos_c[-1][3]
axes[1].annotate(
    f'+{f1_rf-f1_dt:.3f} F1\\npor {n_rf}x arvores',
    xy=(n_rf, f1_rf), xytext=(n_rf-250, f1_rf-0.06),
    fontsize=9, color='#9b59b6',
    arrowprops=dict(arrowstyle='->', color='#9b59b6')
)

plt.suptitle('Ato 1 — Evidencia de Problema Simples', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()

print(f'Saturacao em max_depth={depths[best_idx]}  |  Ganho depth=7→20: {vl_m[-1]-vl_m[best_idx]:+.4f}')
print(f'Ganho RF Tuned vs DT Tuned: +{f1_rf-f1_dt:.4f} F1  com {n_rf} arvores adicionais')"""))

# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 9 — ATO 2: ABLAÇÃO DE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

cells.append(nbformat.v4.new_markdown_cell(
    "## 9. Ato 2 — Ablação de Features\n\n"
    "**Mesmo modelo** (DT com parâmetros do tuning), **três conjuntos de features**:  \n\n"
    "| Experimento | Features | Pergunta |\n"
    "|-------------|----------|----------|\n"
    "| Tabular only | 40 features clínicas | O que os metadados sozinhos conseguem? |\n"
    "| PDI only | 55 features de imagem | Imagem sem contexto clínico |\n"
    "| Tabular + PDI | 95 features | PDI acrescenta ao clínico? |"
))

cells.append(nbformat.v4.new_code_cell(
"""dt_params = dict(
    criterion=search_dt.best_params_['dt__criterion'],
    max_depth=search_dt.best_params_['dt__max_depth'],
    min_samples_leaf=search_dt.best_params_['dt__min_samples_leaf'],
    class_weight=search_dt.best_params_['dt__class_weight'],
    random_state=42
)
print('Parametros fixos do DT:', dt_params)

def treinar_dt_ablacao(X_tr, y_tr, X_vl, y_vl, nome):
    pipe = ImbPipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=5)),
        ('dt',    DecisionTreeClassifier(**dt_params))
    ])
    pipe.fit(X_tr, y_tr)
    return avaliar_modelo(nome, pipe, X_tr, y_tr, X_vl, y_vl)

X_train_pdi = pdi_train_sc.reset_index(drop=True)
X_val_pdi   = pdi_val_sc.reset_index(drop=True)

res_abl_tab = treinar_dt_ablacao(X_train_sel,  y_train, X_val_sel,  y_val, 'DT — Tabular only')
res_abl_pdi = treinar_dt_ablacao(X_train_pdi,  y_train, X_val_pdi,  y_val, 'DT — PDI only')
res_abl_ful = treinar_dt_ablacao(X_train_full, y_train, X_val_full, y_val, 'DT — Tabular+PDI')

resultados_ablacao = [res_abl_tab, res_abl_pdi, res_abl_ful]"""
))

cells.append(nbformat.v4.new_code_cell(
"""cores_abl = ['#3498db', '#e67e22', '#2ecc71']
labels_abl = ['Tabular only', 'PDI only', 'Tabular+PDI']
metricas   = ['F1-macro', 'Acuracia', 'AUC-ROC']
vals       = [[r['f1_vl'], r['acc_vl'], r['auc_vl']] for r in resultados_ablacao]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Barplot metricas gerais
x = np.arange(len(metricas))
w = 0.25
for i, (res, cor, lbl) in enumerate(zip(resultados_ablacao, cores_abl, labels_abl)):
    v = [res['f1_vl'], res['acc_vl'], res['auc_vl']]
    bars = axes[0].bar(x + (i-1)*w, v, w, label=lbl, color=cor, edgecolor='white')
    for bar, val in zip(bars, v):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)
axes[0].set_xticks(x); axes[0].set_xticklabels(metricas)
axes[0].set_ylim(0, 1.1); axes[0].set_ylabel('Score')
axes[0].set_title('Metricas por Conjunto de Features', fontsize=11)
axes[0].legend(); axes[0].grid(axis='y', alpha=0.3)

# F1 por classe
x_cls = np.arange(len(CLASS_NAMES))
for i, (res, cor, lbl) in enumerate(zip(resultados_ablacao, cores_abl, labels_abl)):
    f1_cls = [f1_score(res['y_vl'], res['pred_vl'], labels=[j],
                       average='macro', zero_division=0) for j in range(len(CLASS_NAMES))]
    axes[1].bar(x_cls+(i-1)*w, f1_cls, w, label=lbl, color=cor, edgecolor='white')
axes[1].set_xticks(x_cls); axes[1].set_xticklabels(CLASS_NAMES, fontsize=11)
axes[1].set_ylim(0, 1.1); axes[1].set_ylabel('F1-score')
axes[1].set_title('F1 por Classe — Ablacao', fontsize=11)
axes[1].legend(fontsize=9)
axes[1].axhline(0.5, color='gray', linestyle='--', lw=0.8, alpha=0.5)
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('Ato 2 — Ablacao: Tabular vs PDI vs Tabular+PDI', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()

# Matrizes
plot_confusion(resultados_ablacao, 'Ablacao — Matrizes de Confusao (val)')"""))

cells.append(nbformat.v4.new_code_cell(
"""f1_tab = res_abl_tab['f1_vl']
f1_pdi = res_abl_pdi['f1_vl']
f1_ful = res_abl_ful['f1_vl']

print('Ablacao — resumo:')
header = f"{'Conjunto':<22} {'F1_tr':>7} {'F1_val':>8} {'Delta':>7} {'AUC_val':>9}"
print(header); print('-'*len(header))
for r in resultados_ablacao:
    print(f"{r['nome']:<22} {r['f1_tr']:>7.4f} {r['f1_vl']:>8.4f} "
          f"{r['f1_tr']-r['f1_vl']:>+7.4f} {r['auc_vl']:>9.4f}")
print(f'\\nGanho Tabular+PDI vs Tabular only : {f1_ful-f1_tab:+.4f} F1-macro')
print(f'PDI sozinho vs Tabular only        : {f1_pdi-f1_tab:+.4f} F1-macro')
if f1_ful > f1_tab:
    print('-> PDI agrega valor sobre os metadados clinicos.')
else:
    print('-> PDI nao agrega valor significativo.')"""))

# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 10 — COMPARATIVO FINAL RF vs DT
# ═══════════════════════════════════════════════════════════════════════════════

cells.append(nbformat.v4.new_markdown_cell(
    "## 10. Comparativo Final — RF vs DT\n\n"
    "Síntese de todos os experimentos realizados."
))

cells.append(nbformat.v4.new_code_cell(
"""todos = [res_rf_cw, res_rf_sm, res_rf_tuned, res_dt_tuned]
cores_f = ['#5b9bd5', '#ed7d31', '#9b59b6', '#2ecc71']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
nomes = [r['nome'] for r in todos]
x = np.arange(len(nomes)); w = 0.6

for ax, (key, titulo) in zip(axes, [
    ('f1_vl',  'F1-macro (val)'),
    ('acc_vl', 'Acuracia (val)'),
    ('auc_vl', 'AUC-ROC (val)'),
]):
    vals = [r[key] for r in todos]
    bars = ax.bar(x, vals, w, color=cores_f, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(nomes, rotation=15, ha='right', fontsize=9)
    ax.set_ylim(0, 1.1); ax.set_title(titulo, fontsize=11); ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('Comparativo Final — RF vs DT (Conjunto de Validacao)',
             fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()"""))

cells.append(nbformat.v4.new_code_cell(
"""# Tabela com flag de overfitting
print('Tabela completa:')
header = f"{'Modelo':<28} {'F1_tr':>7} {'F1_val':>8} {'Delta_F1':>9} {'Acc_val':>8} {'AUC_val':>9}"
print(header); print('-'*len(header))
for r in todos:
    delta = r['f1_tr'] - r['f1_vl']
    flag  = '  (!)' if delta > 0.25 else ''
    print(f"{r['nome']:<28} {r['f1_tr']:>7.4f} {r['f1_vl']:>8.4f} "
          f"{delta:>+9.4f}{flag:<5} {r['acc_vl']:>8.4f} {r['auc_vl']:>9.4f}")
print('\\n(!) delta F1 > 0.25 = overfitting relevante')

# F1 por classe — RF Tuned vs DT Tuned
print('\\nF1 por classe — RF Tuned vs DT Tuned:')
header2 = f"{'Classe':<6} {'RF_Tuned':>10} {'DT_Tuned':>10} {'Ganho RF':>10}"
print(header2); print('-'*38)
for i, cls in enumerate(CLASS_NAMES):
    f1_rf = f1_score(res_rf_tuned['y_vl'], res_rf_tuned['pred_vl'],
                     labels=[i], average='macro', zero_division=0)
    f1_dt = f1_score(res_dt_tuned['y_vl'], res_dt_tuned['pred_vl'],
                     labels=[i], average='macro', zero_division=0)
    print(f"{cls:<6} {f1_rf:>10.4f} {f1_dt:>10.4f} {f1_rf-f1_dt:>+10.4f}")"""))

cells.append(nbformat.v4.new_code_cell(
"""# F1 por classe — barplot comparativo
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(CLASS_NAMES)); w = 0.2
for i, (res, cor) in enumerate(zip(todos, cores_f)):
    f1_cls = [f1_score(res['y_vl'], res['pred_vl'], labels=[j],
                       average='macro', zero_division=0) for j in range(len(CLASS_NAMES))]
    ax.bar(x+(i-1.5)*w, f1_cls, w, label=res['nome'], color=cor, edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, fontsize=11)
ax.set_ylim(0, 1.1); ax.set_ylabel('F1-score')
ax.set_title('F1 por Classe — Todos os Modelos (val)', fontsize=12)
ax.legend(fontsize=9); ax.axhline(0.5, color='gray', linestyle='--', lw=0.8, alpha=0.5)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.show()"""))

nb.cells.extend(cells)
nbformat.write(nb, 'notebooks/projeto.ipynb')
print(f'OK — {len(nb.cells)} celulas totais, {len(cells)} adicionadas.')
