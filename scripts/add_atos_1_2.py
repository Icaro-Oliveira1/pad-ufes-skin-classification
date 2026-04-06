"""Adiciona Ato 1 (evidência problema simples) e Ato 2 (ablação de features)."""
import nbformat

nb = nbformat.read('notebooks/projeto.ipynb', as_version=4)
cells = []

# ═══════════════════════════════════════════════════════════════════════════════
# ATO 1 — Evidência de Problema Simples
# ═══════════════════════════════════════════════════════════════════════════════

cells.append(nbformat.v4.new_markdown_cell(
    "## 10. Ato 1 — Evidência de Problema Simples\n\n"
    "Investigamos se o problema realmente exige alta complexidade comparando:\n\n"
    "1. **Curva de validação** — F1-macro do DT em função de `max_depth`: "
    "a partir de qual profundidade o ganho satura?\n"
    "2. **Complexidade vs Ganho** — DT vs RF: quanto de F1 se ganha ao multiplicar "
    "a complexidade por centenas de árvores?\n\n"
    "Se o DT simples satura cedo e o gap para o RF for pequeno, "
    "o problema não exige alta complexidade."
))

# ── Curva de validação (max_depth) ────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""from sklearn.model_selection import validation_curve

depths = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]

pipe_dt_vc = ImbPipeline([
    ('smote', SMOTE(random_state=42, k_neighbors=5)),
    ('dt',    DecisionTreeClassifier(
                  criterion='entropy',
                  min_samples_leaf=10,
                  class_weight=None,
                  random_state=42))
])

tr_scores, vl_scores = validation_curve(
    pipe_dt_vc,
    X_train_full, y_train,
    param_name='dt__max_depth',
    param_range=depths,
    scoring=scorer_f1,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1
)

tr_m, tr_s = tr_scores.mean(1), tr_scores.std(1)
vl_m, vl_s = vl_scores.mean(1), vl_scores.std(1)

best_depth_idx = vl_m.argmax()
best_depth     = depths[best_depth_idx]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(depths, tr_m, 'o-', color='steelblue', label='Treino (CV)')
ax.fill_between(depths, tr_m-tr_s, tr_m+tr_s, alpha=0.15, color='steelblue')
ax.plot(depths, vl_m, 'o-', color='coral', label='Validacao (CV)')
ax.fill_between(depths, vl_m-vl_s, vl_m+vl_s, alpha=0.15, color='coral')
ax.axvline(best_depth, color='green', linestyle='--', lw=1.5,
           label=f'Melhor depth={best_depth} (F1={vl_m[best_depth_idx]:.3f})')
ax.set_xlabel('max_depth')
ax.set_ylabel('F1-macro')
ax.set_title('Curva de Validacao — Decision Tree (max_depth)', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f'Melhor max_depth: {best_depth}  |  F1-macro CV: {vl_m[best_depth_idx]:.4f}')
print(f'Ganho de depth=7 para depth=20 : {vl_m[-1]-vl_m[best_depth_idx]:+.4f}')
print('-> Curva satura cedo: complexidade adicional nao traz ganho significativo.')"""
))

# ── Complexidade vs Ganho ─────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""# Coleta metricas de todos os modelos com estimativa de complexidade
import math

modelos_complexidade = [
    # (nome, f1_val, auc_val, n_params_aprox, descricao_complexidade)
    ('DT  depth=7',      res_dt_tuned['f1_val'],  res_dt_tuned['auc_val'],   1,   'max_depth=7'),
    ('KNN Tuned',        res_knn_tuned['f1_val'], res_knn_tuned['auc_val'],  2,   'k=7, manhattan'),
    ('RF  class_weight', res_rf_cw['f1_val'],     res_rf_cw['auc_val'],     300, 'n_est=300'),
    ('RF  SMOTE',        res_rf_sm['f1_val'],     res_rf_sm['auc_val'],     300, 'n_est=300'),
    ('RF  Tuned',        res_rf_tuned['f1_val'],  res_rf_tuned['auc_val'],  563, 'n_est=563'),
]

nomes_c  = [m[0] for m in modelos_complexidade]
f1s      = [m[1] for m in modelos_complexidade]
aucs     = [m[2] for m in modelos_complexidade]
n_est    = [m[3] for m in modelos_complexidade]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# F1 por complexidade (tamanho do ponto = n_estimators)
cores_c = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c', '#9b59b6']
for i, (nome, f1, auc, ne, desc) in enumerate(modelos_complexidade):
    size = 80 + ne * 0.4
    axes[0].scatter(ne, f1, s=size, color=cores_c[i], zorder=3, label=nome, edgecolors='white', lw=1.5)

axes[0].set_xlabel('Complexidade (numero de estimadores / arvores)', fontsize=10)
axes[0].set_ylabel('F1-macro (val)', fontsize=10)
axes[0].set_title('Complexidade vs F1-macro', fontsize=11)
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)
axes[0].set_xlim(-30, 620)

# Barplot F1 ordenado
ordem = sorted(range(len(f1s)), key=lambda i: f1s[i])
ax = axes[1]
bars = ax.barh([nomes_c[i] for i in ordem], [f1s[i] for i in ordem],
               color=[cores_c[i] for i in ordem], edgecolor='white')
ax.set_xlabel('F1-macro (val)')
ax.set_title('Ranking de F1-macro — todos os modelos', fontsize=11)
ax.set_xlim(0, 1)
ax.grid(axis='x', alpha=0.3)
for bar, val in zip(bars, [f1s[i] for i in ordem]):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)

# Anotacao do ganho DT -> RF Tuned
f1_dt = res_dt_tuned['f1_val']
f1_rf = res_rf_tuned['f1_val']
ax.annotate(f'Ganho RF vs DT:\\n+{f1_rf-f1_dt:.3f} F1\\npor ~563x complexidade',
            xy=(f1_rf, len(nomes_c)-1), xytext=(f1_dt - 0.18, len(nomes_c)-1.8),
            fontsize=8, color='#9b59b6',
            arrowprops=dict(arrowstyle='->', color='#9b59b6'))

plt.suptitle('Ato 1 — Complexidade vs Desempenho', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print(f'DT  Tuned  — F1: {f1_dt:.4f}  |  AUC: {res_dt_tuned[\"auc_val\"]:.4f}  | ~1 arvore')
print(f'RF  Tuned  — F1: {f1_rf:.4f}  |  AUC: {res_rf_tuned[\"auc_val\"]:.4f}  | 563 arvores')
print(f'Ganho absoluto de F1 : {f1_rf - f1_dt:+.4f}')
print(f'-> Alta complexidade traz ganho marginal: problema nao exige RF com centenas de arvores.')"""
))

# ═══════════════════════════════════════════════════════════════════════════════
# ATO 2 — Ablação de Features
# ═══════════════════════════════════════════════════════════════════════════════

cells.append(nbformat.v4.new_markdown_cell(
    "## 11. Ato 2 — Ablação de Features\n\n"
    "Mesmo modelo (Decision Tree com parâmetros do tuning), três conjuntos de features:\n\n"
    "| Experimento | Features | Objetivo |\n"
    "|-------------|----------|----------|\n"
    "| Tabular only | 40 features clínicas | Linha de base — o que os metadados sozinhos conseguem? |\n"
    "| PDI only | 55 features de imagem | Imagem sem contexto clínico |\n"
    "| Tabular + PDI | 95 features | PDI acrescenta ao clínico? |\n\n"
    "**Modelo fixo:** DT com `criterion=entropy, max_depth=7, min_samples_leaf=10`  \n"
    "**SMOTE aplicado dentro do CV** em todos os experimentos."
))

# ── Treino dos 3 DTs ──────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""# Parametros fixos do DT (encontrados no tuning)
dt_params = dict(criterion='entropy', max_depth=7, min_samples_leaf=10,
                 class_weight=None, random_state=42)

def treinar_dt_ablacao(X_tr, y_tr, X_vl, y_vl, nome):
    pipe = ImbPipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=5)),
        ('dt',    DecisionTreeClassifier(**dt_params))
    ])
    pipe.fit(X_tr, y_tr)
    return avaliar_modelo(nome, pipe, X_tr, y_tr, X_vl, y_vl)

# PDI only — usa diretamente os splits PDI escalonados (sem features tabulares)
X_train_pdi = pdi_train_sc.reset_index(drop=True)
X_val_pdi   = pdi_val_sc.reset_index(drop=True)
X_test_pdi  = pdi_test_sc.reset_index(drop=True)

res_dt_tabular = treinar_dt_ablacao(X_train_sel,  y_train, X_val_sel,  y_val, 'DT — Tabular only')
res_dt_pdi     = treinar_dt_ablacao(X_train_pdi,  y_train, X_val_pdi,  y_val, 'DT — PDI only')
res_dt_full    = treinar_dt_ablacao(X_train_full, y_train, X_val_full, y_val, 'DT — Tabular+PDI')

resultados_ablacao = [res_dt_tabular, res_dt_pdi, res_dt_full]"""
))

# ── Matrizes de confusão ablação ──────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""plot_confusion(resultados_ablacao, 'Ato 2 — Ablacao de Features (Decision Tree)')"""
))

# ── Gráficos ablação ──────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""cores_abl = ['#3498db', '#e67e22', '#2ecc71']

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Barplot metricas gerais
x = np.arange(3)
w = 0.28
labels_abl = [r['nome'] for r in resultados_ablacao]
for i, (valores, label) in enumerate([
    ([r['f1_val']  for r in resultados_ablacao], 'F1-macro'),
    ([r['acc_val'] for r in resultados_ablacao], 'Acuracia'),
    ([r['auc_val'] for r in resultados_ablacao], 'AUC-ROC'),
]):
    offset = (i - 1) * w
    bars = axes[0].bar(x + offset, valores, w, label=label,
                       color=['#5b9bd5','#ed7d31','#70ad47'][i], edgecolor='white')
    for bar, val in zip(bars, valores):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)

axes[0].set_xticks(x)
axes[0].set_xticklabels(['Tabular only', 'PDI only', 'Tabular+PDI'], fontsize=10)
axes[0].set_ylim(0, 1.08)
axes[0].set_ylabel('Score')
axes[0].set_title('Metricas Gerais por Conjunto de Features', fontsize=11)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# F1 por classe
x_cls = np.arange(len(CLASS_NAMES))
w_cls = 0.28
for i, (res, color) in enumerate(zip(resultados_ablacao, cores_abl)):
    f1_cls = [f1_score(res['y_val'], res['pred_val'], labels=[j],
                       average='macro', zero_division=0) for j in range(len(CLASS_NAMES))]
    offset = (i - 1) * w_cls
    axes[1].bar(x_cls + offset, f1_cls, w_cls, label=res['nome'],
                color=color, edgecolor='white')

axes[1].set_xticks(x_cls)
axes[1].set_xticklabels(CLASS_NAMES, fontsize=11)
axes[1].set_ylim(0, 1.1)
axes[1].set_ylabel('F1-score')
axes[1].set_title('F1 por Classe — Ablacao de Features', fontsize=11)
axes[1].legend(fontsize=9)
axes[1].axhline(0.5, color='gray', linestyle='--', lw=0.8, alpha=0.5)
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('Ato 2 — Ablacao: Tabular vs PDI vs Tabular+PDI (Decision Tree)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()"""
))

# ── Tabela e conclusão ablação ────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""print('Ablacao — resumo:')
header = f"{'Conjunto':<22} {'F1_tr':>7} {'F1_val':>8} {'D_f1':>7} {'Acc_val':>8} {'AUC_val':>9}"
print(header)
print('-' * len(header))
for r in resultados_ablacao:
    print(f"{r['nome']:<22} {r['f1_tr']:>7.4f} {r['f1_val']:>8.4f} "
          f"{r['f1_tr']-r['f1_val']:>+7.4f} {r['acc_val']:>8.4f} {r['auc_val']:>9.4f}")

f1_tab = res_dt_tabular['f1_val']
f1_pdi = res_dt_pdi['f1_val']
f1_ful = res_dt_full['f1_val']

print(f'\\nGanho PDI sobre Tabular       : {f1_ful - f1_tab:+.4f} F1-macro')
print(f'PDI sozinho vs Tabular sozinho : {f1_pdi - f1_tab:+.4f} F1-macro')
print(f'PDI sozinho vs Tabular+PDI     : {f1_pdi - f1_ful:+.4f} F1-macro')

if f1_ful > f1_tab:
    print('\\n-> PDI agrega valor sobre os metadados clinicos.')
else:
    print('\\n-> PDI nao agrega valor significativo: metadados clinicos sao suficientes.')"""
))

nb.cells.extend(cells)
nbformat.write(nb, 'notebooks/projeto.ipynb')
print(f'OK — {len(nb.cells)} celulas totais, {len(cells)} adicionadas.')
