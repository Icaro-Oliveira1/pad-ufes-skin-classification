"""Adiciona seção de ML Clássico ao notebook."""
import nbformat

nb = nbformat.read('notebooks/projeto.ipynb', as_version=4)

cells = []

# ── Markdown intro ────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell(
    "## 6. ML Clássico — KNN e Random Forest\n\n"
    "### Estratégia\n\n"
    "Duas rodadas para avaliar o impacto do desbalanceamento:\n\n"
    "| Rodada | Estratégia | Modelos |\n"
    "|--------|-----------|--------|\n"
    "| A | `class_weight='balanced'` | RF, KNN* |\n"
    "| B | SMOTE no treino | RF, KNN |\n\n"
    "> *KNN não suporta `class_weight` — na Rodada A serve de baseline sem compensação; "
    "na Rodada B recebe o treino oversampled.\n\n"
    "**Features:** `X_train_full` (95 features: 40 tabular + 55 PDI, com `biopsed`)  \n"
    "**Métricas:** Acurácia, F1-macro, F1 por classe, AUC-ROC, Matriz de Confusão  \n"
    "**Overfitting:** comparar treino vs val em cada modelo"
))

# ── Imports e helpers ─────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""from sklearn.neighbors  import KNeighborsClassifier
from sklearn.ensemble   import RandomForestClassifier
from sklearn.metrics    import (accuracy_score, f1_score,
                                classification_report,
                                confusion_matrix, roc_auc_score)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

CLASS_NAMES = list(le_target.classes_)  # ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']

def avaliar_modelo(nome, modelo, X_tr, y_tr, X_val, y_val):
    pred_tr  = modelo.predict(X_tr)
    pred_val = modelo.predict(X_val)
    prob_val = modelo.predict_proba(X_val) if hasattr(modelo, 'predict_proba') else None

    acc_tr  = accuracy_score(y_tr,  pred_tr)
    acc_val = accuracy_score(y_val, pred_val)
    f1_tr   = f1_score(y_tr,  pred_tr,  average='macro', zero_division=0)
    f1_val  = f1_score(y_val, pred_val, average='macro', zero_division=0)
    auc_val = (roc_auc_score(y_val, prob_val, multi_class='ovr', average='macro')
               if prob_val is not None else float('nan'))

    sep = '=' * 55
    print(sep)
    print(nome)
    print(sep)
    print(f'  Acuracia  treino={acc_tr:.4f}  val={acc_val:.4f}  delta={acc_tr-acc_val:+.4f}')
    print(f'  F1-macro  treino={f1_tr:.4f}  val={f1_val:.4f}  delta={f1_tr-f1_val:+.4f}')
    print(f'  AUC-ROC val (macro-ovr): {auc_val:.4f}')
    print()
    print('Relatorio de classificacao (val):')
    print(classification_report(y_val, pred_val, target_names=CLASS_NAMES, zero_division=0))

    return dict(nome=nome, acc_tr=acc_tr, acc_val=acc_val,
                f1_tr=f1_tr, f1_val=f1_val, auc_val=auc_val,
                pred_val=pred_val, y_val=y_val)


def plot_confusion(resultados, titulo):
    n = len(resultados)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 5))
    if n == 1:
        axes = [axes]
    for ax, res in zip(axes, resultados):
        cm = confusion_matrix(res['y_val'], res['pred_val'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_title(res['nome'], fontsize=11)
        ax.set_xlabel('Predito')
        ax.set_ylabel('Real')
    fig.suptitle(titulo, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_comparacao(todos):
    nomes    = [r['nome']    for r in todos]
    f1_vals  = [r['f1_val']  for r in todos]
    acc_vals = [r['acc_val'] for r in todos]
    x = np.arange(len(nomes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w/2, f1_vals,  w, label='F1-macro (val)',  color='steelblue')
    ax.bar(x + w/2, acc_vals, w, label='Acuracia (val)', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(nomes, rotation=15, ha='right')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.set_title('Comparacao — todos os modelos (validacao)')
    ax.legend()
    ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plt.show()

print('Helpers carregados.')"""
))

# ── Markdown Rodada A ─────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell(
    "### Rodada A — class_weight (sem alterar o dataset)\n\n"
    "- **Random Forest:** `class_weight='balanced'` — penaliza mais erros nas classes raras  \n"
    "- **KNN:** sem compensação (não suportado nativamente) — baseline por distância pura  \n\n"
    "Hiperparâmetros iniciais fixos; tuning somente após comparação entre rodadas."
))

# ── Rodada A: treino ──────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""# Random Forest com class_weight='balanced'
rf_cw = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_cw.fit(X_train_full, y_train)

# KNN baseline (k impar, metrica euclidiana)
knn_base = KNeighborsClassifier(n_neighbors=11, n_jobs=-1)
knn_base.fit(X_train_full, y_train)

res_rf_cw    = avaliar_modelo('RF  — class_weight',    rf_cw,    X_train_full, y_train, X_val_full, y_val)
res_knn_base = avaliar_modelo('KNN — baseline (k=11)', knn_base, X_train_full, y_train, X_val_full, y_val)

resultados_A = [res_rf_cw, res_knn_base]"""
))

# ── Confusão Rodada A ─────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""plot_confusion(resultados_A, 'Rodada A — Matrizes de Confusao (val)')"""
))

# ── Markdown Rodada B ─────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell(
    "### Rodada B — SMOTE no treino\n\n"
    "SMOTE interpola vizinhos sintéticos das classes minoritárias **apenas em `X_train_full`**.  \n"
    "Val e teste permanecem intocados (sem leakage).\n\n"
    "**Overfitting monitorado:** se Δ(treino−val) crescer muito vs Rodada A, "
    "o SMOTE está gerando amostras fáceis demais."
))

# ── Rodada B: SMOTE + treino ──────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""from collections import Counter

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train_full, y_train)

print(f'Treino original : {X_train_full.shape[0]} amostras')
print(f'Treino SMOTE   : {X_train_sm.shape[0]} amostras')
print()
print('Distribuicao por classe:')
dist_antes  = Counter(y_train)
dist_depois = Counter(y_train_sm)
for i, cls in enumerate(CLASS_NAMES):
    print(f'  {cls}: {dist_antes[i]:>4} -> {dist_depois[i]:>4}')

# RF sem class_weight (dados ja balanceados)
rf_sm = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_sm.fit(X_train_sm, y_train_sm)

# KNN com dados balanceados
knn_sm = KNeighborsClassifier(n_neighbors=11, n_jobs=-1)
knn_sm.fit(X_train_sm, y_train_sm)

res_rf_sm  = avaliar_modelo('RF  — SMOTE',        rf_sm,  X_train_sm, y_train_sm, X_val_full, y_val)
res_knn_sm = avaliar_modelo('KNN — SMOTE (k=11)', knn_sm, X_train_sm, y_train_sm, X_val_full, y_val)

resultados_B = [res_rf_sm, res_knn_sm]"""
))

# ── Confusão Rodada B ─────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""plot_confusion(resultados_B, 'Rodada B — Matrizes de Confusao (val)')"""
))

# ── Markdown comparação ───────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell(
    "### Comparação geral — Rodada A vs Rodada B"
))

# ── Tabela comparação ─────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell(
"""todos = resultados_A + resultados_B
plot_comparacao(todos)

header = f"{'Modelo':<25} {'Acc_tr':>7} {'Acc_val':>8} {'D_acc':>7} {'F1_tr':>7} {'F1_val':>8} {'D_f1':>6} {'AUC_val':>9}"
print(header)
print('-' * len(header))
for r in todos:
    print(f"{r['nome']:<25} {r['acc_tr']:>7.4f} {r['acc_val']:>8.4f} {r['acc_tr']-r['acc_val']:>+7.4f} "
          f"{r['f1_tr']:>7.4f} {r['f1_val']:>8.4f} {r['f1_tr']-r['f1_val']:>+6.4f} {r['auc_val']:>9.4f}")"""
))

nb.cells.extend(cells)
nbformat.write(nb, 'notebooks/projeto.ipynb')
print(f'OK — {len(nb.cells)} células totais, {len(cells)} células adicionadas.')
