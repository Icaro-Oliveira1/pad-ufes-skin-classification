#!/usr/bin/env python3
"""
Experimento de validação do impacto de `biopsed` (data leakage).

Treina os DOIS modelos finais do projeto (RF Tuned e DT Tuned) em duas
configurações cada:
  - COM biopsed   (X_*_full      → 95 features)
  - SEM biopsed   (X_*_full_nb   → 94 features)

Replica EXATAMENTE o pré-processamento do notebook:
  - split estratificado 70/15/15 (random_state=42)
  - imputação diferenciada por coluna (mediana por classe / global / moda)
  - OHE em region/background_father/background_mother
  - StandardScaler nas 4 numéricas
  - seleção via Mutual Information (threshold 0.005)
  - extração de 55 features PDI (com cache em disco)
  - SMOTE + RF/DT com os melhores hiperparâmetros do tuning

Uso:
  python scripts/biopsed_experiment.py

Saída:
  - Tabela comparativa F1-macro / Acurácia / AUC-ROC no TESTE
  - F1 por classe (com vs sem biopsed) para os dois modelos
  - Salvo também em outputs/biopsed_experiment.txt
"""
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")

# ── Caminhos ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
METADATA = DATA_DIR / "metadata.csv"
IMAGES_DIR = DATA_DIR / "images"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

PDI_CACHE = OUT_DIR / "pdi_features_cache.npz"
RESULT_FILE = OUT_DIR / "biopsed_experiment.txt"

# ── Constantes do projeto ─────────────────────────────────────────────────────
CLASS_ORDER = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
MALIGNANT = ["BCC", "SCC", "MEL"]

BOOL_COLS = [
    "smoke", "drink", "pesticide", "skin_cancer_history", "cancer_history",
    "has_piped_water", "has_sewage_system",
    "itch", "grew", "hurt", "changed", "bleed", "elevation", "biopsed",
]

# Best params do tuning (notebook cell 73)
RF_BEST_PARAMS = dict(
    n_estimators=563,
    max_depth=15,
    max_features=0.3,
    min_samples_leaf=4,
    class_weight=None,
    random_state=42,
    n_jobs=-1,
)

# DT: faremos GridSearch breve (mesmo grid do notebook cell 78) — params não
# foram salvos no run_log. O grid é pequeno (80 combinações × 5 folds).
DT_PARAM_GRID = {
    "dt__max_depth":        [3, 5, 7, 10, 15],
    "dt__min_samples_leaf": [2, 5, 10, 20],
    "dt__criterion":        ["gini", "entropy"],
    "dt__class_weight":     [None, "balanced"],
}

IMG_SIZE_PDI = (128, 128)

# ── Banner ────────────────────────────────────────────────────────────────────
def banner(msg, char="=", width=70):
    print()
    print(char * width)
    print(msg)
    print(char * width)


# ═════════════════════════════════════════════════════════════════════════════
# 1) CARREGAMENTO + LIMPEZA  (replica cell 4)
# ═════════════════════════════════════════════════════════════════════════════
def load_data():
    banner("1) Carregando dataset")
    df = pd.read_csv(METADATA)

    for col in BOOL_COLS:
        df[col] = df[col].map(
            {"True": True, "False": False, True: True, False: False}
        )

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["diameter_1"] = pd.to_numeric(df["diameter_1"], errors="coerce")
    df["diameter_2"] = pd.to_numeric(df["diameter_2"], errors="coerce")
    df["fitspatrick"] = pd.to_numeric(df["fitspatrick"], errors="coerce")

    df["img_path"] = df["img_id"].apply(
        lambda i: str(IMAGES_DIR / i) if (IMAGES_DIR / i).exists() else None
    )
    df = df.dropna(subset=["img_path"]).reset_index(drop=True)

    print(f"  Shape final: {df.shape}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 2) SPLIT ESTRATIFICADO 70/15/15  (replica cell 34)
# ═════════════════════════════════════════════════════════════════════════════
def split_data(df):
    from sklearn.model_selection import train_test_split

    banner("2) Split estratificado 70/15/15")
    df_train, df_temp = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["diagnostic"]
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=0.50, random_state=42, stratify=df_temp["diagnostic"]
    )
    print(f"  Treino : {len(df_train)}")
    print(f"  Val    : {len(df_val)}")
    print(f"  Teste  : {len(df_test)}")
    return df_train, df_val, df_test


# ═════════════════════════════════════════════════════════════════════════════
# 3) FEATURE ENGINEERING + IMPUTAÇÃO  (replica cells 38, 40)
# ═════════════════════════════════════════════════════════════════════════════
def add_diameter_features(frame):
    f = frame.copy()
    f["diameter_mean"] = (f["diameter_1"] + f["diameter_2"]) / 2
    f["diameter_area_est"] = np.pi * (f["diameter_1"] / 2) * (f["diameter_2"] / 2)
    return f


def preprocess_tabular(df_train, df_val, df_test):
    banner("3) Feature engineering + imputação")

    # Feature engineering
    df_train = add_diameter_features(df_train)
    df_val = add_diameter_features(df_val)
    df_test = add_diameter_features(df_test)

    # Background: BRASIL→BRAZIL, UNK→UNKNOWN, NaN→UNKNOWN
    BACK_FIX = {"BRASIL": "BRAZIL", "UNK": "UNKNOWN"}
    for d in [df_train, df_val, df_test]:
        for col in ["background_father", "background_mother"]:
            d[col] = d[col].replace(BACK_FIX).fillna("UNKNOWN")

    # age: mediana por classe (treino)
    age_median_by_class = df_train.groupby("diagnostic")["age"].median()
    for d in [df_train, df_val, df_test]:
        mask = d["age"].isnull()
        d.loc[mask, "age"] = d.loc[mask, "diagnostic"].map(age_median_by_class)

    # Numéricas: mediana global do treino
    num_cols = ["diameter_mean", "diameter_area_est", "fitspatrick"]
    num_medians = df_train[num_cols].median()
    for col in num_cols:
        df_train[col] = df_train[col].fillna(num_medians[col])
        df_val[col] = df_val[col].fillna(num_medians[col])
        df_test[col] = df_test[col].fillna(num_medians[col])

    # gender: moda
    gmode = df_train["gender"].mode()[0]
    for d in [df_train, df_val, df_test]:
        d["gender"] = d["gender"].fillna(gmode)

    # Booleanas: False
    for col in BOOL_COLS:
        df_train[col] = df_train[col].fillna(False)
        df_val[col] = df_val[col].fillna(False)
        df_test[col] = df_test[col].fillna(False)

    print("  Imputação concluída — sem nulos restantes")
    return df_train, df_val, df_test


# ═════════════════════════════════════════════════════════════════════════════
# 4) ENCODING + SCALING + MI SELECTION  (replica cells 42, 44, 48, 49)
# ═════════════════════════════════════════════════════════════════════════════
def build_features(df_train, df_val, df_test):
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
    from sklearn.feature_selection import mutual_info_classif

    banner("4) Encoding + Scaling + Mutual Information")

    # Gender → 0/1
    gender_map = {"FEMALE": 0, "MALE": 1}
    for d in [df_train, df_val, df_test]:
        d["gender"] = d["gender"].map(gender_map)

    # Target
    le_target = LabelEncoder().fit(df_train["diagnostic"])
    y_train = le_target.transform(df_train["diagnostic"])
    y_val = le_target.transform(df_val["diagnostic"])
    y_test = le_target.transform(df_test["diagnostic"])

    # OHE
    ohe_cols = ["region", "background_father", "background_mother"]
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.float32)
    ohe.fit(df_train[ohe_cols])
    ohe_names = ohe.get_feature_names_out(ohe_cols).tolist()

    def apply_ohe(d):
        return pd.DataFrame(ohe.transform(d[ohe_cols]), columns=ohe_names, index=d.index)

    ohe_train = apply_ohe(df_train)
    ohe_val = apply_ohe(df_val)
    ohe_test = apply_ohe(df_test)

    # StandardScaler nas 4 numéricas
    NUMERIC_COLS = ["age", "fitspatrick", "diameter_mean", "diameter_area_est"]
    BOOL_COLS_OUT = [c for c in BOOL_COLS if c != "biopsed"] + ["biopsed", "gender"]
    scaler = StandardScaler().fit(df_train[NUMERIC_COLS])

    def build_X(d, d_ohe):
        num = pd.DataFrame(scaler.transform(d[NUMERIC_COLS]),
                           columns=NUMERIC_COLS, index=d.index).reset_index(drop=True)
        boo = d[BOOL_COLS_OUT].astype(np.float32).reset_index(drop=True)
        oh = d_ohe.reset_index(drop=True)
        return pd.concat([num, boo, oh], axis=1)

    X_train = build_X(df_train, ohe_train)
    X_val = build_X(df_val, ohe_val)
    X_test = build_X(df_test, ohe_test)

    # MI selection (threshold 0.005)
    bool_and_ohe = set(BOOL_COLS_OUT) | set(ohe_names)
    discrete_mask = np.array([c in bool_and_ohe for c in X_train.columns])
    mi = mutual_info_classif(X_train, y_train, discrete_features=discrete_mask,
                             random_state=42, n_neighbors=5)
    mi_series = pd.Series(mi, index=X_train.columns).sort_values(ascending=False)
    keep = mi_series[mi_series >= 0.005].index.tolist()

    X_train_sel = X_train[keep]
    X_val_sel = X_val[keep]
    X_test_sel = X_test[keep]

    keep_nb = [c for c in keep if c != "biopsed"]
    X_train_nb = X_train_sel[keep_nb]
    X_val_nb = X_val_sel[keep_nb]
    X_test_nb = X_test_sel[keep_nb]

    print(f"  Features após MI (com biopsed) : {X_train_sel.shape[1]}")
    print(f"  Features sem biopsed           : {X_train_nb.shape[1]}")
    print(f"  biopsed mantida pelo MI?       : {'biopsed' in keep}")

    return (X_train_sel, X_val_sel, X_test_sel,
            X_train_nb, X_val_nb, X_test_nb,
            y_train, y_val, y_test, le_target)


# ═════════════════════════════════════════════════════════════════════════════
# 5) EXTRAÇÃO PDI  (com cache)  — replica cells 56-57, 59
# ═════════════════════════════════════════════════════════════════════════════
def extract_pdi_features(df_train, df_val, df_test):
    import cv2
    from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
    from scipy.stats import skew as scipy_skew
    from sklearn.preprocessing import StandardScaler

    banner("5) Extração de features PDI (55 por imagem)")

    PDI_NAMES = (
        [f"rgb_{ch}_{s}" for ch in ["R", "G", "B"] for s in ["mean", "std", "skew"]]
        + [f"hsv_{ch}_{s}" for ch in ["H", "S", "V"] for s in ["mean", "std", "skew"]]
        + ["glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
           "glcm_energy", "glcm_correlation", "glcm_ASM"]
        + [f"lbp_bin_{i}" for i in range(10)]
        + ["edge_density", "sobel_mean", "sobel_std", "asym_horizontal", "asym_vertical"]
        + [f"gabor_t{int(np.degrees(t)):03d}_s{s}_{stat}"
           for t in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
           for s in ["fine", "coarse"]
           for stat in ["mean", "std"]]
    )
    assert len(PDI_NAMES) == 55

    GABOR_PARAMS = [
        (theta, lam, sig)
        for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        for (lam, sig) in [(5, 2), (10, 4)]
    ]

    def extract_one(img_path):
        img_rgb = np.array(Image.open(img_path).convert("RGB").resize(IMG_SIZE_PDI, Image.LANCZOS))
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        feats = []
        # Cor
        rgb = img_rgb.astype(np.float32)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        for arr in [rgb, hsv]:
            for ch in range(3):
                v = arr[:, :, ch].ravel()
                feats += [float(np.mean(v)), float(np.std(v)), float(scipy_skew(v))]
        # GLCM
        gray64 = (gray // 4).astype(np.uint8)
        glcm = graycomatrix(gray64, distances=[1],
                            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                            levels=64, symmetric=True, normed=True)
        for prop in ["contrast", "dissimilarity", "homogeneity",
                     "energy", "correlation", "ASM"]:
            feats.append(float(graycoprops(glcm, prop).mean()))
        # LBP
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
        feats += hist.tolist()
        # Bordas
        edges = cv2.Canny(gray, 50, 150)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobelx ** 2 + sobely ** 2)
        h, w = gray.shape
        feats += [
            float((edges > 0).mean()),
            float(mag.mean()),
            float(mag.std()),
            float(np.abs(gray[:, :w // 2].mean() - gray[:, w // 2:].mean())),
            float(np.abs(gray[:h // 2, :].mean() - gray[h // 2:, :].mean())),
        ]
        # Gabor
        img_f = gray.astype(np.float64)
        for theta, lam, sig in GABOR_PARAMS:
            kernel = cv2.getGaborKernel((21, 21), sig, theta, lam, 0.5, 0, cv2.CV_64F)
            resp = np.abs(cv2.filter2D(img_f, cv2.CV_64F, kernel))
            feats += [float(resp.mean()), float(resp.std())]
        return np.array(feats, dtype=np.float32)

    def extract_split(df_split, name):
        rows = []
        n = len(df_split)
        t0 = time.time()
        for i, p in enumerate(df_split["img_path"]):
            rows.append(extract_one(p))
            if (i + 1) % 200 == 0:
                print(f"    {name}: {i + 1}/{n}  ({time.time() - t0:.1f}s)")
        return pd.DataFrame(rows, columns=PDI_NAMES, index=df_split.index)

    # Cache check
    if PDI_CACHE.exists():
        print(f"  Cache encontrado: {PDI_CACHE}")
        cache = np.load(PDI_CACHE, allow_pickle=True)
        # Validar que o split bate
        if (len(cache["train"]) == len(df_train)
                and len(cache["val"]) == len(df_val)
                and len(cache["test"]) == len(df_test)):
            print("  Cache válido (mesmos tamanhos de split). Reusando.")
            pdi_train = pd.DataFrame(cache["train"], columns=PDI_NAMES, index=df_train.index)
            pdi_val = pd.DataFrame(cache["val"], columns=PDI_NAMES, index=df_val.index)
            pdi_test = pd.DataFrame(cache["test"], columns=PDI_NAMES, index=df_test.index)
        else:
            print("  Cache inválido (tamanhos divergem). Re-extraindo.")
            pdi_train = extract_split(df_train, "treino")
            pdi_val = extract_split(df_val, "val   ")
            pdi_test = extract_split(df_test, "teste ")
            np.savez(PDI_CACHE, train=pdi_train.values, val=pdi_val.values, test=pdi_test.values)
    else:
        print("  Cache não encontrado — extraindo (~2 min)")
        pdi_train = extract_split(df_train, "treino")
        pdi_val = extract_split(df_val, "val   ")
        pdi_test = extract_split(df_test, "teste ")
        np.savez(PDI_CACHE, train=pdi_train.values, val=pdi_val.values, test=pdi_test.values)
        print(f"  Cache salvo em {PDI_CACHE}")

    # Scaling PDI
    pdi_scaler = StandardScaler().fit(pdi_train)
    pdi_train_sc = pd.DataFrame(pdi_scaler.transform(pdi_train), columns=PDI_NAMES, index=pdi_train.index)
    pdi_val_sc = pd.DataFrame(pdi_scaler.transform(pdi_val), columns=PDI_NAMES, index=pdi_val.index)
    pdi_test_sc = pd.DataFrame(pdi_scaler.transform(pdi_test), columns=PDI_NAMES, index=pdi_test.index)

    return pdi_train_sc, pdi_val_sc, pdi_test_sc


# ═════════════════════════════════════════════════════════════════════════════
# 6) MONTAR X_full / X_full_nb
# ═════════════════════════════════════════════════════════════════════════════
def assemble_full(X_tab, pdi):
    return pd.concat([X_tab.reset_index(drop=True),
                      pdi.reset_index(drop=True)], axis=1)


# ═════════════════════════════════════════════════════════════════════════════
# 7) TREINO + AVALIAÇÃO
# ═════════════════════════════════════════════════════════════════════════════
def train_and_eval(name, model_pipe, X_train, y_train, X_test, y_test, class_names):
    from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

    t0 = time.time()
    model_pipe.fit(X_train, y_train)
    elapsed = time.time() - t0

    pred = model_pipe.predict(X_test)
    prob = model_pipe.predict_proba(X_test)
    f1m = f1_score(y_test, pred, average="macro", zero_division=0)
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, prob, multi_class="ovr", average="macro")
    f1_per_class = f1_score(y_test, pred, average=None,
                            labels=list(range(len(class_names))), zero_division=0)

    print(f"  [{name}] treinado em {elapsed:.1f}s — "
          f"F1={f1m:.4f}  Acc={acc:.4f}  AUC={auc:.4f}")
    return dict(name=name, f1=f1m, acc=acc, auc=auc, f1_per_class=f1_per_class)


def make_rf_pipeline():
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    from sklearn.ensemble import RandomForestClassifier

    return ImbPipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=5)),
        ("rf", RandomForestClassifier(**RF_BEST_PARAMS)),
    ])


def make_dt_pipeline_search(X_train, y_train):
    """Roda GridSearchCV para encontrar os melhores params do DT (mesmo grid do notebook)."""
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.metrics import f1_score, make_scorer

    pipe = ImbPipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=5)),
        ("dt", DecisionTreeClassifier(random_state=42)),
    ])
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score, average="macro", zero_division=0)
    search = GridSearchCV(pipe, param_grid=DT_PARAM_GRID, scoring=scorer,
                          cv=cv5, refit=True, n_jobs=-1, verbose=0)
    print("  Rodando GridSearchCV do DT (80 combinações × 5 folds)...")
    t0 = time.time()
    search.fit(X_train, y_train)
    print(f"  GridSearch DT concluído em {time.time() - t0:.1f}s  "
          f"|  Melhor CV F1: {search.best_score_:.4f}")
    print(f"  Melhores params: {search.best_params_}")
    return search.best_params_


def make_dt_pipeline(best_params):
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    from sklearn.tree import DecisionTreeClassifier

    dt_kwargs = {k.replace("dt__", ""): v for k, v in best_params.items()}
    dt_kwargs["random_state"] = 42
    return ImbPipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=5)),
        ("dt", DecisionTreeClassifier(**dt_kwargs)),
    ])


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()

    df = load_data()
    df_train, df_val, df_test = split_data(df)
    df_train, df_val, df_test = preprocess_tabular(df_train, df_val, df_test)

    (X_train_sel, X_val_sel, X_test_sel,
     X_train_nb, X_val_nb, X_test_nb,
     y_train, y_val, y_test, le_target) = build_features(df_train, df_val, df_test)

    pdi_train_sc, pdi_val_sc, pdi_test_sc = extract_pdi_features(df_train, df_val, df_test)

    # Vetores finais
    X_train_full = assemble_full(X_train_sel, pdi_train_sc)
    X_test_full = assemble_full(X_test_sel, pdi_test_sc)
    X_train_full_nb = assemble_full(X_train_nb, pdi_train_sc)
    X_test_full_nb = assemble_full(X_test_nb, pdi_test_sc)

    print(f"\n  X_train_full    : {X_train_full.shape}")
    print(f"  X_train_full_nb : {X_train_full_nb.shape}")

    class_names = list(le_target.classes_)

    # ── DT: rodar GridSearch UMA vez na versão com biopsed (canônica) ─────────
    banner("6) GridSearch DT (com biopsed) para fixar hiperparâmetros")
    dt_best_params = make_dt_pipeline_search(X_train_full, y_train)

    # ── Treinar os 4 modelos ──────────────────────────────────────────────────
    banner("7) Treinando os 4 modelos finais")
    results = []

    print("\n  RF Tuned (best params do tuning original):")
    print(f"    {RF_BEST_PARAMS}")

    results.append(train_and_eval(
        "RF com biopsed", make_rf_pipeline(),
        X_train_full, y_train, X_test_full, y_test, class_names
    ))
    results.append(train_and_eval(
        "RF sem biopsed", make_rf_pipeline(),
        X_train_full_nb, y_train, X_test_full_nb, y_test, class_names
    ))
    results.append(train_and_eval(
        "DT com biopsed", make_dt_pipeline(dt_best_params),
        X_train_full, y_train, X_test_full, y_test, class_names
    ))
    results.append(train_and_eval(
        "DT sem biopsed", make_dt_pipeline(dt_best_params),
        X_train_full_nb, y_train, X_test_full_nb, y_test, class_names
    ))

    # ── Tabela comparativa ────────────────────────────────────────────────────
    banner("8) RESULTADOS — IMPACTO DE biopsed (CONJUNTO DE TESTE)")

    lines = []
    lines.append(f"{'Modelo':<20} {'F1-macro':>10} {'Acurácia':>10} {'AUC-ROC':>10}")
    lines.append("-" * 55)
    for r in results:
        lines.append(f"{r['name']:<20} {r['f1']:>10.4f} {r['acc']:>10.4f} {r['auc']:>10.4f}")
    lines.append("")

    # Deltas
    rf_com, rf_sem = results[0], results[1]
    dt_com, dt_sem = results[2], results[3]
    lines.append("Deltas (com - sem biopsed):")
    lines.append(f"  RF :  ΔF1 = {rf_com['f1'] - rf_sem['f1']:+.4f}   "
                 f"ΔAUC = {rf_com['auc'] - rf_sem['auc']:+.4f}")
    lines.append(f"  DT :  ΔF1 = {dt_com['f1'] - dt_sem['f1']:+.4f}   "
                 f"ΔAUC = {dt_com['auc'] - dt_sem['auc']:+.4f}")
    lines.append("")

    # Interpretação
    def interpret(delta_f1, name):
        if delta_f1 > 0.05:
            return f"  {name}: queda > 5pp → leakage RELEVANTE — reportar ambas as versões"
        elif delta_f1 > 0.02:
            return f"  {name}: queda 2–5pp → contribuição MODERADA — incluir nota"
        else:
            return f"  {name}: queda < 2pp → diferença NEGLIGENCIÁVEL — biopsed é preditor legítimo"

    lines.append("Interpretação:")
    lines.append(interpret(rf_com["f1"] - rf_sem["f1"], "RF"))
    lines.append(interpret(dt_com["f1"] - dt_sem["f1"], "DT"))
    lines.append("")

    # F1 por classe
    lines.append("F1 por classe (com vs sem biopsed):")
    lines.append(f"  {'Classe':<8} {'RF com':>8} {'RF sem':>8} {'Δ':>8}   "
                 f"{'DT com':>8} {'DT sem':>8} {'Δ':>8}")
    lines.append("  " + "-" * 60)
    for j, cls in enumerate(class_names):
        lines.append(
            f"  {cls:<8} "
            f"{rf_com['f1_per_class'][j]:>8.3f} {rf_sem['f1_per_class'][j]:>8.3f} "
            f"{rf_com['f1_per_class'][j] - rf_sem['f1_per_class'][j]:>+8.3f}   "
            f"{dt_com['f1_per_class'][j]:>8.3f} {dt_sem['f1_per_class'][j]:>8.3f} "
            f"{dt_com['f1_per_class'][j] - dt_sem['f1_per_class'][j]:>+8.3f}"
        )

    output = "\n".join(lines)
    print()
    print(output)

    # Salvar
    with open(RESULT_FILE, "w") as f:
        f.write(f"Experimento de validação do impacto de `biopsed`\n")
        f.write(f"Gerado por scripts/biopsed_experiment.py\n\n")
        f.write(output + "\n")
    print(f"\n  → Resultados salvos em: {RESULT_FILE}")

    print(f"\nTempo total: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
