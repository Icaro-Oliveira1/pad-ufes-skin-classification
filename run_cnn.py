#!/usr/bin/env python3
# Execução do pipeline PAD-UFES: dados → RF/DT → CNN → comparativo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os, json as _json
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'notebooks'))
OUT_DIR = '../outputs'
os.makedirs(OUT_DIR, exist_ok=True)
_fig_counter = [0]
def _savefig(name=None):
    _fig_counter[0] += 1
    fname = name or f'fig_{_fig_counter[0]:03d}'
    plt.savefig(f'{OUT_DIR}/{fname}.png', dpi=100, bbox_inches='tight')
    plt.close('all')


# === CELL 1 ===
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from PIL import Image

sns.set_theme(style='whitegrid', palette='muted')
pd.set_option('display.max_columns', None)

CLASS_ORDER  = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
CLASS_COLORS = dict(zip(CLASS_ORDER, sns.color_palette('tab10', len(CLASS_ORDER))))

MALIGNANT = ['BCC', 'SCC', 'MEL']
BENIGN    = ['ACK', 'NEV', 'SEK']

CLASS_LABELS = {
    'ACK': 'Actinic Keratosis',
    'BCC': 'Basal Cell Carcinoma',
    'MEL': 'Melanoma',
    'NEV': 'Nevus',
    'SCC': 'Squamous Cell Carcinoma',
    'SEK': 'Seborrheic Keratosis'
}

# === CELL 2 ===
BASE_DIR   = Path('..') / 'data' / 'raw'
METADATA   = BASE_DIR / 'metadata.csv'
IMAGES_DIR = BASE_DIR / 'images'

print(f'Metadata : {METADATA.resolve()}')
print(f'Imagens  : {IMAGES_DIR.resolve()}')
print(f'Metadata existe        : {METADATA.exists()}')
print(f'Pasta de imagens existe: {IMAGES_DIR.exists()}')

# === CELL 4 ===
df = pd.read_csv(METADATA)

bool_cols = [
    'smoke', 'drink', 'pesticide', 'skin_cancer_history', 'cancer_history',
    'has_piped_water', 'has_sewage_system', 'itch', 'grew', 'hurt',
    'changed', 'bleed', 'elevation', 'biopsed'
]
for col in bool_cols:
    df[col] = df[col].map({'True': True, 'False': False, True: True, False: False})

df['age']         = pd.to_numeric(df['age'],         errors='coerce')
df['diameter_1']  = pd.to_numeric(df['diameter_1'],  errors='coerce')
df['diameter_2']  = pd.to_numeric(df['diameter_2'],  errors='coerce')
df['fitspatrick'] = pd.to_numeric(df['fitspatrick'], errors='coerce')

df['grupo'] = df['diagnostic'].apply(lambda x: 'Maligno' if x in MALIGNANT else 'Benigno')

def resolve_image(img_id):
    path = IMAGES_DIR / img_id
    return str(path) if path.exists() else None

df['img_path'] = df['img_id'].apply(resolve_image)
df = df.dropna(subset=['img_path']).reset_index(drop=True)

print(f'Shape final: {df.shape}')
df.head(3)

# === CELL 32 ===
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection  import mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.impute   import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# === CELL 34 ===
# ⚠️  biopsed: potencial data leakage
# A biópsia é frequentemente solicitada PORQUE a lesão já é suspeita → correlacionada
# com diagnósticos malignos. Será incluída no conjunto de features, mas os modelos
# serão avaliados COM e SEM essa coluna para medir o impacto real.

# Split inicial: 70% treino, 30% temporário
df_train, df_temp = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df['diagnostic']
)
# Divide o temporário em 50/50 → 15% val + 15% teste
df_val, df_test = train_test_split(
    df_temp, test_size=0.50, random_state=42, stratify=df_temp['diagnostic']
)

print(f'Treino : {len(df_train):>5}  ({len(df_train)/len(df)*100:.1f}%)')
print(f'Val    : {len(df_val):>5}  ({len(df_val)/len(df)*100:.1f}%)')
print(f'Teste  : {len(df_test):>5}  ({len(df_test)/len(df)*100:.1f}%)')
print()
print('Proporção por classe (treino / val / teste):')
comp = pd.DataFrame({
    'treino_%': df_train['diagnostic'].value_counts(normalize=True).reindex(CLASS_ORDER)*100,
    'val_%'   : df_val['diagnostic'].value_counts(normalize=True).reindex(CLASS_ORDER)*100,
    'teste_%' : df_test['diagnostic'].value_counts(normalize=True).reindex(CLASS_ORDER)*100,
}).round(1)
print(comp.to_string())


# === CELL 38 ===
def add_diameter_features(frame):
    """Cria diameter_mean e diameter_area_est. Operação segura com NaN."""
    f = frame.copy()
    f['diameter_mean']     = (f['diameter_1'] + f['diameter_2']) / 2
    f['diameter_area_est'] = np.pi * (f['diameter_1'] / 2) * (f['diameter_2'] / 2)
    return f

df_train = add_diameter_features(df_train)
df_val   = add_diameter_features(df_val)
df_test  = add_diameter_features(df_test)

print('Features criadas: diameter_mean, diameter_area_est')
print(df_train[['diameter_1','diameter_2','diameter_mean','diameter_area_est']].describe().round(2))

# === CELL 40 ===
# ── Estratégia por coluna ────────────────────────────────────────────────
# age            → mediana por classe (distribuição assimétrica)
# diameter_mean/area → mediana global (correlação alta, poucas diferenças entre classes)
# fitspatrick    → moda global (ordinal com valor dominante)
# gender         → moda global
# background_*   → "UNKNOWN" (categórico, nulo é informativo)
# booleanas      → False (ausência de registro = ausência do sintoma/histórico)

bool_feature_cols = [
    "smoke", "drink", "pesticide", "skin_cancer_history", "cancer_history",
    "has_piped_water", "has_sewage_system",
    "itch", "grew", "hurt", "changed", "bleed", "elevation", "biopsed"
]

# 4.1 Limpeza de categorias duplicadas em background
BACK_FIX = {"BRASIL": "BRAZIL", "UNK": "UNKNOWN"}

def clean_background(frame):
    f = frame.copy()
    for col in ["background_father", "background_mother"]:
        f[col] = f[col].replace(BACK_FIX).fillna("UNKNOWN")
    return f

df_train = clean_background(df_train)
df_val   = clean_background(df_val)
df_test  = clean_background(df_test)

# 4.2 Imputar age: mediana por classe calculada no TREINO
age_median_by_class = df_train.groupby("diagnostic")["age"].median()

def impute_age(frame, median_map):
    f = frame.copy()
    mask = f["age"].isnull()
    f.loc[mask, "age"] = f.loc[mask, "diagnostic"].map(median_map)
    return f

df_train = impute_age(df_train, age_median_by_class)
df_val   = impute_age(df_val,   age_median_by_class)
df_test  = impute_age(df_test,  age_median_by_class)

# 4.3 Imputar numéricos: mediana global do TREINO
# NOTA: usar atribuição explícita — fillna(inplace=True) em chained
# indexing (split[col].fillna) não propaga para o DataFrame original.
num_impute_cols = ["diameter_mean", "diameter_area_est", "fitspatrick"]
num_medians = df_train[num_impute_cols].median()

for col in num_impute_cols:
    df_train[col] = df_train[col].fillna(num_medians[col])
    df_val[col]   = df_val[col].fillna(num_medians[col])
    df_test[col]  = df_test[col].fillna(num_medians[col])

# 4.4 Imputar gender: moda do TREINO
gender_mode = df_train["gender"].mode()[0]
df_train["gender"] = df_train["gender"].fillna(gender_mode)
df_val["gender"]   = df_val["gender"].fillna(gender_mode)
df_test["gender"]  = df_test["gender"].fillna(gender_mode)

# 4.5 Imputar booleanas: False
for col in bool_feature_cols:
    df_train[col] = df_train[col].fillna(False)
    df_val[col]   = df_val[col].fillna(False)
    df_test[col]  = df_test[col].fillna(False)

# Verificação de integridade
cols_to_check = num_impute_cols + ["age", "gender"] + bool_feature_cols
remaining_nulls = pd.concat([df_train, df_val, df_test])[cols_to_check].isnull().sum()
remaining_nulls = remaining_nulls[remaining_nulls > 0]
if remaining_nulls.empty:
    print("Nenhum nulo restante nas features imputadas.")
else:
    print("Nulos ainda presentes:")
    print(remaining_nulls)


# === CELL 42 ===
# 5.1 Gender: FEMALE→0, MALE→1
gender_map = {'FEMALE': 0, 'MALE': 1}
for split in [df_train, df_val, df_test]:
    split['gender'] = split['gender'].map(gender_map)

# 5.2 Target: LabelEncoder (ACK→0, BCC→1, MEL→2, NEV→3, SCC→4, SEK→5)
le_target = LabelEncoder()
le_target.fit(df_train['diagnostic'])

y_train = le_target.transform(df_train['diagnostic'])
y_val   = le_target.transform(df_val['diagnostic'])
y_test  = le_target.transform(df_test['diagnostic'])

print('Mapeamento de classes:')
for i, cls in enumerate(le_target.classes_):
    print(f'  {i} → {cls}')

# 5.3 OHE para region, background_father, background_mother
# Ajustado no TREINO — categorias desconhecidas no val/test recebem zeros
ohe_cols = ['region', 'background_father', 'background_mother']

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.float32)
ohe.fit(df_train[ohe_cols])

ohe_feature_names = ohe.get_feature_names_out(ohe_cols).tolist()

def apply_ohe(frame, encoder, cols, feat_names):
    encoded = encoder.transform(frame[cols])
    return pd.DataFrame(encoded, columns=feat_names, index=frame.index)

df_train_ohe = apply_ohe(df_train, ohe, ohe_cols, ohe_feature_names)
df_val_ohe   = apply_ohe(df_val,   ohe, ohe_cols, ohe_feature_names)
df_test_ohe  = apply_ohe(df_test,  ohe, ohe_cols, ohe_feature_names)

print(f'\nFeatures após OHE: {len(ohe_feature_names)}')
print('Exemplos:', ohe_feature_names[:8], '...')

# === CELL 44 ===
NUMERIC_COLS  = ['age', 'fitspatrick', 'diameter_mean', 'diameter_area_est']
BOOL_COLS_OUT = [c for c in bool_feature_cols if c != 'biopsed'] + ['biopsed', 'gender']

scaler = StandardScaler()
scaler.fit(df_train[NUMERIC_COLS])

def build_feature_matrix(frame, frame_ohe, num_cols, bool_cols):
    """Monta a matriz de features: numéricas escaladas + booleanas + OHE.
    frame_ohe deve ter o mesmo índice que frame.
    """
    num_scaled = pd.DataFrame(
        scaler.transform(frame[num_cols]),
        columns=num_cols,
        index=frame.index
    )
    bool_part = frame[bool_cols].astype(np.float32).reset_index(drop=True)
    num_part  = num_scaled.reset_index(drop=True)
    ohe_part  = frame_ohe.reset_index(drop=True)
    return pd.concat([num_part, bool_part, ohe_part], axis=1)

X_train = build_feature_matrix(df_train, df_train_ohe, NUMERIC_COLS, BOOL_COLS_OUT)
X_val   = build_feature_matrix(df_val,   df_val_ohe,   NUMERIC_COLS, BOOL_COLS_OUT)
X_test  = build_feature_matrix(df_test,  df_test_ohe,  NUMERIC_COLS, BOOL_COLS_OUT)

# Garantia explícita: nenhum NaN deve chegar nos modelos
assert X_train.isnull().sum().sum() == 0, 'NaN em X_train!'
assert X_val.isnull().sum().sum()   == 0, 'NaN em X_val!'
assert X_test.isnull().sum().sum()  == 0, 'NaN em X_test!'

print(f'X_train : {X_train.shape}  |  NaN: 0')
print(f'X_val   : {X_val.shape}  |  NaN: 0')
print(f'X_test  : {X_test.shape}  |  NaN: 0')
print(f'\nComposição das {X_train.shape[1]} features:')
print(f'  Numéricas (escaladas) : {len(NUMERIC_COLS)}')
print(f'  Booleanas             : {len(BOOL_COLS_OUT)}')
print(f'  OHE                   : {len(ohe_feature_names)}')


# === CELL 46 ===
classes_arr = np.unique(y_train)
weights     = compute_class_weight('balanced', classes=classes_arr, y=y_train)

# Converter para int Python nativo — numpy.int64 pode causar incompatibilidade
# em alguns estimadores sklearn que esperam chaves int no class_weight dict.
class_weight_dict = {int(k): float(v) for k, v in zip(classes_arr, weights)}

print('Pesos por classe:')
for idx, cls_name in enumerate(le_target.classes_):
    grp = 'Maligno' if cls_name in MALIGNANT else 'Benigno'
    print(f'  {idx} ({cls_name} / {grp}): peso = {weights[idx]:.3f}')

fig, ax = plt.subplots(figsize=(7, 3))
bar_colors = ['#e07070' if c in MALIGNANT else '#70a0e0' for c in le_target.classes_]
ax.bar(le_target.classes_, weights, color=bar_colors, edgecolor='white')
ax.set_title('Pesos de Classe (balanced) — quanto maior, mais penalizado o erro')
ax.set_ylabel('Peso')
ax.set_xlabel('Diagnóstico')
for i_bar, (cls, w) in enumerate(zip(le_target.classes_, weights)):
    ax.text(i_bar, w + 0.02, f'{w:.2f}', ha='center', fontsize=9)
plt.tight_layout()
plt.show()


# === CELL 48 ===
# Máscara de features discretas: booleanas e OHE são discretas (0/1)
# Passá-las como discrete_features melhora a estimativa de MI para essas colunas.
bool_and_ohe_cols = set(BOOL_COLS_OUT) | set(ohe_feature_names)
discrete_mask = np.array([col in bool_and_ohe_cols for col in X_train.columns])

mi_scores = mutual_info_classif(
    X_train, y_train,
    discrete_features=discrete_mask,
    random_state=42,
    n_neighbors=5
)
mi_series = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

top25 = mi_series.head(25)
top25.plot(kind='barh', ax=axes[0], color='steelblue', edgecolor='white')
axes[0].invert_yaxis()
axes[0].set_title('Top 25 Features — Mutual Information com Target', fontsize=11)
axes[0].set_xlabel('MI Score')

bot15 = mi_series.tail(15)
bot15.plot(kind='barh', ax=axes[1], color='salmon', edgecolor='white')
axes[1].invert_yaxis()
axes[1].set_title('Bottom 15 Features — Menor Relevância', fontsize=11)
axes[1].set_xlabel('MI Score')

plt.tight_layout()
plt.show()

# Escolha do threshold: ponto de inflexão na curva de MI (elbow)
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(range(len(mi_series)), mi_series.values, marker='.', markersize=4)
ax.axhline(y=0.005, color='red', linestyle='--', label='Threshold = 0.005')
ax.set_title('Curva de MI — todas as features ordenadas (elbow para escolha do threshold)')
ax.set_xlabel('Feature rank')
ax.set_ylabel('MI Score')
ax.legend()
plt.tight_layout()
plt.show()

THRESHOLD_MI = 0.005
low_mi = mi_series[mi_series < THRESHOLD_MI]
print(f'Features com MI < {THRESHOLD_MI}: {len(low_mi)}')
print(low_mi.round(4).to_string())


# === CELL 49 ===
features_keep = mi_series[mi_series >= THRESHOLD_MI].index.tolist()
features_drop = mi_series[mi_series <  THRESHOLD_MI].index.tolist()

X_train_sel = X_train[features_keep]
X_val_sel   = X_val[features_keep]
X_test_sel  = X_test[features_keep]

# Versão sem biopsed — para experimento de leakage
# Avalia o quanto biopsed está inflando a performance dos modelos.
features_no_biopsed = [f for f in features_keep if f != 'biopsed']
X_train_nb = X_train_sel[features_no_biopsed]
X_val_nb   = X_val_sel[features_no_biopsed]
X_test_nb  = X_test_sel[features_no_biopsed]

print(f'Features antes da seleção         : {X_train.shape[1]}')
print(f'Features após seleção (MI >= 0.005): {X_train_sel.shape[1]}')
print(f'Features sem biopsed              : {X_train_nb.shape[1]}')
print(f'Features descartadas              : {len(features_drop)}')
print(f'\nDescartadas: {features_drop}')
print(f'\nbiopsed está em features_keep: {"biopsed" in features_keep}')


# === CELL 51 ===
IMG_SIZE_PDI = (128, 128)   # Para extração de features PDI
IMG_SIZE_CNN = (224, 224)   # Para CNN (padrão ImageNet / transfer learning)

# Estatísticas ImageNet — usadas na normalização para modelos pré-treinados
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image_pdi(img_path: str) -> np.ndarray:
    """Carrega, redimensiona e normaliza [0,1] para extração de features PDI."""
    img = Image.open(img_path).convert('RGB').resize(IMG_SIZE_PDI, Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0

def preprocess_image_cnn(img_path: str, imagenet_norm: bool = True) -> np.ndarray:
    """Carrega, redimensiona e normaliza com stats ImageNet para CNN."""
    img = Image.open(img_path).convert('RGB').resize(IMG_SIZE_CNN, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    if imagenet_norm:
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr

# Teste rápido de sanidade
sample_path = df_train['img_path'].iloc[0]
pdi_sample  = preprocess_image_pdi(sample_path)
cnn_sample  = preprocess_image_cnn(sample_path)

print(f'PDI sample — shape: {pdi_sample.shape}  min: {pdi_sample.min():.3f}  max: {pdi_sample.max():.3f}')
print(f'CNN sample — shape: {cnn_sample.shape}  min: {cnn_sample.min():.3f}  max: {cnn_sample.max():.3f}')

# Visualização lado a lado: original vs pdi vs cnn (revertido para exibição)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
orig = np.array(Image.open(sample_path).convert('RGB'))
axes[0].imshow(orig);        axes[0].set_title(f'Original\n{orig.shape[1]}×{orig.shape[0]}px'); axes[0].axis('off')
axes[1].imshow(pdi_sample);  axes[1].set_title(f'PDI resize\n{IMG_SIZE_PDI[0]}×{IMG_SIZE_PDI[1]}px'); axes[1].axis('off')
cnn_vis = np.clip(cnn_sample * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
axes[2].imshow(cnn_vis);     axes[2].set_title(f'CNN resize (revertido)\n{IMG_SIZE_CNN[0]}×{IMG_SIZE_CNN[1]}px'); axes[2].axis('off')
plt.suptitle('Pipeline de Imagem — Exemplo', fontsize=11)
plt.tight_layout()
plt.show()

# === CELL 56 ===
import cv2
import skimage
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy.stats import skew as scipy_skew
import time
from tqdm.auto import tqdm

# Confirma versões
print(f'OpenCV    : {cv2.__version__}')
print(f'scikit-image: {skimage.__version__}')

PDI_FEATURE_NAMES = (
    # Cor RGB (9)
    [f'rgb_{ch}_{stat}' for ch in ['R','G','B'] for stat in ['mean','std','skew']] +
    # Cor HSV (9)
    [f'hsv_{ch}_{stat}' for ch in ['H','S','V'] for stat in ['mean','std','skew']] +
    # GLCM (6)
    ['glcm_contrast','glcm_dissimilarity','glcm_homogeneity',
     'glcm_energy','glcm_correlation','glcm_ASM'] +
    # LBP (10)
    [f'lbp_bin_{i}' for i in range(10)] +
    # Bordas/Forma (5)
    ['edge_density','sobel_mean','sobel_std','asym_horizontal','asym_vertical'] +
    # Gabor (16: 4 orientações × 2 escalas × mean+std)
    [f'gabor_t{int(np.degrees(t)):03d}_s{s}_{stat}'
     for t in [0, np.pi/4, np.pi/2, 3*np.pi/4]
     for s in ['fine','coarse']
     for stat in ['mean','std']]
)

assert len(PDI_FEATURE_NAMES) == 55, f'Esperado 55, obtido {len(PDI_FEATURE_NAMES)}'
print(f'\nTotal features PDI: {len(PDI_FEATURE_NAMES)}')

# === CELL 57 ===
def extract_color_features(img_rgb: np.ndarray) -> list:
    """18 features: mean/std/skew por canal em RGB e HSV."""
    rgb  = img_rgb.astype(np.float32)
    hsv  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    feats = []
    for arr in [rgb, hsv]:
        for ch in range(3):
            vals = arr[:, :, ch].ravel()
            feats += [float(np.mean(vals)), float(np.std(vals)), float(scipy_skew(vals))]
    return feats  # 18 features


def extract_texture_features(gray: np.ndarray) -> list:
    """16 features: GLCM (6 props, 4 ângulos médios) + LBP histograma (10 bins)."""
    # GLCM: reduz para 64 níveis para menor custo computacional sem perda significativa
    gray64 = (gray // 4).astype(np.uint8)
    glcm   = graycomatrix(gray64, distances=[1],
                          angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                          levels=64, symmetric=True, normed=True)
    feats = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        feats.append(float(graycoprops(glcm, prop).mean()))  # média sobre os 4 ângulos

    # LBP: P=8 pontos, R=1 pixel, método 'uniform' → P+2=10 bins
    lbp  = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    feats += hist.tolist()  # 10 bins normalizados
    return feats  # 16 features


def extract_edge_features(gray: np.ndarray) -> list:
    """5 features: densidade Canny, magnitude Sobel (mean/std), assimetria H e V."""
    edges  = cv2.Canny(gray, threshold1=50, threshold2=150)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag    = np.sqrt(sobelx**2 + sobely**2)

    h, w = gray.shape
    asym_h = float(np.abs(gray[:, :w//2].mean() - gray[:, w//2:].mean()))
    asym_v = float(np.abs(gray[:h//2, :].mean() - gray[h//2:, :].mean()))

    return [
        float((edges > 0).mean()),  # densidade de bordas
        float(mag.mean()),          # magnitude média do gradiente
        float(mag.std()),           # desvio da magnitude
        asym_h,                     # assimetria horizontal
        asym_v,                     # assimetria vertical
    ]  # 5 features


# Banco de filtros Gabor: 4 orientações × 2 escalas
GABOR_PARAMS = [
    (theta, lam, sig)
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]
    for (lam, sig) in [(5, 2), (10, 4)]   # fine: alta freq | coarse: baixa freq
]

def extract_gabor_features(gray: np.ndarray) -> list:
    """16 features: mean e std da resposta absoluta de 8 filtros Gabor."""
    img_f  = gray.astype(np.float64)
    feats  = []
    for theta, lam, sig in GABOR_PARAMS:
        kernel = cv2.getGaborKernel((21, 21), sig, theta, lam, 0.5, 0, cv2.CV_64F)
        resp   = np.abs(cv2.filter2D(img_f, cv2.CV_64F, kernel))
        feats += [float(resp.mean()), float(resp.std())]
    return feats  # 16 features


def extract_pdi_features(img_path: str) -> np.ndarray:
    """Função mestre: carrega, redimensiona e extrai todas as 55 features PDI."""
    img_rgb = np.array(Image.open(img_path).convert('RGB').resize(IMG_SIZE_PDI, Image.LANCZOS))
    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    feats = (
        extract_color_features(img_rgb) +   # 18
        extract_texture_features(gray)    +  # 16
        extract_edge_features(gray)       +  # 5
        extract_gabor_features(gray)         # 16
    )
    return np.array(feats, dtype=np.float32)


# Sanidade: verifica shape e ausência de NaN em uma imagem de exemplo
sample_feats = extract_pdi_features(df_train['img_path'].iloc[0])
assert sample_feats.shape == (55,), f'Shape inesperado: {sample_feats.shape}'
assert not np.isnan(sample_feats).any(), 'NaN nas features PDI!'
print(f'Features PDI por imagem: {len(sample_feats)}')
print(f'NaN: {np.isnan(sample_feats).sum()}  |  Inf: {np.isinf(sample_feats).sum()}')
print(f'Exemplo (primeiras 10): {sample_feats[:10].round(4)}')

# === CELL 59 ===
def extract_split(df_split: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Extrai features PDI para um split inteiro. Retorna DataFrame."""
    rows = []
    for path in tqdm(df_split['img_path'], desc=f'PDI {split_name}', unit='img'):
        rows.append(extract_pdi_features(path))
    return pd.DataFrame(rows, columns=PDI_FEATURE_NAMES, index=df_split.index)


t0 = time.time()
pdi_train = extract_split(df_train, 'treino')
pdi_val   = extract_split(df_val,   'val   ')
pdi_test  = extract_split(df_test,  'teste ')
elapsed = time.time() - t0

print(f'\nExtração concluída em {elapsed:.1f}s')
print(f'pdi_train : {pdi_train.shape}  | NaN: {pdi_train.isnull().sum().sum()}')
print(f'pdi_val   : {pdi_val.shape}  | NaN: {pdi_val.isnull().sum().sum()}')
print(f'pdi_test  : {pdi_test.shape}  | NaN: {pdi_test.isnull().sum().sum()}')

# === CELL 66 ===
# Normalizar features PDI com o scaler ajustado no treino
pdi_scaler = StandardScaler()
pdi_scaler.fit(pdi_train)

def scale_pdi(pdi_df):
    return pd.DataFrame(
        pdi_scaler.transform(pdi_df),
        columns=PDI_FEATURE_NAMES,
        index=pdi_df.index
    )

pdi_train_sc = scale_pdi(pdi_train)
pdi_val_sc   = scale_pdi(pdi_val)
pdi_test_sc  = scale_pdi(pdi_test)

# Concatena tabular + PDI (alinhamento por reset_index)
X_train_full = pd.concat(
    [X_train_sel.reset_index(drop=True), pdi_train_sc.reset_index(drop=True)], axis=1
)
X_val_full = pd.concat(
    [X_val_sel.reset_index(drop=True), pdi_val_sc.reset_index(drop=True)], axis=1
)
X_test_full = pd.concat(
    [X_test_sel.reset_index(drop=True), pdi_test_sc.reset_index(drop=True)], axis=1
)

# Versão sem biopsed + PDI
X_train_full_nb = pd.concat(
    [X_train_nb.reset_index(drop=True), pdi_train_sc.reset_index(drop=True)], axis=1
)
X_val_full_nb = pd.concat(
    [X_val_nb.reset_index(drop=True), pdi_val_sc.reset_index(drop=True)], axis=1
)
X_test_full_nb = pd.concat(
    [X_test_nb.reset_index(drop=True), pdi_test_sc.reset_index(drop=True)], axis=1
)

# Garantia
for name, X in [('train_full', X_train_full), ('val_full', X_val_full), ('test_full', X_test_full)]:
    assert X.isnull().sum().sum() == 0, f'NaN em {name}!'

print('Composição do vetor final (com biopsed):')
print(f'  Features tabulares : {X_train_sel.shape[1]}')
print(f'  Features PDI       : {len(PDI_FEATURE_NAMES)}')
print(f'  Total              : {X_train_full.shape[1]}')
print(f'\nX_train_full : {X_train_full.shape}')
print(f'X_val_full   : {X_val_full.shape}')
print(f'X_test_full  : {X_test_full.shape}')

# === CELL 69 ===
from sklearn.neighbors       import KNeighborsClassifier
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

print('Helpers carregados.')

# === RF e DT com best params conhecidos (pula RandomizedSearchCV) ===
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import time

print('\nTreinando RF com best params...')
t0 = time.time()
pipe_rf_fixed = ImbPipeline([
    ('smote', SMOTE(random_state=42, k_neighbors=5)),
    ('rf',    RandomForestClassifier(
        n_estimators=563, max_depth=15, min_samples_leaf=4,
        max_features=0.3, class_weight=None, random_state=42, n_jobs=-1
    ))
])
pipe_rf_fixed.fit(X_train_full, y_train)
res_rf_tuned = avaliar_modelo('RF Tuned (SMOTE+CV)', pipe_rf_fixed,
                               X_train_full, y_train, X_val_full, y_val)
print(f'RF treinado em {time.time()-t0:.1f}s')

print('\nTreinando DT com best params...')
t0 = time.time()
pipe_dt_fixed = ImbPipeline([
    ('smote', SMOTE(random_state=42, k_neighbors=5)),
    ('dt',    DecisionTreeClassifier(
        criterion='entropy', max_depth=7, min_samples_leaf=10,
        class_weight=None, random_state=42
    ))
])
pipe_dt_fixed.fit(X_train_full, y_train)
res_dt_tuned = avaliar_modelo('DT Tuned (SMOTE+CV)', pipe_dt_fixed,
                               X_train_full, y_train, X_val_full, y_val)
print(f'DT treinado em {time.time()-t0:.1f}s')

# Avaliação no TESTE
print('\n' + '='*60)
print('AVALIAÇÃO NO CONJUNTO DE TESTE — ML Clássico')
print('='*60)
from sklearn.metrics import precision_score, recall_score
res_teste = {}
for nome, modelo, res_val in [('RF Tuned', pipe_rf_fixed, res_rf_tuned),
                               ('DT Tuned', pipe_dt_fixed, res_dt_tuned)]:
    pred_te = modelo.predict(X_test_full)
    prob_te = modelo.predict_proba(X_test_full)
    f1_te   = f1_score(y_test, pred_te, average='macro', zero_division=0)
    acc_te  = accuracy_score(y_test, pred_te)
    auc_te  = roc_auc_score(y_test, prob_te, multi_class='ovr', average='macro')
    res_teste[nome] = dict(pred=pred_te, prob=prob_te, f1=f1_te, acc=acc_te, auc=auc_te)
    print(f'\n{nome}:')
    print(f'  F1-macro  val={res_val["f1_vl"]:.4f} → test={f1_te:.4f}  (delta={f1_te-res_val["f1_vl"]:+.4f})')
    print(f'  Acurácia  val={res_val["acc_vl"]:.4f} → test={acc_te:.4f}')
    print(f'  AUC-ROC   val={res_val["auc_vl"]:.4f} → test={auc_te:.4f}')
    print(classification_report(y_test, pred_te, target_names=CLASS_NAMES, zero_division=0))


# === CELL 101 ===
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import random

# ── Reprodutibilidade total ──────────────────────────────────────────────────
CNN_SEED = 42
random.seed(CNN_SEED)
np.random.seed(CNN_SEED)
torch.manual_seed(CNN_SEED)
torch.cuda.manual_seed_all(CNN_SEED)
# deterministic ops — ligeiramente mais lento, mas garante reprodução exata
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
N_CLASSES  = len(CLASS_NAMES)   # 6

print(f'Device  : {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU     : {torch.cuda.get_device_name(0)}')
    print(f'VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'Seed    : {CNN_SEED}  (random, numpy, torch, cuda — deterministic=True)')

# ── Transforms ──────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Dataset ──────────────────────────────────────────────────────────────────
class SkinDataset(Dataset):
    def __init__(self, df_split, labels, transform):
        self.paths     = df_split.reset_index(drop=True)['img_path'].tolist()
        self.labels    = labels
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img), int(self.labels[idx])

# ── DataLoaders ───────────────────────────────────────────────────────────────
sample_weights = torch.tensor(
    [class_weight_dict[int(y)] for y in y_train], dtype=torch.float32
)
sampler = torch.utils.data.WeightedRandomSampler(
    sample_weights, num_samples=len(sample_weights), replacement=True,
    generator=torch.Generator().manual_seed(CNN_SEED)
)

ds_train = SkinDataset(df_train, y_train, train_tf)
ds_val   = SkinDataset(df_val,   y_val,   val_tf)
ds_test  = SkinDataset(df_test,  y_test,  val_tf)

dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, sampler=sampler,
                      num_workers=0, pin_memory=True)
dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=0, pin_memory=True)
dl_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=0, pin_memory=True)

print(f'\nSplits — treino: {len(ds_train)}  val: {len(ds_val)}  teste: {len(ds_test)}')
print(f'Batches por época (treino): {len(dl_train)}')

# === CELL 102 ===
# ── Modelo ───────────────────────────────────────────────────────────────────
def build_model():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, N_CLASSES)
    )
    return model.to(DEVICE)

# ── Loss com pesos de classe ──────────────────────────────────────────────────
class_weights_tensor = torch.tensor(
    [class_weight_dict[i] for i in range(N_CLASSES)], dtype=torch.float32
).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# ── Treino: retorna loss E F1 de treino para acompanhar overfitting ───────────
def train_epoch(model, loader, optimizer, scaler):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(labels)
        all_preds.extend(logits.argmax(1).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    tr_loss = total_loss / len(loader.dataset)
    tr_f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return tr_loss, tr_f1

# ── Avaliação ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(imgs)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        all_preds.extend(probs.argmax(axis=1))
        all_probs.append(probs)
        all_labels.extend(labels.numpy())
    all_preds  = np.array(all_preds)
    all_probs  = np.vstack(all_probs)
    all_labels = np.array(all_labels)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    return f1, acc, auc, all_preds, all_probs, all_labels

# ── Loop de treino com early stop por F1-val ──────────────────────────────────
def train_stages(model, optimizer, scheduler, n_epochs, stage_name):
    scaler  = torch.amp.GradScaler()
    best_f1, best_state, history = 0.0, None, []
    PATIENCE, no_improve = 5, 0

    print(f'\n{"="*65}')
    print(f'{stage_name}  ({n_epochs} épocas, patience={PATIENCE})')
    print(f'{"="*65}')
    print(f'{"Ep":>4} {"Loss_tr":>9} {"F1_tr":>7} {"F1_val":>8} {"AUC_val":>9}  {"Gap":>6}')
    print('-' * 55)

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_f1  = train_epoch(model, dl_train, optimizer, scaler)
        f1_v, acc_v, auc_v, _, _, _ = eval_epoch(model, dl_val)
        scheduler.step()

        gap = tr_f1 - f1_v
        history.append(dict(epoch=epoch, tr_loss=tr_loss,
                            tr_f1=tr_f1, vl_f1=f1_v, vl_acc=acc_v,
                            vl_auc=auc_v, gap=gap))

        flag = ''
        if f1_v > best_f1:
            best_f1, best_state = f1_v, {k: v.cpu().clone()
                                          for k, v in model.state_dict().items()}
            no_improve = 0; flag = ' ←'
        else:
            no_improve += 1

        ovf_flag = ' ⚠OVF' if gap > 0.15 else ''
        print(f'{epoch:>4} {tr_loss:>9.4f} {tr_f1:>7.4f} {f1_v:>8.4f} '
              f'{auc_v:>9.4f}  {gap:>+6.3f}{flag}{ovf_flag}')

        if no_improve >= PATIENCE:
            print(f'Early stop na época {epoch} (sem melhora em F1-val há {PATIENCE} épocas).')
            break

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    print(f'\nMelhor F1-val: {best_f1:.4f}')
    return history

print('Infraestrutura CNN carregada.')

# === CELL 104 — Stage 1 ===
model = build_model()

# Congela backbone — só o classificador treina
for param in model.features.parameters():
    param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f'Parâmetros treináveis: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)')

optimizer_s1 = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
)
scheduler_s1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s1, T_max=15, eta_min=1e-5)

history_s1 = train_stages(model, optimizer_s1, scheduler_s1,
                           n_epochs=15, stage_name='Stage 1 — Feature Extraction')

# === CELL 106 — Stage 2 ===
# Descongela os últimos 3 blocos do backbone
for block in list(model.features)[-3:]:
    for param in block.parameters():
        param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f'Parâmetros treináveis: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)')

# LR diferenciado: backbone desbloqueado com LR menor
optimizer_s2 = optim.Adam([
    {'params': model.features[-3:].parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(),    'lr': 1e-4},
])
scheduler_s2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=10, eta_min=1e-6)

history_s2 = train_stages(model, optimizer_s2, scheduler_s2,
                           n_epochs=10, stage_name='Stage 2 — Fine-Tuning')

# === CELL 108 ===

df_s1 = pd.DataFrame(history_s1)
df_s2 = pd.DataFrame(history_s2)
df_s2 = df_s2.copy(); df_s2['epoch'] = df_s2['epoch'] + df_s1['epoch'].max()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for df_h, ls, label_prefix in [
    (df_s1, '-',  'Stage 1'),
    (df_s2, '--', 'Stage 2'),
]:
    axes[0].plot(df_h['epoch'], df_h['tr_f1'], 'o'+ls, color='steelblue', lw=2,
                 label=f'{label_prefix} — Treino')
    axes[0].plot(df_h['epoch'], df_h['vl_f1'], 's'+ls, color='coral', lw=2,
                 label=f'{label_prefix} — Val')
    axes[1].plot(df_h['epoch'], df_h['tr_loss'], 'o'+ls, color='steelblue', lw=2,
                 label=f'{label_prefix} — Treino')

axes[0].axvline(df_s1['epoch'].max() + 0.5, ls=':', color='gray', lw=1)
axes[1].axvline(df_s1['epoch'].max() + 0.5, ls=':', color='gray', lw=1)
axes[0].set_xlabel('Época'); axes[0].set_ylabel('F1-macro')
axes[0].set_title('F1-macro — Treino vs Validação (overfitting)')
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

axes[1].set_xlabel('Época'); axes[1].set_ylabel('Loss (treino)')
axes[1].set_title('Loss de Treino')
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

fig.suptitle('EfficientNetB0 — Curvas de Aprendizado', fontsize=13, fontweight='bold')
plt.tight_layout(); _savefig("cnn_learning_curves")

# ── Análise de overfitting: gap F1 por época ─────────────────────────────────
df_all = pd.concat([df_s1, df_s2], ignore_index=True)
max_gap   = df_all['gap'].max()
last_gap  = df_all['gap'].iloc[-1]
best_vl   = df_all['vl_f1'].max()
best_ep   = df_all.loc[df_all['vl_f1'].idxmax(), 'epoch']

print('Análise de overfitting (Gap = F1_treino − F1_val):')
print(f'  Gap máximo       : {max_gap:+.4f}')
print(f'  Gap no checkpoint: {last_gap:+.4f}  (época {df_all["epoch"].iloc[-1]})')
print(f'  Melhor F1-val    : {best_vl:.4f}  na época {best_ep}')
if max_gap < 0.10:
    print('  → Sem overfitting relevante: modelo generaliza bem.')
elif max_gap < 0.20:
    print('  → Overfitting moderado: augmentation e early stop contiveram o problema.')
else:
    print('  → Overfitting relevante: considerar mais augmentation ou dropout maior.')

# === CELL 110 ===
# Avaliar CNN no conjunto de teste
f1_cnn, acc_cnn, auc_cnn, pred_cnn, prob_cnn, _ = eval_epoch(model, dl_test)

print('=' * 60)
print('COMPARATIVO FINAL — CONJUNTO DE TESTE')
print('=' * 60)

resultados_finais = [
    ('RF Tuned',        res_teste['RF Tuned']['f1'],  res_teste['RF Tuned']['acc'],  res_teste['RF Tuned']['auc']),
    ('DT Tuned',        res_teste['DT Tuned']['f1'],  res_teste['DT Tuned']['acc'],  res_teste['DT Tuned']['auc']),
    ('EfficientNetB0',  f1_cnn,                        acc_cnn,                       auc_cnn),
]

print(f'\n{"Modelo":<22} {"F1-macro":>9} {"Acurácia":>9} {"AUC-ROC":>9}')
print('-' * 52)
for nome, f1, acc, auc in resultados_finais:
    print(f'{nome:<22} {f1:>9.4f} {acc:>9.4f} {auc:>9.4f}')

# Gráfico comparativo
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
nomes_f = [r[0] for r in resultados_finais]
cores_f = ['#9b59b6', '#2ecc71', '#e74c3c']
x = np.arange(len(nomes_f)); w = 0.6

for ax, (col_idx, titulo, ymin) in zip(
    axes,
    [(1, 'F1-macro (Teste)', 0.0),
     (2, 'Acurácia (Teste)', 0.0),
     (3, 'AUC-ROC (Teste)',  0.8)]
):
    vals = [r[col_idx] for r in resultados_finais]
    bars = ax.bar(x, vals, w, color=cores_f, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(nomes_f, rotation=10, ha='right')
    ax.set_ylim(ymin, min(1.0, max(vals) + 0.12))
    ax.set_title(titulo, fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

fig.suptitle('Comparativo Final — ML Clássico vs CNN (Conjunto de Teste)',
             fontsize=13, fontweight='bold')
plt.tight_layout(); _savefig()

# F1 por classe
fig, ax = plt.subplots(figsize=(12, 5))
xc = np.arange(len(CLASS_NAMES)); w = 0.25
for i, (nome, cor, pred_model) in enumerate([
    ('RF Tuned',       '#9b59b6', res_teste['RF Tuned']['pred']),
    ('DT Tuned',       '#2ecc71', res_teste['DT Tuned']['pred']),
    ('EfficientNetB0', '#e74c3c', pred_cnn),
]):
    f1_cls = [f1_score(y_test, pred_model, labels=[j], average='macro', zero_division=0)
              for j in range(N_CLASSES)]
    ax.bar(xc + (i-1)*w, f1_cls, w, label=nome, color=cor, edgecolor='white')

ax.set_xticks(xc); ax.set_xticklabels(CLASS_NAMES, fontsize=11)
ax.set_ylim(0, 1.1); ax.set_ylabel('F1-score')
ax.set_title('F1 por Classe — Todos os Modelos (Teste)', fontsize=12)
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); _savefig()

# Matriz de confusão — CNN
fig, ax = plt.subplots(figsize=(7, 5))
cm_cnn = confusion_matrix(y_test, pred_cnn)
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Reds',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_title('EfficientNetB0 — Matriz de Confusão (Teste)', fontsize=11)
ax.set_xlabel('Predito'); ax.set_ylabel('Real')
plt.tight_layout(); _savefig()
print(classification_report(y_test, pred_cnn, target_names=CLASS_NAMES, zero_division=0))

# === SALVA RESULTADOS ===
f1_cnn_val = max(r['vl_f1'] for r in history_s1 + history_s2)
results = {
    'rf': {'f1_val': float(res_rf_tuned['f1_vl']), 'f1_test': float(res_teste['RF Tuned']['f1']),
            'acc_test': float(res_teste['RF Tuned']['acc']), 'auc_test': float(res_teste['RF Tuned']['auc'])},
    'dt': {'f1_val': float(res_dt_tuned['f1_vl']), 'f1_test': float(res_teste['DT Tuned']['f1']),
            'acc_test': float(res_teste['DT Tuned']['acc']), 'auc_test': float(res_teste['DT Tuned']['auc'])},
    'cnn': {'f1_val': float(f1_cnn_val), 'f1_test': float(f1_cnn),
             'acc_test': float(acc_cnn), 'auc_test': float(auc_cnn)},
    'history_s1': [dict(epoch=int(r['epoch']), tr_f1=float(r['tr_f1']),
                        vl_f1=float(r['vl_f1']), vl_auc=float(r['vl_auc']),
                        gap=float(r['gap'])) for r in history_s1],
    'history_s2': [dict(epoch=int(r['epoch']), tr_f1=float(r['tr_f1']),
                        vl_f1=float(r['vl_f1']), vl_auc=float(r['vl_auc']),
                        gap=float(r['gap'])) for r in history_s2],
}
with open('../outputs/results.json', 'w') as f:
    _json.dump(results, f, indent=2)
print('\nResultados salvos em outputs/results.json')
print('Plots salvos em outputs/')
