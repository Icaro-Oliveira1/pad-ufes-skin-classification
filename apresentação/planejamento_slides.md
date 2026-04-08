# Planejamento dos Slides — Parte 1: Apresentação do Dataset

> **Contexto:** primeira parte de uma apresentação de **15 minutos**.
> Esta seção (dataset) deve ocupar **~3 a 4 minutos**, em **4 slides**.
> Objetivo: dar ao público o mínimo necessário para entender o resto da
> apresentação (pré-processamento, PDI, modelo).

---

## Slide 1 — O problema e o dataset

**Título:** PAD-UFES-20 — Classificação de Lesões de Pele

**Layout:**
- Coluna esquerda (texto): contexto + números-chave
- Coluna direita (imagem): 1 foto de lesão (exemplo BCC ou MEL) — extraída de `data/raw/images/`

**Conteúdo textual (bullets curtos):**
- Programa de assistência dermatológica da UFES (Espírito Santo)
- Imagens **clínicas** (smartphone), não dermatoscópicas
- **2298 imagens** · **1373 pacientes** · **1641 lesões** · **6 classes**
- Tarefa: classificar a lesão a partir da imagem

**Gráfico:** nenhum. Apenas números grandes em destaque + 1 imagem real.

---

## Slide 2 — As 6 classes e o desbalanceamento

**Título:** Seis classes — três malignas, uma pré-maligna, duas benignas

**Layout:**
- Gráfico de barras ocupando ~60% do slide (esquerda)
- Tabela compacta à direita: sigla → nome → tipo

**Gráfico:** **Barras das classes** (já existe no notebook `01_entendimento_dataset.ipynb`, célula da seção 2).
- Eixo X: classes em ordem `[BCC, ACK, NEV, SEK, SCC, MEL]`
- Eixo Y: número de imagens
- Cores por tipo (vermelho/laranja = maligno, amarelo = pré, verde/ciano = benigno)
- Anotação em cima de cada barra: `n (%)`

**Dados a aparecer no gráfico:**

| Classe | Nome | Tipo | n | % |
|---|---|---|---:|---:|
| BCC | Basal Cell Carcinoma | maligno | 845 | 36.8 |
| ACK | Actinic Keratosis | pré-maligno | 730 | 31.8 |
| NEV | Nevus (pinta) | benigno | 244 | 10.6 |
| SEK | Seborrheic Keratosis | benigno | 235 | 10.2 |
| SCC | Squamous Cell Carcinoma | maligno | 192 | 8.4 |
| **MEL** | **Melanoma** | **maligno** | **52** | **2.3** |

**Destaque visual:** caixa/seta sobre a barra do MEL com o texto
**"o mais letal — só 52 amostras"**.

**Mensagem-chave:** razão maior/menor classe ≈ **16×**. Desbalanceamento extremo na classe que mais importa.

---

## Slide 3 — O que cada amostra contém

**Título:** Imagem + 26 colunas de metadados clínicos

**Layout:** dois blocos lado a lado.

**Bloco esquerdo — "Imagem":**
- Mini-grade de 3 exemplos (qualquer classe, mas variando aspecto)
- Bullets:
  - PNG, **tamanho variável: 189 a 3096 px** (mediana ~772)
  - Maioria quadrada, mas **mistura RGB e RGBA**
  - Smartphone → iluminação irregular, pelo, sombras, fundo de pele

**Bloco direito — "Metadados (26 colunas)":**
- Agrupados em 4 categorias (não listar uma a uma):
  - **Paciente** — idade (6–94, média 60), sexo, **Fitzpatrick (1–6)**, histórico de câncer
  - **Lesão** — região do corpo (14 regiões), diâmetros (mm)
  - **Sintomas (booleanos)** — coça, cresceu, dói, mudou, sangra, elevação
  - **Diagnóstico** — alvo

**Gráfico:** opcional — se sobrar espaço, **mini-histograma das larguras das imagens** (gerado na seção 7 do notebook). Caso contrário, só os blocos textuais bastam.

**Mensagem-chave:** temos **duas fontes complementares de informação — imagem e metadados clínicos — e o projeto vai usar as duas**. Da imagem extraímos features por PDI (cor, textura, forma); dos metadados aproveitamos os sinais clínicos (idade, Fitzpatrick, sintomas, região). O modelo final combina os dois.

---

## Slide 4 — Três armadilhas que moldam o projeto

**Título:** O que descobrimos explorando os dados

**Layout:** 3 cards horizontais, um por armadilha. Cada card: ícone + título curto + 1 frase + decisão.

**Card 1 — Vazamento por paciente**
- Mesma lesão tem várias fotos; mesmo paciente tem várias lesões
- **Decisão:** split treino/teste **por paciente** (não por imagem)

**Card 2 — `biopsed` é leak**
- Crosstab: **100% dos malignos** (BCC, MEL, SCC) foram biopsiados
- **Decisão:** remover `biopsed` das features

**Dados do crosstab para mostrar (mini-tabela no card):**

| Classe | biopsed=False | biopsed=True |
|---|---:|---:|
| BCC | 0 | 845 |
| MEL | 0 | 52 |
| SCC | 0 | 192 |
| ACK | 552 | 178 |
| NEV | 184 | 60 |
| SEK | 220 | 15 |

**Card 3 — 35% de metadados ausentes em bloco**
- 13 colunas têm exatamente **804 NaNs (~35%)** — dois formulários distintos (short vs long)
- **Descoberta crítica:** esse bloco de 804 linhas **não contém nenhuma lesão maligna** (0 BCC, 0 MEL, 0 SCC) → a missingness é **MNAR outcome-dependent**, proxy quase perfeito do diagnóstico
- **Decisão:** dropar as 13 colunas (detalhes no slide de pré-processamento)

**Gráfico:** nenhum gráfico novo. Os dados estão nas mini-tabelas dos cards.

**Mensagem-chave (slide de transição):**
> "Essas 3 descobertas justificam o pipeline que vamos apresentar a seguir."

---

## Resumo de tempos

| Slide | Tempo alvo |
|---|---:|
| 1. Problema e dataset | ~45s |
| 2. Classes e desbalanceamento | ~60s |
| 3. O que cada amostra contém | ~60s |
| 4. Armadilhas | ~75s |
| **Total parte 1** | **~4 min** |

Sobram ~11 min para o restante (pré-processamento, PDI, modelo, resultados, conclusões).

---

# Planejamento dos Slides — Parte 2: Pré-processamento dos Metadados

> **Contexto:** segunda parte da apresentação.
> Esta seção ocupa **2 slides** e **~2:30 minutos** no total.
> Objetivo: (1) mostrar que as decisões de pré-processamento são **consequências
> diretas** das armadilhas apresentadas no slide 4 da Parte 1, e (2) deixar
> explícito quais **features finais** vão alimentar o modelo.

---

## Slide 5 — Pipeline completo de pré-processamento

**Título:** Pré-processamento dos Metadados — pipeline e decisões

**Layout:** fluxograma vertical grande ocupando ~55% do slide (esquerda) + 2 cards de justificativa empilhados (direita).

### Fluxograma (coluna esquerda — vertical, 8 passos)

```
  ┌─────────────────────────────┐
  │  1. CSV bruto               │  2298 × 26 colunas
  │     metadata.csv            │  13 cols com 35% missing em bloco
  └──────────────┬──────────────┘
                 │
  ┌──────────────▼──────────────┐
  │  2. Verificação de qualidade│  ✓ 0 linhas duplicadas
  │     (duplicatas, img_id)    │  ✓ img_id único (chave de merge p/ PDI)
  └──────────────┬──────────────┘  ✓ 512 lesões multi-foto → split por paciente
                 │
  ┌──────────────▼──────────────┐
  │  3. Drop de leak            │  − biopsed (causal reverso)
  │     + bloco MNAR            │  − 13 colunas outcome-dependent
  └──────────────┬──────────────┘
                 │
  ┌──────────────▼──────────────┐
  │  4. UNK handling            │  grew / changed: 17% UNK
  │     + feature engineering   │  → flags grew_unknown, changed_unknown
  │                             │  + symptom_count (soma ABCDE)
  └──────────────┬──────────────┘
                 │
  ┌──────────────▼──────────────┐
  │  5. Análise de outliers     │  53 outliers em age (94% NEV jovem)
  │     (documentação)          │  → preservar: sinal clínico, não ruído
  └──────────────┬──────────────┘
                 │
  ┌──────────────▼──────────────┐
  │  6. Split por paciente      │  StratifiedGroupKFold 64/16/20
  │     + drift check           │  zero overlap · Δage = 0 · Δregion ≤ 5.7pp
  └──────────────┬──────────────┘
                 │
  ┌──────────────▼──────────────┐
  │  7. Encoding + scaling      │  OneHotEncoder(region, min_freq=10)
  │     via ColumnTransformer   │  StandardScaler(age)
  │     (fit SÓ no treino)      │  passthrough(sintomas + flags)
  │                             │  LabelEncoder(diagnostic) — target
  └──────────────┬──────────────┘
                 │
  ┌──────────────▼──────────────┐
  │  8. Persistência            │  train/val/test.parquet +
  │                             │  pipeline.joblib + class_weights.joblib
  └─────────────────────────────┘
```

### Destaque sobre encoding (caixa discreta abaixo do fluxograma)

**Duas estratégias de encoding, cada uma para um propósito:**

- **`OneHotEncoder` em `region`** — 14 categorias nominais viram 14 colunas binárias, evita ordem espúria. Parâmetro `min_frequency=10` agrupa regiões raras (LIP, FOOT, SCALP) em uma única categoria `infrequent_sklearn` para evitar overfit. `handle_unknown='ignore'` protege contra categoria inédita em val/test.
- **`LabelEncoder` em `diagnostic`** — a variável alvo vira inteiros `0–5` (ACK=0, BCC=1, MEL=2, NEV=3, SCC=4, SEK=5). É a API esperada pelos classificadores do sklearn. Não introduz ordem porque o modelo trata cada valor como rótulo independente.

### Card 1 (direita, topo) — Por que dropar as 13 colunas MNAR

**Título curto:** "Imputar seria leak mascarado"

Três pontos (cada um uma linha):
- **Ma & Zhang (NeurIPS 2021):** MNAR não é identificável — mesmo com infinitas amostras, a imputação fica enviesada
- **Sisk et al. (Diagn Progn Res 2023):** missing indicators em outcome-dependent missingness são *harmful*
- **Consequência:** qualquer imputação (mediana, KNN, MICE, missForest) criaria uma assinatura que RF/XGBoost detectariam como proxy do diagnóstico

**Caixa discreta abaixo do card:**
> *Sinal biológico real (fotótipo, diâmetros) será recuperado via PDI da imagem no próximo notebook.*

### Card 2 (direita, base) — Decisões-chave do pipeline

Três bullets curtos:
- **Zero imputação** — após os drops o dataset fica 100% completo, nada a imputar
- **Split por `patient_id`** — garante que fotos quase idênticas da mesma lesão não vazem entre splits (512 lesões têm ≥2 fotos)
- **`fit_transform` só no treino** — `ColumnTransformer` do sklearn impede leak estrutural por construção

### Mensagem-chave do slide

> "Todo o pipeline é basicamente responder 'como não trapacear?' — no split, no `biopsed`, nas 13 colunas MNAR, e no `fit` do scaler. Zero imputação, zero leak."

**Gráfico:** apenas o fluxograma à esquerda e os dois cards à direita. Nada mais.

---

## Slide 6 — Features finais e compensação do desbalanceamento

**Título:** 24 Features Finais + Class Weights

**Layout:** tabela de features (esquerda, ~55%) + gráfico de pesos de classe (direita, ~45%).

### Bloco esquerdo — Tabela de features agrupadas

**Nota introdutória no topo:**
> *24 features finais após o pipeline. Nenhuma feature clínica sensível ao protocolo de coleta (MNAR) entra no modelo.*

**Tabela (agrupada por tipo):**

| Grupo | # | Features | Transformação |
|---|---:|---|---|
| **Numérica** | 1 | `age` | `StandardScaler` |
| **Categórica (OHE)** | 14 | `region_*` (14 regiões, raras agrupadas via `min_frequency=10`) | `OneHotEncoder` |
| **Sintomas binários** | 6 | `itch, grew, hurt, changed, bleed, elevation` | passthrough |
| **Flags de unknown** | 2 | `grew_unknown, changed_unknown` | passthrough |
| **Derivada** | 1 | `symptom_count` (soma dos 6 sintomas = ABCDE-like) | passthrough |
| **Total** | **24** | | |

**Caixa destacada abaixo da tabela — "o que NÃO entra":**
> ❌ `biopsed` (leak) · ❌ `fitspatrick`, `diameter_1/2` (MNAR — serão recuperados via PDI) · ❌ `smoke`, `drink`, `gender`, histórico familiar, socioeconômicos, ancestralidade (MNAR, sem contrapartida visual)

**Linha sobre a target (abaixo da caixa anterior, estilo rodapé):**
> 🎯 **Target:** `diagnostic` (6 classes) → `LabelEncoder` → `{ACK:0, BCC:1, MEL:2, NEV:3, SCC:4, SEK:5}`. Fora do `ColumnTransformer` por convenção do sklearn.

### Bloco direito — Class weights (gráfico de barras)

**Gráfico:** barras verticais, 6 classes em ordem `[ACK, BCC, MEL, NEV, SCC, SEK]`, altura = peso balanced. Cores: vermelho para malignos (BCC, MEL, SCC), verde para benignos (ACK, NEV, SEK). Linha horizontal tracejada em `y=1.0` ("peso neutro"). Valor em cima de cada barra.

**Dados do gráfico (calculados a partir do `y_train`):**

| Classe | n treino | peso |
|---|---:|---:|
| ACK | 467 | 0.525 |
| BCC | 541 | 0.453 |
| **MEL** | **33** | **7.424** |
| NEV | 156 | 1.571 |
| SCC | 123 | 1.992 |
| SEK | 150 | 1.633 |

**Texto de apoio à direita do gráfico (2 linhas):**
- **Razão 16.4×** entre MEL (raro) e BCC (comum) — um erro em MEL "dói" 16× mais durante o treino
- Fórmula: `w[c] = n_samples / (n_classes × count[c])` — sem tocar no dataset

### Mensagem-chave do slide

> "O dataset fica intacto — os 33 melanomas continuam 33. Mas com `class_weight='balanced'`, o modelo é **forçado** a prestar atenção neles desde o primeiro gradiente."

**Artefatos persistidos (rodapé do slide, letra pequena):**
`train/val/test.parquet` · `pipeline.joblib` · `label_encoder.joblib` · `class_weights.joblib` · `feature_names.json` · `split_info.json`

---

## Resumo de tempos — atualizado

| Slide | Tempo alvo |
|---|---:|
| 1. Problema e dataset | ~45s |
| 2. Classes e desbalanceamento | ~60s |
| 3. O que cada amostra contém | ~60s |
| 4. Armadilhas | ~75s |
| **5. Pipeline de pré-processamento** | **~90s** |
| **6. Features finais + class weights** | **~60s** |
| **Total partes 1 + 2** | **~6:30 min** |

Sobram ~8:30 min para PDI, modelo, resultados e conclusões.

---

# Planejamento dos Slides — Parte 3: Pré-processamento das Imagens (PDI)

> **Contexto:** terceira parte da apresentação.
> Esta seção ocupa **1 slide** e **~90 segundos**.
> Objetivo: mostrar o pipeline de 4 etapas aplicado às imagens, explicar
> brevemente o Otsu (que é a etapa técnica mais interessante), e reportar
> a qualidade da segmentação.

---

## Slide 7 — Pré-processamento das Imagens (PDI)

**Título:** Pré-processamento das Imagens — 4 etapas + Otsu

**Layout:** fluxograma horizontal de 4 caixas no topo + caixa de destaque
do Otsu no meio (com mini-histograma ilustrativo) + 2 mini-cards de
resultados no rodapé.

### Fluxograma (topo, horizontal — 4 caixas)

```
  raw PNG           1. Hair removal      2. Color constancy    3. Segmentação       4. Crop + resize
  189–3096 px   →   DullRazor        →   Shades of Gray    →   Otsu no LAB L*   →   bbox quadrada
  RGB / RGBA        (blackhat +          (Finlayson, p=6)      + fallback            + padding 10%
                     inpaint)            normaliza celular     center-crop           → 256×256
```

**Destaque visual:** seta abaixo de cada etapa indicando o "problema que
ela resolve":
- Hair removal → *elimina pelo que contamina features de textura*
- Color constancy → *normaliza balanço de branco entre smartphones*
- Segmentação → *isola a lesão — não extrair features de pele + fundo*
- Crop + resize → *preserva aspect ratio — não distorce assimetria*

### Caixa central — "Como o Otsu funciona" (explicação compacta)

**Layout:** caixa larga com título + 3 bullets curtos à esquerda + mini
histograma à direita.

**Título:** Otsu (1979) — threshold binário automático

**Bullets:**
- No canal **L\* do LAB**, o histograma de uma foto dermatológica é
  **bimodal**: um pico de pele clara, um pico de lesão escura
- Otsu varre os 256 valores possíveis de threshold e escolhe aquele que
  **maximiza a variância *entre* os dois grupos** (= minimiza a dispersão
  dentro de cada um)
- **Zero treinado**, ~5 ms por imagem, interpretável — não depende de
  dataset de segmentação nem de GPU

**Mini-histograma ilustrativo:** duas colinas bem separadas, com uma linha
vermelha tracejada no vale indicando `T_ótimo`. Abaixo, label:
> *maximiza `σ²_entre = ω₀·ω₁·(μ₀ − μ₁)²`*

**Caixa discreta abaixo do histograma:**
> *Fallback quando Otsu falha (máscara < 5% ou > 95% da imagem): retângulo
> central 60% × 60%. Aciona em ~1.5% dos casos.*

### Card 1 (rodapé esquerdo) — Qualidade da segmentação

Três bullets:
- **2298 / 2298** imagens processadas, zero erros
- **1.5% fallback total** (35 imagens) — bem abaixo do 5-10% típico em
  datasets dermatológicos
- Distribuição dos fallbacks: **NEV 9%** (pintas uniformes, esperado
  clinicamente), **MEL 3.8%** (apenas 2/52 — crítico, dentro do aceitável)

### Card 2 (rodapé direito) — Output do pipeline

Três bullets:
- **2298 × `.npz`** contendo `{image 256×256×3, mask 256×256}`
- **326 MB** cacheados em disco (execução ~20 min, reuso ~5 ms / imagem)
- Helper público: `load_processed(img_id) → (image, mask)`

### Mensagem-chave do slide

> "Cada imagem sai do pipeline com uma máscara que isola a lesão real —
> o próximo passo vai extrair features **dentro da máscara**, não de uma
> mistura de lesão + pele + pelo + fundo."

**Gráfico extra:** nenhum. O fluxograma + mini-histograma + 2 cards já
preenchem o slide sem sobra.

---

## Resumo de tempos — atualizado

| Slide | Tempo alvo |
|---|---:|
| 1. Problema e dataset | ~45s |
| 2. Classes e desbalanceamento | ~60s |
| 3. O que cada amostra contém | ~60s |
| 4. Armadilhas | ~75s |
| 5. Pipeline de pré-processamento | ~90s |
| 6. Features finais + class weights | ~60s |
| **7. Pré-processamento das imagens (PDI)** | **~90s** |
| **Total partes 1 + 2 + 3** | **~8:00 min** |

Sobram ~7:00 min para extração de features, modelagem, resultados e conclusões.
