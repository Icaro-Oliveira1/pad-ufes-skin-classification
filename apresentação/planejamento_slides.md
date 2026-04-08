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
> Esta seção ocupa **1 slide** e **~90 segundos**.
> Objetivo: mostrar que as decisões de pré-processamento são **consequências diretas**
> das armadilhas apresentadas no slide 4 da Parte 1 — não escolhas arbitrárias.

---

## Slide 5 — Pré-processamento dos metadados

**Título:** Pré-processamento dos Metadados — decisões fundamentadas

**Layout:** fluxograma horizontal no topo + 2 cards de destaque embaixo.

### Fluxograma (topo, horizontal)

```
   raw             drop leak        drop MNAR         feat. eng.         split agrupado       pipeline sklearn
   2298×26    →    −biopsed    →    −13 colunas   →   +symptom_count →   64/16/20 por      →   OHE region
                                    (MNAR block)      +flags UNK         paciente              scale age
                                                                                                (fit só no treino)
```

**Resultado final:** `train / val / test` em Parquet + `pipeline.joblib` reutilizável.

### Card 1 — Por que dropar as 13 colunas MNAR (a decisão mais importante)

**Título:** "Imputar seria leak mascarado"

Três pontos (curtos, cada um uma linha):

- **Ma & Zhang (NeurIPS 2021):** MNAR não é identificável — mesmo com infinitas amostras a imputação fica enviesada
- **Sisk et al. (Diagn Progn Res 2023):** missing indicators em outcome-dependent missingness são *harmful*
- **Consequência prática:** qualquer imputação (mediana, KNN, MICE, missForest) criaria uma assinatura que RF/XGBoost detectariam como proxy do diagnóstico

**Destaque visual:** caixa discreta embaixo com:
> *O sinal biológico real das colunas úteis (fotótipo, diâmetros) será recuperado via PDI da imagem no próximo notebook.*

### Card 2 — Split por paciente + pipeline sklearn

Dois mini-bullets:

- **`StratifiedGroupKFold` com `groups=patient_id`** → zero overlap entre splits, 33/9/10 melanomas em train/val/test
- **`ColumnTransformer` ajustado só no treino** → OHE(`region`, min_frequency=10) + StandardScaler(`age`) + passthrough dos sintomas → **24 features finais**, salvo em `pipeline.joblib`

### Mensagem-chave do slide

> "Todo o notebook é basicamente responder 'como não trapacear?' — seja no split, no biopsed, ou nas 13 colunas MNAR. Zero imputação, zero leak."

**Gráfico:** nenhum novo. Só o fluxograma no topo e os dois cards.

---

## Resumo de tempos — atualizado

| Slide | Tempo alvo |
|---|---:|
| 1. Problema e dataset | ~45s |
| 2. Classes e desbalanceamento | ~60s |
| 3. O que cada amostra contém | ~60s |
| 4. Armadilhas | ~75s |
| **5. Pré-processamento metadados** | **~90s** |
| **Total partes 1 + 2** | **~5:30 min** |

Sobram ~9:30 min para PDI, modelo, resultados e conclusões.
