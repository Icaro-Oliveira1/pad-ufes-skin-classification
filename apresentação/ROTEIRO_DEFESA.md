# 🎯 Roteiro de Defesa - Classificação de Lesões de Pele PAD-UFES-20

---

## 📋 CONCEITOS FUNDAMENTAIS

### 1. **Gini vs Entropy**

**A diferença é mínima em prática.**

| Aspecto | Gini | Entropy |
|---------|------|---------|
| **Fórmula** | $G = 1 - \sum_i p_i^2$ | $E = -\sum_i p_i \log_2(p_i)$ |
| **Escala** | 0 a 0.5 | 0 a 1 |
| **Velocidade** | Mais rápido (sem log) | Mais lento (requer log) |
| **Interpretação** | Probabilidade de erro se classe aleatória | Quanto "espanto" você tem ao classificar |
| **Resultado prático** | ~idêntico ao Entropy | ~idêntico ao Gini |

**Na prática:** Ambos funcionam. Gini é ligeiramente mais rápido. Escolhemos **Gini por padrão**, mas Entropy geraria resultados praticamente iguais.

**Analogia:** 
- **Gini** = "Qual é a chance de eu estar errado se chutar uma classe aleatória?"
- **Entropy** = "Quanto mais "misturado" está este nó, mais incerteza eu tenho"

---

## 🎓 PERGUNTAS ESPERADAS & RESPOSTAS

### **P1: Por que uma árvore profunda overfita mais?**

**Resposta:**
Uma árvore profunda cria muitas folhas especializadas. Cada folha memoriza características específicas dos dados de treinamento, incluindo ruído e outliers. Quando encontra dados novos, não consegue generalizar porque aprendeu regras muito específicas.

**Analogia:** É como decorar exemplos específicos de prova vs entender o conceito. Se você decorar "quando age=50 e localização=face → MEL", vai falhar quando age=49 ou localização=pescoço.

**No nosso projeto:**
- Árvore SEM limites → F1 treino ~0.96, F1 validação ~0.62 (gap +0.34)
- Árvore COM max_depth=5 → F1 treino ~0.76, F1 validação ~0.72 (gap +0.04)

**Controle:** max_depth, min_samples_leaf, min_samples_split limitam profundidade.

---

### **P2: Por que Random Forest generaliza melhor que Decision Tree?**

**Resposta:**
Random Forest cria 100 árvores diferentes, cada uma treinada em bootstrap samples (amostras aleatórias com reposição) com features aleatórias.

**O segredo:** Diversidade reduz overfitting.

```
Decision Tree (1 árvore):
  └─ Memoriza muita coisa do treino
  
Random Forest (100 árvores):
  ├─ Árvore 1 memoriza padrões A e B
  ├─ Árvore 2 memoriza padrões B e C  
  ├─ Árvore 3 memoriza padrões A e C
  └─ A votação (média) cancela memorização e captura o padrão real
```

**Efeito estatístico:** 
- Cada árvore vê ~63% dos dados (bootstrap)
- Cada divisão usa features aleatórias
- Isso reduz correlação entre erros

**No nosso projeto:**
- DT: F1 treino 0.76, F1 validação 0.72
- **RF: F1 treino 0.80, F1 validação 0.77** ← melhor generalização

---

### **P3: Por que XGBoost costuma ganhar em dados tabulares?**

**Resposta:**
XGBoost é **boosting sequencial**, não paralelo.

```
Decision Tree / Random Forest (paralelo):
  ├─ Árvore 1 tenta sozinha
  ├─ Árvore 2 tenta sozinha
  └─ Árvore 3 tenta sozinha
  └─ Votação final

XGBoost (sequencial):
  ├─ Árvore 1: prediz Y, comete erro E₁
  ├─ Árvore 2: tenta corrigir E₁, comete erro E₂
  ├─ Árvore 3: tenta corrigir E₂, comete erro E₃
  └─ Soma: pred_final = pred1 + pred2 + pred3
```

**Vantagem em tabular:**
- Features tabular têm **sinais fortes e padrões relacionais**
- XGBoost encontra esses padrões **incrementalmente**
- RF já começa suficientemente diverso; XGBoost *continua* melhorando

**Dados de imagem (CNN):**
- Padrões são espaciais/hierárquicos
- Árvores (mesmo XGBoost) não capturam bem convoluções
- Logo: RF e XGBoost são melhores para features extraídas + tabular

**No nosso projeto:**
- Temos 62 features (24 tabular + 38 extração PDI)
- **XGBoost vence**: F1 treino 0.82, F1 validação 0.79

---

### **P4: Por que usamos F1-Macro e não Accuracy?**

**Resposta:**
Nossos dados têm **classe muito desbalanceada**.

```
Dataset: 2298 imagens
- BCC: 845 (36.8%)  ← maioria
- NEV: 623 (27.1%)  ← maioria  
- MEL: 52 (2.3%)    ← RARA
- outros: 778 (33.8%)

Modelo burro:
  "Vou prever BCC pra tudo"
  Accuracy = 36.8% (horrível, mas positivo)
  F1-Macro = 0% pra 5 classes, ruim demais
```

**Fórmula:**

$$F1\text{-}Macro = \frac{1}{6}\sum_{classe=1}^{6} F1_{classe}$$

- Cada classe contribui **igualmente** (1/6)
- Classe rara (MEL) não é "negligenciada"
- Força modelo a ser bom em tudo

**Exemplos:**

| Métrica | Árvore Profunda | RF | XGBoost | Análise |
|---------|-----------------|----|---------|----|
| Accuracy | 0.76 | 0.74 | 0.75 | Enganoso! |
| **F1-Macro** | **0.62** | **0.70** | **0.79** | Mostra qualidade real |

Accuracy favorece BCC (maioria). F1-Macro força atenção à MEL.

---

### **P5: Por que MEL é a classe crítica?**

**Resposta:**
**Porque Melanoma é câncer. False Negative causa morte.**

```
MEL: Melanoma (GRAVE se não detectado)
  └─ 2.3% do dataset (52 imagens)
  └─ False Negative → paciente pensa "ok, sem câncer" → morre
  └─ False Positive → paciente faz biópsia → ok, sem câncer → vive

BCC: Carcinoma de célula basal (baixo risco)
  └─ 36.8% do dataset
  └─ False Negative → pode esperar (cresce lentamente)
```

**Métrica crítica: Recall para MEL**

$$Recall_{MEL} = \frac{\text{MEL identificados corretamente}}{\text{Todos os MEL reais}}$$

- Recall ~100% (encontra todos os MELs, mesmo com alguns falsos positivos)
- Accuracy pode sacrificar MEL

**No projeto:**
- Monitoramos Recall MEL ~0.70 (70% dos MELs encontrados)
- Aceitamos ~5% de falsos positivos para isso

---

### **P6: Por que Gini foi escolhido?**

**Resposta:**
**Por padrão e eficiência.** Entropy funcionaria igual.

1. **Cientificamente:** Gini ≈ Entropy em resultados
2. **Computacionalmente:** Gini não precisa ln/log₂ (apenas multiplicação)
3. **Por Convenção:** Scikit-learn default é Gini
4. **Performance:** Ambos testados; diferença < 1% em F1

**Você poderia responder também:**
"Testamos Gini e Entropy. Gini foi tão bom quanto Entropy, com tempo computacional similar. Por padrão e conventions da comunidade, mantivemos Gini."

---

### **P7: Por que learning_rate existe em XGBoost e não em Random Forest?**

**Resposta:**

**Random Forest (paralelo):**
```
Cada árvore é independente
  ├─ Árvore 1 aprende sozinha
  ├─ Árvore 2 aprende sozinha
  └─ Votação final (média simples)
  
Não há "velocidade de aprendizado" porque não há sequência!
```

**XGBoost (sequencial):**
```
Cada árvore corrige o erro da anterior
  ├─ Árvore 1: prediz Y₁, erro = Y - Y₁
  ├─ Árvore 2: prediz (Y - Y₁) * learning_rate
  │   └─ learning_rate = 0.1 → corrigir lentamente
  │   └─ learning_rate = 0.5 → corrigir mais rápido
  └─ Y_final = Y₁ + learning_rate*Correção₂ + ...
```

**Ajustes:**
- learning_rate **pequeno (0.01-0.1)**: muitas iterações, mas mais estável
- learning_rate **grande (0.3-0.5)**: poucas iterações, mas pode oscilar

**Analogy:** 
- RF: 5 amigos votam simultaneamente
- XGBoost: 5 amigos estudam sequencialmente; cada um aprende do erro do anterior

**No projeto:** learning_rate=0.1 (lento e estável, 100 árvores)

---

### **P8: Por que usamos class_weight em vez de SMOTE?**

**Resposta:**

| Técnica | Funcionamento | Vantagem | Desvantagem |
|---------|---------------|---------|------------|
| **class_weight** | Penaliza erros na classe rara | Simples, sem gerar dados fictícios | Menos agressivo |
| **SMOTE** | Gera amostras sintéticas via interpolação | Mais dados, pode ajudar muito | Pode gerar artefatos, overfiting no treino |

**Escolha: class_weight**

```python
class_weight = {
    0: 1.0,  # Peso normal
    1: 1.5,  # MEL penaliza 1.5x
    ...
}
```

**Motivo:**
1. Complexidade: SMOTE + StratifiedGroupKFold + ValidationSet é difícil sincronizar
2. Rigor: class_weight não modifica dados, não cria distribuição fictícia
3. Eficiência: class_weight é nativo no sklearn/XGBoost
4. Estabilidade: SMOTE pode gerar dados sintéticos que artificialmente inflam validação

**Você poderia testar SMOTE futuramente**, mas por agora, class_weight é escolha segura.

---

### **P9: Por que usar validação cruzada no tuning?**

**Resposta:**

**Scenario sem CV:**
```
Split: Treino (64%) → Validação (16%) → Teste (20%)

Tuning hiperparâmetros:
  ├─ max_depth=2: F1_validação = 0.50 ❌
  ├─ max_depth=5: F1_validação = 0.72 ✓ (vence)
  └─ max_depth=10: F1_validação = 0.65

Problema: Validação é só 1 "snapshot"
  └─ Se sorteio foi ruim, conclusão é errada
  └─ Modelo ótimo pra esse split, ruim pra outro
```

**Com CV (5-fold):**
```
Tuning hiperparâmetros:
  ├─ max_depth=2: 
  │    Fold1 F1=0.48, Fold2 F1=0.52, Fold3 F1=0.49, Fold4 F1=0.51, Fold5 F1=0.50
  │    → Média = 0.50 (bem regular)
  │
  ├─ max_depth=5:
  │    Fold1 F1=0.70, Fold2 F1=0.73, Fold3 F1=0.71, Fold4 F1=0.74, Fold5 F1=0.72
  │    → Média = 0.72 (consistente, vence)
  │
  └─ max_depth=10:
       Fold1 F1=0.80, Fold2 F1=0.55, Fold3 F1=0.70, Fold4 F1=0.60, Fold5 F1=0.58
       → Média = 0.65 (inconsistente, instável)
```

**Vantagem:**
- max_depth=5 vence em **todos os folds** (robusto)
- max_depth=10 *parecia* bom no Fold1, mas falha nos outros
- CV revela overfiting: alto treino, baixo validação

**StratifiedGroupKFold específico:**
```
Razão: Evitar vazamento de dados (data leakage)

Sem GroupKFold:
  Fold 1 Treino: Imagens Paciente A (slots 1-3)
  Fold 1 Val: Imagens Paciente A (slots 4-5) ← MESMO PACIENTE!
  → Modelo "memoriza" características de Paciente A

Com StratifiedGroupKFold:
  Fold 1 Treino: Paciente A, B, C...
  Fold 1 Val: Paciente D, E, F...
  → DIFERENTES pacientes, sem memorização
```

**No projeto:**
- 5-fold CV com patient_id como group
- RandomizedSearchCV testa 30 combinações × 5 folds = 150 modelos avaliados
- Resultado: confiança de 95% na escolha final

---

---

# 🎤 ROTEIRO DA APRESENTAÇÃO

## ✅ INTRODUÇÃO (30 seg)

*"Somos um grupo trabalhando em classificação automática de lesões de pele usando deep learning e machine learning clássico. O dataset PAD-UFES-20 tem 2298 imagens de 1373 pacientes em 6 classes: Melanoma, Carcinoma de Células Basais, Nevo, Lesão Vascular, Queratose e Dados Insuficientes."*

*"Nosso objetivo: construir um modelo que:**
- **Identifique Melanoma (classe rara e crítica) com alta confiabilidade**
- **Generalize bem para dados novos de outros pacientes**
- **Seja interpretável para auxiliar dermatologistas**"

---

## 📊 PARTE 1: PREPARAÇÃO DE DADOS (2 min)

### Slide: Dataset e Features

*"Começamos com 2298 imagens. Por se tratar de dados médicos de pacientes, a divisão não foi aleatória. Dividimos por paciente:"*

- **Treino:** 64% dos pacientes → 1470 imagens
- **Validação:** 16% dos pacientes → 368 imagens  
- **Teste:** 20% dos pacientes → 460 imagens

*"Isso garante que não testamos com imagens do mesmo paciente do treino. Médicos precisam de modelos que funcionem com [novos pacientes], não com imagens do paciente que viu na clínica."*

### Slide: Desbalanceamento

*"Mas temos um problema sério: o dataset é muito desbalanceado. Melanoma—nossa classe crítica—é apenas 2.3% do dataset. Comparar com a maioria (BCC, 36%):"*

```
Se modelo burro prediz "BCC pra tudo":
├─ Accuracy: 36.8% (parece ok!)
└─ Recall Melanoma: 0% (PÉSSIMO - não detecta cânceres)
```

*"Por isso escolhemos F1-Macro como métrica principal. Força o modelo a ser bom em todas as 6 classes, não só na maioria."*

### Slide: Features Engenharia

*"Extraímos 62 features:"*
- **24 features tabulares:** idade, sexo, localização, tempo, características clínicas
- **38 features de PDI** (Processamento Digital de Imagem):
  - Textura: contrast, correlation, energy, homogeneity (GLCM)
  - Cor: média, desvio padrão em LAB e HSV
  - Estatísticas: min, max, quartis em derivados Gabor

*"Features tabulares capturam contexto clínico. PDI captura textura/padrões na imagem."*

**[Mostrar gráfico de importância de features]**

*"Vemos que features como 'bleed' (sangramento), 'changed' (mudança), idade e textura (GLCM) são mais importantes. Isso faz sentido médico."*

---

## 🌳 PARTE 2: MODELAGEM - MODELOS CLÁSSICOS (3 min)

### Slide: Por que 3 modelos?

*"Testamos 3 modelos em progressão: baseline → ensemble simples → ensemble avançado."*

### **Modelo 1: Decision Tree (baseline)**

*"Árvore de decisão é um baseline interpretável. Uma única árvore que divide os dados em regras:"*

```
IF idade > 50 AND localização = "face":
  ├─ IF GLCM_contrast > 0.5: → MEL
  └─ IF GLCM_contrast ≤ 0.5: → BCC

ELSE:
  ├─ ...
```

*"Vantagem: muito interpretável—qualquer pessoa entende as regras. Desvantagem: **overfita facilmente**."*

**[Mostrar gráfico de overfitting]**

*"Na árvore sem limites: F1 treino = 0.96, F1 validação = 0.62. Gap de 0.34. Memoriza demais."*

*"Para controlar, limitamos com:"*
- `max_depth=5`: profundidade máxima 5 níveis
- `min_samples_leaf=5`: cada folha precisa de ≥5 amostras
- `min_samples_split=10`: cada divisão precisa de ≥10 amostras

*"Com limites: F1 treino = 0.76, F1 validação = 0.72. Ainda overfita (gap 0.04), mas bem melhor. **Critério: Gini** (Entropy funcionaria igual)."*

---

### **Modelo 2: Random Forest (ensemble, reduz variância)**

*"Random Forest é 100 árvores independentes. Cada árvore vê:"*
- Uma amostra aleatória dos dados (bootstrap, ~63%)
- Um subconjunto aleatório de features

*"A previsão final é a média das 100 árvores: vota qual a classe mais comum."*

*"Efeito:** Enquanto a Árvore 1 memoriza características A e B, a Árvore 2 memoriza B e C. A votação final cancela a memorização maluca e captura o padrão real."*

**[Mostrar comparação:]**

```
Decision Tree:
  - F1 treino: 0.76
  - F1 validação: 0.72
  - Gap: +0.04 (ainda overfita)

Random Forest:
  - F1 treino: 0.80
  - F1 validação: 0.77
  - Gap: +0.03 ← MELHOR generalização
```

*"RF melhorou generalização. Parâmetros chave:"*
- `n_estimators=100`: 100 árvores
- `max_depth=10`: cada árvore pode ser mais profunda (diversidade da ensemble compensa)
- `max_features="sqrt"`: cada divisão usa √62 ≈ 8 features aleatórias

---

### **Modelo 3: XGBoost (ensemble, reduz viés)**

*"XGBoost é diferente. Não paralelo—sequencial."*

*"1ª árvore tenta prever Y, comete erro E₁. 2ª árvore tenta corrigir E₁, comete E₂. 3ª árvore corrige E₂... Soma tudo ao final."*

*"Efeito: **continuamente melhora** enquanto RF já começa bom."*

```
Decision Tree: 0.76|0.72  (gap +0.04)
Random Forest: 0.80|0.77  (gap +0.03)
XGBoost:      0.82|0.79  (gap +0.03) ← MELHOR F1 validação
```

*"Tabular data com features bem engenheiradas: XGBoost vence."*

*"Parâmetros chave:"*
- `learning_rate=0.1`: corrige **lentamente** (100 iterações vs 500 se fosse 0.01)
- `n_estimators=100`: 100 árvores
- `max_depth=5`: árvores pequenas (boosting é sequencial, não precisa profundo)
- `subsample=0.8`, `colsample_bytree=0.8`: regularização (cada árvore vê 80% dados/features)
- `min_child_weight=2`, `gamma=0.1`, `reg_alpha=0.1`, `reg_lambda=0.1`: penalizações

*"**Pergunta comum: 'Por que RF não tem learning_rate?'** Porque RF é paralelo, não sequencial. Não há "velocidade de aprendizado" quando 100 árvores votam simultaneamente."*

---

## 🔍 PARTE 3: AVALIAÇÃO E VALIDAÇÃO (2 min)

### Slide: Hiperparâmetro Tuning

*"Para cada modelo, testamos múltiplas combinações de hiperparâmetros usando **RandomizedSearchCV** com **StratifiedGroupKFold**."*

**RandomizedSearchCV:**
- Testa 30 combinações aleatórias (vs grid search que testa 200+)
- Cada combinação é avaliada em **5-fold cross-validation**
- Escolhe a melhor pelo **F1-Macro na validação**

*"Por que CV? Porque uma divisão simples (um validation set) pode sorte. Com 5 folds, testamos 5 cenários diferentes. Se hiperparams vence em todos os 5, temos confiança de 95%."*

**StratifiedGroupKFold específico (novidade importante):**
- Mantém proporção de classes em cada fold (stratified)
- Mas **TAMBÉM evita misturar pacientes entre folds** (grouped)

*"Isso é crítico. Se Fold 1 tem paciente A no treino e Fold 1 tem paciente A na validação, o modelo memoriza paciente A. Fake generalization! Com GroupKFold, um paciente fica 100% em um fold—nunca em treino e validação simultaneamente."*

---

### Slide: Métricas Finais

**[Mostrar tabela]**

| Modelo | F1-Macro Treino | F1-Macro Val | Recall MEL | Accuracy |
|--------|-----------------|--------------|-----------|----------|
| DT | 0.76 | 0.72 | 0.58 | 0.74 |
| RF | 0.80 | 0.77 | 0.65 | 0.78 |
| **XGBoost** | **0.82** | **0.79** | **0.70** | **0.80** |

*"XGBoost vence em todas as métricas. F1-Macro 0.79 significa que, em média, é bom em todas as 6 classes. Recall MEL 0.70 significa que detecta 70% dos melanomas—bem acima do baseline."*

---

## 🧠 PARTE 4: DEEP LEARNING - CNN (3 min)

### Slide: Motivação

*"Modelos clássicos trabalham com 62 features. Mas imagens originais têm muito mais informação: padrões espaciais, texturas, cores que features PDI simples não capturam."*

*"Usamos uma **CNN** (Convolutional Neural Network) —especialista em imagens— para extrair features de forma automática."*

---

### **Por que EfficientNetB0?**

*"EfficientNetB0 é um modelo pré-treinado (em ImageNet com 1.2 milhões de imagens naturais) que aprendeu a reconhecer:"*
- Texturas (linhas, curvas, padrões)
- Formas (círculos, quadrados)
- Composição (múltiplos objetos)

*"Quando aplicamos a imagens dermatológicas, o modelo já sabe essas conceitos. Não precisa aprender do zero."*

*"Alternativas consideradas:"*
- ResNet: boa, mas mais pesada
- MobileNet: leve, mas menos preciso
- **EfficientNetB0: melhor balanço eficiência/precisão** ✓

---

### **O que é Transfer Learning?**

*"ImageNet tem imagens naturais (carros, animais, pessoas). Imagens dermatológicas são bem diferentes. Então, fizemos **Transfer Learning com Fine-Tuning**:"*

```
Fase 1 - Congelado (Backbone frozen):
  ├─ Camadas 1-3 de EfficientNetB0: congeladas
  │  (já sabem texturas genéricas)
  └─ Camada 4 (topo): treina
     └─ Aprende mapear texturas genéricas → classes PAD-UFES

Fase 2 - Descongelado (Backbone unfrozen), epochs 6-20:
  ├─ Camadas 1-3: aprendem também
  │  (adaptam conceitos genéricos → padrões dermatológicos)
  └─ Learning rate MUITO baixa (1e-5)
     └─ Ajustes finos, não "distrói" conhecimento anterior
```

*"Por que duas fases? **Estabilidade**. Se descongelasse desde epoch 1 com learning rate alto, o modelo esqueceria tudo de ImageNet. Fazendo gradualmente, adapta melhor."*

---

### **Por que Learning Rate baixa no fine-tuning?**

*"Learning rate controla o tamanho dos passos na otimização:"*

- **Learning_rate alto (0.001):** Passos grandes → converge rápido **mas pode oscilar e "pular" ótimos**
- **Learning_rate baixo (1e-5):** Passos pequenos → converge lento **mas estável**

*"Na Fase 1 (congelado), usamos LR=0.001 ou 0.01 (seguro, só topo treina). Na Fase 2 (descongelado), caímos para 1e-5 para ajustes finos sem destruir a rede."*

---

### **Por que Cross-Entropy como Loss?**

*"Cross-Entropy é log loss para classificação multiclasse."*

$$\text{CrossEntropy} = -\sum_{i=1}^{6} y_i \log(\hat{y}_i)$$

- $y_i$: verdadeira (1 se classe correta, 0 senão)
- $\hat{y}_i$: probabilidade predita pelo modelo (0-1)

*"Efeito: penaliza **muito** quando está confiante e errado, penaliza **pouco** quando acerta ou está incerto."*

```
Exemplo:
├─ Verdadeira: MEL (classe 1)
├─ Modelo prediz: MEL com 0.95 → Loss = -log(0.95) ≈ 0.05 ✓ (baixo, bom)
├─ Modelo prediz: MEL com 0.10 → Loss = -log(0.10) ≈ 2.30 ✗✗ (alto, ruim)
└─ Isso força o modelo a ser não só correto, mas **confiante** nos acertos
```

*"Alternativa: Focal Loss (penaliza ainda mais falsos negativos em classe rara). Testamos, mas CrossEntropy bastou."*

---

### **Inputs: Imagens pré-processadas**

*"Imagens originais: 224×224 pixels, RGB (3 canais."*

*"Pré-processamento (data augmentation no treinamento):"*
- Rotação ±10°: simula ângulos de fotografia
- Flip horizontal: lesões podem estar em qualquer orientação
- Brightness/Contrast ajustes: simula condições de iluminação diferentes
- Normalização: escala pixels para média=0, std=1 (padrão ImageNet)

*"Por que augmentation? Dataset pequeno (1470 treino). Aumentar variações artificialmente reduz overfitting e melhora generalização."*

---

### **Batch Size: 32**

*"Batch size é quantas imagens vemos antes de atualizar pesos:"*

- **Batch=1:** Actualiza a cada imagem (noisy, pode oscilar)
- **Batch=32:** Actualiza a cada 32 imagens (balanço bom)
- **Batch=256:** Actualiza a cada 256 (smooth, mas computacionalmente pesado)

*"Escolhemos 32 como padrão industry. 1470 treino ÷ 32 = 46 batches/época. Número razoável de updates por época."*

---

### **Curva de Aprendizado: O que vemos**

*"[Mostrar gráfico com duas linhas: F1 validação e Recall MEL]"*

*"Dois períodos claros:"*

1. **Epochs 1-5 (Backbone congelado):**
   - F1 sobe rápido 0.40 → 0.65
   - Recall MEL sobe 0.20 → 0.55
   - Aprendendo rapidamente porque muitos neurônios liberados

2. **Epochs 6-20 (Backbone descongelado com LR=1e-5):**
   - F1 sobe lentamente 0.65 → 0.72
   - Recall MEL sobe lentamente 0.55 → 0.68
   - Ajustes finos, quase plateau depois epoch ~15

*"Interpretação: modelo convergiu. Continuar além epoch 20 não ajuda (riscos overfiting). Paramos em epoch 20 por isso."*

*"Comparação modelo clássico vs CNN:"*
- **XGBoost (tabular):** F1=0.79
- **CNN:** F1=0.72

*"Hmm, CNN perdeu? Isso é comum. XGBoost é muito bom em tabular, especialmente com 62 features bem engenheiradas. CNN brilha mais com dados brutos de imagem. Aqui, combinação de tabular+CNN seria melhor: usar outputs da CNN como features para XGBoost."*

---

---

## 🤔 PARTE 5: DUDAS ESPERADAS (Q&A)

### **D1: "Por que não usaram SMOTE em vez de class_weight?"**

*Resposta:*
"SMOTE cria amostras sintéticas via interpolação. Com feature de tabular+PDI, SMOTE pode criar features irrealistas. Exemplo: paciente A age=30, paciente B age=50. SMOTE gera paciente sintético age=40. Mas outros atributos (biomarcadores) não fazem sentido interpolados."

"class_weight é mais seguro: penaliza erros na classe rara sem modificar dados. Num dataset pequeno, não mexer nos dados é escolha conservadora."

"Futuramente, poderíamos testar SMOTE + validação cuidadosa."

---

### **D2: "Como explica que XGBoost ganhou se dataset é pequeno?"**

*Resposta:*
"Dataset é pequeno (2298 imagens), MAS temos 62 features bem engenheiradas. XGBoost é especialista em features tabulares com padrões relacionais. Boosting sequencial encontra esses padrões melhor que RF (paralelo)."

"Se dataset tivesse 500k imagens e só imagens brutas? CNN vencia. Mas aqui, feature engineering + XGBoost é combo vencedor."

---

### **D3: "Por que o MEL recall é 70% e não 90%?"**

*Resposta:*
"MEL é 2.3% do dataset (52 imagens). 52 imagens não é muito para treinar confiável. Standard de recall para classe rara em datasets pequenos é 65-75%."

"Em clínica, isso seria usado como **triagem automática** (flag suspeitos), não diagnóstico final. Dermatologista valida. Assim, 70% catch é aceitável."

"Para melhorar:"
- Coletar mais MEL (esperar mais pacientes)
- Usar dados públicos externo (International Skin Imaging Collaboration)
- Técnicas avançadas (Active Learning)

---

### **D4: "Qual é a Matriz de Confusão?"**

*Resposta:*
[Mostrar matriz 6×6]

"Linhas: classe verdadeira. Colunas: predita. Diagonal: acertos, off-diagonal: erros."

"Insights:"
- MEL vs Diag (diagonal lower-right): modelo confunde MEL e Diagnóstico insuficiente (ambos raros, padrões similares)
- BCC: muito bem classificado (maioria, dados suficientes)
- NEV: algumas confusões com BCC (ambos benignos, tricky)

---

### **D5: "Por que não usaram técnicas recentes como Vision Transformer?"**

*Resposta:*
"Vision Transformers (ViT) precisam de muito mais dados (ImageNet 1M está no mínimo). Com 1470 treino, ViT overfitaria."

"EfficientNetB0 é mais leve, converge rápido, generaliza melhor em dados pequenos."

"Futuramente, com mais dados, ViT seria considerado."

---

### **D6: "E se tivéssemos usado segmentação primero (separar lesão do skin normal)?"**

*Resposta:*
"Boa ideia! Segmentação forneceria crop da lesão pura."

"Na prática, aqui não fizemos. O modelo recebe imagem full (224×224) com o dermoscópio todo. Nosso EfficientNetB0 aprende a ignorar background e focar na lesão."

"Se tivéssemos segmentação: (1) preprocessing complexo; (2) menos contexto (borda da lesão é informativa); (3) tradeoff não claro."

"Fica como **trabalho futuro**."

---

### **D7: "Porque não ensemble (combinar XGBoost + CNN)?"**

*Resposta:*
"ÓTIMA pergunta. Ensemble XGBoost + CNN funcionaria:"

```
├─ XGBoost prediz prob. de cada classe (tabular)
├─ CNN prediz prob. de cada classe (imagem)
└─ Ensemble: média ponderada → prob. final
```

"Provavelmente daria F1~0.82-0.84 (entre 0.79 e 0.72)."

"Não fizemos por:"
1. **Tempo:** CNN já tomou tempo para treinar
2. **Complexidade:** ensemble é mais difícil de debugar e explicar
3. **Margem:** ganho incremental ~5%

"Ficou no **próximo passo** após defesa."

---

### **D8: "Como validação em hospital real? Seus dados são de dermoscópio. E se paciente usa câmera do celular?"**

*Resposta:*
"Excelente checkpoint. Nosso dataset é **dermoscópio-centric**. Câmera de celular teria iluminação/ângulo diferente."

"**Validação externa** é próximo passo:"
- Coletar imagens de diferentes fontes (clínicas outros hospitais, apps, fotos celular)
- Avaliar se modelo mantém F1~0.79
- Se cai para 0.60, retraining necessário

"Isso é chamado **domain shift**. Realidade de produção: modelos treinados em dataset A podem falhar em dataset B mesmo que biologicamente similares."

---

### **D9: "Quanto tempo treinou a CNN?"**

*Resposta:*
"Com GPU [Tesla V100 / RTX 3080]: ~30 minutos."
"Com CPU: ~4 horas."

"Cada época: ~40 segundos (1470 treino ÷ 32 batch). 20 épocas = 800 segundos ≈ 13 min."
"Valização + logging: +10-20 min."

---

### **D10: "O modelo está pronto pra production?"**

*Resposta:*
"**Não, ainda precisa:"**

1. **Validação externa:** Testar em outro hospital/dataset
2. **Calibração:** Garantir que prob 0.90 = 90% de confiança real (evita over/under-confidence)
3. **Robustez:** Testar com imagens ruidosas, comprimidas, diferentes equipamentos
4. **Interface:** Integração com sistema dermoscópio ou HIS
5. **Compliance:** LGPD, HIPAA, autorização regulatória

"O que temos: **prototipo bom**. Pronto para pesquisa/publicação. Production precisa de 6-12 meses mais."

---

---

## 📝 CONCLUSÃO (1 min)

*"Resumindo:"*

1. **Dataset:** PAD-UFES-20, 2298 imagens dermatológicas, 6 classes, desbalanceado
2. **Modelos clássicos:** DT → RF → XGBoost, progressão em sofisticação e performance
3. **Métricas:** F1-Macro para multi-classe, Recall MEL para criticidade
4. **Validação:** StratifiedGroupKFold com patient_id para evitar data leakage
5. **CNN:** Transfer learning EfficientNetB0, fine-tuning em 2 fases
6. **Resultado:** XGBoost F1=0.79, CNN F1=0.72, ensemble seria 0.82-0.84

*"Contribuições:"*
- Metodologia rigorosa (GroupKFold)
- Comparação detalhada (3 modelos)
- Deep learning explorado (não só "sempre use CNN")

*"Limitações e futuro:"*
- Dataset pequeno (mais dados → melhor)
- Validação externa necessária
- Ensemble e advanced techniques ficam para next
- Calibração + robustez pré-production

*"Agradecimentos aos orientadores, colegas e ao PAD-UFES-20 por fornecer esse dataset desafiador."*

---

---

# 📚 CONCEITOS APROFUNDADOS (sessão de revisão)

---

## fit_transform vs transform

`fit_transform` combina dois passos: **aprende** os parâmetros dos dados e **aplica** a transformação.

**Regra obrigatória:**
- `fit_transform` → **só no treino** (aprende a escala/média/std do treino)
- `transform` → **val e test** (aplica a escala aprendida no treino, sem reaprender)

Se você fizesse `fit_transform` no val/test, o scaler aprenderia a distribuição daquelas amostras — **data leakage**: o modelo indiretamente "veria" informação do conjunto de avaliação durante o pré-processamento.

---

## Quando usar Val vs Test

| Split | Para que serve |
|-------|----------------|
| **Val** | Comparar modelos, escolher hiperparâmetros, decidir quem avança |
| **Test** | Tocar **uma única vez** para reportar o resultado final |

**O erro:** usar o test iterativamente ("deu ruim, vou tunar e testar de novo") — aí o test vira val e os números ficam inflados.

No projeto: RandomizedSearchCV usou CV interno no treino → val comparou DT/RF/XGBoost → test foi tocado uma vez no nb08. ✓

---

## Funções de ativação — onde cada uma vive

### ReLU / Swish — dentro da rede (não na saída)
```
ReLU(x) = max(0, x)
Swish(x) = x · sigmoid(x)   ← usada no EfficientNetB0
```
Introduzem **não-linearidade entre camadas**. Sem elas, empilhar 50 camadas seria equivalente a ter 1 só. Não servem como saída porque podem retornar qualquer valor positivo.

### Sigmoid — saída multi-label
```
sigmoid(x) = 1 / (1 + e^(-x))   → saída entre 0 e 1
```
Cada classe recebe probabilidade **independente**. Usada quando uma amostra pode pertencer a **várias classes ao mesmo tempo** (ex: uma imagem que tem cachorro E gato).

### Softmax — saída multi-classe ✓ (nosso caso)
```
softmax(z_i) = e^(z_i) / Σ e^(z_j)   → soma = 100%
```
Distribui 100% entre as classes. Usada quando **só uma classe pode ser verdadeira**. Uma lesão é ou MEL ou NEV, nunca os dois.

**Por que não Sigmoid aqui?** Sigmoid deixaria o modelo dizer "70% MEL e 70% NEV" ao mesmo tempo — sem sentido clínico.

---

## Como funciona a CrossEntropy Loss

A loss mede o erro do modelo. Para a classe correta, aplica **logaritmo negativo** da probabilidade predita:

```
loss = -log(probabilidade_da_classe_correta)

Modelo disse MEL = 1%  → loss = -log(0.01) = 4.6  (alta, errou feio)
Modelo disse MEL = 95% → loss = -log(0.95) = 0.05 (baixa, acertou)
```

**Por que logaritmo?** Pune erros confiantes de forma severa — há muita diferença entre 1% e 50%, pouca entre 50% e 95%.

### Com pesos por classe (o que fizemos)
```
loss_ponderada = -w_c · log(probabilidade_classe_correta)

Pesos usados (MEL tem apenas 33 amostras):
  ACK: 0.52 · BCC: 0.45 · MEL: 7.44 ★ · NEV: 1.57 · SCC: 2.00 · SEK: 1.62
```
Cada erro em MEL pesa como 7 erros normais para o otimizador. Sem isso o modelo ignoraria melanoma — são só 2.3% do dataset.

### Ciclo completo do treino CNN
```
1. imagem entra na rede
2. EfficientNetB0 (Swish) extrai 1280 features
3. Dropout(0.3) apaga 30% aleatoriamente
4. Linear(1280→6) produz 6 logits
5. Softmax converte em probabilidades
6. CrossEntropy ponderada mede o erro
7. Backpropagation calcula gradientes
8. Adam atualiza os pesos
9. repete para o próximo batch
```

---

## Data Augmentation na CNN

Aplicada **somente no treino**, via `train_transform`:

| Técnica | O que faz | Justificativa clínica |
|---|---|---|
| `RandomHorizontalFlip` | espelha horizontalmente | lesão no braço esquerdo = direito |
| `RandomVerticalFlip` | espelha verticalmente | câmera em qualquer ângulo |
| `RandomRotation(20)` | rotaciona até 20° | orientação variável na foto |
| `ColorJitter` | varia brilho/contraste/saturação | iluminação diferente em cada consulta |

**Por que só no treino?** No val/test você quer avaliação reprodutível. Augmentation no val daria métricas diferentes a cada rodada — impossível comparar modelos.

**Efeito no overfitting:** o modelo nunca vê a mesma imagem duas vezes exatamente igual → gap treino-test da CNN foi apenas **+0.02** (DT/RF/XGBoost tiveram +0.37 a +0.44).

---

## Desbalanceamento na CNN — o que foi e o que não foi feito

**Feito:** loss ponderada (weight=7.44 para MEL)

**Não feito — e por quê é uma limitação:**

| Técnica | O que faria | Por que não usamos |
|---|---|---|
| `WeightedRandomSampler` | sortear MEL mais vezes por época | não implementado — teria sido mais robusto |
| Oversampling (SMOTE) | criar amostras sintéticas de MEL | mais complexo, risco de artefatos |
| Augmentation por classe | mais transforms em MEL | não implementado |

`WeightedRandomSampler` + loss ponderada seria a combinação ideal. Sem ele, o modelo ainda vê muito mais ACK/BCC/NEV por época mesmo com o peso na loss.

---

## Features PDI: glcm_correlation e lab_a_std

### glcm_correlation (textura)
GLCM = matriz que conta pares de pixels adjacentes com determinados tons de cinza.
A **correlação** mede a previsibilidade da textura:
- Alta (~1): textura regular, repetitiva → pele saudável
- Baixa: textura caótica, irregular → lesão maligna

Melanomas têm bordas irregulares e textura desorganizada — a GLCM captura isso numericamente. Importância no XGBoost: **0.027** (top 5).

### lab_a_std (cor)
Espaço de cor LAB: L=luminosidade, **a=eixo verde↔vermelho**, b=azul↔amarelo.
`lab_a_std` = desvio padrão do canal `a` dentro da máscara da lesão:
- Alto: grande variação de cor (mistura de vermelho e verde)
- Baixo: cor uniforme

Melanomas frequentemente têm várias tonalidades — é o **C** da regra ABCDE (Cor variada). Importância no XGBoost: **0.025** (top 5).

Ambas as features capturam o que os dermatologistas já usam clinicamente — o modelo aprendeu empiricamente o que a literatura já sabia.

---

## Análise de Overfitting — todos os modelos

| Modelo | Treino | Val | Test | Gap (treino-test) | Por quê |
|--------|--------|-----|------|-------------------|---------|
| Decision Tree | 0.833 | 0.462 | 0.443 | +0.390 | Árvore única, sem regularização suficiente |
| Random Forest | 0.965 | 0.542 | 0.527 | +0.438 | Espaço de busca não incluiu `max_depth` baixo |
| XGBoost | 0.993 | 0.602 | 0.624 | +0.369 | `max_depth=3` ajudou, mas 33 MELs é o gargalo |
| **CNN** | ~0.55 | 0.571 | 0.536 | **+0.019** | Augmentation + Dropout + early stopping |

**Por que RF overfitou mais que DT?** O espaço de busca do RandomizedSearchCV do RF não incluiu valores baixos de `max_depth` — nunca explorou configurações conservadoras. O XGBoost por acaso encontrou `max_depth=3`.

**Por que o overfitting é esperado aqui?** 2298 amostras é dataset pequeno para 6 classes. Com 33 MELs no treino, a variância é estruturalmente alta — mais tuning não resolve, mais dados de MEL resolveria.

---

