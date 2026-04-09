# Roteiro Falado — Parte 1: Apresentação do Dataset

> Tempo alvo: **~4 minutos**. Leitura natural, sem decorar. As frases entre
> colchetes são notas de palco (pausas, ênfase, gestos), não para falar.

---

## Slide 1 — O problema e o dataset  *(~45s)*

> "Bom dia. O dataset que usamos no projeto é o **PAD-UFES-20**, coletado
> pelo programa de assistência dermatológica da Universidade Federal do
> Espírito Santo.
>
> A particularidade dele é que as imagens **não são dermatoscópicas** — são
> fotos clínicas, tiradas com smartphone comum, do jeito que um agente de
> saúde fotografaria a lesão de um paciente no atendimento.
>
> No total temos **2298 imagens**, vindas de **1373 pacientes** e cobrindo
> **1641 lesões diferentes**, distribuídas em **6 classes diagnósticas**.
> Nosso objetivo é, dada uma dessas imagens, classificar em qual das seis
> categorias a lesão se encaixa."

[transição: "Vamos ver quais são essas seis classes."]

---

## Slide 2 — As 6 classes e o desbalanceamento  *(~60s)*

> "As seis classes se dividem em três grupos: **três malignas — BCC, SCC e
> melanoma —**, uma pré-maligna que é a ceratose actínica (ACK), e duas
> benignas, o nevo (a 'pinta' comum) e a ceratose seborreica.
>
> Olhando o gráfico, o problema do dataset salta aos olhos: ele é
> **fortemente desbalanceado**. O carcinoma basocelular tem 845 amostras, e
> aqui [apontar a barra do MEL] o **melanoma — que é justamente o câncer
> de pele mais letal — tem só 52 amostras**, ou seja, 2,3% do total. A
> razão entre a maior e a menor classe é de cerca de **16 vezes**.
>
> Isso já define um desafio central do projeto: qualquer modelo otimizado
> pra acurácia bruta vai simplesmente ignorar o melanoma. Nós vamos
> precisar usar **métricas por classe** e estratégias específicas pra
> classe minoritária — isso volta na parte de modelagem."

[transição: "E o que exatamente nós temos pra cada amostra?"]

---

## Slide 3 — O que cada amostra contém  *(~60s)*

> "Cada amostra do dataset traz **duas fontes de informação**.
>
> Do lado da **imagem**, são arquivos PNG com tamanho bem variável — vão
> de 189 até 3000 pixels de lado, com mediana em torno de 770. A maioria é
> quadrada, mas alguns vêm em RGB e outros em RGBA. E como foram tiradas
> com celular, têm iluminação irregular, presença de pelo, sombras e
> fundo de pele saudável ao redor da lesão. Tudo isso vai precisar ser
> normalizado antes da extração de features.
>
> Do lado dos **metadados**, são 26 colunas que a gente agrupou em quatro
> categorias: dados do **paciente** — idade, sexo, escala de Fitzpatrick,
> histórico de câncer; dados da **lesão** — em que região do corpo está,
> os diâmetros em milímetros; **sintomas booleanos** — se coça, se
> cresceu, se sangra, se mudou; e o **diagnóstico**, que é o nosso alvo.
>
> E é importante deixar claro: **o projeto vai usar as duas fontes
> juntas**. Da imagem, vamos extrair features por processamento digital
> de imagens — cor, textura, forma, bordas. Dos metadados, vamos
> aproveitar os sinais clínicos que o dermatologista usa na prática:
> idade, tipo de pele, região do corpo, sintomas. **No modelo final, as
> features de PDI e os metadados entram juntos**, porque cada fonte
> captura uma parte diferente do problema."

[transição: "E ao explorar o dataset, encontramos três armadilhas que
moldaram quase todas as decisões do projeto."]

---

## Slide 4 — Três armadilhas que moldam o projeto  *(~75s)*

> "Três descobertas da exploração que são importantes pra entender o
> resto da apresentação.
>
> **Primeira: vazamento por paciente.** Como a gente viu, são 2298
> imagens mas só 1641 lesões e 1373 pacientes. Isso significa que existem
> **várias fotos da mesma lesão**, e vários pacientes com mais de uma
> lesão. Se a gente fizer um split aleatório de treino e teste, é quase
> garantido que duas fotos quase idênticas da mesma lesão caiam, uma no
> treino e outra no teste — e o modelo vai parecer ótimo, mas estaria
> trapaceando. **Decisão: split por paciente**, não por imagem.
>
> **Segunda armadilha: a coluna `biopsed` é um vazamento.** A tabelinha
> aqui mostra: **100% das lesões malignas — BCC, melanoma e SCC — foram
> biopsiadas**. Faz sentido na vida real: você só pede biópsia depois de
> já desconfiar. Mas pro modelo, essa coluna praticamente entrega a
> resposta. **Decisão: remover `biopsed` das features**.
>
> **Terceira: 35% dos metadados clínicos estão ausentes em bloco.**
> Treze colunas têm exatamente 804 valores nulos cada — dois formulários
> distintos de coleta. E ao investigar descobrimos uma coisa mais grave:
> **nenhuma dessas 804 linhas contém lesão maligna**. Zero BCC, zero
> melanoma, zero SCC. Ou seja, ter ou não ter metadados é um proxy quase
> perfeito do diagnóstico — o que na literatura se chama **MNAR
> outcome-dependent**. **Decisão: dropar essas 13 colunas**. Eu explico
> o porquê no próximo slide, mas a razão curta é que **imputar seria
> leak mascarado**.
>
> Essas três descobertas justificam todo o pipeline que vou apresentar a
> seguir."

[transição para o próximo bloco da apresentação: "Então, partindo desse
entendimento, o primeiro passo do pipeline foi o pré-processamento das
imagens..."]

---

## Checklist mental antes de subir ao palco — Parte 1

- [ ] Lembrar de **apontar o MEL** no slide 2 — é o ponto emocional
- [ ] No slide 4, **falar devagar** nas decisões (split / biopsed / focar na imagem) — são as 3 coisas que o público precisa levar pra casa
- [ ] Não ler bullets — usar como apoio
- [ ] Manter o ritmo: **4 minutos no total**, não passar de 5

---

# Roteiro Falado — Parte 2: Pré-processamento dos Metadados

> Tempo alvo: **~2:30 minutos** no total, em **2 slides**.
> Slide 5 (~90s) mostra o pipeline completo e justifica a decisão mais polêmica
> (dropar 35% do dataset) com literatura. Slide 6 (~60s) mostra as features
> finais que entram no modelo e o tratamento do desbalanceamento.

---

## Slide 5 — Pipeline completo de pré-processamento  *(~90s)*

> "O pré-processamento dos metadados é basicamente a resposta pra uma
> pergunta: **como não trapacear?** O fluxograma à esquerda mostra os
> oito passos do pipeline.
>
> Começamos do CSV bruto e a primeira coisa que fazemos é uma
> **verificação de qualidade**: confirmamos que não há linhas
> duplicadas e que o `img_id` é único — isso é crítico porque o
> `img_id` vai ser a chave de merge com o notebook de PDI mais pra
> frente. Também descobrimos que **512 lesões têm mais de uma foto**
> — essas não são duplicatas a remover, são fotos legítimas do mesmo
> caso, e é justamente por isso que o split precisa ser agrupado por
> paciente.
>
> Aí vêm os **drops**: removemos o `biopsed` pelo leak que eu já
> mostrei, e o bloco de **13 colunas com missing**. Essa é a decisão
> mais impactante do notebook.
>
> [pausa curta, apontar pro Card 1]
>
> A primeira reação natural seria imputar esses valores, mas a
> literatura fecha essa porta. **Ma e Zhang, no NeurIPS 2021**,
> mostraram que dados MNAR — que é o nosso caso — não são
> identificáveis: mesmo com infinitas amostras, a imputação fica
> enviesada. E **Sisk e colegas, num paper de 2023**, mostraram que
> adicionar uma flag de missingness em dados outcome-dependent é
> literalmente *harmful* — re-introduz o leak com outro nome. Então
> a única saída honesta é dropar. Preservamos as 2298 linhas,
> incluindo os 52 melanomas, e o sinal biológico real — fotótipo e
> diâmetro — vai ser recuperado via PDI da imagem no próximo notebook.
>
> Depois disso tratamos os sintomas: descobrimos que `grew` e
> `changed` têm 17% de valor *unknown*, então criamos flags específicas
> pra preservar os três estados. Criamos também o `symptom_count` pra
> condensar os critérios ABCDE. Fazemos análise de outliers em `age`
> — tem 53 outliers estatísticos, mas 94% deles são nevos em pacientes
> jovens: não é ruído, é o sinal clínico da classe. **Preservamos.**
>
> Aí vem o **split por paciente** com `StratifiedGroupKFold`, zero
> overlap verificado, e rodamos um drift check — a mediana de idade
> é idêntica nos três splits. Por fim, tudo entra num
> `ColumnTransformer` ajustado **exclusivamente no treino**, com
> **duas estratégias de encoding**: `OneHotEncoder` na `region`,
> porque são 14 categorias nominais sem ordem natural — e o sklearn
> ainda agrupa as regiões raras automaticamente via `min_frequency`;
> e `LabelEncoder` na variável alvo, que é o padrão pra classificação
> multiclasse do sklearn. `age` passa pelo `StandardScaler`, os
> sintomas vão direto. Todo o pipeline é serializado em disco como
> `joblib` — o próximo notebook só precisa carregar. Zero imputação,
> zero leak."

[transição curta: "E as features que isso gera estão no próximo
slide..."]

---

## Slide 6 — Features finais e compensação do desbalanceamento  *(~60s)*

> "Depois de tudo isso, o que alimenta o modelo são **24 features**,
> divididas em cinco grupos.
>
> [apontar pra tabela]
>
> Uma única feature numérica — a `age`, escalada. **Catorze colunas
> one-hot** da região do corpo — o sklearn agrupa regiões raras como
> `LIP` e `SCALP` automaticamente via `min_frequency=10`. **Seis
> sintomas booleanos** da lesão. **Duas flags de unknown** pra
> preservar a informação "paciente não sabe". E o `symptom_count`
> derivado. Nada mais.
>
> Repare no que **não entra**: `biopsed` pelo leak, `fitspatrick` e
> diâmetros pelo bloco MNAR, e toda a anamnese do paciente — tabagismo,
> álcool, histórico, condições socioeconômicas, ancestralidade —
> porque são exatamente as colunas outcome-dependent que a gente
> removeu.
>
> [apontar pro gráfico de class weights]
>
> E como o dataset é muito desbalanceado — BCC tem 541 amostras no
> treino, melanoma tem só 33 — calculamos os **class weights
> balanceados** aqui mesmo. A fórmula é simples: cada classe recebe
> um peso inversamente proporcional à frequência. O melanoma fica
> com peso **7.4**, contra **0.45** do BCC — ou seja, **um erro em
> melanoma 'dói' 16 vezes mais** durante o treino. E o detalhe
> importante: o **dataset fica intacto**. Os 33 melanomas continuam
> 33. A compensação acontece na função de perda, não nos dados. Isso
> evita o risco de overfit que métodos como SMOTE teriam com tão
> poucas amostras.
>
> O pipeline inteiro sai em disco como artefato — parquets, scaler,
> encoder, pesos de classe — e o próximo notebook, que é o de PDI,
> só precisa carregar."

[transição para o próximo bloco: "Com os metadados resolvidos,
partimos pro pré-processamento das imagens..."]

---

## Checklist mental — Parte 2

- [ ] **Slide 5:** falar devagar no Card 1 (literatura) — é a defesa da decisão mais impopular
- [ ] **Slide 5:** não detalhar `StratifiedGroupKFold` — só mencionar "zero overlap" e seguir
- [ ] **Slide 5:** se o ritmo apertar, cortar a parte dos autores (Ma & Zhang / Sisk) e só dizer "a literatura mostra que imputar seria leak mascarado"
- [ ] **Slide 6:** fazer o gesto de "dói 16× mais" apontando o gráfico — é a mensagem visual mais forte do slide
- [ ] **Slide 6:** deixar claro que os dados ficam **intactos** — isso diferencia class_weight de SMOTE/undersampling
- [ ] Manter os dois slides dentro de ~2:30 no total — se passar, cortar no 5, não no 6 (o 6 é visual e rápido)

---

# Roteiro Falado — Parte 3: Pré-processamento das Imagens (PDI)

> Tempo alvo: **~90 segundos**. **1 slide só.** A ideia é mostrar o
> pipeline de 4 etapas, explicar brevemente o Otsu (sem entrar em
> matemática) e reportar a qualidade da segmentação. O público precisa
> sair sabendo: "o pipeline isola a lesão real e tem 1.5% de falha".

---

## Slide 7 — Pré-processamento das imagens (PDI)  *(~90s)*

> "Agora o pré-processamento das imagens. O objetivo é transformar as
> fotos de smartphone cruas — com pelo, iluminação irregular, resolução
> entre 189 e 3000 pixels — numa forma padronizada, onde as features
> possam ser extraídas da lesão real, e não de uma mistura de lesão,
> pele, pelo e fundo.
>
> O pipeline tem quatro etapas. Primeiro, **hair removal** via DullRazor,
> que detecta fios escuros com uma operação morfológica chamada blackhat
> e reconstrói os pixels por inpainting. Segundo, **color constancy**
> com Shades of Gray, que normaliza o balanço de branco entre smartphones
> diferentes. Terceiro, **segmentação da lesão** com Otsu — eu já volto
> nesse —, e por último, **crop quadrado da bounding box** com 10% de
> padding e resize pra 256 por 256, preservando o aspect ratio pra não
> distorcer a assimetria.
>
> [apontar pra caixa central do Otsu]
>
> Sobre o Otsu, rapidamente: é um algoritmo de 1979 que encontra o
> threshold binário ótimo automaticamente. No canal L* do LAB, o
> histograma de uma foto dermatológica é tipicamente **bimodal** — tem
> um pico de pixels claros que é a pele, e um pico de pixels escuros
> que é a lesão. O Otsu varre os 256 valores possíveis de threshold e
> escolhe aquele que **maximiza a variância entre os dois grupos** —
> ou seja, o que deixa os dois picos o mais separados possível. É um
> algoritmo com zero parâmetros treinados, roda em 5 milissegundos por
> imagem, e é totalmente interpretável. Quando ele falha — lesão
> uniforme, iluminação estranha — temos um fallback que é simplesmente
> um retângulo central cobrindo 60% da imagem.
>
> [apontar pros cards de resultado no rodapé]
>
> Nos resultados, processamos as **2298 imagens com zero erros**, e
> tivemos só **1.5% de fallback total** — bem abaixo do 5 a 10% típico
> em datasets dermatológicos. A classe com mais fallback foi o nevo,
> com 9%, que faz sentido clinicamente porque pintas tendem a ser
> uniformes e se misturam mais com a pele. O melanoma, que é a classe
> crítica, teve só 2 fallbacks em 52 amostras — ou seja, **96% dos
> melanomas foram segmentados com sucesso**.
>
> Cada imagem vira um arquivo `.npz` contendo a imagem processada mais
> a máscara, e o próximo notebook — de extração de features — vai
> consumir isso via uma função helper e extrair features apenas dos
> pixels dentro da máscara."

[transição para o próximo bloco: "Feito o pré-processamento das duas
fontes, agora a gente extrai as features e junta tudo num dataset
final pra modelagem..."]

---

## Checklist mental — Parte 3

- [ ] **Não entrar em matemática do Otsu** — só a frase "maximiza a variância entre os dois grupos" e segue
- [ ] **Falar devagar na parte do fallback** — é onde o público pode perder o fio ("eles têm um plano B quando Otsu falha")
- [ ] **Enfatizar o 96% dos melanomas** — é a mensagem que o público deve levar pra casa: "a classe crítica foi bem segmentada"
- [ ] **Não mostrar exemplos de máscara visual no slide** — se existir tempo, é melhor na parte de resultados finais; aqui fica no esquema do fluxograma
- [ ] Se estiver indo longo, cortar a explicação do DullRazor (é a parte menos importante do pipeline) e só dizer "remove pelo"
- [ ] **90 segundos no máximo** — se passar, o resto da apresentação aperta

---

# Roteiro Falado — Parte 4: Extração de Features + Merge

> Tempo alvo: **~90 segundos**. **1 slide só.** A ideia é mostrar as 5
> famílias de features PDI de forma rápida, enfatizar que foram **escolhidas
> pelo critério ABCDE clínico** (não arbitrariamente), e reportar o achado
> empírico mais forte do MI ranking — que `lab_a_std` ficou em 2º lugar,
> confirmando a decisão de priorizar LAB.

---

## Slide 8 — Extração de features + merge  *(~90s)*

> "Com as imagens processadas, o próximo passo é transformar cada uma
> num vetor numérico de features. Extraímos **38 features handcrafted
> por imagem, todas dentro da máscara da lesão** — porque é exatamente
> isso que a segmentação do Otsu permitiu. Se tivéssemos rodado sobre
> a imagem inteira, estaríamos medindo pele normal junto com lesão.
>
> [apontar pra tabela à esquerda]
>
> As 38 features se dividem em **cinco grupos**, cada um mapeado a um
> critério do ABCDE clínico. **Doze features de cor** em LAB e HSV,
> medindo a variegação — que é o C do ABCDE. **Seis features GLCM**
> de Haralick, medindo textura via co-ocorrência. **Dez features LBP**,
> que capturam textura de forma robusta a iluminação. **Três features
> de borda**, que correspondem ao B do ABCDE — densidade de bordas
> interna, irregularidade do contorno comparada a um círculo, e quão
> difusa é a transição lesão/pele. E **sete features de forma**
> calculadas direto da máscara — compactness, solidity, eccentricity,
> e principalmente a **assimetria após rotação pro eixo principal** —
> que é o A do ABCDE. **Essas sete de forma eram impossíveis sem a
> segmentação** do notebook anterior — são o maior ganho técnico do
> refactor.
>
> Originalmente eu tinha proposto 81 features, mas cortei pra 38
> removendo redundâncias: RGB é redundante com LAB, LBP multi-raio com
> alta correlação interna, Gabor redundante com GLCM+LBP. Em dataset
> pequeno, menos features boas é melhor que mais features médias.
>
> [apontar pro bloco direito — Top 5 por MI]
>
> A validação empírica veio pelo **ranking de Mutual Information**.
> `age` ficou em primeiro, o que é esperado. Mas em **segundo lugar
> ficou `lab_a_std` — com MI de 0.20, acima de todas as features de
> sintomas clínicos**. Isso é exatamente a variação no eixo vermelho-verde
> do LAB, que é o critério C de variegação de cor. **Confirmou
> empiricamente a decisão de priorizar LAB em vez de RGB.** Além
> disso, quatro das sete features de forma também ficaram no top 20,
> todas novas em relação ao notebook original.
>
> [apontar pro rodapé]
>
> O merge final é simples: 24 features tabulares do nb02 + 38 PDI
> deste notebook = **62 features finais**, por `img_id` com validação
> one_to_one. Zero NaN, zero leak residual — o máximo MI de qualquer
> feature sozinha foi 0.23, bem abaixo do threshold de 0.7 que
> indicaria vazamento."

[transição para o próximo bloco: "Com o dataset final pronto, passamos
para a modelagem..."]

---

## Checklist mental — Parte 4

- [ ] **Enfatizar o "38 features, todas dentro da máscara"** — o público precisa conectar que extrair dentro da máscara é o que diferencia do notebook original
- [ ] **Falar devagar na parte do mapeamento ABCDE** — é o que dá legitimidade clínica às escolhas
- [ ] **`lab_a_std` em 2º lugar é a punchline do slide** — fazer pausa breve após dizer o MI de 0.20, deixar assentar
- [ ] **Não listar cada uma das 38 features uma por uma** — o público não precisa decorar, só precisa ver a estrutura dos 5 grupos
- [ ] **Se apertar o tempo:** cortar a parte sobre o corte de 81→38 (é explicação de processo, não de resultado)
- [ ] **90 segundos no máximo** — densidade de informação é alta, falar pausado mas sem enrolar
