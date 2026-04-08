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

> Tempo alvo: **~90 segundos**. **1 slide só.** A ideia é mostrar que as
> decisões são consequência direta do que foi apresentado no slide 4 da
> Parte 1 — e justificar a mais polêmica (dropar 35% do dataset) com
> literatura.

---

## Slide 5 — Pré-processamento dos metadados  *(~90s)*

> "O pré-processamento dos metadados é basicamente a resposta pra uma
> pergunta: **como não trapacear?** O fluxograma resume os passos.
>
> Primeiro, removemos o `biopsed` — já expliquei o porquê no slide
> anterior. Em seguida, e essa é a decisão mais impactante do notebook,
> **removemos o bloco de 13 colunas com missing**.
>
> [pausa curta, apontar pro Card 1]
>
> Aqui vale uma justificativa técnica, porque a primeira reação natural
> seria imputar esses valores. A literatura, no entanto, fecha essa
> porta. **Ma e Zhang, no NeurIPS de 2021**, mostraram que dados MNAR
> — que é o nosso caso — não são identificáveis: mesmo com infinitas
> amostras, a imputação fica enviesada. **E Sisk e colegas, num paper
> de 2023 sobre modelos clínicos de predição**, mostraram que adicionar
> uma flag de missingness em dados outcome-dependent é literalmente
> 'harmful' — é re-introduzir o leak com outro nome.
>
> Na prática, qualquer imputação — mediana, KNN, MICE, missForest —
> criaria um valor característico que Random Forest e XGBoost
> reconheceriam como proxy do diagnóstico. Então a única saída honesta
> é dropar. **Preservamos as 2298 linhas, incluindo os 52 melanomas**,
> e o sinal biológico real que a gente perde — fotótipo e diâmetro —
> vai ser recuperado via PDI da imagem no próximo notebook.
>
> Depois disso, tudo é padrão: criamos uma feature derivada
> `symptom_count` pra condensar os critérios ABCDE, fazemos o
> **split por paciente** com `StratifiedGroupKFold` — zero overlap
> entre treino, validação e teste —, e empacotamos todo o resto num
> `ColumnTransformer` ajustado **exclusivamente no treino**:
> `OneHotEncoder` pra região, `StandardScaler` pra idade, e passthrough
> nos sintomas. **Resultado: 24 features finais**, salvas em Parquet
> e com o pipeline serializado em `joblib` pra reuso.
>
> Zero imputação, zero leak."

[transição para o próximo bloco: "Com os metadados resolvidos, o
próximo passo foi processar as imagens..."]

---

## Checklist mental — Parte 2

- [ ] **Falar devagar no Card 1** — a justificativa da literatura é o que defende a decisão
- [ ] Não entrar em detalhes técnicos do `StratifiedGroupKFold` — só mencionar "zero overlap"
- [ ] **Não passar de 90 segundos** — é o slide mais "árido" da apresentação, público perde atenção rápido
- [ ] Se sentir que está indo longo, cortar a parte dos autores (Ma & Zhang / Sisk) e só dizer "a literatura mostra que imputar seria leak mascarado"
