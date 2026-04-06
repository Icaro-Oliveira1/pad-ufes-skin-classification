# PAD-UFES — Classificação de Lesões Cutâneas

Projeto de classificação de 6 tipos de lesões cutâneas usando o dataset PAD-UFES-20,
comparando ML clássico (features tabulares + PDI) com CNN end-to-end (EfficientNetB0).

## Estrutura

```
projeto_pad_ufes/
├── data/
│   └── raw/
│       ├── metadata.csv        # Metadados clínicos (2298 amostras)
│       └── images/             # Imagens PNG — baixar separadamente (3.4 GB)
├── notebooks/
│   └── projeto.ipynb           # Notebook principal com todo o pipeline
├── outputs/                    # Gráficos gerados pelo run_cnn.py
├── scripts/                    # Scripts auxiliares usados na construção do notebook
├── run_cnn.py                  # Script para rodar o pipeline completo
└── requirements.txt
```

## Configuração

### 1. Obter o dataset

Baixe o dataset PAD-UFES-20 e coloque os arquivos em:
- `data/raw/metadata.csv`
- `data/raw/images/*.png` (2298 imagens)

O dataset está disponível publicamente no Mendeley Data:
> Pacheco, Andre G. C. et al. (2020). PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones. Mendeley Data.

### 2. Criar e ativar o ambiente virtual

```bash
python3 -m venv .venv
source .venv/bin/activate      # Linux/Mac
# ou
.venv\Scripts\activate         # Windows
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

> **GPU (recomendado):** se tiver CUDA disponível, instale o PyTorch com suporte a GPU
> seguindo as instruções em [pytorch.org](https://pytorch.org/get-started/locally/).
> O script detecta automaticamente se há GPU disponível.

### 4. Rodar o pipeline completo

```bash
python run_cnn.py 2>&1 | tee outputs/run_log.txt
```

Tempo estimado: ~17-20 min com GPU. Sem GPU pode levar mais de 1h.

O script executa na ordem:
1. Carregamento e pré-processamento dos dados
2. Extração de features PDI
3. Treinamento RF + DT
4. Treinamento CNN em dois stages (backbone frozen → fine-tuning)
5. Comparativo final e geração de gráficos em `outputs/`

### 5. Explorar o notebook

```bash
jupyter lab
```

Abra `notebooks/projeto.ipynb` para ver o pipeline completo com EDA,
decisões documentadas, resultados e discussões.
