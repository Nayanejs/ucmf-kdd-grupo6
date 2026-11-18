# UCMF — Predição da condição cardíaca de crianças e adolescentes

Mineração de Dados (INF/UFG) — Grupo 6: João Guilherme da Silva Chaveiro, Luiany Gonçalves Carvalho,
Nayane Januário Santos e Victor Pereira Sousa Pires.

Classificação binária (Normal × Anormal) no conjunto UCMF (Real Hospital Português, Recife), seguindo o processo KDD.
O atributo `SOPRO` é excluído dos modelos por vazamento de dados (sozinho, acerta 92,2% do teste).

## Como reproduzir

```bash
pip install -r requirements.txt
python ucmf_pipeline.py data/UCMF_raw.csv results   # limpeza, CV, modelos, métricas (~7 min)
python make_figures.py results paper                # figuras do artigo
python fill_numbers.py                              # tabelas do artigo (paper/numbers.tex)
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Ou abra o notebook no Google Colab (já salvo com as saídas): https://colab.research.google.com/github/Nayanejs/ucmf-kdd-grupo6/blob/main/ucmf_kdd_grupo6.ipynb

`RANDOM_STATE = 42` em todas as etapas aleatórias.

## Estrutura

| Arquivo | Conteúdo |
|---|---|
| `ucmf_pipeline.py` | Limpeza determinística, pipeline scikit-learn (ajustado só no treino), GridSearchCV, avaliação, vazamento, erros e importância |
| `make_figures.py` | Figuras do relatório |
| `fill_numbers.py` | Gera as tabelas LaTeX a partir de `results/results.json` |
| `ucmf_kdd_grupo6.ipynb` | Notebook com a execução completa |
| `results/` | `results.json` e probabilidades no teste |
| `paper/` | Artigo no template IEEE (IEEEtran) |
| `data/UCMF_raw.csv` | Dados brutos (uso acadêmico autorizado pelos comitês de ética do RHP e da U. Porto) |

## Resultados (teste, n = 2.279)

| Modelo | AUC-ROC | Acur. balanceada | Sensibilidade | Especificidade |
|---|---|---|---|---|
| Classe majoritária | 0,500 | 50,0% | 0,0% | 100,0% |
| Regressão Logística | 0,798 | 74,0% | 66,0% | 81,9% |
| Random Forest | 0,809 | 73,2% | 58,5% | 87,9% |
| **XGBoost** | **0,811** | **74,5%** | **66,7%** | 82,2% |
| XGBoost + SOPRO (vazamento) | 0,956 | 92,0% | 87,3% | 96,8% |
