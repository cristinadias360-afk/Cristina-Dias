# Cristina-Dias
# MVP — Classificação de Risco de Crédito: Modelos Tradicionais vs. Árvores com XAI (SHAP)

## 📌 Sobre o projeto
MVP de pós-graduação que desenvolve um framework preditivo para classificação de inadimplência em crédito, confrontando um modelo tradicional (Regressão Logística) com modelos baseados em árvore (Random Forest, Gradient Boosting, XGBoost e LightGBM), utilizando **SHAP (SHapley Additive exPlanations)** para extrair explicabilidade global e local dos modelos não lineares.

## 🎯 Objetivo
Avaliar se modelos baseados em árvore superam a Regressão Logística em poder preditivo o suficiente para justificar o custo adicional de governança, e se o SHAP consegue entregar um nível de transparência equivalente ao dos coeficientes de um modelo linear.

## 🗂️ Dataset
[Credit Risk Dataset (Kaggle)](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) — dados de clientes de crédito (idade, renda, histórico de crédito, finalidade e valor do empréstimo) com a variável-alvo `loan_status` (inadimplência).

## 🧱 Estrutura do notebook
1. Escopo e definição do problema
2. Carga, qualidade e engenharia de atributos
3. Análise exploratória (EDA)
4. Pré-processamento (pipeline `scikit-learn`)
5. Modelagem: baseline + candidatos (Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM)
6. Ablação: efeito de variáveis contínuas vs. faixas categóricas por família de modelo
7. Otimização de hiperparâmetros (`RandomizedSearchCV`)
8. Avaliação final (ROC-AUC, KS, PR-AUC, matriz de confusão, threshold)
9. Explicabilidade (SHAP): importância global e explicações locais
10. Conclusões e artefatos salvos

## 🏆 Principais resultados

| Modelo | ROC-AUC | KS | F1 (Inadimplente) |
|---|---|---|---|
| Regressão Logística (ajustada) | 0.888 | 0.647 | 0.67 |
| **XGBoost (ajustado)** | **0.950** | **0.765** | **0.83** |

O XGBoost supera a Regressão Logística em todas as métricas, com destaque para a precisão (0.88 vs. 0.59), reduzindo significativamente falsos positivos sem perder recall.

## 🔍 Explicabilidade (SHAP)
As variáveis mais relevantes para o modelo — `person_income`, `loan_int_rate`, `loan_percent_income`/faixas de renda e `loan_grade` — foram validadas via SHAP global (beeswarm) e local (waterfall), e comparadas ao ranking de coeficientes da Regressão Logística.

## 🛠️ Tecnologias utilizadas
- Python, pandas, scikit-learn
- XGBoost, LightGBM
- SHAP
- matplotlib, seaborn

## 📁 Arquivos
- `MVP_ML_&_Analytics_Credito_Risco_Cristina_Silva_Dias.ipynb` — notebook completo
- `modelo_xgboost_final.pkl` — modelo XGBoost treinado
- `modelo_logistic_regression_final.pkl` — modelo de Regressão Logística treinado
- `credit_risk_dataset.csv` — dataset utilizado

## 👩‍💻 Autora
Cristina Silva Dias — Pós-graduação em Machine Learning & Analytics
