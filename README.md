# Predicción de Abandono de Clientes (Customer Churn) — Telco

Ejercicio 2: comparación de 5 clasificadores supervisados para predecir el abandono de clientes en una empresa de telecomunicaciones, selección del mejor con ANOVA + Tukey HSD y sistema de predicción de nuevos clientes.

**Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7.043 clientes, 21 variables), cargado desde el [mirror público de IBM](https://github.com/IBM/telco-customer-churn-on-icp4d).

## Contenido

| Archivo | Descripción |
|---|---|
| `Churn_Telco_Prediccion.ipynb` | Cuaderno completo (Colab): negocio, EDA, preprocesamiento, SMOTE, 5 modelos, métricas, ANOVA, selección, sistema de predicción e interpretación. |
| `app.py` | App web en Streamlit para predecir el churn de un cliente nuevo. |
| `modelo_churn.pkl` | Pipeline final (OneHotEncoder + StandardScaler + Logistic Regression) generado por el cuaderno. |
| `metadata_churn.json` | Variables, opciones, valores por defecto y métricas del modelo. |

## Resultados (conjunto de prueba, 1.409 clientes)

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.7374 | 0.5034 | **0.7968** | **0.6170** | **0.8399** |
| KNN | 0.6962 | 0.4534 | 0.7032 | 0.5514 | 0.7685 |
| SVM | **0.7559** | **0.5295** | 0.7193 | 0.6100 | 0.8192 |
| Decision Tree | 0.7197 | 0.4754 | 0.5428 | 0.5069 | 0.6630 |
| Naive Bayes | 0.6962 | 0.4596 | 0.8209 | 0.5893 | 0.8066 |

ANOVA sobre 10 folds de validación cruzada: F1-Score F = 74.72, p = 2.76e-19, por lo que se rechaza H₀. Tukey HSD: Logistic Regression es significativamente superior en ROC-AUC a los otros 4 modelos.

## Ejecutar la app localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```
