# Predicción de Abandono de Clientes (Customer Churn) - Telco

Ejercicio 2: comparación de 5 clasificadores supervisados para predecir el abandono de clientes en una empresa de telecomunicaciones, validación estadística (ANOVA exploratorio, prueba de Friedman y t-test corregido de Nadeau y Bengio con Holm), calibración de probabilidades y sistema de predicción de nuevos clientes.

**Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7.043 clientes, 21 variables), cargado desde el [mirror público de IBM](https://github.com/IBM/telco-customer-churn-on-icp4d).

## Contenido

| Archivo | Descripción |
|---|---|
| `JAMN_Churn_Telco_Prediccion_Corregido.ipynb` | Cuaderno completo (Colab): negocio, EDA, preprocesamiento, SMOTENC, 5 modelos, métricas, pruebas estadísticas, selección, calibración, sistema de predicción e interpretación. |
| `app.py` | App web en Streamlit para predecir el churn de un cliente nuevo. |
| `modelo_churn.pkl` | Modelo final: Regresión Logística (StandardScaler + SMOTENC + One-Hot) con probabilidades calibradas (`CalibratedClassifierCV`, sigmoid). |
| `metadata_churn.json` | Variables, opciones, valores por defecto, umbral y métricas del modelo. |

Abrir en Colab: https://colab.research.google.com/github/jamn95/churn-telco/blob/main/JAMN_Churn_Telco_Prediccion_Corregido.ipynb

## Resultados (conjunto de prueba, 1.409 clientes, clase Churn = Yes)

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.7381 | 0.5044 | **0.7647** | **0.6079** | **0.8327** |
| KNN | 0.7175 | 0.4786 | 0.7166 | 0.5739 | 0.7757 |
| SVM | **0.7537** | **0.5265** | 0.7166 | 0.6070 | 0.8185 |
| Decision Tree | 0.7168 | 0.4733 | 0.5936 | 0.5267 | 0.6783 |
| Naive Bayes | 0.7381 | 0.5045 | 0.7540 | 0.6045 | 0.8198 |

Validación cruzada de 10 folds: la prueba de Friedman rechaza H₀ en F1 (p = 1.36e-06) y en ROC-AUC (p = 1.08e-07). Con el t-test corregido y Holm, la Regresión Logística es significativamente mejor que los otros 4 modelos en ROC-AUC; en F1 no se diferencia significativamente de SVM ni de Naive Bayes.

Salidas generadas con el entorno de Colab (Python 3.13, scikit-learn 1.6.1). Al re-ejecutar en otro entorno, algunas métricas pueden variar en el tercer decimal sin que cambien las conclusiones. El modelo de la app se generó con el mismo cuaderno usando scikit-learn 1.9.1.

## Ejecutar la app localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```
