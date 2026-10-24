"""App Streamlit: predicción de abandono de clientes (Telco Customer Churn).

Carga el pipeline (preprocesamiento + modelo) y los metadatos generados por
el cuaderno Churn_Telco_Prediccion.ipynb.
"""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
SERVICIOS_INTERNET = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                      "TechSupport", "StreamingTV", "StreamingMovies"]
PRINCIPALES = ["Contract", "TechSupport", "InternetService"]
ETIQUETAS = {
    "gender": "Género", "SeniorCitizen": "Adulto mayor", "Partner": "Tiene pareja",
    "Dependents": "Tiene dependientes", "PhoneService": "Servicio telefónico",
    "MultipleLines": "Múltiples líneas", "InternetService": "Servicio de internet",
    "OnlineSecurity": "Seguridad en línea", "OnlineBackup": "Respaldo en línea",
    "DeviceProtection": "Protección de equipo", "TechSupport": "Soporte técnico",
    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming películas",
    "Contract": "Tipo de contrato", "PaperlessBilling": "Factura electrónica",
    "PaymentMethod": "Método de pago",
}

st.set_page_config(page_title="Predicción de Churn · Telco", page_icon="📉", layout="wide")


@st.cache_resource
def cargar():
    modelo = joblib.load(BASE / "modelo_churn.pkl")
    meta = json.loads((BASE / "metadata_churn.json").read_text(encoding="utf-8"))
    return modelo, meta


def ajustar_consistencia(cliente):
    # Evita perfiles imposibles (p. ej. sin internet pero con soporte técnico)
    for s in SERVICIOS_INTERNET:
        if cliente["InternetService"] == "No":
            cliente[s] = "No internet service"
        elif cliente[s] == "No internet service":
            cliente[s] = "No"
    if cliente["PhoneService"] == "No":
        cliente["MultipleLines"] = "No phone service"
    elif cliente["MultipleLines"] == "No phone service":
        cliente["MultipleLines"] = "No"
    return cliente


def nivel_riesgo(prob):
    if prob >= 0.70:
        return "ALTO", "#C0392B"
    if prob >= 0.40:
        return "MEDIO", "#D68910"
    return "BAJO", "#1E8449"


def recomendaciones(c):
    acciones = []
    if c["Contract"] == "Month-to-month":
        acciones.append("Ofrecer migración a contrato de 1 o 2 años con un incentivo (descuento o beneficio).")
    if c["tenure"] <= 12:
        acciones.append("Incluir en el programa de bienvenida: los primeros 12 meses concentran el mayor abandono.")
    if c["InternetService"] != "No" and c["TechSupport"] == "No":
        acciones.append("Ofrecer soporte técnico gratuito por algunos meses.")
    if c["InternetService"] != "No" and c["OnlineSecurity"] == "No":
        acciones.append("Ofrecer el servicio de seguridad en línea como beneficio de retención.")
    if c["PaymentMethod"] == "Electronic check":
        acciones.append("Incentivar el pago automático (tarjeta o transferencia) con un descuento.")
    if c["InternetService"] == "Fiber optic" and c["MonthlyCharges"] >= 80:
        acciones.append("Revisar el plan de fibra: cargo mensual alto, posible sensibilidad al precio.")
    return acciones


def contribuciones(modelo, X_nuevo):
    # Solo para modelos lineales: aporte de cada variable al log-odds
    clf = modelo.named_steps["modelo"]
    if not hasattr(clf, "coef_"):
        return None
    prep = modelo.named_steps["prep"]
    valores = prep.transform(X_nuevo)[0]
    nombres = [n.replace("num__", "").replace("cat__", "") for n in prep.get_feature_names_out()]
    aporte = pd.Series(clf.coef_[0] * valores, index=nombres)
    return aporte[aporte != 0].sort_values()


modelo, meta = cargar()
opciones, defaults = meta["opciones"], meta["defaults"]
NUM_COLS, CAT_COLS = meta["num_cols"], meta["cat_cols"]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Modelo")
    st.write(f"**{meta['modelo']}** (pipeline: One-Hot + StandardScaler + SMOTE en entrenamiento)")
    st.caption("Métricas en el conjunto de prueba (1.409 clientes):")
    m = meta["metricas_test"]
    c1, c2 = st.columns(2)
    c1.metric("Recall", f"{m['Recall']:.1%}")
    c2.metric("ROC-AUC", f"{m['ROC-AUC']:.3f}")
    c1.metric("F1-Score", f"{m['F1-Score']:.3f}")
    c2.metric("Precision", f"{m['Precision']:.1%}")
    st.metric("Accuracy", f"{m['Accuracy']:.1%}")
    umbral = st.slider("Umbral de decisión", 0.10, 0.90, float(meta["umbral"]), 0.05,
                       help="Probabilidad a partir de la cual el cliente se clasifica como 'abandona'. "
                            "Subirlo aumenta la Precision; bajarlo aumenta el Recall.")
    st.caption("Dataset: Telco Customer Churn (IBM / Kaggle).")

# ---------------------------------------------------------------- formulario
st.title("📉 Predicción de abandono de clientes")
st.write("Ingrese los datos del cliente para estimar la **probabilidad de abandono** y la **clase predicha**.")

with st.form("cliente"):
    st.subheader("Variables principales")
    a, b = st.columns(2)
    tenure = a.number_input("Tenure (meses de permanencia)", 0, 72, 5)
    monthly = b.number_input("MonthlyCharges (cargo mensual, $)", 18.0, 120.0, 95.0, step=0.5)
    a, b, c = st.columns(3)
    iniciales = {"Contract": "Month-to-month", "TechSupport": "No", "InternetService": "Fiber optic"}
    valores = {}
    for col, nombre in zip((a, b, c), PRINCIPALES):
        ops = opciones[nombre]
        valores[nombre] = col.selectbox(f"{nombre} · {ETIQUETAS[nombre]}", ops, index=ops.index(iniciales[nombre]))

    with st.expander("Otras variables (opcional: si no se cambian se usa el valor más frecuente)"):
        otras = [x for x in CAT_COLS if x not in PRINCIPALES]
        cols = st.columns(3)
        for i, nombre in enumerate(otras):
            ops = opciones[nombre]
            valores[nombre] = cols[i % 3].selectbox(f"{nombre} · {ETIQUETAS[nombre]}", ops,
                                                    index=ops.index(defaults[nombre]))
        total = st.number_input("TotalCharges (0 = estimar como tenure × MonthlyCharges)", 0.0, 9000.0, 0.0)

    enviado = st.form_submit_button("Predecir", type="primary", width="stretch")

# ---------------------------------------------------------------- resultado
if enviado:
    cliente = dict(defaults)
    cliente.update(valores)
    cliente.update({"tenure": tenure, "MonthlyCharges": monthly})
    cliente = ajustar_consistencia(cliente)
    cliente["TotalCharges"] = total if total > 0 else tenure * monthly

    X_nuevo = pd.DataFrame([cliente])[NUM_COLS + CAT_COLS]
    prob = float(modelo.predict_proba(X_nuevo)[0, 1])
    clase = int(prob >= umbral)
    nivel, color = nivel_riesgo(prob)

    st.divider()
    izq, der = st.columns([1, 1])
    with izq:
        texto = ("Cliente con ALTA probabilidad de abandono" if clase
                 else "Cliente con BAJA probabilidad de abandono")
        st.markdown(f"<h3 style='color:{color};margin-bottom:0'>Predicción: {texto}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:1.6rem;margin:0.3rem 0'>Probabilidad estimada: <b>{prob:.0%}</b></p>",
                    unsafe_allow_html=True)
        st.progress(prob)
        k1, k2, k3 = st.columns(3)
        k1.metric("Clase predicha", "Churn = Yes" if clase else "Churn = No")
        k2.metric("Nivel de riesgo", nivel)
        k3.metric("Umbral", f"{umbral:.0%}")

        acciones = recomendaciones(cliente)
        if clase and acciones:
            st.subheader("Acciones de retención sugeridas")
            for x in acciones:
                st.markdown(f"- {x}")
        elif not clase:
            st.success("Cliente estable: mantener el seguimiento habitual.")

    with der:
        aporte = contribuciones(modelo, X_nuevo)
        if aporte is not None:
            st.subheader("¿Qué explica esta predicción?")
            st.caption("Aporte de cada variable al riesgo (log-odds). Rojo aumenta el riesgo; verde lo reduce.")
            top = pd.concat([aporte.head(5), aporte.tail(5)]).drop_duplicates()
            tabla = pd.DataFrame({"Variable": top.index, "Aporte": np.round(top.values, 3)})
            grafico = alt.Chart(tabla).mark_bar().encode(
                x=alt.X("Aporte:Q", title="Aporte al log-odds"),
                y=alt.Y("Variable:N", sort="-x", title=None),
                color=alt.condition("datum.Aporte > 0", alt.value("#C0392B"), alt.value("#1E8449")),
                tooltip=["Variable", "Aporte"],
            ).properties(height=320)
            st.altair_chart(grafico, width="stretch")

    with st.expander("Ver datos completos enviados al modelo"):
        st.dataframe(X_nuevo.T.astype(str).rename(columns={0: "Valor"}), width="stretch")
