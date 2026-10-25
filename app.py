"""App Streamlit: predicción de abandono de clientes (Telco Customer Churn).

Carga el pipeline (preprocesamiento + modelo) y los metadatos generados por
el cuaderno Churn_Telco_Prediccion.ipynb. Incluye exploración del dataset y
comparación de los 5 modelos evaluados.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent
URL_DATASET = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
               "master/data/Telco-Customer-Churn.csv")

# Colores de series (validados para modo oscuro: CVD y contraste)
COLOR_NO = "#3987e5"       # cliente continúa
COLOR_YES = "#d95926"      # cliente abandona
COLOR_SUBE = "#e66767"     # aporte que aumenta el riesgo
COLOR_BAJA = "#3987e5"     # aporte que reduce el riesgo
COLOR_MUTED = "#4a5163"
RIESGO = {"ALTO": "#d03b3b", "MEDIO": "#fab219", "BAJO": "#0ca30c"}
TINTA, TINTA_2, GRID = "#f0efec", "#c3c2b7", "#2c2f3a"

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

# Resultados de la validación cruzada (10 folds) del cuaderno: (media, desviación)
CV = {
    "Logistic Regression": {"Accuracy": (0.7542, 0.0178), "Precision": (0.5247, 0.0233), "Recall": (0.7953, 0.0361),
                            "F1-Score": (0.6320, 0.0250), "ROC-AUC": (0.8460, 0.0151)},
    "SVM": {"Accuracy": (0.7668, 0.0111), "Precision": (0.5467, 0.0176), "Recall": (0.7191, 0.0285),
            "F1-Score": (0.6207, 0.0131), "ROC-AUC": (0.8252, 0.0115)},
    "Naive Bayes": {"Accuracy": (0.7043, 0.0177), "Precision": (0.4688, 0.0177), "Recall": (0.8408, 0.0250),
                    "F1-Score": (0.6017, 0.0154), "ROC-AUC": (0.8201, 0.0149)},
    "KNN": {"Accuracy": (0.6913, 0.0121), "Precision": (0.4501, 0.0136), "Recall": (0.7325, 0.0290),
            "F1-Score": (0.5573, 0.0156), "ROC-AUC": (0.7599, 0.0098)},
    "Decision Tree": {"Accuracy": (0.7270, 0.0159), "Precision": (0.4885, 0.0283), "Recall": (0.5365, 0.0187),
                      "F1-Score": (0.5108, 0.0169), "ROC-AUC": (0.6666, 0.0119)},
}

st.set_page_config(page_title="Predicción de Churn - Telco", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1350px;}
.hero h1 {font-size: 2.6rem; font-weight: 800; margin: 0; padding: 0;
          background: linear-gradient(90deg, #00c2ff 0%, #7c5cff 50%, #ff4b8b 100%);
          -webkit-background-clip: text; background-clip: text; color: transparent;}
.hero p {color: #c3c2b7; font-size: 1.05rem; margin: .35rem 0 0 0;}
.kpi {background: #161a24; border: 1px solid rgba(255,255,255,.08); border-radius: 14px;
      padding: 1rem 1.2rem; border-top: 3px solid var(--acento); height: 100%;}
.kpi .lbl {color: #9a9fae; font-size: .8rem; text-transform: uppercase; letter-spacing: .04em;}
.kpi .val {color: #f0efec; font-size: 1.85rem; font-weight: 700; line-height: 1.3;}
.kpi .sub {color: #9a9fae; font-size: .82rem;}
.resultado {background: #161a24; border-radius: 16px; padding: 1.1rem 1.4rem;
            border: 1px solid rgba(255,255,255,.08); border-left: 6px solid var(--acento);}
.resultado h3 {margin: 0 0 .4rem 0; color: #f0efec; font-size: 1.3rem;}
.chip {display: inline-block; padding: .15rem .7rem; border-radius: 999px; font-weight: 700;
       font-size: .85rem; color: #0b0b0b; background: var(--acento);}
.similares {color: #c3c2b7; font-size: .95rem; margin-top: .7rem;}
div[data-testid="stTabs"] button p {font-size: 1.02rem; font-weight: 600;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ datos
@st.cache_resource
def cargar_modelo():
    modelo = joblib.load(BASE / "modelo_churn.pkl")
    meta = json.loads((BASE / "metadata_churn.json").read_text(encoding="utf-8"))
    return modelo, meta


@st.cache_data(show_spinner="Cargando dataset...")
def cargar_dataset():
    df = pd.read_csv(URL_DATASET)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["churn"] = (df["Churn"] == "Yes").astype(int)
    df["Estado"] = df["Churn"].map({"No": "Continúa", "Yes": "Abandona"})
    return df


# ------------------------------------------------------------------ utilidades
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
        return "ALTO"
    if prob >= 0.40:
        return "MEDIO"
    return "BAJO"


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


def estilo(fig, alto=380, leyenda=False):
    fig.update_layout(
        template="plotly_dark", height=alto, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TINTA_2, size=13), margin=dict(l=10, r=10, t=30, b=10), barcornerradius=4,
        showlegend=leyenda, hoverlabel=dict(bgcolor="#1f2430", font_color=TINTA, bordercolor="#3a3f4c"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="#383835")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="#383835")
    return fig


def grafico(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def kpi(col, etiqueta, valor, sub, acento):
    col.markdown(f"<div class='kpi' style='--acento:{acento}'><div class='lbl'>{etiqueta}</div>"
                 f"<div class='val'>{valor}</div><div class='sub'>{sub}</div></div>", unsafe_allow_html=True)


def miles(n):
    return f"{n:,.0f}".replace(",", ".")


def tasa_por(df, col):
    t = df.groupby(col).agg(clientes=("churn", "size"), tasa=("churn", "mean")).reset_index()
    return t.sort_values("tasa", ascending=False)


def barras_tasa(t, col, tasa_global, alto=360):
    fig = go.Figure(go.Bar(
        x=t[col].astype(str), y=t["tasa"] * 100, marker_color=COLOR_YES,
        text=[f"{v:.1f}%" for v in t["tasa"] * 100], textposition="outside", textfont=dict(color=TINTA),
        customdata=t["clientes"], cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>Tasa de churn: %{y:.1f}%<br>Clientes: %{customdata:,}<extra></extra>",
    ))
    fig.add_hline(y=tasa_global, line_dash="dash", line_color="#9a9fae", line_width=1,
                  annotation_text=f"Promedio {tasa_global:.1f}%", annotation_position="top right",
                  annotation_font_color=TINTA_2)
    fig.update_yaxes(title="% de clientes que abandonan", range=[0, max(60, t["tasa"].max() * 115)])
    fig.update_xaxes(title=None)
    return estilo(fig, alto)


modelo, meta = cargar_modelo()
opciones, defaults = meta["opciones"], meta["defaults"]
NUM_COLS, CAT_COLS = meta["num_cols"], meta["cat_cols"]

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.subheader("Modelo en producción")
    st.write(f"**{meta['modelo']}**")
    st.caption("One-Hot + StandardScaler, entrenado con SMOTE. Métricas en el conjunto de prueba (1.409 clientes):")
    m = meta["metricas_test"]
    c1, c2 = st.columns(2)
    c1.metric("Recall", f"{m['Recall']:.1%}")
    c2.metric("ROC-AUC", f"{m['ROC-AUC']:.3f}")
    c1.metric("F1-Score", f"{m['F1-Score']:.3f}")
    c2.metric("Precision", f"{m['Precision']:.1%}")
    st.metric("Accuracy", f"{m['Accuracy']:.1%}")
    st.divider()
    umbral = st.slider("Umbral de decisión", 0.10, 0.90, float(meta["umbral"]), 0.05,
                       help="Probabilidad a partir de la cual el cliente se clasifica como 'abandona'. "
                            "Subirlo aumenta la Precision; bajarlo aumenta el Recall.")
    st.caption("Dataset: Telco Customer Churn (IBM / Kaggle), 7.043 clientes.")

# ------------------------------------------------------------------ encabezado
st.markdown("<div class='hero'><h1>Predicción de abandono de clientes</h1>"
            "<p>Telecomunicaciones | Customer Churn | Modelo de Regresión Logística</p></div>",
            unsafe_allow_html=True)
st.write("")

tab_pred, tab_datos, tab_modelos = st.tabs(["Predicción", "Explorar datos", "Comparar modelos"])

# ================================================================== PREDICCION
with tab_pred:
    with st.form("cliente"):
        st.markdown("##### Variables principales")
        a, b = st.columns(2)
        tenure = a.number_input("Tenure (meses de permanencia)", 0, 72, 5)
        monthly = b.number_input("MonthlyCharges (cargo mensual, $)", 18.0, 120.0, 95.0, step=0.5)
        a, b, c = st.columns(3)
        iniciales = {"Contract": "Month-to-month", "TechSupport": "No", "InternetService": "Fiber optic"}
        valores = {}
        for col, nombre in zip((a, b, c), PRINCIPALES):
            ops = opciones[nombre]
            valores[nombre] = col.selectbox(f"{nombre} ({ETIQUETAS[nombre]})", ops,
                                            index=ops.index(iniciales[nombre]))

        with st.expander("Otras variables (opcional: si no se cambian se usa el valor más frecuente)"):
            otras = [x for x in CAT_COLS if x not in PRINCIPALES]
            cols = st.columns(3)
            for i, nombre in enumerate(otras):
                ops = opciones[nombre]
                valores[nombre] = cols[i % 3].selectbox(f"{nombre} ({ETIQUETAS[nombre]})", ops,
                                                        index=ops.index(defaults[nombre]))
            total = st.number_input("TotalCharges (0 = estimar como tenure x MonthlyCharges)", 0.0, 9000.0, 0.0)

        enviado = st.form_submit_button("Predecir", type="primary", width="stretch")

    if enviado:
        cliente = dict(defaults)
        cliente.update(valores)
        cliente.update({"tenure": tenure, "MonthlyCharges": monthly})
        cliente = ajustar_consistencia(cliente)
        cliente["TotalCharges"] = total if total > 0 else tenure * monthly

        X_nuevo = pd.DataFrame([cliente])[NUM_COLS + CAT_COLS]
        prob = float(modelo.predict_proba(X_nuevo)[0, 1])
        clase = int(prob >= umbral)
        nivel = nivel_riesgo(prob)
        acento = RIESGO[nivel]

        st.write("")
        izq, der = st.columns([1, 1.15], gap="large")
        with izq:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100,
                number=dict(suffix="%", valueformat=".0f", font=dict(size=56, color=TINTA)),
                title=dict(text="Probabilidad de abandono", font=dict(size=16, color=TINTA_2)),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#6b7080", ticksuffix="%"),
                    bar=dict(color=acento, thickness=0.32), bgcolor="#161a24", borderwidth=0,
                    steps=[dict(range=[0, 40], color="rgba(12,163,12,.16)"),
                           dict(range=[40, 70], color="rgba(250,178,25,.16)"),
                           dict(range=[70, 100], color="rgba(208,59,59,.18)")],
                    threshold=dict(line=dict(color=TINTA, width=3), thickness=0.85, value=umbral * 100),
                ),
            ))
            grafico(estilo(gauge, alto=290))

            texto = ("Cliente con ALTA probabilidad de abandono" if clase
                     else "Cliente con BAJA probabilidad de abandono")
            st.markdown(
                f"<div class='resultado' style='--acento:{acento}'>"
                f"<h3>Predicción: {texto}</h3>"
                f"<div>Probabilidad estimada: <b style='font-size:1.3rem;color:{TINTA}'>{prob:.0%}</b> "
                f"&nbsp; <span class='chip'>Riesgo {nivel}</span></div>"
                f"<div class='similares'>Clase predicha: <b>Churn = {'Yes' if clase else 'No'}</b> "
                f"(umbral {umbral:.0%}; la línea blanca del indicador marca el umbral)</div></div>",
                unsafe_allow_html=True)

            df = cargar_dataset()
            similares = df[(df["Contract"] == cliente["Contract"])
                           & (df["InternetService"] == cliente["InternetService"])
                           & (df["tenure"].sub(tenure).abs() <= 6)]
            if len(similares) >= 10:
                st.markdown(
                    f"<div class='similares'>En el dataset hay <b>{miles(len(similares))}</b> clientes con el mismo "
                    f"contrato, el mismo servicio de internet y una permanencia parecida (hasta 6 meses de diferencia): el "
                    f"<b>{similares['churn'].mean():.0%}</b> de ellos abandonó.</div>", unsafe_allow_html=True)

        with der:
            aporte = contribuciones(modelo, X_nuevo)
            if aporte is not None:
                st.markdown("##### ¿Qué explica esta predicción?")
                st.caption("Aporte de cada variable al riesgo frente a un cliente promedio. "
                           "Rojo aumenta el riesgo; azul lo reduce.")
                top = pd.concat([aporte.head(5), aporte.tail(5)])
                top = top[~top.index.duplicated()].sort_values()
                fig = go.Figure(go.Bar(
                    x=top.values, y=top.index, orientation="h",
                    marker_color=[COLOR_SUBE if v > 0 else COLOR_BAJA for v in top.values],
                    hovertemplate="<b>%{y}</b><br>Aporte: %{x:+.3f}<extra></extra>",
                ))
                fig.update_xaxes(title="Aporte al log-odds")
                grafico(estilo(fig, alto=340))

            acciones = recomendaciones(cliente)
            if clase and acciones:
                st.markdown("##### Acciones de retención sugeridas")
                for x in acciones:
                    st.markdown(f"- {x}")
            elif not clase:
                st.success("Cliente estable: mantener el seguimiento habitual.")

        with st.expander("Ver datos completos enviados al modelo"):
            st.dataframe(X_nuevo.T.astype(str).rename(columns={0: "Valor"}), width="stretch")

# ================================================================== EXPLORAR DATOS
with tab_datos:
    df = cargar_dataset()
    tasa_global = df["churn"].mean() * 100
    perdida = df.loc[df["churn"] == 1, "MonthlyCharges"].sum()

    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Clientes", miles(len(df)), "registros en el dataset", "#00c2ff")
    kpi(k2, "Tasa de abandono", f"{tasa_global:.1f}%", f"{miles(df['churn'].sum())} clientes se fueron", COLOR_YES)
    kpi(k3, "Facturación mensual perdida", f"${miles(perdida)}",
        f"{perdida / df['MonthlyCharges'].sum():.1%} de la facturación total", "#ff4b8b")
    kpi(k4, "Pérdida anual estimada", f"${perdida * 12 / 1e6:.2f} M", "si no se retiene a ninguno", "#7c5cff")
    st.write("")

    c1, c2 = st.columns([1.35, 1], gap="large")
    with c1:
        st.markdown("##### Tasa de abandono por variable")
        variable = st.selectbox("Variable", ["Contract", "InternetService", "PaymentMethod", "TechSupport",
                                             "OnlineSecurity", "SeniorCitizen", "PaperlessBilling",
                                             "Partner", "Dependents", "gender"],
                                format_func=lambda v: f"{v} ({ETIQUETAS[v]})", label_visibility="collapsed")
        t = tasa_por(df, variable)
        grafico(barras_tasa(t, variable, tasa_global))
        with st.expander("Ver tabla"):
            st.dataframe(t.assign(tasa=lambda d: (d["tasa"] * 100).round(1))
                         .rename(columns={variable: ETIQUETAS[variable], "clientes": "Clientes",
                                          "tasa": "% churn"}), hide_index=True, width="stretch")
    with c2:
        st.markdown("##### Tasa de abandono por permanencia")
        st.caption("El primer año concentra el mayor riesgo.")
        tramos = pd.cut(df["tenure"], bins=[-1, 12, 24, 36, 48, 60, 72],
                        labels=["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"])
        t2 = (df.assign(Meses=tramos).groupby("Meses", observed=True)
              .agg(clientes=("churn", "size"), tasa=("churn", "mean")).reset_index())
        fig = barras_tasa(t2, "Meses", tasa_global, alto=375)
        fig.update_xaxes(title="Meses como cliente", categoryorder="array", categoryarray=t2["Meses"].astype(str))
        grafico(fig)

    st.markdown("##### Clientes que abandonan frente a clientes que se quedan")
    c3, c4 = st.columns(2, gap="large")
    for col_ui, var, titulo, inicio, paso in [(c3, "MonthlyCharges", "Cargo mensual ($)", 10, 10),
                                              (c4, "tenure", "Meses de permanencia", 0, 6)]:
        with col_ui:
            fig = go.Figure()
            for estado, color in [("Continúa", COLOR_NO), ("Abandona", COLOR_YES)]:
                datos = df.loc[df["Estado"] == estado, var]
                fig.add_trace(go.Histogram(
                    x=datos, name=f"{estado} (mediana {datos.median():.0f})", marker_color=color,
                    histnorm="percent", xbins=dict(start=inicio, size=paso),
                    hovertemplate=f"<b>{estado}</b><br>{titulo}: %{{x}}<br>%{{y:.1f}}% de su grupo<extra></extra>",
                ))
            fig.update_layout(barmode="group", bargap=0.18, bargroupgap=0.08)
            fig.update_xaxes(title=titulo)
            fig.update_yaxes(title="% de clientes del grupo")
            grafico(estilo(fig, alto=340, leyenda=True))

# ================================================================== COMPARAR MODELOS
with tab_modelos:
    st.markdown("##### Desempeño de los 5 modelos (validación cruzada estratificada, 10 folds)")
    metrica = st.radio("Métrica", ["F1-Score", "ROC-AUC", "Recall", "Precision", "Accuracy"],
                       horizontal=True, label_visibility="collapsed")
    orden = sorted(CV, key=lambda n: -CV[n][metrica][0])
    medias = [CV[n][metrica][0] for n in orden]
    desv = [CV[n][metrica][1] for n in orden]
    fig = go.Figure(go.Bar(
        x=orden, y=medias, error_y=dict(type="data", array=desv, color="#9a9fae", thickness=1.2, width=6),
        marker_color=["#00c2ff" if n == meta["modelo"] else COLOR_MUTED for n in orden],
        text=[f"{v:.3f}" for v in medias], textposition="inside", insidetextanchor="start",
        textfont=dict(color=["#0b0b0b" if n == meta["modelo"] else TINTA for n in orden], size=15),
        hovertemplate="<b>%{x}</b><br>" + metrica + ": %{y:.4f}<extra></extra>",
    ))
    fig.update_yaxes(title=f"{metrica} (media de 10 folds)", range=[0, 1])
    grafico(estilo(fig, alto=380))
    st.caption("Barra resaltada: modelo seleccionado. Las líneas indican la desviación estándar entre los 10 folds.")

    a, b, c = st.columns(3)
    kpi(a, "ANOVA (F1-Score)", "F = 72,20", "p-value = 5,37e-19", "#00c2ff")
    kpi(b, "Decisión", "Se rechaza H₀", "sí hay diferencias significativas entre modelos", "#ff4b8b")
    kpi(c, "Tukey HSD (ROC-AUC)", "LR es el mejor", "significativamente superior a los otros 4", "#7c5cff")
    st.write("")
    st.markdown(
        "**¿Por qué Regresión Logística?** Tiene el mejor F1 y el mejor ROC-AUC, casi no sobreajusta "
        "(brecha train-test < 0,02) y es la más interpretable: cada coeficiente se traduce en un *odds ratio* "
        "que el área de marketing puede entender. SVM empata en F1, pero es significativamente peor en ROC-AUC "
        "y en Recall. Naive Bayes detecta más clientes que abandonan, a costa de muchas más falsas alarmas.")
