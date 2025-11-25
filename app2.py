import streamlit as st
st.set_page_config(page_title="Breast Cancer App", page_icon="🩺", layout="wide")

import numpy as np
import pandas as pd
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# SHAP fix para Streamlit
shap.initjs()

# ============================
# CSS PERSONALIZADO
# ============================
st.markdown("""
<style>
:root {
    --morado: #8b5cf6;
    --morado-claro: #c4b5fd;
    --celeste: #38bdf8;
    --celeste-claro: #a5e8ff;
}

/* Fondo gris oscuro */
body, .stApp { background: #1e1e1e !important; }

/* Títulos */
h1, h2, h3 {
    color: var(--morado) !important;
    font-weight: 800 !important;
}

/* Contenedores */
.block-container { padding-top: 2rem; }

/* Tarjetas */
.card {
    background: #2b2b2b;
    border-radius: 12px;
    padding: 20px;
    border: 2px solid var(--morado-claro);
    box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.4);
    margin-bottom: 20px;
    color: white;
}

/* Botón */
.stButton>button {
    background: linear-gradient(135deg, var(--morado), var(--celeste));
    color: white;
    font-weight: bold;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
}
.stButton>button:hover {
    background: linear-gradient(135deg, var(--celeste), var(--morado));
    transform: scale(1.02);
}

/* Inputs */
input, textarea, .stNumberInput>div>div>input {
    border-radius: 8px !important;
    border: 2px solid var(--morado-claro);
    background-color: #333333 !important;
    color: white !important;
}

/* Tabs */
.stTabs [role="tablist"] { justify-content: center; }
.stTabs [role="tab"] {
    font-size: 18px;
    font-weight: bold;
    color: var(--morado);
}
.stTabs [role="tab"][aria-selected="true"] {
    border-bottom: 3px solid var(--morado);
    color: var(--celeste);
}

/* Texto tarjetas */
.card * { color: white !important; }

/* SHAP fix */
.js-plotly-plot .plotly { zoom: 0.9; }
</style>
""", unsafe_allow_html=True)

# ============================
# TÍTULO
# ============================
st.title("Aplicación de Diagnóstico de Cáncer de Mama")

# ============================
# CARGAR DATASET
# ============================
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
target = data.target

# ============================
# ENTRENAR MODELOS
# ============================

features = data.feature_names
df_selected = df[features]

labels = []
X_blocks = []

for _, row in df_selected.iterrows():
    X_blocks.append(row.iloc[0:10].values)
    labels.append("Mean")

    X_blocks.append(row.iloc[10:20].values)
    labels.append("Error")

    X_blocks.append(row.iloc[20:30].values)
    labels.append("Worst")

X_blocks = np.array(X_blocks)
labels = np.array(labels)

# Modelo Mean/Error/Worst
scaler_blocks = StandardScaler()
X_blocks_scaled = scaler_blocks.fit_transform(X_blocks)

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_blocks_scaled, labels, test_size=0.2, random_state=42
)

model_blocks = KNeighborsClassifier(n_neighbors=5)
model_blocks.fit(X_train_b, y_train_b)

# Modelo Benigno/Maligno
mean_features = features[:10]
X_cancer = df[mean_features].values

scaler_cancer = StandardScaler()
X_cancer_scaled = scaler_cancer.fit_transform(X_cancer)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cancer_scaled, target, test_size=0.2, random_state=42
)

model_cancer = SVC(kernel="linear", probability=True)
model_cancer.fit(X_train_c, y_train_c)

# SHAP Explainer
explainer = shap.Explainer(model_cancer, X_train_c)
shap_values = explainer(X_test_c)

# ============================
# TABS
# ============================
tab1, tab2, tab3 = st.tabs(
    ["🩺 Predicción", "📊 Gráficos del Dataset", "🧠 Interpretación SHAP"]
)

# ============================================================
# TAB 1 — PREDICCIÓN
# ============================================================
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Predicción del Tumor")
    st.subheader("Ingrese las 10 características del tumor:")

    inputs = []
    names = [
        "Radio", "Textura", "Perímetro", "Área", "Suavidad",
        "Compacidad", "Concavidad", "Puntos cóncavos",
        "Simetría", "Dimensión fractal"
    ]

    cols = st.columns(2)
    for i, n in enumerate(names):
        val = cols[i % 2].number_input(n, min_value=0.0, value=1.0)
        inputs.append(val)

    inputs = np.array(inputs).reshape(1, -1)

    if st.button("Realizar Predicción"):
        try:
            X_user_b = scaler_blocks.transform(inputs)
            block_pred = model_blocks.predict(X_user_b)[0]

            X_user_c = scaler_cancer.transform(inputs)
            cancer_pred = model_cancer.predict(X_user_c)[0]
            cancer_label = "Benigno" if cancer_pred == 1 else "Maligno"

            st.success(f"🔹 Tipo de características detectadas: **{block_pred}**")
            st.info(f"🔹 Diagnóstico estimado: **{cancer_label}**")

            st.session_state["user_input"] = inputs

        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 2 — GRÁFICOS
# ============================================================
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📊 Análisis del Dataset")

    st.subheader("Distribución del Radio (mean radius)")
    fig1, ax1 = plt.subplots()
    sns.histplot(df["mean radius"], kde=True, ax=ax1, color="#8b5cf6")
    st.pyplot(fig1)

    st.subheader("Relación entre Radio y Área")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=df["mean radius"], y=df["mean area"], hue=target, ax=ax2, palette="cool")
    st.pyplot(fig2)

    st.subheader("Matriz de Correlación")
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

    st.subheader("Comparación de Radio: Mean / Error / Worst")
    comparison = pd.DataFrame({
        "Tipo": ["Mean", "Error", "Worst"],
        "Valor": [
            df["mean radius"].mean(),
            df["radius error"].mean(),
            df["worst radius"].mean()
        ]
    })
    fig4, ax4 = plt.subplots()
    sns.barplot(data=comparison, x="Tipo", y="Valor", ax=ax4,
                palette=["#8b5cf6", "#38bdf8", "#a5e8ff"])
    st.pyplot(fig4)

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 3 — SHAP
# ============================================================
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header(" Interpretación del Modelo con SHAP")

    st.subheader(" Importancia global de características")

    plt.close("all")
    shap.plots.bar(shap_values, show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 5)
    st.pyplot(fig)

    st.subheader(" Explicación de la predicción del usuario")

    if "user_input" in st.session_state:
        user = st.session_state["user_input"]
        scaled_user = scaler_cancer.transform(user)
        shap_user = explainer(scaled_user)

        plt.close("all")
        shap.plots.waterfall(shap_user[0], show=False)
        fig_w = plt.gcf()
        fig_w.set_size_inches(10, 5)
        st.pyplot(fig_w)

    else:
        st.info("Realiza primero una predicción en la pestaña Predicción.")

    st.markdown('</div>', unsafe_allow_html=True)