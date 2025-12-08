import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

# --- CONFIGURACIÓN ---
FEATURES_MEAN = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension"
]
FEATURES_SE = [
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error"
]
FEATURES_WORST = [
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

ALL_FEATURES = FEATURES_MEAN + FEATURES_SE + FEATURES_WORST

def load_model():
    model_path = 'models/mlp_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"⚠️ No se encontró el modelo en: {model_path}")
        return None

def generar_valores_aleatorios():
    for feature in ALL_FEATURES:
        st.session_state[f"mlp_{feature}"] = np.round(np.random.uniform(-3, 3), 2)

def render_input_group(features_list):
    cols = st.columns(5)
    for i, feature in enumerate(features_list):
        col_index = i % 5
        key = f"mlp_{feature}"
        if key not in st.session_state: st.session_state[key] = 0.00
        
        with cols[col_index]:
            st.number_input(
                label=feature.replace("mean ", "").replace("worst ", "").replace(" error", "").capitalize(),
                value=float(st.session_state[key]), 
                step=0.1, 
                format="%.2f", 
                key=key
            )

def mostrar():
    st.title("🧪 Probador MLP")
    st.markdown("Diagnóstico utilizando Redes Neuronales.")
    
    model = load_model()

    if 'historial_mlp' not in st.session_state:
        st.session_state['historial_mlp'] = []

    # --- ENCABEZADO ---
    c_title, _, c_rand = st.columns([3, 4, 1.5], gap="small")
    with c_rand:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲 Cargar Ejemplo", key="rand_mlp", use_container_width=True):
            generar_valores_aleatorios()

    # --- INPUTS ---
    with st.container(border=True):
        tab1, tab2, tab3 = st.tabs(["📏 Medias", "📉 Errores", "🚩 Peores"])
        with tab1: render_input_group(FEATURES_MEAN)
        with tab2: render_input_group(FEATURES_SE)
        with tab3: render_input_group(FEATURES_WORST)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- BOTÓN ---
    _, c_btn, _ = st.columns([2, 2, 2])
    with c_btn:
        submitted = st.button("🧠 Ejecutar Análisis", type="primary", use_container_width=True)

    # --- RESULTADO ---
    if submitted and model:
        # Recolectamos los valores de los inputs
        input_values = [st.session_state[f"mlp_{f}"] for f in ALL_FEATURES]
        
        try:
            features_array = np.array(input_values).reshape(1, -1)
            prediction = model.predict(features_array)[0]
            
            proba = 0.0
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features_array).max()

            # --- AQUÍ ESTÁ EL CAMBIO ---
            res_txt = "MALIGNO" if prediction == 1 else "BENIGNO"
            conf_str = f"{proba:.2%}"
            
            # 1. Creamos el registro básico
            nuevo_registro = {
                "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                "Diagnóstico": res_txt, 
                "Confianza": conf_str
            }
            
            # 2. Agregamos TODOS los datos de entrada al registro
            for feature, val in zip(ALL_FEATURES, input_values):
                nuevo_registro[feature] = val
            
            # 3. Guardamos en el historial
            st.session_state['historial_mlp'].append(nuevo_registro)
            # ---------------------------

            st.divider()
            
            # --- TARJETA ---
            col_icon, col_text, col_chart = st.columns([1, 2, 3], gap="medium")
            
            with col_icon:
                icon = "🚨" if prediction == 1 else "✅"
                st.markdown(f"<div style='text-align: center; font-size: 80px; line-height: 1;'>{icon}</div>", unsafe_allow_html=True)
            
            with col_text:
                st.markdown("#### Resultado Neural:")
                if prediction == 1:
                    st.markdown("""<h2 style='color: #d32f2f; margin-top: -10px;'>MALIGNO</h2><p>Alta probabilidad patológica.</p>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<h2 style='color: #2e7d32; margin-top: -10px;'>BENIGNO</h2><p>Patrones normales detectados.</p>""", unsafe_allow_html=True)

            with col_chart:
                st.markdown("#### Certeza:")
                st.metric("Probabilidad", f"{proba:.1%}")
                color = "#d32f2f" if prediction == 1 else "#2e7d32"
                st.markdown(f"""<div style="background-color: #e0e0e0; border-radius: 10px; height: 15px; width: 100%;"><div style="background-color: {color}; width: {proba*100}%; height: 100%; border-radius: 10px;"></div></div>""", unsafe_allow_html=True)
            
            st.toast("Análisis completado", icon="✅")

        except Exception as e:
            st.error(f"Error técnico: {e}")

    # --- HISTORIAL ---
    if len(st.session_state['historial_mlp']) > 0:
        st.divider()
        c_h1, c_h2 = st.columns([4, 1])
        with c_h1: st.subheader("📂 Historial MLP")
        with c_h2:
            if st.button("🗑️ Limpiar Historial", key="cls_mlp"):
                st.session_state['historial_mlp'] = []
                st.rerun()
        
        df = pd.DataFrame(st.session_state['historial_mlp'])
        st.dataframe(df, use_container_width=True, height=200)
        
        col_d1, col_d2 = st.columns([4, 1])
        with col_d2:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Descargar CSV", csv, f"mlp_diag_{datetime.now().strftime('%H%M')}.csv", "text/csv", use_container_width=True)

if __name__ == "__main__":
    mostrar()