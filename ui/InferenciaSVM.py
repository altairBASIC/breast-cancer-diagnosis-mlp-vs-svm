import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

# --- CONFIGURACIÓN Y CONSTANTES ---
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
    model_path = 'models/svm_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"⚠️ No se encontró el modelo en: {model_path}")
        return None

def generar_valores_aleatorios():
    for feature in ALL_FEATURES:
        st.session_state[f"input_{feature}"] = np.round(np.random.uniform(-3, 3), 2)

def render_input_group(features_list):
    """Renderiza inputs en 5 columnas."""
    cols = st.columns(5)
    for i, feature in enumerate(features_list):
        col_index = i % 5
        key = f"input_{feature}"
        if key not in st.session_state: st.session_state[key] = 0.00
        
        with cols[col_index]:
            st.number_input(
                # Limpiamos el nombre para que se vea bien en la UI
                label=feature.replace("mean ", "").replace("worst ", "").replace(" error", "").capitalize(),
                value=float(st.session_state[key]), 
                step=0.1, 
                format="%.2f", 
                key=key
            )

def mostrar():
    model = load_model()

    # --- INICIALIZAR HISTORIAL ---
    if 'historial_predicciones' not in st.session_state:
        st.session_state['historial_predicciones'] = []

    # --- ENCABEZADO ---
    c_title, c_fill, c_rand = st.columns([3, 4, 1.5], gap="small")
    with c_title:
        st.title("🧪 Probador SVM")
        st.caption("Ingrese datos normalizados para predicción.")
    with c_rand:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲 Cargar Ejemplo", use_container_width=True):
            generar_valores_aleatorios()

    # --- PANEL DE INPUTS ---
    with st.container(border=True):
        tab1, tab2, tab3 = st.tabs(["📏 Mediciones Medias", "📉 Errores Estándar", "🚩 Peores Valores"])
        with tab1: render_input_group(FEATURES_MEAN)
        with tab2: render_input_group(FEATURES_SE)
        with tab3: render_input_group(FEATURES_WORST)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- BOTÓN DE ACCIÓN ---
    _, c_btn, _ = st.columns([2, 2, 2])
    with c_btn:
        submitted = st.button("🔍 Analizar Patrones Clínicos", type="primary", use_container_width=True)

    # --- RESULTADO Y GUARDADO ---
    if submitted and model:
        # 1. Recolectar valores
        input_values = [st.session_state[f"input_{f}"] for f in ALL_FEATURES]
        
        try:
            features_array = np.array(input_values).reshape(1, -1)
            prediction = model.predict(features_array)[0]
            proba = model.predict_proba(features_array).max() if hasattr(model, "predict_proba") else 0.0

            # 2. Guardar en Historial (Con claves en Español para evitar duplicados)
            resultado_texto = "MALIGNO" if prediction == 1 else "BENIGNO"
            confianza_str = f"{proba:.2%}"

            nuevo_registro = {
                "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Diagnóstico": resultado_texto,
                "Confianza": confianza_str,
            }
            # Agregar características
            for feature, val in zip(ALL_FEATURES, input_values):
                nuevo_registro[feature] = val
            
            st.session_state['historial_predicciones'].append(nuevo_registro)

            # 3. Mostrar Resultado Visual
            st.divider()
            col_icon, col_text, col_chart = st.columns([1, 2, 3], gap="medium")
            
            with col_icon:
                icon = "🚨" if prediction == 1 else "✅"
                st.markdown(f"<div style='text-align: center; font-size: 80px; line-height: 1;'>{icon}</div>", unsafe_allow_html=True)
            
            with col_text:
                st.markdown("#### Resultado:")
                if prediction == 1:
                    st.markdown("""<h2 style='color: #d32f2f; margin-top: -10px;'>MALIGNO</h2><p>Se recomienda revisión oncológica.</p>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<h2 style='color: #2e7d32; margin-top: -10px;'>BENIGNO</h2><p>No se detectan anomalías críticas.</p>""", unsafe_allow_html=True)

            with col_chart:
                st.markdown("#### Certeza:")
                st.metric("Probabilidad", f"{proba:.1%}")
                color = "#d32f2f" if prediction == 1 else "#2e7d32"
                st.markdown(f"""<div style="background-color: #e0e0e0; border-radius: 10px; height: 15px; width: 100%;"><div style="background-color: {color}; width: {proba*100}%; height: 100%; border-radius: 10px;"></div></div>""", unsafe_allow_html=True)
            
            st.toast("✅ Resultado guardado", icon="💾")

        except Exception as e:
            st.error(f"Error técnico: {e}")

    # --- SECCIÓN HISTORIAL ---
    if len(st.session_state['historial_predicciones']) > 0:
        st.divider()
        
        # Encabezado con botón de borrar
        c_hist_title, c_hist_clear = st.columns([4, 1])
        with c_hist_title:
            st.subheader("📂 Historial de Sesión")
        with c_hist_clear:
            if st.button("🗑️ Borrar Historial", type="secondary", use_container_width=True):
                st.session_state['historial_predicciones'] = []
                st.rerun() # Recargar para limpiar la vista
        
        if len(st.session_state['historial_predicciones']) > 0:
            # Convertir a DataFrame
            df_history = pd.DataFrame(st.session_state['historial_predicciones'])
            
            # Mostrar tabla (sin formateo extra para evitar error de string)
            st.dataframe(df_history, use_container_width=True, height=200)
            
            # Botón Descarga
            col_d1, col_d2 = st.columns([4, 1])
            with col_d2:
                csv = df_history.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar CSV",
                    data=csv,
                    file_name=f"diagnosticos_svm_{datetime.now().strftime('%H%M%S')}.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )

if __name__ == "__main__":
    mostrar()