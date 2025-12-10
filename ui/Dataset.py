import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURACIÓN DE RUTAS ---
# Ajusta estas rutas si tus archivos están en otro lugar
PATH_RAW = "data/breast_cancer.csv"
PATH_PROC = "data/processed/"


@st.cache_data(show_spinner=False)
def load_data():
    """Carga los datasets y metadatos."""
    data = {}
    
    # 1. Cargar Original
    if os.path.exists(PATH_RAW):
        data["raw"] = pd.read_csv(PATH_RAW)
    
    # 2. Cargar Procesados (CSV es más rápido para visualizar que .npy)
    train_path = os.path.join(PATH_PROC, "train_data.csv")
    test_path = os.path.join(PATH_PROC, "test_data.csv")
    
    if os.path.exists(train_path):
        data["train"] = pd.read_csv(train_path)
    if os.path.exists(test_path):
        data["test"] = pd.read_csv(test_path)
        
    # 3. Cargar Reportes
    report_path = os.path.join(PATH_PROC, "preprocessing_report.json")
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            data["report"] = json.load(f)
            
    return data

def plot_target_distribution(df, title, target_col="diagnosis"):
    """Grafica la distribución de clases."""
    if target_col not in df.columns: return None
    
    counts = df[target_col].value_counts().reset_index()
    counts.columns = [target_col, "count"]
    
    fig = px.pie(
        counts, 
        names=target_col, 
        values="count", 
        title=title,
        hole=0.4,
        color_discrete_sequence=['#66b3ff', '#ff9999'] # Azul/Rojo suave
    )
    return fig

def mostrar():
    st.caption("Inicio > Explorador de Datos")
    st.markdown(
        '<h1 style="color:#4A148C;">📂 Explorador de Datos</h1>',
        unsafe_allow_html=True,
    )
    st.markdown("Esta vista permite explorar el dataset original y las particiones train/test generadas tras el preprocesamiento.")

    data = load_data()

    # --- TABS PARA ORGANIZAR ---
    tab1, tab2, tab3 = st.tabs([
        "📊 Dataset Original",
        "✂️ Train/Test Split",
        "📝 Reporte Técnico",
    ])
    st.caption("Consejo: use las pestañas para alternar entre vista general, particiones y reporte técnico de preprocesamiento.")

    # TAB 1: DATASET ORIGINAL
    with tab1:
        df = data.get("raw")
        if df is not None:
            st.markdown(f"**Dimensiones:** {df.shape[0]} filas, {df.shape[1]} columnas")
            
            # Vista previa
            st.dataframe(df.head(10), use_container_width=True)
            
            c1, c2 = st.columns([1, 2])
            with c1:
                # Gráfico de distribución
                fig = plot_target_distribution(df, "Distribución de Diagnóstico")
                if fig: st.plotly_chart(fig, use_container_width=True)
            with c2:
                # Estadísticas básicas
                st.markdown("###### Estadísticas Descriptivas")
                st.dataframe(df.describe(), use_container_width=True, height=300)
                
            # Botón de descarga
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Descargar Dataset Original", csv, "breast_cancer_raw.csv", "text/csv")
        else:
            st.warning(f"No se encontró el archivo en `{PATH_RAW}`. Por favor verifica que esté en la carpeta `data/`.")

    # TAB 2: TRAIN / TEST
    with tab2:
        df_tr = data.get("train")
        df_te = data.get("test")
        
        if df_tr is not None and df_te is not None:
            col_tr, col_te = st.columns(2)
            
            with col_tr:
                st.info(f"### Set de Entrenamiento ({df_tr.shape[0]} muestras)")
                st.dataframe(df_tr.head(), use_container_width=True)
                
            with col_te:
                st.info(f"### Set de Prueba ({df_te.shape[0]} muestras)")
                st.dataframe(df_te.head(), use_container_width=True)

            st.divider()
            
            # Comparativa visual de balance
            st.markdown("#### Balance de Clases: Entrenamiento vs Prueba")
            
            # Preparar datos para gráfico agrupado
            target = "diagnosis" # Asumiendo que la columna se llama 'diagnosis' en el CSV procesado
            if target in df_tr.columns:
                tr_counts = df_tr[target].value_counts().reset_index()
                tr_counts["Set"] = "Entrenamiento"
                te_counts = df_te[target].value_counts().reset_index()
                te_counts["Set"] = "Prueba"
                
                # Unificar nombres de columnas para concatenar
                tr_counts.columns = [target, "count", "Set"]
                te_counts.columns = [target, "count", "Set"]
                
                df_all = pd.concat([tr_counts, te_counts])
                
                fig_bar = px.bar(
                    df_all, x="Set", y="count", color=target, 
                    barmode="group",
                    text="count",
                    color_discrete_sequence=['#1f77b4', '#ff7f0e'] # Colores SVM/Test
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.error(f"Faltan archivos en `{PATH_PROC}`. Asegúrate de tener `train_data.csv` y `test_data.csv`.")

    # TAB 3: REPORTE
    with tab3:
        report = data.get("report")
        if report:
            st.markdown("### 📋 Detalles del Preprocesamiento")
            st.json(report)
        else:
            st.info("No se encontró el archivo `preprocessing_report.json`.")

if __name__ == "__main__":
    mostrar()