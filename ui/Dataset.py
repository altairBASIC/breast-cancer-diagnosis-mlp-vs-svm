import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px

# --- CONFIGURACIÓN DE RUTAS ---
PATH_RAW = "data/breast_cancer.csv"
PATH_PROC = "data/processed/"

@st.cache_data(show_spinner=False)
def load_data():
    """Carga los datasets y metadatos."""
    data = {}
    if os.path.exists(PATH_RAW):
        # Intentamos cargar con separador , o ; por seguridad
        try:
            data["raw"] = pd.read_csv(PATH_RAW, sep=',')
            if data["raw"].shape[1] < 2: # Si cargó mal, intentamos con ;
                data["raw"] = pd.read_csv(PATH_RAW, sep=';')
        except:
            pass
    
    train_path = os.path.join(PATH_PROC, "train_data.csv")
    test_path = os.path.join(PATH_PROC, "test_data.csv")
    
    if os.path.exists(train_path):
        data["train"] = pd.read_csv(train_path)
    if os.path.exists(test_path):
        data["test"] = pd.read_csv(test_path)
        
    report_path = os.path.join(PATH_PROC, "preprocessing_report.json")
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            data["report"] = json.load(f)
            
    return data

def plot_target_distribution(df, title, target_col="diagnosis"):
    """
    Grafica la distribución usando el DataFrame directo para evitar errores de conteo.
    """
    if target_col not in df.columns: 
        return None
    
    # Creamos una copia para no afectar el dataframe original con el mapeo
    df_plot = df.copy()
    
    # Aseguramos que los valores sean string para que el mapa funcione bien
    df_plot[target_col] = df_plot[target_col].astype(str)
    
    # Mapeo de etiquetas para que se vea bonito en el gráfico
    mapeo = {
        "0": "Benigno", "1": "Maligno", 
        "B": "Benigno", "M": "Maligno",
        "0.0": "Benigno", "1.0": "Maligno"
    }
    df_plot["Etiqueta"] = df_plot[target_col].map(mapeo).fillna(df_plot[target_col])
    
    # --- CAMBIO CLAVE: Usamos el DF completo, Plotly cuenta solo ---
    fig = px.pie(
        df_plot, 
        names="Etiqueta", 
        title=title, 
        hole=0.4,
        color="Etiqueta",
        color_discrete_map={"Benigno": "#2ca02c", "Maligno": "#d62728"}
    )
    return fig

def mostrar():
    st.caption("Inicio > Explorador de Datos")
    st.markdown('<h1 style="color:#4A148C;">📂 Explorador de Datos</h1>', unsafe_allow_html=True)

    data = load_data()

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Dataset Original", "✂️ Train/Test Split", "📝 Reporte Técnico"])

    # TAB 1: ORIGINAL
    with tab1:
        df = data.get("raw")
        if df is not None:
            st.markdown(f"**Dimensiones:** {df.shape[0]} filas, {df.shape[1]} columnas")
            
            # Verificación rápida de la columna diagnosis
            if "diagnosis" in df.columns:
                conteo_real = df["diagnosis"].value_counts()
                st.caption(f"Conteo real en datos: {conteo_real.to_dict()}")
            
            st.dataframe(df.head(), use_container_width=True)
            
            c1, c2 = st.columns([1, 2])
            with c1:
                # El gráfico corregido se llama aquí
                fig = plot_target_distribution(df, "Distribución Real del Dataset")
                if fig: st.plotly_chart(fig, use_container_width=True)
                else: st.warning("No se encontró la columna 'diagnosis'.")
            with c2:
                st.markdown("###### Estadísticas")
                st.dataframe(df.describe(), use_container_width=True, height=300)
        else:
            st.warning("No se encontró el archivo raw.")

    # TAB 2: TRAIN / TEST (SOLO TABLA)
    with tab2:
        df_tr = data.get("train")
        df_te = data.get("test")
        
        if df_tr is not None and df_te is not None:
            col_tr, col_te = st.columns(2)
            with col_tr:
                st.info(f"### Entrenamiento ({len(df_tr)})")
                st.dataframe(df_tr.head(), use_container_width=True)
            with col_te:
                st.info(f"### Prueba ({len(df_te)})")
                st.dataframe(df_te.head(), use_container_width=True)

            st.divider()
            st.markdown("### 📋 Conteo Detallado de Clases")

            # --- LÓGICA DE TABLA ---
            target = "diagnosis"
            
            def contar_clases(df, col):
                if col not in df.columns: return 0, 0
                # Convertimos a string para asegurar que atrapamos 0, '0', 0.0, etc.
                vals = df[col].astype(str)
                benignos = vals.isin(['0', '0.0', 'B', 'Benigno']).sum()
                malignos = vals.isin(['1', '1.0', 'M', 'Maligno']).sum()
                return benignos, malignos

            b_train, m_train = contar_clases(df_tr, target)
            b_test, m_test = contar_clases(df_te, target)

            tabla_resumen = pd.DataFrame({
                "Conjunto": ["Entrenamiento", "Prueba", "TOTAL"],
                "🟢 Benignos": [b_train, b_test, b_train + b_test],
                "🔴 Malignos": [m_train, m_test, m_train + m_test],
                "Total": [len(df_tr), len(df_te), len(df_tr) + len(df_te)]
            })

            st.table(tabla_resumen.set_index("Conjunto"))

        else:
            st.error("Faltan los archivos procesados.")

    # TAB 3: REPORTE
    with tab3:
        report = data.get("report")
        if report: st.json(report)
        else: st.info("No hay reporte disponible.")

if __name__ == "__main__":
    mostrar()