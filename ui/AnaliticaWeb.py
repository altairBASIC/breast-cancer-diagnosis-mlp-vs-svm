import streamlit as st
import pandas as pd
import plotly.express as px


COLOR_SVM = "#1f77b4"  # Azul consistente para SVM
COLOR_MLP = "#6A1B9A"  # Morado consistente para MLP
COLOR_PURPLE_BG = "#F3E5F5"  # Fondo suave morado para resaltar secciones
COLOR_PURPLE_TEXT = "#4A148C"  # Texto morado oscuro para títulos/destacados


def _get_historial_svm() -> pd.DataFrame:
    data = st.session_state.get("historial_predicciones", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    # Normalizar posible nombre de columna de diagnóstico
    if "Diagnostico" in df.columns and "Diagnóstico" not in df.columns:
        df = df.rename(columns={"Diagnostico": "Diagnóstico"})
    return df


def _get_historial_mlp() -> pd.DataFrame:
    data = st.session_state.get("historial_mlp", [])
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


def mostrar() -> None:
    st.caption("Inicio > Analítica Web")
    st.markdown(
        '<h1 style="color:#4A148C;">📊 Analítica de Sesión</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Resumen de las predicciones realizadas en esta sesión para los modelos **SVM** y **MLP**."
    )

    # Banda descriptiva en tonos morados para dar identidad visual a la vista
    st.markdown(
        f"""
        <div style="background-color:{COLOR_PURPLE_BG}; padding: 0.75rem 1rem; border-radius: 0.6rem; margin-bottom: 0.5rem;">
            <span style="color:{COLOR_PURPLE_TEXT}; font-weight: 600;">
                Vista general del uso de los probadores durante la sesión actual.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_svm = _get_historial_svm()
    df_mlp = _get_historial_mlp()

    if df_svm.empty and df_mlp.empty:
        st.info(
            "Aún no hay predicciones registradas en esta sesión. "
            "Utiliza los probadores SVM o MLP para generar casos."
        )
        return

    # --- Resumen numérico ---
    total_svm = len(df_svm)
    total_mlp = len(df_mlp)
    total = total_svm + total_mlp

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total de predicciones", total, help="Suma de todos los casos evaluados en esta sesión (SVM + MLP).")
    with c2:
        st.metric("Predicciones SVM", total_svm, help="Cantidad de predicciones realizadas usando el probador SVM.")
    with c3:
        st.metric("Predicciones MLP", total_mlp, help="Cantidad de predicciones realizadas usando el probador MLP.")

    st.divider()

    # --- Distribución Benigno/Maligno por modelo ---
    rows = []

    if not df_svm.empty and "Diagnóstico" in df_svm.columns:
        for diag, count in df_svm["Diagnóstico"].value_counts().items():
            rows.append({"Modelo": "SVM", "Diagnóstico": str(diag), "Cantidad": int(count)})

    if not df_mlp.empty:
        col_diag_mlp = "Diagnóstico" if "Diagnóstico" in df_mlp.columns else "Diagnostico" if "Diagnostico" in df_mlp.columns else None
        if col_diag_mlp is not None:
            for diag, count in df_mlp[col_diag_mlp].value_counts().items():
                rows.append({"Modelo": "MLP", "Diagnóstico": str(diag), "Cantidad": int(count)})

    st.subheader("⚖️ Balance Benigno/Maligno por modelo (sesión)")

    if rows:
        df_counts = pd.DataFrame(rows)
        # Gráfico en Plotly usando la paleta consistente (SVM azul, MLP morado)
        fig = px.bar(
            df_counts,
            x="Diagnóstico",
            y="Cantidad",
            color="Modelo",
            barmode="group",
            color_discrete_map={"SVM": COLOR_SVM, "MLP": COLOR_MLP},
            title="Distribución de diagnósticos por modelo",
        )
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=50, b=30),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay suficientes datos para graficar la distribución de diagnósticos.")

    st.divider()

    # --- Historial reciente con filtros ---
    st.subheader("📂 Historial reciente de la sesión")

    modelo_filtro = st.selectbox(
        "Modelo",
        ["Ambos", "Solo SVM", "Solo MLP"],
        index=0,
        help="Permite limitar el historial a un modelo específico o ver ambos.",
    )
    diag_filtro = st.selectbox(
        "Diagnóstico",
        ["Todos", "BENIGNO", "MALIGNO"],
        index=0,
        help="Filtra los casos por tipo de diagnóstico reportado por el modelo.",
    )
    n_casos = st.slider(
        "Número máximo de casos a mostrar",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        help="Controla cuántos registros recientes se muestran en la tabla inferior.",
    )

    frames = []
    if modelo_filtro in ("Ambos", "Solo SVM") and not df_svm.empty:
        d = df_svm.copy()
        d["Modelo"] = "SVM"
        frames.append(d)
    if modelo_filtro in ("Ambos", "Solo MLP") and not df_mlp.empty:
        d = df_mlp.copy()
        d["Modelo"] = "MLP"
        frames.append(d)

    if not frames:
        st.info("No hay registros que coincidan con el filtro seleccionado.")
        return

    df_all = pd.concat(frames, ignore_index=True)

    # Columna de diagnóstico normalizada para filtrar y mostrar
    diag_col = "Diagnóstico" if "Diagnóstico" in df_all.columns else "Diagnostico" if "Diagnostico" in df_all.columns else None
    if diag_col is not None:
        df_all["Diagnóstico_norm"] = df_all[diag_col].astype(str)
        if diag_filtro != "Todos":
            df_all = df_all[df_all["Diagnóstico_norm"] == diag_filtro]

    # Ordenar por fecha si existe
    if "Fecha" in df_all.columns:
        df_all = df_all.sort_values("Fecha", ascending=False)

    df_all = df_all.head(n_casos)

    if df_all.empty:
        st.info("No hay registros que coincidan con el filtro actual.")
        return

    # Seleccionar columnas principales para mostrar
    columnas_base = [c for c in ["Fecha", "Modelo", "Diagnóstico_norm", "Confianza"] if c in df_all.columns]
    if columnas_base:
        st.dataframe(df_all[columnas_base], use_container_width=True, height=300)
    else:
        st.dataframe(df_all, use_container_width=True, height=300)
