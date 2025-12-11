import streamlit as st
import pandas as pd
import altair as alt


COLOR_SVM = "#1f77b4"   # Azul consistente para SVM
COLOR_MLP = "#6A1B9A"   # Morado consistente para MLP

def _get_historial_svm() -> pd.DataFrame:
    """Historial que viene del probador SVM."""
    data = st.session_state.get("historial_predicciones", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    # Normalizar posibles nombres de columnas
    if "Diagnostico" in df.columns and "Diagnóstico" not in df.columns:
        df = df.rename(columns={"Diagnostico": "Diagnóstico"})
    return df


def _get_historial_mlp() -> pd.DataFrame:
    """Historial que viene del probador MLP."""
    data = st.session_state.get("historial_mlp", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "Diagnostico" in df.columns and "Diagnóstico" not in df.columns:
        df = df.rename(columns={"Diagnostico": "Diagnóstico"})
    return df


def mostrar() -> None:
    st.caption("Inicio > Analítica Web")
    st.markdown(
        """
        <h1 style="color:#4A148C;">📊 Analítica de la sesión</h1>
        <p style="color:#555;">
            Resumen de todos los diagnósticos realizados con los probadores
            <strong>SVM</strong> y <strong>MLP</strong> durante esta sesión.
        </p>
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

    # Unificamos todo en un solo DataFrame
    frames = []
    if not df_svm.empty:
        d = df_svm.copy()
        d["Modelo"] = "SVM"
        frames.append(d)
    if not df_mlp.empty:
        d = df_mlp.copy()
        d["Modelo"] = "MLP"
        frames.append(d)

    df_all = pd.concat(frames, ignore_index=True)

    # Asegurarnos de que exista la columna Diagnóstico
    if "Diagnóstico" not in df_all.columns:
        st.error("No se encontró la columna 'Diagnóstico' en el historial.")
        st.dataframe(df_all, use_container_width=True)
        return

    # --- Resumen numérico ---
    total_svm = len(df_all[df_all["Modelo"] == "SVM"])
    total_mlp = len(df_all[df_all["Modelo"] == "MLP"])
    total = len(df_all)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Total de predicciones",
            total,
            help="Suma de todos los casos evaluados en esta sesión (SVM + MLP).",
        )
    with c2:
        st.metric(
            "Predicciones SVM",
            total_svm,
            help="Cantidad de predicciones realizadas usando el probador SVM.",
        )
    with c3:
        st.metric(
            "Predicciones MLP",
            total_mlp,
            help="Cantidad de predicciones realizadas usando el probador MLP.",
        )

    st.divider()

    # --- Distribución Benigno/Maligno por modelo ---
    st.subheader("⚖️ Balance Benigno/Maligno por modelo (sesión)")

    col_svm, col_mlp = st.columns(2)

    # ===== Gráfico SVM =====
    with col_svm:
        st.markdown("#### SVM")
        if df_svm.empty or "Diagnóstico" not in df_svm.columns:
            st.info("Aún no hay predicciones SVM en esta sesión.")
        else:
            counts_svm = (
                df_svm["Diagnóstico"]
                .astype(str)
                .value_counts()
                .reset_index()
            )
            counts_svm.columns = ["Diagnóstico", "Cantidad"]

            chart_svm = (
                alt.Chart(counts_svm)
                .mark_bar()
                .encode(
                    x=alt.X("Diagnóstico:N", title="Diagnóstico"),
                    y=alt.Y("Cantidad:Q", title="Cantidad"),
                    color=alt.value(COLOR_SVM),
                )
                .properties(height=250)
            )
            st.altair_chart(chart_svm, use_container_width=True)

    # ===== Gráfico MLP =====
    with col_mlp:
        st.markdown("#### MLP")
        if df_mlp.empty or "Diagnóstico" not in df_mlp.columns:
            st.info("Aún no hay predicciones MLP en esta sesión.")
        else:
            counts_mlp = (
                df_mlp["Diagnóstico"]
                .astype(str)
                .value_counts()
                .reset_index()
            )
            counts_mlp.columns = ["Diagnóstico", "Cantidad"]

            chart_mlp = (
                alt.Chart(counts_mlp)
                .mark_bar()
                .encode(
                    x=alt.X("Diagnóstico:N", title="Diagnóstico"),
                    y=alt.Y("Cantidad:Q", title="Cantidad"),
                    color=alt.value(COLOR_MLP),
                )
                .properties(height=250)
            )
            st.altair_chart(chart_mlp, use_container_width=True)

    st.divider()

    # --- Historial reciente con filtros ---
    st.subheader("📂 Historial reciente de la sesión")

    # Filtros básicos
    modelos = ["Ambos", "Solo SVM", "Solo MLP"]
    modelo_filtro = st.selectbox(
        "Modelo",
        modelos,
        index=0,
        help="Permite limitar el historial a un modelo específico o ver ambos.",
    )

    diag_unicos = sorted(df_all["Diagnóstico"].dropna().astype(str).unique().tolist())
    diag_label = ["Todos"] + diag_unicos
    diag_filtro = st.selectbox(
        "Diagnóstico",
        diag_label,
        index=0,
        help="Filtra el historial por tipo de diagnóstico.",
    )
    # Aplicar filtros
    df_filtrado = df_all.copy()

    if modelo_filtro == "Solo SVM":
        df_filtrado = df_filtrado[df_filtrado["Modelo"] == "SVM"]
    elif modelo_filtro == "Solo MLP":
        df_filtrado = df_filtrado[df_filtrado["Modelo"] == "MLP"]

    if diag_filtro != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Diagnóstico"].astype(str) == diag_filtro]

    # Ordenar por fecha si existe esa columna
    if "Fecha" in df_filtrado.columns:
        df_filtrado = df_filtrado.sort_values("Fecha", ascending=False)

    max_casos = len(df_filtrado)

    if max_casos == 0:
        st.info("No hay registros que coincidan con el filtro actual.")
        return

    # Si hay 1 solo registro, no usamos slider (Streamlit no deja min == max)
    if max_casos == 1:
        n_casos = 1
    else:
        n_casos = st.slider(
            "Número máximo de registros a mostrar",
            min_value=1,
            max_value=max_casos,
            value=min(50, max_casos),
            step=1,
            help="Define cuántos registros recientes quieres ver en la tabla.",
        )

    df_filtrado = df_filtrado.head(n_casos)

    if modelo_filtro == "Solo SVM":
        df_filtrado = df_filtrado[df_filtrado["Modelo"] == "SVM"]
    elif modelo_filtro == "Solo MLP":
        df_filtrado = df_filtrado[df_filtrado["Modelo"] == "MLP"]

    if diag_filtro != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Diagnóstico"].astype(str) == diag_filtro]

    # Ordenar por fecha si existe esa columna
    if "Fecha" in df_filtrado.columns:
        df_filtrado = df_filtrado.sort_values("Fecha", ascending=False)

    df_filtrado = df_filtrado.head(n_casos)

    if df_filtrado.empty:
        st.info("No hay registros que coincidan con el filtro actual.")
        return

    columnas_base = [
        c for c in ["Fecha", "Modelo", "Diagnóstico", "Confianza"]
        if c in df_filtrado.columns
    ]
    if columnas_base:
        st.dataframe(df_filtrado[columnas_base], use_container_width=True, height=300)
    else:
        st.dataframe(df_filtrado, use_container_width=True, height=300)


if __name__ == "__main__":
    mostrar()
