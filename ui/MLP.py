import streamlit as st
import pandas as pd
import json
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import time

# --- CONFIGURACIÓN Y CONSTANTES ---

KEY_MAPPING = {
    "acc": "accuracy", "prec": "precision", "rec": "recall",
    "f1": "f1_score", "roc": "auc", "bal": "balanced_accuracy",
    "ap": "avg_precision", "spec": "specificity", "kappa": "cohens_kappa"
}

METRIC_DEFINITIONS = {
    "accuracy": "Exactitud: % de predicciones correctas totales.",
    "precision": "Precisión: De los predichos positivos, cuántos eran reales.",
    "recall": "Sensibilidad (recall): De los casos reales, cuántos detectó.",
    "f1_score": "F1-Score: Balance entre precisión y sensibilidad.",
    "auc": "AUC: Capacidad de distinguir clases (1.0 es perfecto).",
    "specificity": "Especificidad: Capacidad de detectar casos negativos.",
    "balanced_accuracy": "Exactitud balanceada.",
    "mcc": "Coeficiente de Matthews.",
    "avg_precision": "Precisión promedio (Average Precision).",
    "cohens_kappa": "Nivel de acuerdo excluyendo el azar."
}


def get_metric_value(metrics_dict, long_name):
    """Busca una métrica usando el nombre largo o sus abreviaturas."""
    if not metrics_dict:
        return 0.0

    # Si ya viene con el nombre largo
    if long_name in metrics_dict:
        return metrics_dict[long_name]

    # Buscar clave corta equivalente
    short_name = next((k for k, v in KEY_MAPPING.items() if v == long_name), None)
    if short_name and short_name in metrics_dict:
        return metrics_dict[short_name]

    return 0.0


# --- CARGA DE DATOS ---

@st.cache_resource(show_spinner=False)
def load_data():
    json_path = "models/mlp_training_report.json"
    model_path = "models/mlp_model.pkl"

    report = None
    model = None

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            report = json.load(f)

    if os.path.exists(model_path):
        model = joblib.load(model_path)

    return report, model


# --- GRÁFICOS PLOTLY ---


def plot_confusion_matrix_plotly(cm_data, title):
    """Matriz de confusión interactiva con Plotly."""
    if not cm_data:
        return None

    z = [
        [cm_data.get("true_negatives", 0), cm_data.get("false_positives", 0)],
        [cm_data.get("false_negatives", 0), cm_data.get("true_positives", 0)],
    ]

    x = ["Benigno", "Maligno"]
    y = ["Benigno", "Maligno"]

    z_text = [[str(v) for v in row] for row in z]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            text=z_text,
            texttemplate="%{text}",
            textfont={"size": 20},
            colorscale="Purples",
            showscale=False,
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title="Predicción",
        yaxis_title="Realidad",
        width=350,
        height=350,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    fig["layout"]["yaxis"]["autorange"] = "reversed"

    return fig


def plot_metrics_comparison_plotly(report):
    """Gráfico de barras comparativo Train vs Test."""
    target_metrics = ["accuracy", "precision", "recall", "f1_score"]
    perf = report.get("performance_metrics", {})
    train_data = perf.get("train", {})
    test_data = perf.get("test", {})

    y_train = [get_metric_value(train_data, m) for m in target_metrics]
    y_test = [get_metric_value(test_data, m) for m in target_metrics]
    x_labels = [m.replace("_", " ").capitalize() for m in target_metrics]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=y_train,
            name="Entrenamiento",
            marker_color="#1f77b4",
            text=[f"{v:.2f}" for v in y_train],
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=y_test,
            name="Prueba",
            marker_color="#ff7f0e",
            text=[f"{v:.2f}" for v in y_test],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="Comparación de Rendimiento",
        barmode="group",
        yaxis=dict(range=[0, 1.1]),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    return fig


def plot_class_distribution_plotly(report):
    """Gráfico de Donut interactivo: distribución real en test."""
    cm = report.get("confusion_matrix", {}).get("test", {})
    if not cm:
        return None

    benignos = cm.get("true_negatives", 0) + cm.get("false_positives", 0)
    malignos = cm.get("true_positives", 0) + cm.get("false_negatives", 0)

    labels = ["Benignos", "Malignos"]
    values = [benignos, malignos]
    colors = ["#66b3ff", "#ff9999"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                marker=dict(colors=colors),
                textinfo="label+percent",
                insidetextorientation="radial",
            )
        ]
    )

    fig.update_layout(
        title="Distribución Real (Test)",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
    )
    return fig


def plot_learning_curve_plotly(report, current_step=None):
    """
    Curva de aprendizaje del MLP.

    Soporta dos formatos de JSON:
    - MLP actual:   learning_curve = { "iterations": [...], "loss": [...] }
    - Estilo SVM:   learning_curve = { "train_sizes": [...], "train_scores": [...], "cv_scores": [...] }
    """
    lc = report.get("learning_curve", {})
    if not lc:
        return None

    # --- Caso 1: MLP actual (iterations + loss) ---
    if "iterations" in lc and "loss" in lc:
        xs_all = lc.get("iterations", [])
        train_all = lc.get("loss", [])
        cv_all = lc.get("val_loss")  # opcional, por si algún día lo agregas

        x_label = "Iteraciones"
        y_label = "Pérdida (Loss)"
        train_name = "Pérdida (Entrenamiento)"
        cv_name = "Pérdida (Validación)"

    # --- Caso 2: estilo SVM (por si lo reutilizas a futuro) ---
    elif "train_sizes" in lc and "train_scores" in lc:
        xs_all = lc.get("train_sizes", [])
        train_all = lc.get("train_scores", [])
        cv_all = lc.get("cv_scores") or lc.get("test_scores")

        x_label = "Ejemplos Procesados"
        y_label = "Score (Precisión)"
        train_name = "Entrenamiento"
        cv_name = "Validación (Prueba)"
    else:
        return None

    if not xs_all or not train_all:
        return None

    n = len(xs_all)
    if current_step is None:
        current_step = n
    current_step = max(1, min(current_step, n))

    xs = xs_all[:current_step]
    ys_train = train_all[:current_step]
    ys_cv = cv_all[:current_step] if cv_all is not None else None

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys_train,
            mode="lines+markers",
            name=train_name,
            line=dict(color="#d62728", width=3),
            marker=dict(size=8),
        )
    )

    if ys_cv is not None:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys_cv,
                mode="lines+markers",
                name=cv_name,
                line=dict(color="#2ca02c", width=3),
                marker=dict(size=8),
            )
        )

    # Rango dinámico
    all_y = ys_train + (ys_cv if ys_cv is not None else [])
    min_y = min(all_y)
    max_y = max(all_y)
    padding = (max_y - min_y) * 0.1 if max_y > min_y else 0.02

    fig.update_layout(
        title="Evolución del Aprendizaje",
        xaxis_title=x_label,
        yaxis_title=y_label,
        yaxis=dict(range=[min_y - padding, max_y + padding]),
        xaxis=dict(range=[min(xs_all) * 0.9, max(xs_all) * 1.05]),
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


# --- SIMULACIÓN DEL ENTRENAMIENTO (ANIMACIÓN) ---


def run_simulation(report, placeholder_bar, placeholder_plot):
    """Ejecuta la animación de entrenamiento para el MLP."""
    lc = report.get("learning_curve", {})
    if not lc:
        return

    if "iterations" in lc:
        total_steps = len(lc.get("iterations", []))
    elif "train_sizes" in lc:
        total_steps = len(lc.get("train_sizes", []))
    else:
        return

    if total_steps == 0:
        return

    for i in range(1, total_steps + 1):
        progress = i / total_steps
        placeholder_bar.progress(
            progress,
            text=f"Entrenando modelo MLP... ({int(progress * 100)}%)",
        )

        fig = plot_learning_curve_plotly(report, current_step=i)
        if fig:
            placeholder_plot.plotly_chart(fig, use_container_width=True)

        time.sleep(0.15)

    placeholder_bar.empty()
    st.toast("¡Entrenamiento del MLP finalizado con éxito!", icon="🧠")
    st.session_state["mlp_is_trained"] = True


# --- COMPONENTES VISUALES ---


def mostrar_estadisticas(report):
    """Tabla de métricas finales (test) en un expander."""
    metrics_raw = report.get("performance_metrics", {}).get("test", {})
    metrics_normalized = {}

    # Normalizar claves usando KEY_MAPPING
    for k, v in metrics_raw.items():
        new_key = KEY_MAPPING.get(k, k)
        metrics_normalized[new_key] = v

    with st.expander("📊 Métricas Detalladas (Clic para desplegar)", expanded=False):
        _, col_centro, _ = st.columns([1, 2, 1])
        with col_centro:
            with st.container(border=True):
                h1, h2 = st.columns([2, 1])
                h1.markdown("**Métrica**")
                h2.markdown("**Valor**")
                st.divider()
                for metric_name, value in metrics_normalized.items():
                    c1, c2 = st.columns([2, 1])
                    c1.markdown(
                        f"{metric_name}",
                        help=METRIC_DEFINITIONS.get(metric_name, ""),
                    )
                    if isinstance(value, (int, float)):
                        c2.markdown(f"`{value:.4f}`")
                    else:
                        c2.markdown(f"`{value}`")


def generar_explicacion_matriz(cm, tipo=""):
    if not cm:
        return ""
    tn = cm.get("true_negatives", 0)
    fp = cm.get("false_positives", 0)
    fn = cm.get("false_negatives", 0)
    tp = cm.get("true_positives", 0)
    total = tn + fp + fn + tp
    return f"""
    **Análisis {tipo}:** (Total: {total})
    * ✅ **Aciertos: {tn + tp}**
    * ❌ **Errores: {fn + fp}** (FN: {fn}, FP: {fp})
    """


def mostrar_resultados_post_entrenamiento(report):
    """Sección completa de resultados, igual al flujo del SVM."""

    # 1. Curva de aprendizaje (final)
    lc = report.get("learning_curve", {})
    if lc:
        st.markdown("### 1. Curva de Aprendizaje")
        fig_lc = plot_learning_curve_plotly(report)
        if fig_lc:
            st.plotly_chart(fig_lc, use_container_width=True)

    st.divider()

    # 2. Matrices de confusión
    st.markdown("### 2. Matrices de confusión")
    cm = report.get("confusion_matrix", {})

    col1, col2 = st.columns(2)
    with col1:
        fig_train = plot_confusion_matrix_plotly(
            cm.get("train"), "Matriz Entrenamiento"
        )
        if fig_train:
            st.plotly_chart(fig_train, use_container_width=True)
        st.info(generar_explicacion_matriz(cm.get("train"), "Entrenamiento"))

    with col2:
        fig_test = plot_confusion_matrix_plotly(cm.get("test"), "Matriz Prueba")
        if fig_test:
            st.plotly_chart(fig_test, use_container_width=True)
        st.success(generar_explicacion_matriz(cm.get("test"), "Prueba"))

    st.divider()

    # 3. Rendimiento global
    st.markdown("### 3. Rendimiento global")
    c3, c4 = st.columns([2, 1])
    with c3:
        fig_bar = plot_metrics_comparison_plotly(report)
        if fig_bar:
            st.plotly_chart(fig_bar, use_container_width=True)
    with c4:
        fig_pie = plot_class_distribution_plotly(report)
        if fig_pie:
            st.plotly_chart(fig_pie, use_container_width=True)

    # 4. Tabla de métricas
    st.divider()
    mostrar_estadisticas(report)


# --- ENTRADA PRINCIPAL ---


def mostrar():
    st.caption("Inicio > MLP (análisis)")
    st.markdown(
        '<h1 style="color:#4A148C;">Red Neuronal (MLP)</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Esta vista permite simular el entrenamiento y analizar el rendimiento "
        "del modelo MLP sobre el dataset preprocesado."
    )

    report, model = load_data()
    if not report:
        st.error("No se encontró el archivo 'mlp_training_report.json'.")
        return

    # Estado de simulación
    if "mlp_is_trained" not in st.session_state:
        st.session_state["mlp_is_trained"] = False

    st.markdown("---")
    st.caption(
        "Nota: la simulación recorre la curva de aprendizaje del MLP paso a paso "
        "para ilustrar cómo evoluciona el entrenamiento."
    )

    col_act, col_viz = st.columns([1, 3])

    with col_act:
        st.markdown("#### Panel de control")
        st.markdown(
            "Inicie o repita la simulación de entrenamiento para actualizar las "
            "métricas mostradas a continuación."
        )

        btn_label = (
            "▶️ Iniciar Entrenamiento"
            if not st.session_state["mlp_is_trained"]
            else "🔄 Re-entrenar Modelo"
        )

        if st.button(btn_label, type="primary", use_container_width=True):
            st.session_state["mlp_is_trained"] = False
            with col_viz:
                bar_ph = st.empty()
                plot_ph = st.empty()
                run_simulation(report, bar_ph, plot_ph)
                st.rerun()

    if st.session_state["mlp_is_trained"]:
        mostrar_resultados_post_entrenamiento(report)
    else:
        with col_viz:
            st.info(
                "El modelo MLP está listo para ser entrenado. "
                "Presiona el botón de la izquierda para comenzar la simulación."
            )


if __name__ == "__main__":
    mostrar()
