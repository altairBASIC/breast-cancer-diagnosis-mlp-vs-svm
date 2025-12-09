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
    "recall": "Sensibilidad: De los casos reales, cuántos detectó.",
    "f1_score": "F1-Score: Balance entre precisión y sensibilidad.",
    "auc": "AUC: Capacidad de distinguir clases (1.0 es perfecto).",
    "specificity": "Especificidad: Capacidad de detectar casos negativos.",
    "balanced_accuracy": "Exactitud balanceada.",
    "mcc": "Coeficiente de Matthews.",
    "avg_precision": "Precisión promedio.",
    "cohens_kappa": "Nivel de acuerdo excluyendo el azar."
}

def get_metric_value(metrics_dict, long_name):
    if not metrics_dict: return 0.0
    if long_name in metrics_dict: return metrics_dict[long_name]
    short_name = next((k for k, v in KEY_MAPPING.items() if v == long_name), None)
    if short_name and short_name in metrics_dict: return metrics_dict[short_name]
    return 0.0

def load_data():
    json_path = 'models/svm_training_report.json'
    model_path = 'models/svm_model.pkl'
    
    report = None
    model = None

    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            report = json.load(f)
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    
    return report, model

# --- GRÁFICOS CON PLOTLY ---

def plot_confusion_matrix_plotly(cm_data, title):
    """Matriz de confusión interactiva con Plotly."""
    if not cm_data: return None
    
    # Preparar datos
    z = [[cm_data.get('true_negatives', 0), cm_data.get('false_positives', 0)],
         [cm_data.get('false_negatives', 0), cm_data.get('true_positives', 0)]]
    
    x = ['Benigno', 'Maligno']
    y = ['Benigno', 'Maligno']
    
    # Texto para mostrar dentro de las celdas
    z_text = [[str(y) for y in x] for x in z]

    # Crear Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y,
        text=z_text,
        texttemplate="%{text}",
        textfont={"size": 20},
        colorscale='Blues',
        showscale=False
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Predicción",
        yaxis_title="Realidad",
        width=350, height=350,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    # Invertir eje Y para que coincida con la convención estándar (TN arriba izq)
    fig['layout']['yaxis']['autorange'] = "reversed"
    
    return fig

def plot_metrics_comparison_plotly(report):
    """Gráfico de barras comparativo Train vs Test."""
    target_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    perf = report.get('performance_metrics', {})
    train_data = perf.get('train', {})
    test_data = perf.get('test', {})

    y_train = [get_metric_value(train_data, m) for m in target_metrics]
    y_test  = [get_metric_value(test_data, m) for m in target_metrics]
    x_labels = [m.replace('_', ' ').capitalize() for m in target_metrics]

    fig = go.Figure()
    
    # Barra Train
    fig.add_trace(go.Bar(
        x=x_labels, y=y_train,
        name='Entrenamiento',
        marker_color='#1f77b4',
        text=[f"{v:.2f}" for v in y_train],
        textposition='auto'
    ))
    
    # Barra Test
    fig.add_trace(go.Bar(
        x=x_labels, y=y_test,
        name='Prueba',
        marker_color='#ff7f0e',
        text=[f"{v:.2f}" for v in y_test],
        textposition='auto'
    ))

    fig.update_layout(
        title="Comparación de Rendimiento",
        barmode='group',
        yaxis=dict(range=[0, 1.1]),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_class_distribution_plotly(report):
    """Gráfico de Donut interactivo."""
    cm = report.get('confusion_matrix', {}).get('test', {})
    if not cm: return None
    
    benignos = cm.get('true_negatives', 0) + cm.get('false_positives', 0)
    malignos = cm.get('true_positives', 0) + cm.get('false_negatives', 0)
    
    labels = ['Benignos', 'Malignos']
    values = [benignos, malignos]
    colors = ['#66b3ff', '#ff9999']

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=.5,
        marker=dict(colors=colors),
        textinfo='label+percent',
        insidetextorientation='radial'
    )])

    fig.update_layout(
        title="Distribución Real (Test)",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    return fig

def plot_learning_curve_plotly(train_sizes, train_scores, cv_scores, current_step=None):
    """Crea la figura de la curva de aprendizaje."""
    if current_step is None:
        current_step = len(train_sizes)

    # Recortar datos según el paso de la animación
    ts = train_sizes[:current_step]
    tr_sc = train_scores[:current_step]
    cv_sc = cv_scores[:current_step]

    fig = go.Figure()

    # Línea Entrenamiento
    fig.add_trace(go.Scatter(
        x=ts, y=tr_sc,
        mode='lines+markers',
        name='Entrenamiento',
        line=dict(color='#d62728', width=3),
        marker=dict(size=8)
    ))

    # Línea Validación
    fig.add_trace(go.Scatter(
        x=ts, y=cv_sc,
        mode='lines+markers',
        name='Validación (Prueba)',
        line=dict(color='#2ca02c', width=3),
        marker=dict(size=8)
    ))

    # Configuración de ejes fija para que no "baile"
    min_y = min(min(train_scores), min(cv_scores)) * 0.98
    
    fig.update_layout(
        title="Evolución del Aprendizaje",
        xaxis_title="Ejemplos Procesados",
        yaxis_title="Score (Precisión)",
        yaxis=dict(range=[min_y, 1.005]),
        xaxis=dict(range=[min(train_sizes)*0.9, max(train_sizes)*1.05]),
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

# --- LÓGICA DE SIMULACIÓN ---
def run_simulation(report, placeholder_bar, placeholder_plot):
    """Ejecuta la animación y actualiza el estado al finalizar."""
    lc = report.get('learning_curve', {})
    if not lc: return

    train_sizes = lc.get('train_sizes')
    train_scores = lc.get('train_scores')
    cv_scores = lc.get('cv_scores')
    if not cv_scores and 'test_scores' in lc: cv_scores = lc['test_scores']
    
    if not train_sizes: return

    # Animación
    for i in range(1, len(train_sizes) + 1):
        # Actualizar barra
        progress = i / len(train_sizes)
        placeholder_bar.progress(progress, text=f"Entrenando modelo... ({int(progress*100)}%)")
        
        # Actualizar gráfico
        fig = plot_learning_curve_plotly(train_sizes, train_scores, cv_scores, current_step=i)
        placeholder_plot.plotly_chart(fig, use_container_width=True)
        
        time.sleep(0.15) # Velocidad de animación

    # Finalizar
    placeholder_bar.empty() # Quitar barra
    st.toast("¡Entrenamiento finalizado con éxito!", icon="🚀")
    
    # Guardar en sesión que ya se entrenó
    st.session_state['svm_is_trained'] = True

# --- COMPONENTES VISUALES ---

def mostrar_estadisticas(report):
    metrics_raw = report.get('performance_metrics', {}).get('test', {})
    metrics_normalized = {}
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
                    c1.markdown(f"{metric_name}", help=METRIC_DEFINITIONS.get(metric_name, ""))
                    if isinstance(value, (int, float)):
                        c2.markdown(f"`{value:.4f}`")
                    else:
                        c2.markdown(f"`{value}`")

def generar_explicacion_matriz(cm, tipo=""):
    if not cm: return ""
    tn = cm.get('true_negatives', 0)
    fp = cm.get('false_positives', 0)
    fn = cm.get('false_negatives', 0)
    tp = cm.get('true_positives', 0)
    total = tn + fp + fn + tp
    return f"""
    **Análisis {tipo}:** (Total: {total})
    * ✅ **Aciertos: {tn + tp}**
    * ❌ **Errores: {fn + fp}** (FN: {fn}, FP: {fp})
    """

def mostrar_resultados_post_entrenamiento(report):
    """Muestra todo el contenido analítico después de la simulación."""
    
    # 1. Curva de Aprendizaje (Estática Final)
    lc = report.get('learning_curve', {})
    if lc:
        st.markdown("### 1. Curva de Aprendizaje")
        fig_lc = plot_learning_curve_plotly(
            lc.get('train_sizes'), 
            lc.get('train_scores'), 
            lc.get('cv_scores') if 'cv_scores' in lc else lc.get('test_scores')
        )
        st.plotly_chart(fig_lc, use_container_width=True)

    st.divider()

    # 2. Matrices de Confusión
    st.markdown("### 2. Matrices de Confusión")
    cm = report.get('confusion_matrix', {})
    
    col1, col2 = st.columns(2)
    with col1:
        fig_train = plot_confusion_matrix_plotly(cm.get('train'), "Matriz Entrenamiento")
        if fig_train: st.plotly_chart(fig_train, use_container_width=True)
        st.info(generar_explicacion_matriz(cm.get('train'), "Entrenamiento"))

    with col2:
        fig_test = plot_confusion_matrix_plotly(cm.get('test'), "Matriz Prueba")
        if fig_test: st.plotly_chart(fig_test, use_container_width=True)
        st.success(generar_explicacion_matriz(cm.get('test'), "Prueba"))
    
    st.divider()

    # 3. Rendimiento Global
    st.markdown("### 3. Rendimiento Global")
    c3, c4 = st.columns([2, 1])
    with c3:
        fig_bar = plot_metrics_comparison_plotly(report)
        if fig_bar: st.plotly_chart(fig_bar, use_container_width=True)
    with c4:
        fig_pie = plot_class_distribution_plotly(report)
        if fig_pie: st.plotly_chart(fig_pie, use_container_width=True)
    
    # 4. Tabla de Métricas
    st.divider()
    mostrar_estadisticas(report)

def mostrar():
    st.title("Support Vector Machine (SVM)")
    st.markdown("Plataforma de entrenamiento y análisis de vectores de soporte.")
    
    report, model = load_data()
    
    if not report:
        return

    # Inicializar estado si no existe
    if 'svm_is_trained' not in st.session_state:
        st.session_state['svm_is_trained'] = False

    # --- SECCIÓN SUPERIOR: ENTRENAMIENTO ---
    st.markdown("---")
    
    # Contenedor principal de acción
    col_act, col_viz = st.columns([1, 3])
    
    with col_act:
        st.markdown("#### Panel de Control")
        st.markdown("Inicie el proceso para generar el modelo y visualizar las métricas.")
        
        # Botón de acción
        # Si ya está entrenado, el botón permite "Re-entrenar"
        btn_label = "▶️ Iniciar Entrenamiento" if not st.session_state['svm_is_trained'] else "🔄 Re-entrenar Modelo"
        
        if st.button(btn_label, type="primary", use_container_width=True):
            # Resetear estado para forzar animación
            st.session_state['svm_is_trained'] = False
            # Crear placeholders
            with col_viz:
                bar_ph = st.empty()
                plot_ph = st.empty()
                run_simulation(report, bar_ph, plot_ph)
                # Al terminar run_simulation, se pone svm_is_trained = True
                st.rerun()

    # --- MOSTRAR RESULTADOS (Solo si ya se entrenó) ---
    if st.session_state['svm_is_trained']:
        mostrar_resultados_post_entrenamiento(report)
    else:
        with col_viz:
            # Estado inicial vacío o imagen placeholder
            st.info("El modelo está listo para ser entrenado. Presione el botón a la izquierda.")

if __name__ == "__main__":
    mostrar()