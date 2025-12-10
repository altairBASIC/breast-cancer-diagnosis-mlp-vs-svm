import streamlit as st
import pandas as pd
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURACIÓN Y CONSTANTES ---
# Listas de posibles claves para cada métrica (SOLUCIÓN AL KEYERROR)
METRIC_KEYS = {
    "accuracy": ["accuracy", "acc", "test_accuracy"],
    "precision": ["precision", "prec", "ppv"],
    "recall": ["recall", "rec", "sensitivity"],
    "f1_score": ["f1_score", "f1", "f1-score"],
    "auc": ["auc", "roc", "roc_auc"],
    "specificity": ["specificity", "spec"]
}

METRIC_DEFINITIONS = {
    "accuracy": "Exactitud: Porcentaje total de predicciones correctas.",
    "precision": "Precisión: De los predichos positivos, cuántos eran reales.",
    "recall": "Sensibilidad: Capacidad de detectar casos positivos reales.",
    "f1_score": "F1-Score: Balance armónico entre precisión y recall.",
    "auc": "AUC: Capacidad de distinción entre clases.",
    "specificity": "Especificidad: Capacidad de detectar casos negativos.",
    "balanced_accuracy": "Promedio de recall por clase.",
    "mcc": "Coeficiente de Matthews (calidad global).",
    "cohens_kappa": "Acuerdo excluyendo el azar."
}

def get_best_metric_value(metrics_dict, metric_type):
    """Busca el valor de una métrica probando varias claves posibles."""
    if not metrics_dict: return 0.0
    candidates = METRIC_KEYS.get(metric_type, [metric_type])
    for key in candidates:
        if key in metrics_dict:
            return metrics_dict[key]
    return 0.0

@st.cache_resource(show_spinner=False)
def load_data():
    # Rutas específicas para el MLP
    json_path = 'models/mlp_training_report.json'
    model_path = 'models/mlp_model.pkl'
    
    report = None
    model = None

    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            report = json.load(f)
    else:
        st.error(f"⚠️ No se encontró el reporte en: {json_path}")
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        st.error(f"⚠️ No se encontró el modelo en: {model_path}")
        
    return report, model

def mostrar_estadisticas(report):
    if not report: return
    metrics_raw = report.get('performance_metrics', {}).get('test', {})
    
    # Preparamos las métricas para la tabla con nombres bonitos
    pretty_metrics = {}
    for key, val in metrics_raw.items():
        pretty_name = key
        for main_name, variants in METRIC_KEYS.items():
            if key in variants:
                pretty_name = main_name
                break
        pretty_metrics[pretty_name] = val

    with st.expander("📊 Métricas del Modelo (Clic para desplegar)", expanded=False):
        _, col_centro, _ = st.columns([1, 2, 1])
        with col_centro:
            with st.container(border=True):
                h1, h2 = st.columns([2, 1])
                h1.markdown("**Métrica**")
                h2.markdown("**Valor (Test)**")
                st.divider() 
                for metric_name, value in pretty_metrics.items():
                    c1, c2 = st.columns([2, 1])
                    c1.markdown(f"{metric_name}", help=METRIC_DEFINITIONS.get(metric_name, ""))
                    if isinstance(value, (int, float)):
                        c2.markdown(f"`{value:.4f}`")
                    else:
                        c2.markdown(f"`{value}`")

def plot_confusion_matrix(cm_data, title):
    if not cm_data:
        return None
    try:
        cm_array = np.array([
            [cm_data.get("true_negatives", 0), cm_data.get("false_positives", 0)],
            [cm_data.get("false_negatives", 0), cm_data.get("true_positives", 0)],
        ])
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        # Usamos 'Purples' para diferenciar del SVM (que es azul)
        sns.heatmap(
            cm_array,
            annot=True,
            fmt="d",
            cmap="Purples",
            cbar=False,
            ax=ax,
            annot_kws={"size": 10},
            xticklabels=["Benigno", "Maligno"],
            yticklabels=["Benigno", "Maligno"],
        )

        ax.set_ylabel("Realidad", fontsize=8)
        ax.set_xlabel("Predicción", fontsize=8)
        ax.set_title(title, fontsize=10, pad=8)
        ax.tick_params(axis="both", which="major", labelsize=8)
        plt.tight_layout()
        return fig
    except Exception:
        return None

def generar_explicacion_matriz(cm, tipo=""):
    if not cm: return ""
    tn = cm.get('true_negatives', 0)
    fp = cm.get('false_positives', 0)
    fn = cm.get('false_negatives', 0)
    tp = cm.get('true_positives', 0)
    total = tn + fp + fn + tp
    
    return f"""
    **{tipo} ({total}):**
    * ✅ Aciertos: **{tn + tp}**
    * ❌ Errores: **{fn + fp}** (FN: {fn}, FP: {fp})
    """

def plot_metrics_comparison(report):
    target_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    perf = report.get('performance_metrics', {})
    train_data = perf.get('train', {})
    test_data = perf.get('test', {})

    # Extracción robusta de datos
    y_train = [get_best_metric_value(train_data, m) for m in target_metrics]
    y_test  = [get_best_metric_value(test_data, m) for m in target_metrics]
    
    x = np.arange(len(target_metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(5, 3))
    # Colores personalizados para MLP
    rects1 = ax.bar(x - width/2, y_train, width, label='Train', color='#6A1B9A') # Morado oscuro
    rects2 = ax.bar(x + width/2, y_test, width, label='Test', color='#AB47BC')   # Morado claro
    
    ax.set_title('Comparación Train vs Test', fontsize=10)
    ax.set_xticks(x)
    clean_labels = [m.replace('_', ' ').capitalize() for m in target_metrics]
    ax.set_xticklabels(clean_labels, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='lower right', fontsize=7)
    
    ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=7)
    ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=7)
    
    plt.tight_layout()
    return fig

def plot_class_distribution(report):
    cm = report.get('confusion_matrix', {}).get('test', {})
    if not cm: return None
    benignos = cm.get('true_negatives', 0) + cm.get('false_positives', 0)
    malignos = cm.get('true_positives', 0) + cm.get('false_negatives', 0)
    
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.pie([benignos, malignos], labels=['Benig', 'Malig'], 
           colors=['#E1BEE7','#BA68C8'], autopct='%1.0f%%', startangle=90, # Tonos morados
           textprops={'fontsize': 8})
    ax.set_title('Distribución (Test)', fontsize=9)
    plt.tight_layout()
    return fig

def plot_learning_curve(report):
    lc = report.get('learning_curve', {})
    if not lc: return None
    train_sizes = lc.get('train_sizes')
    train_scores = lc.get('train_scores')
    cv_scores = lc.get('cv_scores')
    if not cv_scores and 'test_scores' in lc: cv_scores = lc['test_scores']
    if not train_sizes: return None

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(train_sizes, train_scores, 'o-', color="#4A148C", label="Train", markersize=4)
    ax.plot(train_sizes, cv_scores, 'o-', color="#7B1FA2", label="CV", markersize=4)

    ax.set_title("Curva de Aprendizaje (MLP)", fontsize=10)
    ax.set_xlabel("Ejemplos", fontsize=9)
    ax.set_ylabel("Score", fontsize=9)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

def mostrar_graficos(report):
    if not report: return
    cm = report.get('confusion_matrix', {})
    
    with st.expander("📈 Visualización de Resultados", expanded=True):
        # 1. Matrices de Confusión
        st.markdown("##### 1. Matrices de Confusión")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            fig = plot_confusion_matrix(cm.get('train'), "Matriz Entrenamiento")
            if fig: st.pyplot(fig, use_container_width=True)
            st.info(generar_explicacion_matriz(cm.get('train'), "Entrenamiento"))

        with col2:
            fig = plot_confusion_matrix(cm.get('test'), "Matriz Prueba")
            if fig: st.pyplot(fig, use_container_width=True)
            st.success(generar_explicacion_matriz(cm.get('test'), "Prueba"))
            
        st.divider()
        
        # 2. Rendimiento Global
        st.markdown("##### 2. Rendimiento Global")
        c5, c6 = st.columns([2, 1])
        with c5:
            fig = plot_metrics_comparison(report)
            if fig: st.pyplot(fig, use_container_width=True)
        with c6:
            fig = plot_class_distribution(report)
            if fig: st.pyplot(fig, use_container_width=True)

        # 3. Curva de Aprendizaje
        if 'learning_curve' in report:
            st.divider()
            st.markdown("##### 3. Curva de Aprendizaje")
            _, center, _ = st.columns([1, 4, 1])
            with center:
                fig_lc = plot_learning_curve(report)
                if fig_lc: st.pyplot(fig_lc, use_container_width=True)

def mostrar():
    st.caption("Inicio > MLP (análisis)")
    st.markdown(
        '<h1 style="color:#4A148C;">Red Neuronal (MLP)</h1>',
        unsafe_allow_html=True,
    )
    st.markdown("Esta vista muestra métricas de entrenamiento y prueba del modelo MLP sobre el dataset preprocesado.")
    st.caption("Sugerencia: revise primero las matrices de confusión y luego la curva de aprendizaje para entender el comportamiento del modelo.")
    report, model = load_data()
    mostrar_estadisticas(report)
    mostrar_graficos(report)

if __name__ == "__main__":
    mostrar()