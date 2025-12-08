import streamlit as st
import pandas as pd
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURACIÓN Y CONSTANTES ---
KEY_MAPPING = {
    "acc": "accuracy", "prec": "precision", "rec": "recall",
    "f1": "f1_score", "roc": "auc", "bal": "balanced_accuracy",
    "ap": "avg_precision", "spec": "specificity", "kappa": "cohens_kappa"
}

METRIC_DEFINITIONS = {
    "accuracy": "Exactitud: % de predicciones correctas totales.",
    "precision": "Precisión: De los predichos positivos, cuántos son reales.",
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
    # Solo necesitamos el JSON para las métricas, pero mantenemos la estructura
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            report = json.load(f)
        return report
    else:
        st.error(f"⚠️ No se encontró: {json_path}")
        return None

def mostrar_estadisticas(report):
    if not report: return
    metrics_raw = report.get('performance_metrics', {}).get('test', {})
    metrics_normalized = {}
    for k, v in metrics_raw.items():
        new_key = KEY_MAPPING.get(k, k) 
        metrics_normalized[new_key] = v

    with st.expander("📊 Métricas del Modelo (Clic para desplegar)", expanded=False):
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

def plot_confusion_matrix(cm_data, title):
    if not cm_data: return None
    try:
        cm_array = np.array([
            [cm_data.get('true_negatives',0), cm_data.get('false_positives',0)],
            [cm_data.get('false_negatives',0), cm_data.get('true_positives',0)]
        ])
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    annot_kws={"size": 12},
                    xticklabels=['Benigno', 'Maligno'], yticklabels=['Benigno', 'Maligno'])
        
        ax.set_ylabel('Realidad', fontsize=9)
        ax.set_xlabel('Predicción', fontsize=9)
        ax.set_title(title, fontsize=11, pad=10)
        plt.tight_layout()
        return fig
    except: return None

def generar_explicacion_matriz(cm, tipo=""):
    if not cm: return ""
    tn = cm.get('true_negatives', 0)
    fp = cm.get('false_positives', 0)
    fn = cm.get('false_negatives', 0)
    tp = cm.get('true_positives', 0)
    total = tn + fp + fn + tp
    
    return f"""
    **Análisis de {tipo}:** (Total: {total} casos)
    * ✅ **Aciertos ({tn + tp}):**
        * Detectó **{tn}** casos benignos.
        * Detectó **{tp}** casos malignos.
    * ❌ **Errores ({fn + fp}):**
        * **Falsos Negativos ({fn}):** Casos cáncer no detectados.
        * **Falsos Positivos ({fp}):** Falsas alarmas.
    """

def plot_metrics_comparison(report):
    target_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    perf = report.get('performance_metrics', {})
    train_data = perf.get('train', {})
    test_data = perf.get('test', {})

    y_train = [get_metric_value(train_data, m) for m in target_metrics]
    y_test  = [get_metric_value(test_data, m) for m in target_metrics]
    
    x = np.arange(len(target_metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(5, 3))
    rects1 = ax.bar(x - width/2, y_train, width, label='Train', color='#1f77b4')
    rects2 = ax.bar(x + width/2, y_test, width, label='Test', color='#ff7f0e')
    
    ax.set_title('Comparación Train vs Test', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in target_metrics], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='lower right', fontsize=8)
    ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=7)
    ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=7)
    plt.tight_layout()
    return fig

def plot_class_distribution(report):
    cm = report.get('confusion_matrix', {}).get('test', {})
    if not cm: return None
    benignos = cm.get('true_negatives', 0) + cm.get('false_positives', 0)
    malignos = cm.get('true_positives', 0) + cm.get('false_negatives', 0)
    
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie([benignos, malignos], labels=['Benig', 'Malig'], 
           colors=['#66b3ff','#ff9999'], autopct='%1.0f%%', startangle=90)
    ax.set_title('Distribución (Test)', fontsize=10)
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
    ax.plot(train_sizes, train_scores, 'o-', color="#d62728", label="Train", markersize=4)
    ax.plot(train_sizes, cv_scores, 'o-', color="#2ca02c", label="CV", markersize=4)

    ax.set_title("Curva de Aprendizaje", fontsize=10)
    ax.set_xlabel("Ejemplos", fontsize=9)
    ax.set_ylabel("Score", fontsize=9)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

def mostrar_graficos(report):
    if not report: return
    cm = report.get('confusion_matrix', {})
    
    with st.expander("📈 Visualización de Resultados", expanded=True): # Expanded True para que se vea directo
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
        st.markdown("##### 2. Rendimiento Global")
        c5, c6 = st.columns([2, 1])
        with c5:
            fig = plot_metrics_comparison(report)
            if fig: st.pyplot(fig, use_container_width=True)
        with c6:
            fig = plot_class_distribution(report)
            if fig: st.pyplot(fig, use_container_width=True)

        if 'learning_curve' in report:
            st.divider()
            st.markdown("##### 3. Curva de Aprendizaje")
            _, center, _ = st.columns([1, 4, 1])
            with center:
                fig_lc = plot_learning_curve(report)
                if fig_lc: st.pyplot(fig_lc, use_container_width=True)

def mostrar():
    st.title("Análisis SVM: Métricas y Gráficos")
    st.markdown("Visualización detallada del rendimiento del modelo.")
    report = load_data()
    mostrar_estadisticas(report)
    mostrar_graficos(report)

if __name__ == "__main__":
    mostrar()