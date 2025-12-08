import streamlit as st
import pandas as pd
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# --- CONFIGURACIÓN Y CONSTANTES ---
# Mapeo para leer métricas sin importar si vienen como 'acc' o 'accuracy'
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
    """Busca el valor de una métrica de forma robusta."""
    if not metrics_dict: return 0.0
    # 1. Intento directo
    if long_name in metrics_dict: return metrics_dict[long_name]
    # 2. Intento por alias corto
    short_name = next((k for k, v in KEY_MAPPING.items() if v == long_name), None)
    if short_name and short_name in metrics_dict: return metrics_dict[short_name]
    return 0.0

def load_data():
    """Carga reporte y modelo."""
    json_path = 'models/svm_training_report.json'
    model_path = 'models/svm_model.pkl'
    
    report = None
    model = None

    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            report = json.load(f)
    else:
        st.error(f"⚠️ No se encontró: {json_path}")
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    
    return report, model

# --- SECCIÓN 1: TABLA DE MÉTRICAS ---
def mostrar_estadisticas(report):
    if not report: return
    metrics_raw = report.get('performance_metrics', {}).get('test', {})
    
    # Normalizar nombres para la tabla
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

# --- SECCIÓN 2: GRÁFICOS ---
def plot_confusion_matrix(cm_data, title):
    if not cm_data: return None
    try:
        cm_array = np.array([
            [cm_data.get('true_negatives',0), cm_data.get('false_positives',0)],
            [cm_data.get('false_negatives',0), cm_data.get('true_positives',0)]
        ])
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    annot_kws={"size": 10},
                    xticklabels=['Benigno', 'Maligno'], yticklabels=['Benigno', 'Maligno'])
        
        ax.set_ylabel('Realidad', fontsize=8)
        ax.set_xlabel('Predicción', fontsize=8)
        ax.set_title(title, fontsize=10, pad=8)
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
    **Análisis {tipo}:** ({total} casos)
    * ✅ **Aciertos: {tn + tp}**
    * ❌ **Errores: {fn + fp}** (FN: {fn}, FP: {fp})
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
    ax.bar(x - width/2, y_train, width, label='Train', color='#1f77b4')
    ax.bar(x + width/2, y_test, width, label='Test', color='#ff7f0e')
    
    ax.set_title('Comparación Train vs Test', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in target_metrics], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='lower right', fontsize=7)
    ax.bar_label(plt.gca().containers[0], padding=3, fmt='%.2f', fontsize=7)
    ax.bar_label(plt.gca().containers[1], padding=3, fmt='%.2f', fontsize=7)
    plt.tight_layout()
    return fig

def plot_class_distribution(report):
    cm = report.get('confusion_matrix', {}).get('test', {})
    if not cm: return None
    benignos = cm.get('true_negatives', 0) + cm.get('false_positives', 0)
    malignos = cm.get('true_positives', 0) + cm.get('false_negatives', 0)
    
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie([benignos, malignos], labels=['Benig', 'Malig'], 
           colors=['#66b3ff','#ff9999'], autopct='%1.0f%%', startangle=90, 
           textprops={'fontsize': 8})
    ax.set_title('Distribución (Test)', fontsize=9)
    plt.tight_layout()
    return fig

# --- LÓGICA DE ANIMACIÓN ---
def animate_learning_curve(report, placeholder):
    """Anima la curva sincronizada con una barra de progreso."""
    lc = report.get('learning_curve', {})
    if not lc: return
    
    train_sizes = lc.get('train_sizes')
    train_scores = lc.get('train_scores')
    cv_scores = lc.get('cv_scores')
    if not cv_scores and 'test_scores' in lc: cv_scores = lc['test_scores']
    
    if not train_sizes: return

    # Crear la barra de progreso encima del gráfico
    progress_bar = st.progress(0, text="Iniciando entrenamiento...")

    # Iterar y dibujar progresivamente
    for i in range(1, len(train_sizes) + 1):
        # Actualizar barra
        progress_pct = i / len(train_sizes)
        progress_bar.progress(progress_pct, text=f"Entrenando modelo... ({int(progress_pct*100)}%)")

        fig, ax = plt.subplots(figsize=(6, 3))
        
        # Dibujar hasta el punto actual
        ax.plot(train_sizes[:i], train_scores[:i], 'o-', color="#d62728", label="Entrenamiento", markersize=5, linewidth=2)
        ax.plot(train_sizes[:i], cv_scores[:i], 'o-', color="#2ca02c", label="Prueba (Validación)", markersize=5, linewidth=2)
        
        # Fijar ejes para evitar saltos
        ax.set_xlim(min(train_sizes)*0.9, max(train_sizes)*1.05)
        min_y = min(min(train_scores), min(cv_scores))
        ax.set_ylim(min_y*0.98, 1.005)
        
        ax.set_title("Evolución del Aprendizaje", fontsize=10)
        ax.set_xlabel("Ejemplos Procesados", fontsize=9)
        ax.set_ylabel("Precisión (Score)", fontsize=9)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # Actualizar gráfico
        placeholder.pyplot(fig)
        
        # Velocidad de animación
        time.sleep(0.15) 
        plt.close(fig)

    # Al terminar, limpiamos la barra y dejamos el gráfico final
    progress_bar.empty() # Borra la barra
    
    # Mostrar gráfico final estático
    fig_final = plot_learning_curve_static(report)
    placeholder.pyplot(fig_final)
    
    # Mensaje de éxito temporal
    st.toast("¡Entrenamiento completado exitosamente!", icon="✅")
def plot_learning_curve_static(report):
    lc = report.get('learning_curve', {})
    if not lc: return None
    train_sizes = lc.get('train_sizes')
    train_scores = lc.get('train_scores')
    cv_scores = lc.get('cv_scores')
    if not cv_scores and 'test_scores' in lc: cv_scores = lc['test_scores']
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(train_sizes, train_scores, 'o-', color="#d62728", label="Entrenamiento", markersize=5, linewidth=2)
    ax.plot(train_sizes, cv_scores, 'o-', color="#2ca02c", label="Prueba (Validación)", markersize=5, linewidth=2)

    ax.set_title("Curva de Aprendizaje Final (SVM)", fontsize=10)
    ax.set_xlabel("Ejemplos", fontsize=9)
    ax.set_ylabel("Score", fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

# --- FUNCIÓN PRINCIPAL DE VISUALIZACIÓN ---
def mostrar_graficos(report):
    if not report: return
    cm = report.get('confusion_matrix', {})
    
    with st.expander("📈 Visualización de Resultados", expanded=True):
        st.markdown("##### 1. Matrices de Confusión")
        # ... (código de matrices igual) ...
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
        # ... (código de rendimiento igual) ...
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
            
            c_btn, c_plot, _ = st.columns([1, 4, 1])
            
            with c_btn:
                st.markdown("<br><br><br>", unsafe_allow_html=True)
                run_anim = st.button("▶️ Entrenar Modelo", help="Visualizar proceso")
            
            with c_plot:
                plot_area = st.empty()
                
                if run_anim:
                    # La función animate ahora se encarga de la barra de progreso
                    animate_learning_curve(report, plot_area)
                else:
                    # Mostrar estado inicial vacío o mensaje
                    plot_area.info("Presiona 'Entrenar Modelo' para iniciar la simulación.")
def mostrar():
    st.title("Support Vector Machine (SVM)")
    st.markdown("Análisis de rendimiento: Vectores de Soporte.")
    report, model = load_data()
    mostrar_estadisticas(report)
    mostrar_graficos(report)

if __name__ == "__main__":
    mostrar()