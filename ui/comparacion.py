import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURACIÓN ---
# Colores consistentes: SVM (Azul), MLP (Morado)
COLORS = {"SVM": "#1f77b4", "MLP": "#6A1B9A"}

METRIC_KEYS = {
    "accuracy": ["accuracy", "acc", "test_accuracy"],
    "precision": ["precision", "prec", "ppv"],
    "recall": ["recall", "rec", "sensitivity"],
    "f1_score": ["f1_score", "f1", "f1-score"],
    "auc": ["auc", "roc", "roc_auc"]
}

def get_metric(data, metric_key):
    """Busca una métrica en el diccionario de forma robusta."""
    if not data: return 0.0
    candidates = METRIC_KEYS.get(metric_key, [metric_key])
    for key in candidates:
        if key in data:
            return data[key]
    return 0.0

def load_reports():
    """Carga ambos reportes JSON."""
    paths = {
        "SVM": "models/svm_training_report.json",
        "MLP": "models/mlp_training_report.json"
    }
    data = {}
    for model_name, path in paths.items():
        if os.path.exists(path):
            with open(path, 'r') as f:
                data[model_name] = json.load(f)
        else:
            data[model_name] = None
    return data

def plot_benchmark(df_metrics):
    """Genera un gráfico de barras agrupado para comparar."""
    if df_metrics.empty: return None
    
    # Preparamos datos para plotear
    metrics = df_metrics.index.tolist()
    svm_scores = df_metrics['SVM'].tolist()
    mlp_scores = df_metrics['MLP'].tolist()
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 4))
    rects1 = ax.bar(x - width/2, svm_scores, width, label='SVM', color=COLORS['SVM'])
    rects2 = ax.bar(x + width/2, mlp_scores, width, label='MLP', color=COLORS['MLP'])
    
    ax.set_ylabel('Puntaje (0-1)')
    ax.set_title('Rendimiento: SVM vs MLP')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0.85, 1.02) # Zoom para ver diferencias pequeñas
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Etiquetas
    ax.bar_label(rects1, padding=3, fmt='%.3f', fontsize=8)
    ax.bar_label(rects2, padding=3, fmt='%.3f', fontsize=8)
    
    plt.tight_layout()
    return fig

def mostrar():
    st.title("⚖️ Comparación de Modelos")
    st.markdown("Análisis competitivo entre **Support Vector Machine** y **Perceptrón Multicapa**.")
    
    data = load_reports()
    
    # Verificar si tenemos datos
    if not data["SVM"] or not data["MLP"]:
        st.error("⚠️ Faltan reportes de entrenamiento. Asegúrate de haber ejecutado los notebooks de SVM y MLP.")
        return

    # --- 1. PREPARACIÓN DE DATOS ---
    # Extraer métricas de Test
    svm_test = data["SVM"].get("performance_metrics", {}).get("test", {})
    mlp_test = data["MLP"].get("performance_metrics", {}).get("test", {})
    
    # Extraer Tiempos
    t_svm = data["SVM"].get("training_info", {}).get("training_time_seconds", 0)
    # A veces MLP guarda tiempo en "training" o "training_info"
    t_mlp = data["MLP"].get("training", {}).get("training_time_seconds", 0)

    # Crear DataFrame Comparativo
    metrics_to_compare = ["accuracy", "recall", "precision", "f1_score", "auc"]
    
    df = pd.DataFrame(index=metrics_to_compare, columns=["SVM", "MLP"])
    
    for m in metrics_to_compare:
        df.loc[m, "SVM"] = get_metric(svm_test, m)
        df.loc[m, "MLP"] = get_metric(mlp_test, m)

    # Calcular Diferencia
    df["Diferencia"] = df["SVM"] - df["MLP"]
    df["Ganador"] = df.apply(lambda row: "SVM" if row["SVM"] >= row["MLP"] else "MLP", axis=1)

    # --- 2. TARJETAS DE RESUMEN (KPIs) ---
    st.subheader("🏆 Resumen Ejecutivo")
    
    c1, c2, c3 = st.columns(3)
    
    # Ganador en Exactitud
    best_acc_model = df.loc["accuracy", "Ganador"]
    best_acc_val = df.loc["accuracy", best_acc_model]
    
    # Ganador en Sensibilidad (Vital para cáncer)
    best_rec_model = df.loc["recall", "Ganador"]
    best_rec_val = df.loc["recall", best_rec_model]
    
    # Modelo más rápido
    fastest_model = "SVM" if t_svm < t_mlp else "MLP"
    fastest_time = min(t_svm, t_mlp)

    with c1:
        st.metric("Mejor Exactitud", f"{best_acc_model}", f"{best_acc_val:.2%}")
    with c2:
        st.metric("Mejor Sensibilidad", f"{best_rec_model}", f"{best_rec_val:.2%}")
    with c3:
        st.metric("Entrenamiento más Rápido", f"{fastest_model}", f"{fastest_time:.2f} s")

    st.divider()

    # --- 3. GRÁFICO COMPARATIVO ---
    col_graf, col_tabla = st.columns([3, 2])
    
    with col_graf:
        st.markdown("#### 📊 Duelo de Métricas")
        fig = plot_benchmark(df)
        st.pyplot(fig, use_container_width=True)
        
    with col_tabla:
        st.markdown("#### 📝 Detalle Numérico")
        # Estilizamos la tabla para resaltar al ganador
        st.dataframe(
            df.style.highlight_max(axis=1, subset=["SVM", "MLP"], color='#d1e7dd', props='font-weight:bold;'),
            use_container_width=True,
            height=300
        )

    # --- 4. ANÁLISIS DE TIEMPOS ---
    st.divider()
    st.markdown("#### ⏱️ Eficiencia Computacional")
    
    # Barra de progreso comparativa manual
    total_time = t_svm + t_mlp
    pct_svm = t_svm / total_time
    pct_mlp = t_mlp / total_time
    
    c_t1, c_t2 = st.columns(2)
    with c_t1:
        st.info(f"**SVM:** {t_svm:.4f} segundos")
    with c_t2:
        st.info(f"**MLP:** {t_mlp:.4f} segundos")

    # Visualización lineal del tiempo
    st.caption("Proporción de tiempo de entrenamiento:")
    st.markdown(f"""
        <div style="display: flex; width: 100%; height: 20px; border-radius: 10px; overflow: hidden;">
            <div style="background-color: {COLORS['SVM']}; width: {pct_svm*100}%;" title="SVM"></div>
            <div style="background-color: {COLORS['MLP']}; width: {pct_mlp*100}%;" title="MLP"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8em; color: gray;">
            <span>SVM ({pct_svm:.1%})</span>
            <span>MLP ({pct_mlp:.1%})</span>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    mostrar()