import streamlit as st
from PIL import Image
import os

def mostrar():
    # --- TÍTULO PRINCIPAL ---
    st.caption("Ruta: Inicio")
    st.title("🔬 Sistema de Diagnóstico de Cáncer de Mama")
    st.markdown("### Comparativa de Modelos de Machine Learning: MLP vs SVM")
    st.markdown("---")

    # --- INTRODUCCIÓN ---
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### 🎯 Objetivo del Proyecto
        El diagnóstico temprano y preciso del cáncer de mama es crucial para mejorar la supervivencia de los pacientes.
        
        Este sistema utiliza herramientas avanzadas de **Ciencia de Datos** y **Machine Learning** para asistir a profesionales médicos en la clasificación de tumores (Benignos o Malignos) basándose en características nucleares de células obtenidas mediante biopsia (FNA).
        
        **Dataset utilizado:** *Breast Cancer Wisconsin (Diagnostic)*
        """)
        
        st.info("""
        **Enfoque Técnico:**
        El proyecto implementa un pipeline completo de MLOps: desde la gestión de datos, preprocesamiento y entrenamiento, hasta el despliegue de esta interfaz interactiva.
        """)

    with col2:
        # Aquí podrías poner una imagen representativa si tienes una en la carpeta assets/
        # Si no, usamos un placeholder o un gráfico de streamlit
        st.markdown(
            """
            <div style="background-color: #693A66; padding: 20px; border-radius: 10px; text-align: center;">
                <span style="font-size: 50px;">🩺</span>
                <h3>Soporte a la Decisión Médica</h3>
                <p>Inteligencia Artificial aplicada a la salud.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

    st.markdown("---")

    # --- METODOLOGÍA (Pipeline) ---
    st.subheader("🛠️ Arquitectura del Sistema")
    st.markdown("El flujo de trabajo se divide en 4 etapas fundamentales:")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.markdown("#### 1. Datos")
        st.markdown("Gestión y almacenamiento de registros clínicos estructurados usando **MongoDB**.")
    
    with col_m2:
        st.markdown("#### 2. Proceso")
        st.markdown("Limpieza, normalización (StandardScaler) y división de datos con **Scikit-learn**.")
    
    with col_m3:
        st.markdown("#### 3. Modelado")
        st.markdown("Entrenamiento y comparación de modelos **MLP** (Red Neuronal) y **SVM** (Vectores de Soporte).")
    
    with col_m4:
        st.markdown("#### 4. Despliegue")
        st.markdown("Interfaz interactiva para inferencia en tiempo real desarrollada con **Streamlit**.")

    st.markdown("---")

    # --- MODELOS COMPARADOS ---
    st.subheader("🤖 Modelos Implementados")
    
    c_mlp, c_svm = st.columns(2)
    
    with c_mlp:
        with st.container(border=True):
            st.markdown("### MLP (Perceptrón Multicapa)")
            st.markdown("""
            Red neuronal artificial feedforward.
            * Capaz de modelar relaciones no lineales complejas.
            * Ideal para patrones profundos en los datos.
            """)
    
    with c_svm:
        with st.container(border=True):
            st.markdown("### SVM (Support Vector Machine)")
            st.markdown("""
            Algoritmo de aprendizaje supervisado robusto.
            * Busca el hiperplano óptimo para separar las clases.
            * Alta eficacia en espacios de dimensiones altas.
            """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.success("👈 **Utilice el menú lateral para navegar entre el Análisis de Modelos y el Probador de Casos.**")

if __name__ == "__main__":
    mostrar()