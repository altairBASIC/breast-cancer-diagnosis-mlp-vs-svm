import streamlit as st
from PIL import Image
import os

def mostrar():
    # --- TÍTULO PRINCIPAL ---
    st.caption("Inicio")
    st.markdown(
        """
        <h1 style="color:#4A148C; margin-bottom:0.1rem;">
            🔬 Sistema de Diagnóstico de Cáncer de Mama
        </h1>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <h3 style="color:#8E24AA; margin-top:0;">
            Comparativa de Modelos de Machine Learning: MLP vs SVM
        </h3>
        """,
        unsafe_allow_html=True,
    )
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
                <h3 style="color:#000000;">Soporte a la Decisión Médica</h3>
                <p style="color:#000000;">Inteligencia Artificial aplicada a la salud.</p>
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
        st.markdown(
            """
            <div style="background-color:#F3E5F5; padding: 1rem; border-radius: 0.75rem; min-height: 140px;">
                <h4 style="color:#4A148C; margin-top:0; margin-bottom:0.5rem;">1. Datos</h4>
                <p style="margin:0;">Gestión y almacenamiento de registros clínicos estructurados usando <strong>MongoDB</strong>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col_m2:
        st.markdown(
            """
            <div style="background-color:#EDE7F6; padding: 1rem; border-radius: 0.75rem; min-height: 140px;">
                <h4 style="color:#4A148C; margin-top:0; margin-bottom:0.5rem;">2. Proceso</h4>
                <p style="margin:0;">Limpieza, normalización (<strong>StandardScaler</strong>) y división de datos con <strong>Scikit-learn</strong>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col_m3:
        st.markdown(
            """
            <div style="background-color:#F3E5F5; padding: 1rem; border-radius: 0.75rem; min-height: 140px;">
                <h4 style="color:#4A148C; margin-top:0; margin-bottom:0.5rem;">3. Modelado</h4>
                <p style="margin:0;">Entrenamiento y comparación de modelos <strong>MLP</strong> (Red Neuronal) y <strong>SVM</strong> (Vectores de Soporte).</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col_m4:
        st.markdown(
            """
            <div style="background-color:#EDE7F6; padding: 1rem; border-radius: 0.75rem; min-height: 140px;">
                <h4 style="color:#4A148C; margin-top:0; margin-bottom:0.5rem;">4. Despliegue</h4>
                <p style="margin:0;">Interfaz interactiva para inferencia en tiempo real desarrollada con <strong>Streamlit</strong>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # --- MODELOS COMPARADOS ---
    st.subheader("🤖 Modelos Implementados")
    
    c_mlp, c_svm = st.columns(2)
    
    with c_mlp:
        st.markdown(
            """
            <div style="background-color:#F3E5F5; padding: 1rem 1.2rem; border-radius: 0.75rem; min-height: 170px;">
                <h3 style="color:#4A148C; margin-top:0; margin-bottom:0.5rem;">MLP (Perceptrón Multicapa)</h3>
                <p style="margin:0 0 0.4rem 0;">Red neuronal artificial <em>feedforward</em>.</p>
                <ul style="padding-left:1.2rem; margin:0;">
                    <li>Capaz de modelar relaciones no lineales complejas.</li>
                    <li>Ideal para patrones profundos en los datos.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with c_svm:
        st.markdown(
            """
            <div style="background-color:#EDE7F6; padding: 1rem 1.2rem; border-radius: 0.75rem; min-height: 170px;">
                <h3 style="color:#4A148C; margin-top:0; margin-bottom:0.5rem;">SVM (Support Vector Machine)</h3>
                <p style="margin:0 0 0.4rem 0;">Algoritmo de aprendizaje supervisado robusto.</p>
                <ul style="padding-left:1.2rem; margin:0;">
                    <li>Busca el hiperplano óptimo para separar las clases.</li>
                    <li>Alta eficacia en espacios de dimensiones altas.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.success("👈 **Utilice el menú lateral para navegar entre el Análisis de Modelos y el Probador de Casos.**")

if __name__ == "__main__":
    mostrar()