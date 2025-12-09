import streamlit as st
from ui import SVM as svm
from ui import InferenciaSVM as inferencia # 
from ui import inicio  
from ui import MLP as mlp
from ui import MLP as mlp_analysis
from ui import inferenciaMLP as mlp_inference
from ui import comparacion
from ui import Dataset as dataset

st.set_page_config(
    page_title="Breast Cancer Diagnosis",
    layout="wide", 
    page_icon="🩺"
)
def main():
    st.sidebar.title("Navegación")
    pagina = st.sidebar.radio(
        "Ir a:",
        (
            "Inicio", 
            "Explorador de Datos",
            "MLP (análisis)",  
            "MLP (probador)", 
            "SVM (Análisis)",        
            "SVM (Probador)",        
            "Comparación", 
            "Analítica Web"
        )
    )

    if pagina == "Inicio":
        # ... inicio ...
        inicio.mostrar()

        pass
    elif pagina == "SVM (Análisis)":
        svm.mostrar()
    elif pagina == "SVM (Probador)": 
        inferencia.mostrar()
    elif pagina == "Comparación":
        comparacion.mostrar()
    elif pagina == "Analítica Web":
        pass
    elif pagina == "MLP (análisis)":
        mlp_analysis.mostrar()
    elif pagina == "MLP (probador)":
        mlp_inference.mostrar()
    elif pagina == "Explorador de Datos":
        dataset.mostrar()

if __name__ == "__main__":
    main()