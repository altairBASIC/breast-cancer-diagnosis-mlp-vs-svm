import streamlit as st
from ui import SVM as svm
from ui import InferenciaSVM as svm_inference
from ui import inicio
from ui import MLP as mlp_analysis
from ui import inferenciaMLP as mlp_inference
from ui import comparacion
from ui import Dataset as dataset
from ui import AnaliticaWeb as analytics_web


st.set_page_config(
    page_title="Breast Cancer Diagnosis",
    layout="wide",
    page_icon="🩺",
)


PAGES = {
    "Inicio": inicio.mostrar,
    "Explorador de Datos": dataset.mostrar,
    "MLP (análisis)": mlp_analysis.mostrar,
    "MLP (probador)": mlp_inference.mostrar,
    "SVM (Análisis)": svm.mostrar,
    "SVM (Probador)": svm_inference.mostrar,
    "Comparación": comparacion.mostrar,
    "Analítica Web": analytics_web.mostrar,
}


def main() -> None:
    st.sidebar.title("Navegación")
    pagina = st.sidebar.radio("Ir a:", list(PAGES.keys()))

    pagina_funcion = PAGES.get(pagina)
    if pagina_funcion is not None:
        pagina_funcion()
    else:
        st.error("La página seleccionada no está disponible.")


if __name__ == "__main__":
    main()
