# Pipeline de Machine Learning para Diagnóstico de Cáncer de Mama: MLP vs SVM

## Descripción del Proyecto

Sistema completo de diagnóstico asistido por inteligencia artificial que compara el rendimiento de dos familias de modelos de clasificación —**Perceptrón Multicapa (MLP)** y **Máquina de Vectores de Soporte (SVM)**— para predecir si un tumor de mama es benigno o maligno.

El proyecto implementa un pipeline end-to-end de Machine Learning con las siguientes etapas:

- Gestión de datos con MongoDB
- Preprocesamiento y normalización de características (StandardScaler)
- Entrenamiento y comparación de modelos MLP y SVM
- Evaluación exhaustiva con métricas clínicas (Accuracy, Precision, Recall, F1-Score, AUC-ROC)
- Despliegue de un dashboard interactivo con Streamlit para visualización y predicción en tiempo real

**Resultados alcanzados:**

- SVM: 98.2% Accuracy, AUC 0.996, Tiempo: 1.44s (probability=True)
- MLP: 97.4% Accuracy, AUC 0.997, Tiempo: 0.048s

Proyecto académico desarrollado para el curso **INFB6052 - Herramientas para Cs. de Datos**, demostrando competencias en MLOps elemental, ingeniería de datos y despliegue de sistemas predictivos.

## Objetivos

### Objetivo General

Desarrollar un pipeline completo y un prototipo funcional para el diagnóstico de cáncer de mama, abarcando desde la gestión de datos tabulares y el entrenamiento comparativo de modelos de clasificación, hasta el despliegue de una interfaz interactiva de predicción.

### Objetivos Específicos

1. Implementar un sistema de gestión de datos utilizando MongoDB para almacenar los registros de pacientes y sus características clínicas del dataset.
2. Desarrollar un pipeline de preprocesamiento de datos, incluyendo escalado de características (StandardScaler) para preparar los datos numéricos para el entrenamiento de los modelos.
3. Entrenar y comparar el rendimiento de dos modelos de clasificación: un Perceptrón Multicapa (MLP) y una Máquina de Vectores de Soporte (SVM).
4. Desarrollar y desplegar una aplicación web interactiva con Streamlit que permita al usuario ingresar valores de características y obtener predicciones diagnósticas en tiempo real.

## Estructura del Proyecto

```
breast-cancer-diagnosis-mlp-vs-svm/
│
├── app.py                         # Aplicación Streamlit principal con dashboard comparativo
├── install.ps1                    # Script de instalación automatizado (Windows)
├── requirements.txt               # Dependencias del proyecto
├── README.md                      # Documentación del proyecto
├── setup_log.txt                  # Log de configuración y ejecución
│
├── data/                          # Dataset y archivos de datos
│   ├── breast_cancer.csv          # Dataset descargado de UCI ML Repository
│   ├── data_summary.txt           # Resumen estadístico del dataset
│   └── processed/                 # Datos preprocesados (escalados y divididos)
│       ├── X_train.npy            # Features de entrenamiento (455 muestras)
│       ├── X_test.npy             # Features de prueba (114 muestras)
│       ├── y_train.npy            # Etiquetas de entrenamiento
│       ├── y_test.npy             # Etiquetas de prueba
│       ├── train_data.csv         # Datos de entrenamiento en formato CSV
│       ├── test_data.csv          # Datos de prueba en formato CSV (usado por app.py)
│       ├── feature_info.json      # Nombres y metadata de las 30 características
│       └── preprocessing_report.json  # Reporte completo del preprocesamiento
│
├── docs/                          # Documentación del proyecto
│   ├── instrucciones.md
│   ├── narrativa.md
│   ├── prompts_ia.txt
│   └── responsabilidades.md
│
├── models/                        # Modelos entrenados y reportes
│   ├── mlp_model.pkl              # Modelo MLP entrenado (AUC 0.995)
│   ├── svm_model.pkl              # Modelo SVM entrenado (AUC 0.995)
│   ├── mlp_training_report.json   # Métricas detalladas del MLP
│   ├── svm_training_report.json   # Métricas detalladas del SVM
│   └── scalers/                   # Objetos de preprocesamiento
│       ├── standard_scaler.pkl    # StandardScaler ajustado
│       └── label_encoder.pkl      # LabelEncoder (B→0, M→1)
│
├── notebooks/                     # Jupyter notebooks para desarrollo
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_mlp_training.ipynb
│   └── 03_svm_training.ipynb
│
├── scripts/                       # Scripts de Python para el pipeline
│   ├── setup_environment.py       # Configuración y verificación del entorno
│   ├── download_dataset.py        # Descarga del dataset desde UCI
│   ├── preprocessing.py           # Pipeline completo de preprocesamiento
│   ├── train_models.py            # Entrenamiento de MLP y SVM
│   └── load_to_mongo.py           # Carga de datos a MongoDB
│
└── ui/                            # Recursos para la interfaz de usuario
    ├── __init__.py
    ├── AnaliticaWeb.py
    ├── common_features.py
    ├── comparacion.py
    ├── Dataset.py
    ├── inferenciaMLP.py
    ├── InferenciaSVM.py
    ├── inicio.py
    ├── MLP.py
    └── SVM.py
```

## Dataset

**Nombre:** Breast Cancer Wisconsin (Diagnostic)

**Fuente:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

**Características:**

- 569 instancias
- 30 características numéricas
- 2 clases: Maligno (M) y Benigno (B)

**Atributos:** El dataset contiene mediciones computadas de imágenes digitalizadas de aspiración con aguja fina (FNA) de masas mamarias, describiendo características de los núcleos celulares presentes en la imagen.

## Tecnologías Utilizadas

### Core

- **Python 3.10+**: Lenguaje base del proyecto
- **scikit-learn 1.3.2**: Entrenamiento de modelos (MLPClassifier, SVC)
- **pandas 2.1.4**: Manipulación y análisis de datos
- **numpy 1.26.2**: Operaciones numéricas

### Almacenamiento

- **MongoDB 4.6+**: Base de datos NoSQL para gestión de datos
- **pymongo 4.6.1**: Driver de Python para MongoDB

### Visualización y Despliegue

- **Streamlit 1.29.0**: Framework para dashboard interactivo
- **plotly 5.18.0**: Gráficos interactivos (Radar Chart, ROC, Confusion Matrix)
- **matplotlib 3.8.2**: Visualizaciones estáticas
- **seaborn 0.13.0**: Gráficos estadísticos

### Utilidades

- **joblib 1.3.2**: Serialización de modelos
- **requests 2.31.0**: Descarga de datasets

## Requisitos Previos

### Software Necesario

1. **Python 3.10+** (recomendado 3.10.11)
2. **Git**
3. **MongoDB 4.6+** (para la etapa de gestión de datos)

> **Nota para Windows**: El script `install.ps1` automatiza la configuración del entorno virtual e instalación de dependencias.

## Instalación y Ejecución

### Instalación Rápida (Windows)

1. **Clonar el repositorio**

```powershell
git clone https://github.com/altairBASIC/breast-cancer-diagnosis-mlp-vs-svm.git
cd breast-cancer-diagnosis-mlp-vs-svm
```

2. **Ejecutar instalación automatizada**

```powershell
.\install.ps1
```

Este script creará el entorno virtual, instalará dependencias y verificará la configuración.

3. **Activar entorno virtual**

```powershell
.\.venv\Scripts\Activate.ps1
```

### Instalación Manual (Linux/macOS)

```bash
cd breast-cancer-diagnosis-mlp-vs-svm
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Pipeline de Ejecución

### 1. Preparación de Datos

```powershell
# Paso 1: Descargar dataset desde UCI ML Repository
python scripts/download_dataset.py

# Paso 2: Cargar datos a MongoDB (opcional pero recomendado)
python scripts/load_to_mongo.py

# Paso 3: Preprocesar datos (limpieza, escalado, división train/test)
python scripts/preprocessing.py
```

**Salida esperada:**

- `data/breast_cancer.csv`: Dataset original
- `data/processed/`: Archivos `.npy` con datos escalados y divididos
- `models/scalers/`: StandardScaler y LabelEncoder guardados

### 2. Entrenamiento de Modelos

```powershell
# Entrenar MLP y SVM, generar reportes
python scripts/train_models.py
```

**Salida esperada:**

- `models/mlp_model.pkl`: Modelo MLP entrenado
- `models/svm_model.pkl`: Modelo SVM entrenado
- Reportes JSON con métricas detalladas

**Métricas alcanzadas:**

- SVM: Accuracy 98.2%, Precision 100%, Recall 95.2%, F1-Score 97.6%, AUC 0.996
- MLP: Accuracy 97.4%, Precision 100%, Recall 92.9%, F1-Score 96.3%, AUC 0.997

### 3. Exploración con Notebooks (Opcional)

```powershell
jupyter notebook
```

Ejecutar en orden:

1. `notebooks/01_data_preprocessing.ipynb`
2. `notebooks/02_mlp_training.ipynb`
3. `notebooks/03_svm_training.ipynb`

### 4. Despliegue de la Aplicación Streamlit

```powershell
streamlit run app.py
```

La aplicación se abrirá en `http://localhost:8501`

## Funcionalidades de la Aplicación

### Dashboard Comparativo

- **Gráfico Radar**: Comparación multimétrica visual (Accuracy, Precision, Recall, F1-Score)
- **Curvas ROC**: Evaluación de capacidad discriminativa con AUC
- **Matrices de Confusión**: Análisis detallado de errores de clasificación

### Predicción Individual

**Dos modos de entrada:**

1. **Selección Aleatoria**: Carga un caso del conjunto de prueba y muestra el diagnóstico real vs. predicho
2. **Ingreso Manual**: Permite introducir valores de las 30 características manualmente

**Resultados mostrados:**

- Predicción del modelo SVM con distancia al hiperplano
- Predicción del modelo MLP con probabilidad de malignidad
- Verificación con diagnóstico real (en modo aleatorio)

## Resultados Destacados

### Comparación de Modelos

| Métrica | SVM | MLP | Diferencia | Ganador |
|---------|-----|-----|------------|----------|
| **Accuracy** | 98.25% | 97.37% | +0.88% | SVM |
| **Precision** | 100% | 100% | 0% | Empate |
| **Recall** | 95.24% | 92.86% | +2.38% | SVM |
| **F1-Score** | 97.56% | 96.30% | +1.26% | SVM |
| **AUC-ROC** | 0.9960 | 0.9970 | -0.0010 | MLP |

### Eficiencia Computacional

| Modelo | Tiempo de Entrenamiento | Velocidad Relativa |
|--------|------------------------|--------------------|
| **SVM** | 1.440 segundos | Baseline (con probability=True) |
| **MLP** | 0.048 segundos | **30x más rápido** |

**Nota:** El SVM es más lento debido al parámetro `probability=True`, que realiza validación cruzada interna de 5-fold para calibrar probabilidades (Platt Scaling).

### Hallazgos Clave

1. **Ambos modelos alcanzan rendimiento clínico excepcional** (>97% accuracy, AUC >0.99)
2. **Empate técnico en rendimiento**: Diferencias <1% en todas las métricas
3. **SVM ligeramente superior en clasificación**: Mayor Accuracy (98.2% vs 97.4%), Recall y F1-Score
4. **MLP ligeramente mejor en AUC**: Mejor capacidad de ranking probabilístico (0.997 vs 0.996)
5. **Precision perfecta en ambos**: No generan falsos positivos (diagnósticos malignos incorrectos)
6. **Trade-off tiempo vs rendimiento**: SVM logra +0.88% accuracy pero tarda 30x más (1.44s vs 0.048s) debido a `probability=True`

## Metodología

### 1. Preprocesamiento

- Limpieza de datos (eliminación de duplicados y valores faltantes)
- Codificación de etiquetas: Benigno (B) → 0, Maligno (M) → 1
- Escalado con StandardScaler (media=0, desviación=1)
- División estratificada: 80% entrenamiento (455 muestras), 20% prueba (114 muestras)

### 2. Modelos Implementados

**SVM (Support Vector Machine):**

- Kernel: RBF (Radial Basis Function)
- Parámetros: C=1.0, gamma='scale', probability=True
- Algoritmo: LIBSVM con calibración probabilística (Platt Scaling)

**MLP (Multi-Layer Perceptron):**

- Arquitectura: 30 → 100 → 50 → 2 (3,000+ parámetros entrenables)
- Activación: ReLU
- Solver: Adam (optimización adaptativa)
- Regularización: Early stopping con 10% de validación

### 3. Evaluación

- Métricas: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Visualizaciones: Curvas ROC, Matrices de Confusión, Radar Chart
- Validación: Conjunto de prueba independiente (nunca visto durante entrenamiento)
  numpy==1.26.2
  pymongo==4.6.1

## Solución de Problemas

### Error: "ModuleNotFoundError: No module named 'plotly'"

```powershell
pip install plotly==5.18.0
```

### Error: SVM con AUC 0.5 (predicción aleatoria)

**Causa:** Datos escalados dos veces o no escalados correctamente.
**Solución:** Verificar que `preprocessing.py` se ejecutó correctamente y regenerar modelos:

```powershell
python scripts/preprocessing.py
python scripts/train_models.py
```

### MongoDB Connection Error

**Solución:** Verificar que MongoDB está ejecutándose:

```powershell
# Windows
net start MongoDB

# Linux/macOS
sudo systemctl start mongod
```

### Streamlit no carga la aplicación

**Solución:** Verificar que los modelos existen:

```powershell
ls models/*.pkl
# Debe mostrar: mlp_model.pkl, svm_model.pkl
```

## Contribuciones

Este es un proyecto académico. Para sugerencias o mejoras, contactar a los autores.

## Licencia

Este proyecto se proporciona con fines educativos bajo el contexto del curso INFB6052 - Herramientas para Cs. de Datos, UTEM.

## Autores

**Grupo 2 - Segundo Semestre 2025**

- Ignacio Ramírez
- Cristián Vergara
- Antonia Montecinos

**Institución:** Universidad Tecnológica Metropolitana (UTEM)
**Curso:** INFB6052 - Herramientas para Ciencia de Datos
**Carrera:** Ingeniería Civil en Ciencia de Datos

## Referencias

### Dataset

Dua, D. & Graff, C. (2019).
*UCI Machine Learning Repository: Breast Cancer Wisconsin (Diagnostic) Data Set.*
University of California, Irvine.
[https://archive.ics.uci.edu/dataset/17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

### Publicaciones Científicas sobre el Dataset

Wolberg, W. H., & Mangasarian, O. L. (1990).
*Multisurface method of pattern separation for medical diagnosis applied to breast cytology.*
Proceedings of the National Academy of Sciences, 87(23), 9193-9196.
[https://doi.org/10.1073/pnas.87.23.9193](https://doi.org/10.1073/pnas.87.23.9193)

Street, W. N., Mangasarian, O. L., & Wolberg, W. H. (1993).
*Nuclear feature extraction for breast tumor diagnosis.*
IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology, 1905, 861-870.
[https://doi.org/10.1117/12.148698](https://doi.org/10.1117/12.148698)

### Algoritmos de Machine Learning

**Support Vector Machines:**
Cortes, C., & Vapnik, V. (1995).
*Support-vector networks.*
Machine Learning, 20(3), 273-297.
[https://doi.org/10.1007/BF00994018](https://doi.org/10.1007/BF00994018)

**Multi-Layer Perceptron:**
Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).
*Learning representations by back-propagating errors.*
Nature, 323(6088), 533-536.
[https://doi.org/10.1038/323533a0](https://doi.org/10.1038/323533a0)

### Herramientas y Frameworks

- **scikit-learn:** Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12, pp. 2825-2830.
- **Streamlit:** [https://streamlit.io/](https://streamlit.io/)
- **MongoDB:** [https://www.mongodb.com/](https://www.mongodb.com/)
- **Plotly:** [https://plotly.com/python/](https://plotly.com/python/)

---

**Nota:** Este proyecto fue desarrollado como parte del trabajo académico del curso INFB6052. Los modelos y métricas presentados tienen fines educativos y demostrativos, no para uso clínico real.

Harris et al. (2020).
The NumPy Array: A Structure for Efficient Numerical Computation.
Nature.
https://numpy.org/

scikit-learn:

Pedregosa et al. (2011).
Scikit-learn: Machine Learning in Python.
Journal of Machine Learning Research.
https://scikit-learn.org/stable/

MongoDB:

MongoDB Inc.
MongoDB Documentation.
https://www.mongodb.com/docs/

PyMongo:

MongoDB Inc.
PyMongo Documentation.
https://pymongo.readthedocs.io/

Streamlit:

Streamlit Inc.
Streamlit Documentation.
https://docs.streamlit.io/

joblib:

joblib developers.
joblib Documentation.
https://joblib.readthedocs.io/

---

**Nota:** Este proyecto fue desarrollado como parte de un curso de Machine Learning para la Universidad Tecnológica Metropolitana y tiene fines exclusivamente educativos.
