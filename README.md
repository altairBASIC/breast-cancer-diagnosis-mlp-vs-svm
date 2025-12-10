# Breast Cancer Diagnosis: MLP vs SVM

## Descripción del Proyecto

Este proyecto implementa y compara dos modelos de Machine Learning —Multi-Layer Perceptron (MLP) y Support Vector Machine (SVM)— para el diagnóstico de cáncer de mama utilizando el dataset Breast Cancer Wisconsin (Diagnostic) del repositorio UCI Machine Learning.

El trabajo incluye un pipeline completo de ciencia de datos:

- Adquisición y exploración del dataset
- Preprocesamiento y generación de conjuntos de entrenamiento/prueba
- Entrenamiento y evaluación de modelos MLP y SVM
- Generación de reportes de entrenamiento en formato JSON
- Despliegue de una aplicación web interactiva (Streamlit) para explorar métricas y realizar predicciones

El proyecto tiene fines estrictamente académicos y busca reforzar competencias en análisis de datos, modelamiento y despliegue de aplicaciones de Machine Learning.

## Objetivo

Desarrollar un pipeline completo de Machine Learning que incluya:

- Configuración del entorno de desarrollo
- Descarga y preparación del dataset
- (Opcional) Carga y almacenamiento de datos en MongoDB
- Preprocesamiento y análisis exploratorio de datos
- Entrenamiento y evaluación de modelos MLP y SVM
- Comparación de métricas de rendimiento
- Despliegue de una aplicación interactiva con Streamlit

## Estructura del Proyecto (resumen)

```
breast-cancer-diagnosis-mlp-vs-svm/
│
├── data/                          # Dataset y archivos de datos
│   ├── breast_cancer.csv          # Dataset preprocesado/descargado
│   ├── data_summary.txt           # Resumen estadístico del dataset
│   └── processed/                 # Datos ya procesados para modelado
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_test.npy
│       ├── feature_info.json
│       └── preprocessing_report.json
│
├── scripts/                       # Scripts de Python
│   ├── setup_environment.py       # Configuración y verificación del entorno
│   ├── download_dataset.py        # Descarga del dataset
│   ├── preprocessing.py           # Limpieza y preprocesamiento de datos
│   └── load_to_mongo.py           # (Opcional) Carga de datos a MongoDB
│
├── notebooks/                     # Jupyter notebooks para análisis y modelos
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_mlp_training.ipynb
│   └── 03_svm_training.ipynb
│
├── models/                        # Modelos entrenados y reportes
│   ├── mlp_training_report.json
│   └── svm_training_report.json
│
├── ui/                            # Recursos para la interfaz (logos, etc.)
│   └── utem1.png
│
├── app.py                         # Aplicación Streamlit principal
├── requirements.txt               # Dependencias del proyecto
├── install.ps1                    # Script de instalación en Windows (automatizado)
├── INSTALLATION_GUIDE.md          # Guía de instalación detallada
├── setup_log.txt                  # Log de configuración y ejecución
└── README.md                      # Documentación del proyecto
```

## Dataset

**Nombre:** Breast Cancer Wisconsin (Diagnostic)

**Fuente:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

**Características:**

- 569 instancias
- 30 características numéricas
- 2 clases: Maligno (M) y Benigno (B)

**Atributos:** El dataset contiene mediciones computadas de imágenes digitalizadas de aspiración con aguja fina (FNA) de masas mamarias, describiendo características de los núcleos celulares presentes en la imagen.

## Requisitos Previos

### Software Necesario

1. **Python 3.10+** (recomendado 3.10.11)
2. **Git**
3. (Opcional) **MongoDB** si deseas usar la capa de base de datos

> Nota: en Windows se incluye un script `install.ps1` que automatiza parte de la configuración.

## Instalación rápida

### 1. Clonar el repositorio

```powershell
git clone https://github.com/altairBASIC/breast-cancer-diagnosis-mlp-vs-svm.git
cd breast-cancer-diagnosis-mlp-vs-svm
```

### 2. Crear y activar entorno virtual (Windows PowerShell)

```powershell
python -m venv .venv
\.venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. (Opcional) Ejecutar script de instalación

```powershell
./install.ps1
```

Este script puede automatizar pasos como creación de directorios, descarga de datos y verificación del entorno (ver `INSTALLATION_GUIDE.md` para más detalles).

## Uso

### 1. Preparación de datos (scripts)

```powershell
# Verificar/recrear estructura básica
python scripts/setup_environment.py

# Descargar dataset y generar resumen en data/
python scripts/download_dataset.py

# Preprocesar datos y generar data/processed/
python scripts/preprocessing.py

# (Opcional) Cargar datos en MongoDB
python scripts/load_to_mongo.py
```

Después de estos pasos deberías tener:

1. `data/breast_cancer.csv` con el dataset
2. `data/data_summary.txt` con el resumen estadístico
3. Archivos `.npy` y JSON dentro de `data/processed/` listos para modelado
4. (Opcional) Base de datos `breast_cancer_db` en MongoDB con la colección `patients_records`

### 2. Trabajo exploratorio y entrenamiento (notebooks)

Puedes reproducir el flujo de trabajo usando los notebooks en `notebooks/`:

1. `01_data_preprocessing.ipynb`: limpieza, análisis exploratorio, generación de features
2. `02_mlp_training.ipynb`: entrenamiento y evaluación del modelo MLP
3. `03_svm_training.ipynb`: entrenamiento y evaluación del modelo SVM

Cada notebook guarda resultados y, en algunos casos, reportes JSON en la carpeta `models/`.

### 3. Aplicación web con Streamlit

Con el entorno virtual activado desde la raíz del proyecto:

```powershell
streamlit run app.py
```

La aplicación permite:

- Visualizar métricas de rendimiento de MLP y SVM
- Consultar matrices de confusión y (cuando aplica) curvas ROC
- Realizar predicciones interactivas introduciendo características de un caso
- Ver resúmenes del dataset y, si existen, reportes JSON en `models/`
```powershell
python scripts/load_to_mongo.py
```

Este script:

- Conecta con MongoDB
- Crea la colección `patients_records`
- Inserta todos los registros del dataset
- Verifica la inserción correcta

### Verificación de la instalación

Después de ejecutar los scripts anteriores, deberías tener:

1. Archivo `setup_log.txt` con el registro de todas las operaciones
2. Archivo `data/breast_cancer_data.csv` con el dataset
3. Archivo `data/data_summary.txt` con el resumen estadístico
4. Base de datos `breast_cancer_db` en MongoDB con la colección `patients_records`

## Tecnologías Utilizadas

- **Python 3.x**: Lenguaje de programación principal
- **pandas**: Manipulación y análisis de datos
- **NumPy**: Operaciones numéricas
- **scikit-learn**: Algoritmos de Machine Learning
- **MongoDB**: Base de datos NoSQL
- **PyMongo**: Driver de Python para MongoDB
- **Streamlit**: Framework para aplicaciones web interactivas
- **joblib**: Serialización de modelos

## Dependencias

```
pandas==2.1.4
numpy==1.26.2
pymongo==4.6.1
scikit-learn==1.3.2
streamlit==1.29.0
joblib==1.3.2
requests==2.31.0
```

## Roadmap (resumen)

- [X] Configuración del entorno
- [X] Descarga y exploración del dataset
- [X] Preprocesamiento y generación de `data/processed/`
- [X] Entrenamiento de modelos MLP y SVM (notebooks)
- [X] Generación de reportes de entrenamiento en `models/`
- [X] Prototipo de aplicación web con Streamlit (`app.py`)
- [ ] Extender comparación con más modelos
- [ ] Integrar persistencia completa en MongoDB para modelos y predicciones

## Licencia

Este proyecto se proporciona con fines educativos.

## Autores
Ignacio Ramírez
Cristián Vergara
Antonia Montecinos

## Referencias
Dataset utilizado:
Dua, D. & Graff, C. (2019).
UCI Machine Learning Repository: Breast Cancer Wisconsin (Diagnostic) Data Set.
University of California, Irvine.
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

Artículos científicos sobre el dataset

Wolberg, W. H., & Mangasarian, O. L. (1990).
Multisurface method of pattern separation for medical diagnosis applied to breast cytology.
Proceedings of the National Academy of Sciences.
https://doi.org/10.1073/pnas.87.23.9135

Street, W. N., Mangasarian, O. L., & Wolberg, W. H. (1993).
Nuclear feature extraction for breast tumor diagnosis.
IS&T/SPIE International Symposium.
https://doi.org/10.1117/12.148698

Referencias de los algoritmos usados
Support Vector Machine

Cortes, C., & Vapnik, V. (1995).
Support-vector networks.
Machine Learning, 20, 273–297.
https://doi.org/10.1007/BF00994018

Multi-Layer Perceptron

Rosenblatt, F. (1958).
The Perceptron: A probabilistic model for information storage and organization in the brain.
Psychological Review, 65(6), 386–408.
https://doi.org/10.1037/h0042519

Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).
Learning representations by back-propagating errors.
Nature 323, 533–536.
https://doi.org/10.1038/323533a0

Documentación oficial de las herramientas usadas
Python:

Python Software Foundation.
Python Language Reference, v3.x.
https://www.python.org/

Pandas:

The pandas development team (2023).
pandas documentation.
https://pandas.pydata.org/

NumPy:

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
