# Breast Cancer Diagnosis: MLP vs SVM

## Descripcion del Proyecto

Este proyecto implementa y compara dos modelos de Machine Learning —Multi-Layer Perceptron (MLP) y Support Vector Machine (SVM)— para el diagnóstico de cáncer de mama utilizando el dataset Breast Cancer Wisconsin (Diagnostic) del repositorio UCI Machine Learning.
El trabajo incluye un pipeline completo de ciencia de datos, abarcando la adquisición del dataset, el preprocesamiento, el análisis exploratorio de datos, el entrenamiento y evaluación de modelos, y la construcción de una aplicación web interactiva mediante Streamlit.
Los modelos y los registros procesados se almacenan en MongoDB, permitiendo su consulta desde el frontend.
El proyecto tiene fines estrictamente académicos y busca reforzar competencias en análisis de datos, modelamiento y despliegue de aplicaciones de Machine Learning.

## Objetivo

Desarrollar un pipeline completo de Machine Learning que incluya:

- Configuracion del entorno de desarrollo
- Carga y almacenamiento de datos en MongoDB
- Preprocesamiento y analisis exploratorio de datos
- Entrenamiento y evaluacion de modelos MLP y SVM
- Comparacion de metricas de rendimiento
- Despliegue de una aplicacion interactiva con Streamlit

## Estructura del Proyecto

```
breast-cancer-diagnosis-mlp-vs-svm/
│
├── data/                          # Dataset y archivos de datos
│   ├── breast_cancer_data.csv     # Dataset descargado
│   └── data_summary.txt           # Resumen estadistico del dataset
│
├── scripts/                       # Scripts de Python
│   ├── setup_environment.py       # Configuracion y verificacion del entorno
│   ├── download_dataset.py        # Descarga y procesamiento del dataset
│   └── load_to_mongo.py           # Carga de datos a MongoDB
│
├── notebooks/                     # Jupyter notebooks para analisis
│
├── models/                        # Modelos entrenados guardados
│
├── requirements.txt               # Dependencias del proyecto
├── setup_log.txt                  # Log de configuracion y ejecucion
└── README.md                      # Documentacion del proyecto
```

## Dataset

**Nombre:** Breast Cancer Wisconsin (Diagnostic)

**Fuente:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

**Caracteristicas:**

- 569 instancias
- 30 caracteristicas numericas
- 2 clases: Maligno (M) y Benigno (B)

**Atributos:** El dataset contiene mediciones computadas de imagenes digitalizadas de aspiracion con aguja fina (FNA) de masas mamarias, describiendo caracteristicas de los nucleos celulares presentes en la imagen.

Para cada nucleo celular se calculan:

- Media
- Error estandar
- Peor valor (promedio de los tres valores mas grandes)

De las siguientes 10 caracteristicas:

1. Radio
2. Textura
3. Perimetro
4. Area
5. Suavidad
6. Compacidad
7. Concavidad
8. Puntos concavos
9. Simetria
10. Dimension fractal

## Requisitos Previos

### Software Necesario

1. **Python 3.8 o superior**
2. **MongoDB** (instalado y ejecutandose)
   - MongoDB Community Server
   - O MongoDB via Docker

### Instalacion de MongoDB

#### Opcion 1: Instalacion Local (Windows)

1. Descargar MongoDB Community Server desde [mongodb.com](https://www.mongodb.com/try/download/community)
2. Instalar siguiendo el asistente
3. Iniciar el servicio MongoDB:
   ```powershell
   net start MongoDB
   ```

#### Opcion 2: Docker

```bash
docker pull mongo:latest
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

## Instalacion

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/breast-cancer-diagnosis-mlp-vs-svm.git
cd breast-cancer-diagnosis-mlp-vs-svm
```

### 2. Crear Entorno Virtual

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Instalar Dependencias

```powershell
pip install -r requirements.txt
```

## Uso

### Fase 1: Configuracion del Entorno

#### Paso 1: Verificar el Entorno

```powershell
python scripts/setup_environment.py
```

Este script verifica:

- Versiones de las librerias instaladas
- Conexion con MongoDB
- Creacion de la base de datos
- Estructura de directorios

#### Paso 2: Descargar el Dataset

```powershell
python scripts/download_dataset.py
```

Este script:

- Descarga el dataset desde UCI Repository
- Genera un resumen estadistico
- Guarda los archivos en el directorio `data/`

#### Paso 3: Cargar Datos en MongoDB

```powershell
python scripts/load_to_mongo.py
```

Este script:

- Conecta con MongoDB
- Crea la coleccion `patients_records`
- Inserta todos los registros del dataset
- Verifica la insercion correcta

### Verificacion de la Instalacion

Despues de ejecutar los scripts anteriores, deberia tener:

1. Archivo `setup_log.txt` con el registro de todas las operaciones
2. Archivo `data/breast_cancer_data.csv` con el dataset
3. Archivo `data/data_summary.txt` con el resumen estadistico
4. Base de datos `breast_cancer_db` en MongoDB con la coleccion `patients_records`

## Tecnologias Utilizadas

- **Python 3.x**: Lenguaje de programacion principal
- **pandas**: Manipulacion y analisis de datos
- **NumPy**: Operaciones numericas
- **scikit-learn**: Algoritmos de Machine Learning
- **MongoDB**: Base de datos NoSQL
- **PyMongo**: Driver de Python para MongoDB
- **Streamlit**: Framework para aplicaciones web interactivas
- **joblib**: Serializacion de modelos

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

## Roadmap

### Semana 1-2: Fase de Datos y Entorno (Completado)

- [X] Configuracion del entorno
- [X] Descarga del dataset
- [X] Carga de datos en MongoDB
- [X] Generacion de resumenes estadisticos

### Entrenamiento de Modelos

Este proyecto es parte de un trabajo academico. Las sugerencias y mejoras son bienvenidas.

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
