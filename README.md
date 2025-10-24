# Breast Cancer Diagnosis: MLP vs SVM

## Descripcion del Proyecto

Este proyecto implementa y compara dos modelos de Machine Learning (Multi-Layer Perceptron y Support Vector Machine) para el diagnostico de cancer de mama utilizando el dataset Breast Cancer Wisconsin (Diagnostic) del repositorio UCI Machine Learning.

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

## Referencias

## Contacto

Para preguntas o comentarios sobre este proyecto, por favor abra un issue en el repositorio de GitHub.

---

**Nota:** Este proyecto fue desarrollado como parte de un curso de Machine Learning y tiene fines exclusivamente educativos.
