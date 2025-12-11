# Guía de Instalación y Configuración

## Proyecto: Breast Cancer Diagnosis - MLP vs SVM

Esta guía complementa al `README.md` y se centra en los pasos prácticos para dejar el entorno listo en una máquina Windows (PowerShell), incluyendo entorno virtual, dependencias, datos, scripts y, opcionalmente, MongoDB.

### Resumen rápido de pasos

1. Clonar el repositorio.
2. Crear y activar el entorno virtual `.venv`.
3. Instalar dependencias desde `requirements.txt`.
4. Ejecutar los scripts de `scripts/` para preparar los datos.
5. (Opcional) Instalar y configurar MongoDB.
6. Ejecutar los notebooks o lanzar la app con `streamlit run app.py`.

---

## 1. Requisitos previos

- **Sistema operativo**: Windows 10/11
- **Python**: 3.10+ (recomendado 3.10.11)
- **Git** instalado y en el PATH
- (Opcional) **MongoDB** si se usará la base de datos

---

## 2. Clonar el repositorio

```powershell
git clone https://github.com/altairBASIC/breast-cancer-diagnosis-mlp-vs-svm.git
cd breast-cancer-diagnosis-mlp-vs-svm
```

---

## 3. Crear y activar entorno virtual

Desde la carpeta raíz del proyecto:

```powershell
python -m venv .venv
\.venv\Scripts\Activate.ps1
```

Verifica que el prompt de la terminal muestre `(.venv)` al inicio.

Si PowerShell bloquea la ejecución de scripts:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 4. Instalar dependencias

Con el entorno virtual activado:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Puedes verificar las librerías principales con:

```powershell
python -c "import pandas, numpy, sklearn, streamlit; print('Librerías clave instaladas')"
```

---

## 5. Preparar datos

El flujo típico usa los scripts en `scripts/`:

1. **Configurar/verificar entorno base**

   ```powershell
   python scripts/setup_environment.py
   ```

   Este script crea carpetas necesarias, puede escribir `setup_log.txt` y valida versión de Python y paquetes.
2. **Descargar dataset y generar resumen**

   ```powershell
   python scripts/download_dataset.py
   ```

   Al finalizar deberías tener:

   - `data/breast_cancer.csv`
   - `data/data_summary.txt`
3. **Preprocesar datos y generar data/processed/**

   ```powershell
   python scripts/preprocessing.py
   ```

   Este script (según su implementación) genera típicamente:

   - `data/processed/X_train.npy`
   - `data/processed/X_test.npy`
   - `data/processed/y_train.npy`
   - `data/processed/y_test.npy`
   - `data/processed/feature_info.json`
   - `data/processed/preprocessing_report.json`

---

## 6. (Opcional) Instalar y configurar MongoDB

La capa MongoDB es opcional; el proyecto funciona sin DB para entrenamiento y Streamlit. Solo es necesaria si quieres persistir registros en una base de datos.

### 6.1 Instalación local en Windows

1. Descargar MongoDB Community Server desde:
   https://www.mongodb.com/try/download/community
2. Ejecutar el instalador y seguir las instrucciones.
3. Iniciar el servicio MongoDB:

   ```powershell
   net start MongoDB
   ```
4. Verificar que MongoDB está ejecutándose:

   ```powershell
   mongosh
   ```

### 6.2 Usar Docker

1. Tener Docker Desktop instalado y corriendo.
2. Ejecutar MongoDB en un contenedor:

   ```powershell
   docker run -d -p 27017:27017 --name mongodb-breast-cancer mongo:latest
   ```
3. Verificar que el contenedor está activo:

   ```powershell
   docker ps
   ```

### 6.3 Cargar datos en MongoDB

Con MongoDB ejecutándose y el entorno virtual activo:

#### Paso 1: Activar el entorno virtual (si no está activado)

```powershell
python scripts/load_to_mongo.py
```

Este script, según la configuración interna, suele:

- Conectarse a MongoDB
- Crear la base de datos `breast_cancer_db`
- Crear la colección `patients_records`
- Insertar los registros del dataset

Puedes verificar rápidamente en MongoDB:

```powershell
mongosh --eval "use breast_cancer_db; show collections; db.patients_records.countDocuments()"
```

---

## 7. Ejecutar notebooks (opcional pero recomendado)

Para reproducir el análisis paso a paso, abre los notebooks en `notebooks/` desde VS Code o Jupyter:

1. `01_data_preprocessing.ipynb`
2. `02_mlp_training.ipynb`
3. `03_svm_training.ipynb`

Asegúrate de seleccionar como kernel el Python de `.venv`.

---

## 8. Ejecutar la aplicación Streamlit

Desde la raíz del proyecto, con el entorno virtual activado:

```powershell
streamlit run app.py
```

El comando mostrará en la terminal la URL local (por ejemplo `http://localhost:8501`) para abrir la app en el navegador.

La app utiliza, cuando existen, los datos de `data/processed/`; si no, puede recurrir al CSV o incluso al dataset de sklearn como respaldo.

---

## 9. Verificación rápida del entorno

### 9.1 Verificar Python y entorno virtual

```powershell
python --version
where python
```

Debe apuntar al ejecutable dentro de `.venv`.

### 9.2 Verificar datos

```powershell
dir data
dir data\processed
```

Deberías ver al menos `breast_cancer.csv`, `data_summary.txt` y los `.npy`/JSON de `processed/` si corriste el preprocesamiento.

### 9.3 Verificar dependencias

```powershell
pip list
```

Y, si usas MongoDB:

```powershell
mongosh --eval "db.version()"
```

---

## Archivos Generados

### setup_log.txt

Contiene el registro detallado de:

- Verificación de librerías
- Intentos de conexión a MongoDB
- Creación de base de datos
- Resultados de todas las operaciones

### data/data_summary.txt

Contiene:

- Información básica del dataset (569 registros, 30 características)
- Tipos de datos de cada columna
- Valores faltantes (ninguno)
- Distribución de clases (B: Benigno, M: Maligno)
- Estadísticas descriptivas completas
- Primeras 5 filas del dataset

---

## Estructura Final

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

---

## Base de Datos MongoDB

### Nombre: breast_cancer_db

### Colección: patients_records

#### Estructura de Documentos:

```json
{
  "_id": ObjectId("..."),
  "id": "ID del paciente",
  "diagnosis": "M o B",
  "radius_mean": 14.127,
  "texture_mean": 23.45,
  ... (30 características más),
  "inserted_at": ISODate("..."),
  "source": "UCI Machine Learning Repository"
}
```

---

## Solución de Problemas

### Error: "No se puede conectar a MongoDB"

**Solución:**

1. Verificar que MongoDB está instalado
2. Verificar que el servicio está ejecutándose:
   ```powershell
   net start MongoDB
   ```
3. Si usa Docker, verificar que el contenedor está activo:
   ```powershell
   docker ps
   ```

### Error: "Módulo no encontrado"

**Solución:**

1. Verificar que el entorno virtual está activado (debe ver `(venv)` en el prompt)
2. Reinstalar dependencias:
   ```powershell
   python -m pip install -r requirements.txt
   ```

### Error al activar el entorno virtual

**Solución:**
Si PowerShell bloquea la ejecución de scripts:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Siguiente Fase: Semana 2

Una vez completada la configuración, estarás listo para:

1. Análisis Exploratorio de Datos (EDA)
2. Visualizaciones con matplotlib/seaborn
3. Preprocesamiento de datos
4. Normalización y división train/test

---

## Comandos de Referencia Rápida

```powershell
# Activar entorno virtual
\.venv\Scripts\Activate.ps1

# Desactivar entorno virtual
deactivate

# Instalar dependencias
python -m pip install -r requirements.txt

# Ejecutar scripts principales
python scripts/setup_environment.py
python scripts/download_dataset.py
python scripts/preprocessing.py
python scripts/load_to_mongo.py   # opcional

# Lanzar app Streamlit
streamlit run app.py
```

---

**Fecha de última actualización:** 10 de diciembre de 2025
**Autores:** Ignacio Ramírez, Cristián Vergara, Antonia Montecinos
**Proyecto:** Breast Cancer Diagnosis - MLP vs SVM
