# Guía de Instalación y Configuración
## Proyecto: Breast Cancer Diagnosis - MLP vs SVM

Esta guía complementa al `README.md` y se centra en los pasos prácticos para dejar el entorno listo en una máquina Windows (PowerShell), incluyendo entorno virtual, dependencias, datos, scripts y, opcionalmente, MongoDB.

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

2. Levantar un contenedor de MongoDB:

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

## Estructura Final Esperada

```
breast-cancer-diagnosis-mlp-vs-svm/
├── venv/                              [COMPLETADO]
├── data/
│   ├── breast_cancer.csv              [COMPLETADO]
│   └── data_summary.txt               [COMPLETADO]
├── scripts/
│   ├── setup_environment.py           [COMPLETADO]
│   ├── download_dataset.py            [COMPLETADO]
│   └── load_to_mongo.py              [COMPLETADO]
├── notebooks/                         [COMPLETADO]
├── models/                            [COMPLETADO]
├── requirements.txt                   [COMPLETADO]
├── setup_log.txt                      [GENERADO AL EJECUTAR]
└── README.md                          [COMPLETADO]
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

**Fecha de última actualización:** 9 de diciembre de 2025  
**Autores:** Ignacio Ramírez, Cristián Vergara, Antonia Montecinos  
**Proyecto:** Breast Cancer Diagnosis - MLP vs SVM
