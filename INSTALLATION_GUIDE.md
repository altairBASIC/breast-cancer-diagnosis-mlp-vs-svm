# Guía de Instalación y Configuración
## Proyecto: Breast Cancer Diagnosis - MLP vs SVM

### Estado Actual del Proyecto

#### Completado:
1. Entorno virtual creado correctamente en `/venv`
2. Todas las dependencias instaladas:
   - pandas 2.1.4
   - numpy 1.26.2
   - pymongo 4.6.1
   - scikit-learn 1.3.2
   - streamlit 1.29.0
   - joblib 1.3.2
   - requests 2.31.0

3. Dataset descargado correctamente en `data/breast_cancer.csv`
4. Resumen estadístico generado en `data/data_summary.txt`
5. Estructura de directorios completa:
   - /data
   - /scripts
   - /notebooks
   - /models

#### Pendiente:
1. Instalación y configuración de MongoDB
2. Ejecución del script de carga de datos a MongoDB

---

## Pasos para Completar la Configuración

### 1. Instalar MongoDB

#### Opción A: Instalación Local en Windows

1. Descargar MongoDB Community Server desde:
   https://www.mongodb.com/try/download/community

2. Ejecutar el instalador y seguir las instrucciones

3. Iniciar el servicio MongoDB:
   ```powershell
   net start MongoDB
   ```

4. Verificar que MongoDB está ejecutándose:
   ```powershell
   mongosh
   ```

#### Opción B: Usar Docker (Recomendado)

1. Asegúrate de tener Docker Desktop instalado

2. Ejecutar MongoDB en un contenedor:
   ```powershell
   docker run -d -p 27017:27017 --name mongodb-breast-cancer mongo:latest
   ```

3. Verificar que el contenedor está ejecutándose:
   ```powershell
   docker ps
   ```

### 2. Ejecutar Scripts de Configuración

Una vez MongoDB esté instalado y ejecutándose:

#### Paso 1: Activar el entorno virtual (si no está activado)
```powershell
cd "c:\Users\ignac\vscode projects\breast-cancer-diagnosis-mlp-vs-svm"
.\venv\Scripts\Activate.ps1
```

#### Paso 2: Verificar el entorno
```powershell
python scripts/setup_environment.py
```

Este script verificará:
- Versiones de librerías instaladas
- Conexión con MongoDB
- Creación de la base de datos `breast_cancer_db`

#### Paso 3: Cargar datos en MongoDB
```powershell
python scripts/load_to_mongo.py
```

Este script:
- Conectará con MongoDB
- Creará la colección `patients_records`
- Insertará los 569 registros del dataset
- Verificará la inserción correcta

---

## Verificación del Entorno

### Comando Rápido para Verificar Todo:
```powershell
python -c "import pandas, numpy, pymongo, sklearn, streamlit, joblib, requests; print('Todas las librerias instaladas correctamente')"
```

### Verificar Dataset:
```powershell
dir data
```

Deberías ver:
- `breast_cancer.csv` (dataset completo)
- `data_summary.txt` (resumen estadístico)

### Verificar MongoDB:
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
.\venv\Scripts\Activate.ps1

# Desactivar entorno virtual
deactivate

# Listar paquetes instalados
pip list

# Verificar Python
python --version

# Verificar MongoDB
mongosh --version

# Ver bases de datos en MongoDB
mongosh --eval "show dbs"

# Ver colecciones en breast_cancer_db
mongosh --eval "use breast_cancer_db; show collections"

# Contar documentos en patients_records
mongosh --eval "use breast_cancer_db; db.patients_records.countDocuments()"
```

---

**Fecha de creación:** 24 de octubre de 2025
**Autor:** Ignacio
**Proyecto:** Breast Cancer Diagnosis - MLP vs SVM
