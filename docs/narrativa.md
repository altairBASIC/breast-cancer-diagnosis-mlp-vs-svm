# Narrativa del Proyecto

## 1. Contexto y Motivación

El cáncer de mama es una de las principales causas de mortalidad en mujeres a nivel mundial. El diagnóstico temprano y preciso marca una diferencia crítica en la probabilidad de supervivencia, en la calidad de vida de las pacientes y en la planificación de los tratamientos. 

En este proyecto se desarrolla un sistema de apoyo al diagnóstico que utiliza **técnicas de Machine Learning** para clasificar tumores mamarios como **Benignos (B)** o **Malignos (M)** a partir de mediciones cuantitativas extraídas de imágenes médicas.

La motivación principal es **explorar, comparar y explicar** dos modelos clásicos de clasificación supervisada:

- **MLP (Multi-Layer Perceptron)**: una red neuronal artificial de tipo *feedforward*.
- **SVM (Support Vector Machine)**: un clasificador basado en vectores de soporte y márgenes máximos.

Al mismo tiempo, el proyecto busca reforzar competencias en:

- Diseño de pipelines de ciencia de datos.
- Buenas prácticas de preprocesamiento de datos.
- Evaluación, comparación y comunicación de resultados de modelos.
- Despliegue de una aplicación web interactiva para usuarios no técnicos.

---

**Autor:** Antonia Montecinos  
**Fecha:** Diciembre 2025  
**Dataset:** Breast Cancer Wisconsin (Diagnostic) - 569 muestras, 30 características

---

## 2. Resumen del Proyecto

Este repositorio implementa un flujo completo de trabajo de Machine Learning para el diagnóstico de cáncer de mama, desde los datos crudos hasta una interfaz web interactiva:

1. **Preparación del entorno**  
   Scripts para instalar dependencias, configurar el entorno de Python y, opcionalmente, preparar una instancia de MongoDB.

2. **Obtención y almacenamiento de datos**  
   - Descarga del **dataset Breast Cancer Wisconsin (Diagnostic)** desde `sklearn.datasets`.
   - Generación de resúmenes estadísticos y archivos auxiliares en la carpeta `data/`.
   - Carga opcional de los datos en **MongoDB** para prácticas de integración con bases de datos NoSQL.

3. **Preprocesamiento de datos**  
   - Limpieza, análisis exploratorio (EDA) y normalización de características.  
   - Codificación de la variable objetivo (de `B`/`M` a `0`/`1`).  
   - División en conjuntos de **entrenamiento** y **prueba**.  
   - Guardado de los datos procesados en formatos `.npy`, `.csv` y `.json`.

4. **Entrenamiento de modelos**  
   Cuadernos Jupyter para entrenar y analizar:
   - Un modelo **MLP** (red neuronal).  
   - Un modelo **SVM** (máquina de vectores de soporte).  
   Ambos modelos se evalúan y sus métricas se registran en la carpeta `models/`.

5. **Aplicación web (Streamlit)**  
   Una aplicación en `app.py` que permite:
   - Explorar el dataset y las particiones train/test.  
   - Visualizar métricas y matrices de confusión.  
   - Probar el modelo con casos sintéticos o ingresados manualmente.  
   - Consultar analíticas de sesión sobre las predicciones realizadas.

En conjunto, el proyecto funciona como un **laboratorio educativo completo** para estudiar la aplicación de ML a problemas de salud.

## 3. Descripción Detallada del Flujo de Trabajo

### 3.1 Preparación del Entorno

La preparación se realiza principalmente mediante:

- `requirements.txt`: lista de librerías necesarias (pandas, numpy, scikit-learn, streamlit, pymongo, etc.).
- `install.ps1`: script de PowerShell que:
  - Crea y activa un entorno virtual.
  - Instala dependencias.  
  - Descarga el dataset.  
  - Verifica la presencia de MongoDB e indica los pasos a seguir.

Además, existen scripts en `scripts/` como:

- `setup_environment.py`: verifica versiones de librerías y la conexión a MongoDB.
- `download_dataset.py`: descarga y analiza el dataset, generando archivos en `data/`.
- `load_to_mongo.py`: carga los datos a la base `breast_cancer_db` (colección `patients_records`).
- `preprocessing.py`: implementa un pipeline de preprocesamiento en forma de clase (`DataPreprocessor`).

### 3.2 Dataset: Breast Cancer Wisconsin (Diagnostic)

El dataset **Breast Cancer Wisconsin (Diagnostic)** es un conjunto clásico de referencia en la literatura de ML biomédico.

- **Número de instancias:** 569 pacientes.  
- **Número de características:** 30 variables numéricas.  
- **Variable objetivo:** `diagnosis`, con dos posibles valores:  
  - `M` = Maligno.  
  - `B` = Benigno.

Las características se obtienen a partir de imágenes digitalizadas de una **aspiración con aguja fina (FNA)** de masas mamarias. Para cada núcleo celular detectado en la imagen se calculan medidas geométricas y de textura, y luego se agregan en tres grupos por característica base:

1. **`mean` (media)**: valor promedio de la característica.  
2. **`se` o `error` (error estándar)**: variabilidad de la característica.  
3. **`worst` (peor valor)**: media de los tres valores más altos.

Las 10 características base son:

1. `radius`: radio del tumor (distancia promedio del centro al borde).
2. `texture`: variación de intensidad en la imagen (rugosidad visual).
3. `perimeter`: perímetro del contorno del tumor.
4. `area`: área de la región del tumor.
5. `smoothness`: suavidad o irregularidad de las variaciones en el radio.
6. `compactness`: relación perímetro² / área, asociada a la compacidad.
7. `concavity`: profundidad de las partes cóncavas en el contorno.
8. `concave points`: número de puntos cóncavos en el contorno.
9. `symmetry`: simetría de la forma del tumor.
10. `fractal dimension`: medida de la complejidad del borde.

Combinando estos 10 atributos con sus tres versiones (`mean`, `error`, `worst`) se obtienen las 30 características numéricas finales.

### 3.3 Preprocesamiento de Datos

El preprocesamiento implementado incluye:

1. **Carga desde MongoDB** (opcional pero utilizada en el notebook principal):  
   Los registros se leen desde la colección `patients_records` y se convierten en un `DataFrame` de pandas.

2. **Exploración inicial (EDA):**  
   - Cálculo de dimensiones, tipos de datos y valores faltantes.  
   - Análisis de la distribución de `diagnosis` (proporción de casos benignos/malignos).  
   - Estadísticas descriptivas para las características numéricas.

3. **Limpieza:**  
   - Eliminación de columnas auxiliares (`_id`, `inserted_at`, `source`).  
   - Eliminación de duplicados.  
   - Eliminación de filas con valores faltantes (si los hubiera).

4. **Codificación de la variable objetivo:**  
   - Uso de `LabelEncoder` para transformar `diagnosis` en `diagnosis_encoded`:  
     - `B` → `0` (Benigno).  
     - `M` → `1` (Maligno).  
   - El codificador se guarda en `models/scalers/label_encoder.pkl`.

5. **División en entrenamiento y prueba:**  
   - Uso de `train_test_split` (80% train, 20% test), con `stratify=y` para mantener la proporción de clases.

6. **Normalización de características:**  
   - Uso de `StandardScaler` entrenado solo con los datos de entrenamiento.  
   - Transformación de `X_train` y `X_test` a `X_train_scaled` y `X_test_scaled`, con media aproximada 0 y desviación estándar 1.  
   - El scaler se guarda en `models/scalers/standard_scaler.pkl`.

7. **Persistencia de datos preprocesados:**  
   Los datos y metadatos se guardan en `data/processed/`:
   - `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`.  
   - `train_data.csv`, `test_data.csv` (para inspección).  
   - `feature_info.json` (nombres de características, tamaños de los conjuntos).  
   - `preprocessing_report.json` (resumen completo del pipeline).

## 4. Conceptos de Machine Learning Utilizados

### 4.1 MLP (Multi-Layer Perceptron)

Un **MLP** es un tipo de **red neuronal artificial** donde la información fluye en una sola dirección (de la capa de entrada a la de salida), sin ciclos. Se compone de:

- **Capa de entrada**: recibe las 30 características del paciente.  
- **Una o varias capas ocultas**: cada neurona aplica una combinación lineal de las entradas seguida de una función de activación no lineal (por ejemplo, ReLU o tanh).  
- **Capa de salida**: produce una probabilidad o un puntaje para cada clase (benigno/maligno).

La idea central es aprender una función:

$$ f_\theta: \mathbb{R}^{30} \to \{0, 1\} $$

que aproxime correctamente la etiqueta de diagnóstico a partir del vector de características de cada caso.

#### Ejemplo práctico (intuitivo)

1. Un paciente tiene ciertas mediciones: radio promedio, textura, área, etc.  
2. Esas 30 características se introducen en el MLP como un vector numérico.  
3. El MLP combina estas entradas mediante pesos internos (parámetros) y genera una probabilidad de que el caso sea maligno.  
4. Si la probabilidad supera un umbral (por ejemplo 0.5), el modelo predice **Maligno**; en caso contrario, **Benigno**.

Dentro del proyecto, el MLP se entrena en los notebooks y sus resultados (métricas y matrices de confusión) se guardan en `models/mlp_training_report.json`.

### 4.2 SVM (Support Vector Machine)

Una **SVM** es un modelo supervisado que busca encontrar un **hiperplano de separación** que maximice el margen entre dos clases. Intuitivamente:

- Cada paciente se representa como un punto en un espacio de 30 dimensiones.  
- La SVM intenta trazar un plano (o hiperplano) que separe los puntos benignos de los malignos con la mayor separación posible.  
- Los puntos más cercanos al hiperplano se llaman **vectores de soporte** y son los que determinan la frontera de decisión.

Si el problema no es linealmente separable, se usan **kernels** (por ejemplo, RBF) para proyectar los datos a espacios de mayor dimensión donde sí exista una separación más clara.

#### Ejemplo práctico (intuitivo)

1. Supongamos que proyectamos los datos a 2D usando solo dos características: `radius_mean` y `texture_mean`.  
2. La SVM dibuja una línea en ese plano, tratando de dejar los casos malignos a un lado y los benignos al otro.  
3. En la práctica, se permiten ciertos errores de clasificación, controlados por el parámetro **C** (trade-off entre margen amplio y errores).

En el proyecto, el modelo SVM se entrena en su notebook correspondiente y sus resultados se guardan en `models/svm_training_report.json`.

### 4.3 Métricas de Evaluación

Para comparar los modelos, se usan varias métricas, entre ellas:

- **Accuracy (Exactitud)**:  
  Proporción de predicciones correctas sobre el total de casos. Es útil cuando las clases están relativamente balanceadas.

- **Precision (Precisión)**:  
  De todos los casos predichos como malignos, cuántos eran realmente malignos. Importante cuando un falso positivo tiene coste.

- **Recall (Sensibilidad)**:  
  De todos los casos realmente malignos, cuántos detecta el modelo. Crítico en medicina para no "dejar escapar" casos graves.

- **F1-Score**:  
  Media armónica entre precisión y sensibilidad, útil cuando queremos un equilibrio entre ambas.

- **AUC-ROC**:  
  Mide la capacidad del modelo para distinguir entre clases a diferentes umbrales de decisión.

- **Matrices de confusión**:  
  Tablas que muestran verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos. Permiten entender con más detalle los errores.

## 5. Ejemplos de Uso en la Aplicación Web

La aplicación `app.py` expone estos conceptos de manera amigable para el usuario.

### 5.1 Exploración de Datos

En la sección **"Explorador de Datos"** se puede:

- Ver una muestra de las filas del dataset original.  
- Revisar las estadísticas descriptivas por columna.  
- Observar la distribución de diagnósticos (gráficos de torta y barras).  
- Descargar el dataset original en formato CSV.

### 5.2 Análisis de Modelos

Las secciones **"MLP (análisis)"** y **"SVM (Análisis)"** muestran:

- Métricas de entrenamiento y prueba.  
- Matrices de confusión.  
- Curvas de aprendizaje (cómo mejora el modelo al ver más datos).  
- Gráficos comparativos de rendimiento (accuracy, recall, etc.).

Esto permite estudiar, desde una perspectiva visual, en qué escenarios cada modelo rinde mejor.

### 5.3 Probadores de Casos (Inferencia)

Las secciones **"MLP (probador)"** y **"SVM (probador)"** permiten:

1. Introducir manualmente valores para las 30 características (ya normalizadas/escaladas).  
2. O bien generar ejemplos aleatorios sintéticos para probar la respuesta del modelo.  
3. Obtener un diagnóstico estimado (**BENIGNO** o **MALIGNO**) junto con una medida de confianza.  
4. Registrar un historial de las predicciones de la sesión y descargarlo como CSV.

De esta forma, la app no solo muestra resultados estáticos de entrenamiento, sino que también ilustra el proceso de **inferencia** en tiempo real.

### 5.4 Analítica de Sesión

La sección **"Analítica Web"** resume las predicciones realizadas durante la sesión:

- Número total de casos evaluados.  
- Cantidad de predicciones por modelo (SVM y MLP).  
- Distribución de diagnósticos (benigno/maligno) por modelo.  
- Tabla filtrable con el historial de casos recientes.

Esto permite entender de manera agregada cómo se han comportado los modelos en el uso interactivo.

## 6. Conclusión

Este proyecto integra **dataset biomédico real**, **técnicas de preprocesamiento sólidas**, **modelos clásicos de ML** (MLP y SVM) y un **frontend interactivo** con Streamlit. 

Más allá de obtener buenas métricas, el objetivo central es **aprender y explicar** paso a paso cómo construir un sistema de diagnóstico asistido por computador:

- Desde la adquisición y exploración de los datos.  
- Pasando por la ingeniería de características y el preprocesamiento.  
- Hasta el entrenamiento, la evaluación, la comparación de modelos y la presentación de resultados a usuarios finales.

La carpeta `docs/` y este archivo `narrativa.md` complementan el `README.md` ofreciendo una visión más narrativa y conceptual, adecuada tanto para revisores académicos como para estudiantes que deseen entender la lógica completa del proyecto.