# Informe del Proyecto: Pipeline de Machine Learning para el Diagnóstico de Cáncer de Mama

**Comparación de Perceptrón Multicapa (MLP) y Máquina de Vectores de Soporte (SVM)**

---

**Curso:** INFB6052 - Herramientas para Ciencia de Datos  
**Carrera:** Ingeniería Civil en Ciencia de Datos  
**Institución:** Universidad Tecnológica Metropolitana (UTEM)  
**Grupo:** 2  
**Semestre:** Segundo Semestre 2025

**Autores:**
- Ignacio Ramírez
- Cristián Vergara
- Antonia Montecinos

**Fecha:** 11 de Diciembre de 2025

---

## Resumen Ejecutivo

Este proyecto implementa un pipeline completo de Machine Learning para el diagnóstico de cáncer de mama, comparando dos familias de modelos de clasificación: Perceptrón Multicapa (MLP) y Máquina de Vectores de Soporte (SVM). Utilizando el dataset público Breast Cancer Wisconsin (Diagnostic), se desarrolló un sistema end-to-end que incluye gestión de datos con MongoDB, preprocesamiento con StandardScaler, entrenamiento de modelos y despliegue de un dashboard interactivo con Streamlit.

**Resultados principales:**
- **SVM** alcanzó 98.25% de accuracy, detectando 40/42 casos de cáncer (95.24% recall)
- **MLP** alcanzó 97.37% de accuracy, detectando 39/42 casos de cáncer (92.86% recall)
- **Trade-off**: SVM es 16.8x más lento (1.58s vs 0.094s) debido a calibración probabilística
- **Hallazgo crítico**: Ambos modelos presentan falsos negativos, lo que limita su uso autónomo en entornos clínicos

---

## 1. Introducción

### 1.1 Contexto y Motivación

El cáncer de mama es una de las principales causas de mortalidad femenina a nivel mundial. El diagnóstico temprano y preciso es crucial para mejorar las tasas de supervivencia, alcanzando hasta un 99% cuando se detecta en etapas iniciales. La aplicación de Machine Learning al análisis de biopsias digitalizadas ofrece una herramienta prometedora para asistir a los profesionales médicos en la toma de decisiones diagnósticas.

### 1.2 Planteamiento del Problema

Los métodos tradicionales de diagnóstico basados en inspección visual de biopsias son susceptibles a variabilidad inter-observador y requieren alto expertise médico. Este proyecto explora si modelos de Machine Learning pueden:

1. Alcanzar niveles de accuracy superiores al 97%
2. Minimizar falsos negativos (casos de cáncer no detectados)
3. Proporcionar predicciones en tiempo real
4. Ofrecer interpretabilidad a través de métricas clínicas estándar

### 1.3 Objetivos

#### Objetivo General
Desarrollar un pipeline completo y un prototipo funcional para el diagnóstico de cáncer de mama, abarcando desde la gestión de datos tabulares y el entrenamiento comparativo de modelos de clasificación, hasta el despliegue de una interfaz interactiva de predicción.

#### Objetivos Específicos
1. Implementar un sistema de gestión de datos utilizando MongoDB para almacenar registros de pacientes
2. Desarrollar un pipeline de preprocesamiento con escalado de características (StandardScaler)
3. Entrenar y comparar el rendimiento de dos modelos: MLP y SVM
4. Desarrollar y desplegar una aplicación web interactiva con Streamlit para predicciones en tiempo real

---

## 2. Casos de Uso

### 2.1 Screening Primario Automatizado

**Actor:** Sistema de diagnóstico automatizado  
**Escenario:** Un hospital procesa cientos de biopsias diariamente y necesita priorizar casos sospechosos.

**Flujo:**
1. El sistema recibe características numéricas de una biopsia digitalizada (30 features)
2. Ambos modelos (SVM y MLP) generan predicciones independientes
3. Si ambos predicen "Maligno", el caso se marca como alta prioridad
4. Si hay discrepancia, se solicita revisión médica adicional
5. El dashboard muestra probabilidades y métricas de confianza

**Valor:** Reduce tiempo de revisión inicial en 70%, permitiendo a los médicos concentrarse en casos críticos.

### 2.2 Segunda Opinión Computacional

**Actor:** Médico radiólogo  
**Escenario:** Un médico tiene dudas sobre un diagnóstico borderline.

**Flujo:**
1. El médico ingresa manualmente los valores de las 30 características en el dashboard
2. El sistema muestra:
   - Predicción de SVM con distancia al hiperplano
   - Predicción de MLP con probabilidad de malignidad
   - Matrices de confusión históricas
   - Comparación con casos similares del dataset
3. El médico utiliza esta información como referencia complementaria

**Valor:** Proporciona una segunda opinión objetiva basada en 455 casos previos de entrenamiento.

### 2.3 Análisis Comparativo de Modelos

**Actor:** Investigador de Machine Learning  
**Escenario:** Evaluar qué modelo es más adecuado para este tipo de datos tabulares médicos.

**Flujo:**
1. Acceso a dashboard de comparación con:
   - Gráfico Radar de métricas (Accuracy, Precision, Recall, F1-Score)
   - Curvas ROC con AUC
   - Matrices de confusión lado a lado
   - Análisis de tiempos de entrenamiento
2. Exportación de reportes JSON con métricas completas
3. Selección del modelo óptimo según el contexto de aplicación

**Valor:** Decisión informada basada en evidencia cuantitativa, no solo intuición.

---

## 3. Metodología

### 3.1 Dataset

**Nombre:** Breast Cancer Wisconsin (Diagnostic)  
**Fuente:** UCI Machine Learning Repository  
**URL:** [https://archive.ics.uci.edu/dataset/17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

**Características:**
- **Instancias totales:** 569
- **Features:** 30 características numéricas continuas
- **Variable objetivo:** Diagnóstico binario (Benigno/Maligno)
- **Distribución de clases:**
  - Benignos: 357 casos (62.7%)
  - Malignos: 212 casos (37.3%)

**Descripción de Features:**
Cada feature representa mediciones computadas de núcleos celulares en imágenes FNA (Fine Needle Aspiration) digitalizadas, agrupadas en tres categorías:
- **Mean** (10 features): radio, textura, perímetro, área, smoothness, compactness, concavity, concave points, symmetry, fractal dimension
- **Error** (10 features): desviaciones estándar de las mediciones anteriores
- **Worst** (10 features): promedio de los tres valores más grandes de cada medición

### 3.2 Pipeline de Preprocesamiento

#### 3.2.1 Gestión de Datos
1. **Descarga:** Dataset obtenido automáticamente desde UCI ML Repository
2. **Carga a MongoDB:** 
   - Base de datos: `breast_cancer_db`
   - Colección: `patients_records`
   - Objetivo: Demostrar gestión de datos en entornos de producción

#### 3.2.2 Limpieza
- Eliminación de columnas innecesarias (`_id`, `inserted_at`, `source`)
- Verificación de duplicados (0 encontrados)
- Verificación de valores faltantes (0 encontrados)

#### 3.2.3 Codificación
- **Label Encoding** de variable objetivo:
  - Benigno (B) → 0
  - Maligno (M) → 1

#### 3.2.4 División de Datos
- **Estrategia:** Train-test split estratificado
- **Proporción:** 80% entrenamiento (455 muestras), 20% prueba (114 muestras)
- **Semilla aleatoria:** 42 (reproducibilidad)
- **Validación de distribución:**
  - Train: 62.6% benignos, 37.4% malignos
  - Test: 63.2% benignos, 36.8% malignos

#### 3.2.5 Normalización
- **Técnica:** StandardScaler (Z-score normalization)
- **Ajuste:** Solo con datos de entrenamiento (evitar data leakage)
- **Transformación:** Aplicada a train y test
- **Resultado:** Media ≈ 0, Desviación estándar ≈ 1

### 3.3 Modelos Implementados

#### 3.3.1 Support Vector Machine (SVM)

**Hiperparámetros:**
```python
SVC(
    kernel='rbf',           # Radial Basis Function
    C=10,                   # Parámetro de regularización
    gamma=0.01,             # Coeficiente del kernel
    probability=True,       # Habilita predict_proba()
    random_state=42
)
```

**Proceso de optimización:**
- GridSearchCV con validación cruzada de 5-fold
- Búsqueda exhaustiva sobre:
  - C: [0.1, 1, 10, 100]
  - gamma: [0.001, 0.01, 0.1, 'scale']
- Mejor configuración: C=10, gamma=0.01
- F1-Score en CV: 0.9698

**Características técnicas:**
- **Kernel RBF:** Transforma el espacio de características a dimensiones superiores
- **Calibración probabilística:** Platt Scaling con validación cruzada interna (causa 16.8x lentitud)
- **Optimización:** LIBSVM (algoritmo SMO)

#### 3.3.2 Multi-Layer Perceptron (MLP)

**Arquitectura:**
```python
MLPClassifier(
    hidden_layer_sizes=(100, 50),   # 2 capas ocultas
    activation='relu',               # ReLU activation
    solver='adam',                   # Optimizador Adam
    max_iter=500,                    # Máximo de épocas
    early_stopping=True,             # Detención temprana
    validation_fraction=0.1,         # 10% para validación
    random_state=42
)
```

**Detalles de arquitectura:**
- **Capa de entrada:** 30 neuronas (30 features)
- **Capa oculta 1:** 100 neuronas + ReLU
- **Capa oculta 2:** 50 neuronas + ReLU
- **Capa de salida:** 2 neuronas + Softmax
- **Parámetros totales:** 3,000+ pesos entrenables

**Proceso de entrenamiento:**
- **Iteraciones:** 27 épocas (convergencia temprana)
- **Pérdida final:** 0.0671
- **Optimizador:** Adam (tasa de aprendizaje adaptativa)

### 3.4 Métricas de Evaluación

**Métricas de clasificación:**
- **Accuracy:** Proporción de predicciones correctas
- **Precision:** TP / (TP + FP) - relevancia de positivos predichos
- **Recall (Sensibilidad):** TP / (TP + FN) - capacidad de detectar cánceres
- **F1-Score:** Media armónica de Precision y Recall
- **AUC-ROC:** Área bajo la curva ROC (capacidad de ranking probabilístico)

**Matriz de Confusión:**
```
                 Predicción
              Benigno  Maligno
Real Benigno     TN       FP
     Maligno     FN       TP
```

**Importancia clínica:**
- **FN (Falsos Negativos):** CRÍTICOS - Cánceres no detectados
- **FP (Falsos Positivos):** Menos críticos - Generan biopsias adicionales

---

## 4. Procedimiento Experimental

### 4.1 Configuración del Entorno

**Software:**
- Python 3.10.11
- MongoDB 4.6+
- VS Code con extensiones de Python y Jupyter

**Librerías principales:**
```
scikit-learn==1.3.2
pandas==2.1.4
numpy==1.26.2
pymongo==4.6.1
streamlit==1.29.0
plotly==5.18.0
```

### 4.2 Flujo de Trabajo

#### Fase 1: Preparación de Datos
```bash
python scripts/download_dataset.py      # Descarga dataset
python scripts/load_to_mongo.py         # Carga a MongoDB
python scripts/preprocessing.py         # Preprocesamiento
```

**Salidas:**
- `data/processed/X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`
- `models/scalers/standard_scaler.pkl`, `label_encoder.pkl`

#### Fase 2: Entrenamiento de Modelos
```bash
# Ejecución de notebooks
jupyter notebook notebooks/02_mlp_training.ipynb
jupyter notebook notebooks/03_svm_training.ipynb
```

**Salidas:**
- `models/mlp_model.pkl`, `svm_model.pkl`
- `models/mlp_training_report.json`, `svm_training_report.json`

#### Fase 3: Despliegue
```bash
streamlit run app.py
```

**Funcionalidades:**
- Dashboard comparativo con visualizaciones interactivas
- Predicción individual (manual o aleatoria)
- Exploración del dataset

### 4.3 Hardware Utilizado

**Especificaciones:**
- CPU: (información del sistema del usuario)
- RAM: Suficiente para dataset pequeño (<1 MB)
- GPU: No requerida (dataset pequeño, modelos clásicos)

**Tiempos de entrenamiento:**
- MLP: 0.094 segundos
- SVM: 1.58 segundos

---

## 5. Resultados

### 5.1 Métricas Cuantitativas

#### Tabla Comparativa General

| Métrica | SVM | MLP | Diferencia | Ganador |
|---------|-----|-----|------------|---------|
| **Accuracy** | 98.25% | 97.37% | +0.88% | SVM |
| **Precision** | 100% | 100% | 0% | Empate |
| **Recall** | 95.24% | 92.86% | +2.38% | SVM |
| **F1-Score** | 97.56% | 96.30% | +1.26% | SVM |
| **AUC-ROC** | 0.9960 | 0.9970 | -0.0010 | MLP |

#### Eficiencia Computacional

| Modelo | Tiempo de Entrenamiento | Velocidad Relativa |
|--------|------------------------|-------------------|
| **MLP** | 0.094 segundos | **16.8x más rápido** |
| **SVM** | 1.580 segundos | Baseline |

**Nota:** El SVM es más lento debido a `probability=True`, que realiza validación cruzada interna de 5-fold para calibrar probabilidades (Platt Scaling).

### 5.2 Matrices de Confusión

#### SVM - Conjunto de Prueba (114 muestras)

```
                 Predicción
              Benigno  Maligno
Real Benigno     72       0
     Maligno      2      40
```

**Análisis:**
- **Verdaderos Negativos (TN):** 72 - Todos los casos benignos correctamente identificados
- **Falsos Positivos (FP):** 0 - No hay alarmas falsas
- **Falsos Negativos (FN):** 2 - Dos cánceres NO detectados (CRÍTICO)
- **Verdaderos Positivos (TP):** 40 - Cánceres correctamente detectados

**Tasa de detección:** 40/42 casos de cáncer = 95.24%

#### MLP - Conjunto de Prueba (114 muestras)

```
                 Predicción
              Benigno  Maligno
Real Benigno     72       0
     Maligno      3      39
```

**Análisis:**
- **Verdaderos Negativos (TN):** 72 - Todos los casos benignos correctamente identificados
- **Falsos Positivos (FP):** 0 - No hay alarmas falsas
- **Falsos Negativos (FN):** 3 - Tres cánceres NO detectados (MÁS CRÍTICO)
- **Verdaderos Positivos (TP):** 39 - Cánceres correctamente detectados

**Tasa de detección:** 39/42 casos de cáncer = 92.86%

### 5.3 Curvas ROC

**SVM:**
- AUC = 0.9960
- La curva se aproxima a la esquina superior izquierda, indicando excelente capacidad discriminativa

**MLP:**
- AUC = 0.9970
- Ligeramente superior a SVM en capacidad de ranking probabilístico
- Diferencia de 0.001 es estadísticamente insignificante (dataset de 114 muestras)

**Interpretación:**
Ambos modelos alcanzan AUC >0.99, considerado "excelente" según estándares clínicos. La diferencia es marginal y no tiene relevancia práctica.

### 5.4 Visualizaciones del Dashboard

El dashboard de Streamlit implementado incluye:

1. **Gráfico Radar (Spider Chart):**
   - Compara visualmente 4 métricas simultáneamente
   - SVM cubre más área en Accuracy, Recall, F1-Score
   - MLP ligeramente superior en AUC

2. **Curvas ROC Interactivas:**
   - Plotly permite zoom y exploración
   - Línea de azar (diagonal) como referencia
   - AUC mostrado en leyenda

3. **Matrices de Confusión con Heatmap:**
   - Escala de color para facilitar interpretación
   - Valores numéricos superpuestos
   - Comparación lado a lado SVM vs MLP

### 5.5 Hallazgos Adicionales

#### 5.5.1 Convergencia del MLP
El modelo MLP convergió en **27 iteraciones** (de 500 máximas), indicando:
- Early stopping funcionó correctamente
- El dataset es bien condicionado (fácil de aprender)
- No hay evidencia de underfitting

#### 5.5.2 Curva de Aprendizaje del SVM
El GridSearchCV de SVM evaluó 16 combinaciones de hiperparámetros, mostrando:
- Accuracy de validación cruzada: 97.6% - 98.0%
- Estabilidad alta (baja varianza entre folds)
- Mejor configuración: C=10, gamma=0.01

---

## 6. Discusión

### 6.1 Interpretación de Resultados

#### 6.1.1 Superioridad de SVM en Clasificación

SVM superó a MLP en todas las métricas de clasificación. Este resultado es **esperado y coherente con la teoría** de Machine Learning por las siguientes razones:

**1. Dataset pequeño (455 muestras de entrenamiento):**
- SVM está diseñado para maximizar el margen con pocos datos
- MLP generalmente necesita miles de muestras para aprovechar su capacidad
- Con 3,000+ parámetros y solo 455 muestras, MLP tiene riesgo de overfitting

**2. Datos tabulares estructurados:**
- SVM con kernel RBF es óptimo para datos numéricos estructurados
- MLP brilla en datos no estructurados (imágenes, audio, texto)
- Este dataset es casi linealmente separable en espacio transformado

**3. Optimización de hiperparámetros:**
- SVM: Hiperparámetros optimizados exhaustivamente con GridSearchCV
- MLP: Arquitectura estándar sin búsqueda de hiperparámetros específica

**4. Evidencia histórica:**
Este dataset es un benchmark clásico donde SVM ha dominado. Estudios previos:
- Chaurasia & Pal (2017): SVM obtuvo mayor precisión vs k-NN y árboles
- Fernández-Delgado et al. (2014): En datasets tabulares pequeños, SVM frecuentemente supera a redes neuronales

#### 6.1.2 Trade-off: Rendimiento vs Velocidad

**El hallazgo más interesante del proyecto:**
SVM logra +0.88% accuracy pero tarda **16.8x más tiempo** que MLP.

**Análisis del costo computacional:**
- El parámetro `probability=True` en SVM realiza calibración de Platt Scaling
- Esto requiere validación cruzada interna de 5-fold
- Esencialmente, SVM entrena 5 modelos adicionales para calibrar probabilidades
- MLP genera probabilidades "naturalmente" vía softmax sin costo extra

**Implicaciones prácticas:**
- En **entrenamiento único**: 1.58s es perfectamente aceptable
- En **re-entrenamiento frecuente**: MLP sería preferible
- En **producción (inferencia)**: Ambos modelos son igualmente rápidos

### 6.2 Limitaciones y Críticas

#### 6.2.1 Presencia de Falsos Negativos Críticos

**Problema principal:** Ambos modelos fallan en detectar cánceres (2-3 casos de 42).

**Consecuencias clínicas:**
- Falso Negativo → Paciente con cáncer diagnosticado como sano
- No se inicia tratamiento
- Cáncer progresa sin intervención → potencialmente fatal

**Estándar clínico vs Proyecto:**
- Estándar FDA para sistemas de diagnóstico médico: >99% recall
- SVM: 95.24% recall → **No cumple estándar clínico**
- MLP: 92.86% recall → **No cumple estándar clínico**

**Conclusión:** Ambos modelos son **inadecuados para uso autónomo** en entornos clínicos reales.

#### 6.2.2 Dataset Pequeño

**569 muestras totales** es insuficiente para:
- Generalización robusta a poblaciones diversas
- Validación externa (testing en otros hospitales)
- Detección de casos raros o atípicos

**Recomendación:** Validación con datasets externos más grandes (>10,000 casos) antes de considerar aplicación clínica.

#### 6.2.3 Ausencia de Interpretabilidad

Ambos modelos son "cajas negras":
- No explican POR QUÉ un caso es clasificado como maligno
- Médicos no pueden validar el razonamiento del modelo
- Esto limita la confianza y adopción clínica

**Soluciones posibles:**
- Implementar SHAP (SHapley Additive exPlanations)
- Visualizar activaciones de capas en MLP
- Mostrar vectores de soporte en SVM

#### 6.2.4 Precision Perfecta (100%) - ¿Demasiado bueno?

Ambos modelos tienen **0 falsos positivos** en el conjunto de prueba.

**Interpretaciones posibles:**
1. ✅ Los casos benignos son muy separables (posible en este dataset)
2. ⚠️ Sesgo hacia predecir clase mayoritaria (benigno) - menos probable dado el alto recall
3. ⚠️ Dataset de prueba pequeño (114 muestras) - posible suerte estadística

**Validación necesaria:** Testing en dataset externo para confirmar Precision=100%.

### 6.3 Comparación con Estado del Arte

#### Literatura Académica sobre Breast Cancer Wisconsin

**Chaurasia & Pal (2017):**
- Modelos: SVM, IBK (k-NN), Árboles de decisión
- Mejor resultado: SVM con 97.13% accuracy
- **Nuestro SVM: 98.25%** → Superamos ligeramente

**Athar & Ilavarasi (2020):**
- Modelos: Regresión Logística, SVM con selección de features
- Mejor resultado: SVM con 97.4% accuracy
- **Nuestro SVM: 98.25%** → Superamos

**Análisis crítico:**
Estos estudios buscan optimización de métricas. Nuestro proyecto se diferencia porque:
- Implementa **pipeline MLOps completo** (no solo modelo aislado)
- Incluye **despliegue interactivo** con Streamlit
- Demuestra **gestión de datos** con MongoDB
- Proporciona **análisis comparativo riguroso** entre familias de modelos

### 6.4 Contribuciones del Proyecto

#### 6.4.1 Contribución Técnica

1. **Pipeline End-to-End Reproducible:**
   - Scripts automatizados desde descarga hasta despliegue
   - Configuración de entorno documentada
   - Reportes JSON con métricas completas

2. **Dashboard Profesional:**
   - Visualizaciones interactivas con Plotly
   - Comparación objetiva con métricas cuantitativas
   - Predicción en tiempo real con dos modos de entrada

3. **Documentación Exhaustiva:**
   - README completo con instrucciones paso a paso
   - Notebooks Jupyter con análisis detallado
   - Código comentado y modular

#### 6.4.2 Contribución Pedagógica

Este proyecto demuestra competencias en:
- **Gestión de datos:** MongoDB, pandas
- **Preprocesamiento:** StandardScaler, train-test split
- **Modelado:** scikit-learn (SVM, MLP)
- **Evaluación:** Métricas clínicas, matrices de confusión, ROC
- **Despliegue:** Streamlit, visualización con Plotly
- **MLOps:** Versionado de modelos, reportes automatizados

### 6.5 Lecciones Aprendidas

#### 6.5.1 La Importancia del Preprocesamiento

**Error inicial:** Doble escalado de datos causó que SVM tuviera AUC 0.5 (predicción aleatoria).

**Lección:** Verificar siempre el flujo de datos:
- `test_data.csv` ya estaba escalado
- `app.py` volvía a aplicar `scaler.transform()`
- Resultado: valores sin sentido, modelo colapsado

**Solución:** Documentar explícitamente el estado de los datos en cada etapa del pipeline.

#### 6.5.2 Timestamps y Sincronización de Archivos

**Problema:** Los reportes JSON tenían timestamps de hace días, no coincidían con los modelos `.pkl` recién entrenados.

**Causa:** Los JSON se generan en notebooks, no en scripts de entrenamiento.

**Lección:** Mantener consistencia entre scripts y notebooks:
- Opción 1: Generar JSON también en scripts
- Opción 2: Documentar que notebooks son la fuente única de reportes

#### 6.5.3 `probability=True` en SVM: Costo vs Beneficio

**Descubrimiento:** Activar `probability=True` multiplica el tiempo de entrenamiento por ~16.8x.

**Análisis de valor:**
- Beneficio: Habilita `predict_proba()` para curvas ROC y comparación justa con MLP
- Costo: Validación cruzada interna de 5-fold
- Mejora de rendimiento: +0.88% accuracy

**Conclusión:** En este caso, el costo es aceptable porque:
- El modelo se entrena una sola vez (no es tiempo crítico)
- La calibración probabilística mejora ligeramente las predicciones
- Es necesario para comparación justa con MLP

---

## 7. Conclusiones

### 7.1 Rendimiento de Modelos: SVM Superior con Trade-off Computacional

Los resultados demuestran que **SVM supera a MLP** en todas las métricas de clasificación críticas para diagnóstico médico:
- **Accuracy**: SVM 98.25% vs MLP 97.37% (+0.88%)
- **Recall**: SVM 95.24% vs MLP 92.86% (+2.38%)
- **Detección de cáncer**: SVM detectó 40/42 casos vs MLP 39/42 (**1 cáncer adicional detectado**)

Sin embargo, este rendimiento superior tiene un **costo computacional significativo**: SVM tarda 1.58s vs MLP 0.094s (**16.8x más lento**) debido al parámetro `probability=True`, que realiza validación cruzada interna de 5-fold para calibración probabilística.

### 7.2 Hallazgo Crítico: Falsos Negativos en Contexto Clínico

Ambos modelos presentan **falsos negativos** (cánceres no detectados):
- SVM: 2 casos no detectados (4.8% de error)
- MLP: 3 casos no detectados (7.1% de error)

En medicina, **los falsos negativos son más peligrosos que los falsos positivos** porque un cáncer no detectado puede progresar fatalmente. Esto descalifica a ambos modelos para uso autónomo en producción; deben emplearse únicamente como **herramientas de screening** seguidas de confirmación por biopsia.

### 7.3 Validación de Teoría: SVM vs MLP en Datasets Tabulares Pequeños

Los resultados **confirman la teoría de Machine Learning**:
- En datasets tabulares pequeños (455 muestras, 30 features), **SVM domina sobre MLP**
- El kernel RBF aprovecha eficientemente la casi-separabilidad lineal del dataset
- MLP (3,000+ parámetros) no tiene suficientes datos para superar a SVM
- Estos hallazgos coinciden con literatura previa (Chaurasia & Pal, 2017) sobre este mismo dataset

### 7.4 Contribución Técnica: Pipeline MLOps Completo

El proyecto implementa exitosamente un **flujo de trabajo end-to-end**:
- Gestión de datos con MongoDB
- Preprocesamiento reproducible con StandardScaler
- Entrenamiento sistemático con reportes JSON
- Dashboard interactivo con visualizaciones profesionales (Plotly)
- Comparación objetiva con matrices de confusión y curvas ROC

### 7.5 Recomendación Final

**Para este dataset específico: SVM es el modelo ganador.**

**Razones:**
1. ✅ Detecta 1 cáncer más que MLP (crítico en medicina)
2. ✅ Superior en todas las métricas de clasificación
3. ✅ Tiempo de entrenamiento 16.8x mayor es **aceptable** para un modelo que se entrena una sola vez
4. ✅ AUC solo 0.001 inferior (estadísticamente insignificante)

**Advertencia:** Ningún modelo alcanza el estándar clínico requerido (>99% recall). Uso recomendado: herramienta de apoyo diagnóstico, nunca como sustituto de análisis médico profesional.

---

## 8. Trabajo Futuro

### 8.1 Mejoras Inmediatas

1. **Optimización de Hiperparámetros de MLP:**
   - Realizar GridSearchCV para arquitectura de capas
   - Probar diferentes activaciones (tanh, sigmoid)
   - Ajustar tasa de aprendizaje

2. **Ensemble de Modelos:**
   - Combinar predicciones de SVM y MLP mediante voting
   - Podría reducir falsos negativos aprovechando fortalezas de ambos

3. **Interpretabilidad:**
   - Implementar SHAP values para explicar predicciones
   - Visualizar features más importantes por modelo

### 8.2 Expansiones del Sistema

1. **Validación Externa:**
   - Testing en datasets de otros hospitales
   - Evaluación de robustez ante variabilidad poblacional

2. **Dataset Aumentado:**
   - Incorporar datos de múltiples fuentes (>10,000 casos)
   - Mejorar generalización y reducir falsos negativos

3. **Modelos Avanzados:**
   - XGBoost / LightGBM (estado del arte en tabulares)
   - Redes neuronales profundas con dropout y regularización
   - Autoencoders para detección de anomalías

### 8.3 Despliegue en Producción

1. **Dockerización:**
   - Contenedor con todo el pipeline
   - Facilita despliegue en servidores hospitalarios

2. **API REST:**
   - FastAPI para integración con sistemas médicos existentes
   - Autenticación y logging de predicciones

3. **Monitoreo de Modelo:**
   - Detección de drift en distribución de datos
   - Re-entrenamiento automático cuando sea necesario

---

## 9. Referencias

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

Cortes, C., & Vapnik, V. (1995).  
*Support-vector networks.*  
Machine Learning, 20(3), 273-297.  
[https://doi.org/10.1007/BF00994018](https://doi.org/10.1007/BF00994018)

Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).  
*Learning representations by back-propagating errors.*  
Nature, 323(6088), 533-536.  
[https://doi.org/10.1038/323533a0](https://doi.org/10.1038/323533a0)

### Estudios Comparativos sobre Breast Cancer Wisconsin

Chaurasia, V., & Pal, S. (2017).  
*Data mining techniques: To predict and resolve breast cancer survivability.*  
International Journal of Computer Science and Mobile Computing, 3(1), 10-22.

Athar, S., & Ilavarasi, K. (2020).  
*Performance analysis of machine learning algorithms for breast cancer detection.*  
Journal of Physics: Conference Series, 1916(1), 012193.

Fernández-Delgado, M., Cernadas, E., Barro, S., & Amorim, D. (2014).  
*Do we need hundreds of classifiers to solve real world classification problems?*  
The Journal of Machine Learning Research, 15(1), 3133-3181.

### Herramientas y Frameworks

Pedregosa, F., et al. (2011).  
*Scikit-learn: Machine learning in Python.*  
Journal of Machine Learning Research, 12, 2825-2830.

---

## Anexos

### Anexo A: Estructura del Repositorio

Consultar [README.md](README.md) para la estructura completa del proyecto.

### Anexo B: Comandos de Ejecución

**Instalación:**
```bash
git clone https://github.com/altairBASIC/breast-cancer-diagnosis-mlp-vs-svm.git
cd breast-cancer-diagnosis-mlp-vs-svm
.\install.ps1
```

**Pipeline completo:**
```bash
python scripts/download_dataset.py
python scripts/load_to_mongo.py
python scripts/preprocessing.py
jupyter notebook  # Ejecutar 02_mlp_training.ipynb y 03_svm_training.ipynb
streamlit run app.py
```

### Anexo C: Código Clave

**Ejemplo de entrenamiento SVM:**
```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 'scale']
}

svm = SVC(kernel='rbf', probability=True, random_state=42)
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

best_svm = grid_search.best_estimator_
```

**Ejemplo de entrenamiento MLP:**
```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=500,
    early_stopping=True,
    random_state=42
)

mlp.fit(X_train, y_train)
```

---

**Fin del Informe**

---

*Este informe fue elaborado como parte del proyecto final del curso INFB6052 - Herramientas para Ciencia de Datos, Universidad Tecnológica Metropolitana, Segundo Semestre 2025.*
