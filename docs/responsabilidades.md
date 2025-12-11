# Responsabilidades del Equipo

Este documento resume, de forma estructurada, las principales responsabilidades de cada integrante del proyecto.

> Nota: ajusta libremente los puntos que aparecen a continuación para que reflejen **exactamente** el trabajo realizado por cada persona según el acuerdo del equipo o los requisitos del curso.

---

## Ignacio Ramírez

Enfoque general: **infraestructura de datos, preprocesamiento y modelos de Machine Learning**.

- Configuración inicial del proyecto y estructura de carpetas.
- Preparación del entorno de desarrollo (entornos virtuales, `requirements.txt`, scripts de instalación).
- Implementación y mantenimiento de scripts en `scripts/` para:
  - Descarga y análisis del dataset original.
  - Preprocesamiento de datos (limpieza, codificación, normalización, partición train/test).
  - Integración opcional con MongoDB (carga y verificación de colecciones).
- Desarrollo y validación del pipeline de preprocesamiento en notebooks (EDA, generación de datos procesados).
- Entrenamiento y ajuste del/los modelo(s) de clasificación basados en redes neuronales (MLP) y su evaluación principal.
- Colaboración en la definición de métricas de rendimiento y en la interpretación de resultados.

---

## Cristián Vergara

Enfoque general: **modelos SVM, análisis comparativo y soporte en backend**.

- Diseño e implementación del/los modelo(s) de clasificación SVM (Support Vector Machine).
- Experimentación con hiperparámetros de SVM y comparación con el desempeño del MLP.
- Análisis y generación de métricas clave (accuracy, precisión, recall, F1, AUC, etc.).
- Participación en la construcción de notebooks de entrenamiento y evaluación de modelos.
- Apoyo en la integración de los modelos entrenados con la capa de persistencia (archivos en `models/`, uso de `joblib`).
- Contribución al análisis comparativo entre MLP y SVM (matrices de confusión, curvas, tablas resumen).

---

## Antonia Montecinos

Enfoque general: **documentación, interfaz de usuario y coordinación del proyecto**.

- Redacción y organización de la documentación principal del proyecto:
  - `README.md` (descripción general y uso).
  - `docs/instrucciones.md` (guía de instalación y configuración).
  - `docs/narrativa.md` (narrativa extendida, explicación del dataset y conceptos teóricos).
- Diseño y mejora de la interfaz web en Streamlit (`app.py` y módulos en `ui/`):
  - Estructura de navegación.
  - Vistas de exploración de datos, análisis de modelos y probadores de casos.
  - Textos explicativos, ayudas contextuales y mensajes para usuarios no técnicos.
- Apoyo en el análisis exploratorio y en la presentación visual de resultados (gráficos, tablas, resúmenes).
- Coordinación general del trabajo en equipo (planificación, integración de entregables, revisión final antes de la subida a GitHub).

---

## Trabajo Colaborativo

Además de las responsabilidades principales anteriores, todos los integrantes colaboraron en:

- Definir los objetivos del proyecto y el alcance técnico.
- Revisar y validar los resultados de los modelos antes de su presentación.
- Discutir decisiones de diseño (selección de métricas, estructura de la app, organización del repositorio).
- Preparar el proyecto para su entrega/defensa (revisión de código, limpieza del repositorio y documentación).
