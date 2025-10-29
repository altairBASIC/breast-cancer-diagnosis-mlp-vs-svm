"""
Script de Preprocesamiento de Datos
====================================
Este script implementa el pipeline de preprocesamiento para el proyecto de 
diagnóstico de cáncer de mama.

Funcionalidades:
1. Lectura de datos desde MongoDB
2. Análisis exploratorio de datos (EDA)
3. Limpieza de datos
4. Codificación de variables categóricas
5. Normalización/Estandarización de características
6. División en conjuntos de entrenamiento y prueba
7. Guardado de datos preprocesados

Autor: altairBASIC
Fecha: Octubre 2025
"""

import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime
import joblib
import os
import json


class DataPreprocessor:
    """
    Clase para preprocesar datos del dataset de cáncer de mama.
    """
    
    def __init__(self, mongo_uri="mongodb://localhost:27017/", 
                 db_name="breast_cancer_db", 
                 collection_name="patients_records"):
        """
        Inicializa el preprocesador de datos.
        
        Args:
            mongo_uri (str): URI de conexión a MongoDB
            db_name (str): Nombre de la base de datos
            collection_name (str): Nombre de la colección
        """
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        
        # Crear directorio para datos preprocesados si no existe
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('models/scalers', exist_ok=True)
        
    def log(self, message):
        """Imprime y registra un mensaje con timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
    def connect_and_load_data(self):
        """
        Conecta a MongoDB y carga los datos en un DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame con los datos cargados
        """
        self.log("="*70)
        self.log("CARGA DE DATOS DESDE MONGODB")
        self.log("="*70)
        
        try:
            # Conectar a MongoDB
            self.log(f"Conectando a MongoDB: {self.mongo_uri}")
            client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            
            # Verificar conexión
            client.admin.command('ping')
            self.log(" Conexión exitosa a MongoDB")
            
            # Seleccionar base de datos y colección
            db = client[self.db_name]
            collection = db[self.collection_name]
            
            # Contar documentos
            doc_count = collection.count_documents({})
            self.log(f" Documentos encontrados: {doc_count}")
            
            # Cargar datos en DataFrame
            self.log("Cargando datos en DataFrame...")
            cursor = collection.find({})
            self.df = pd.DataFrame(list(cursor))
            
            # Cerrar conexión
            client.close()
            
            self.log(f" Datos cargados: {self.df.shape[0]} filas, {self.df.shape[1]} columnas")
            self.log("")
            
            return self.df
            
        except Exception as e:
            self.log(f" Error al cargar datos: {str(e)}")
            raise
    
    def explore_data(self):
        """
        Realiza un análisis exploratorio de los datos.
        """
        self.log("="*70)
        self.log("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
        self.log("="*70)
        
        if self.df is None:
            self.log(" No hay datos cargados. Ejecute connect_and_load_data() primero.")
            return
        
        # Información general
        self.log(f"\n Información General:")
        self.log(f"   - Dimensiones: {self.df.shape}")
        self.log(f"   - Columnas: {len(self.df.columns)}")
        
        # Tipos de datos
        self.log(f"\n Tipos de Datos:")
        dtype_counts = self.df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            self.log(f"   - {dtype}: {count} columnas")
        
        # Valores faltantes
        self.log(f"\n Valores Faltantes:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            self.log(f"    No hay valores faltantes")
        else:
            missing_cols = missing[missing > 0]
            for col, count in missing_cols.items():
                pct = (count / len(self.df)) * 100
                self.log(f"   - {col}: {count} ({pct:.2f}%)")
        
        # Distribución de la variable objetivo
        if 'diagnosis' in self.df.columns:
            self.log(f"\n Distribución de Diagnóstico:")
            diagnosis_dist = self.df['diagnosis'].value_counts()
            total = len(self.df)
            for diagnosis, count in diagnosis_dist.items():
                pct = (count / total) * 100
                label = "Benigno" if diagnosis == 'B' else "Maligno"
                self.log(f"   - {diagnosis} ({label}): {count} ({pct:.2f}%)")
        
        # Estadísticas descriptivas de características numéricas
        self.log(f"\n Estadísticas de Características Numéricas:")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['_id', 'id']]
        
        if len(numeric_cols) > 0:
            stats = self.df[numeric_cols].describe()
            self.log(f"   - Número de características numéricas: {len(numeric_cols)}")
            self.log(f"   - Rango de medias: [{stats.loc['mean'].min():.4f}, {stats.loc['mean'].max():.4f}]")
            self.log(f"   - Rango de desv. estándar: [{stats.loc['std'].min():.4f}, {stats.loc['std'].max():.4f}]")
        
        self.log("")
        
    def clean_data(self):
        """
        Limpia y prepara los datos para el modelado.
        """
        self.log("="*70)
        self.log("LIMPIEZA DE DATOS")
        self.log("="*70)
        
        if self.df is None:
            self.log(" No hay datos cargados.")
            return
        
        original_shape = self.df.shape
        
        # Eliminar columnas innecesarias
        columns_to_drop = ['_id', 'inserted_at', 'source']
        existing_cols_to_drop = [col for col in columns_to_drop if col in self.df.columns]
        
        if existing_cols_to_drop:
            self.df = self.df.drop(columns=existing_cols_to_drop)
            self.log(f" Columnas eliminadas: {existing_cols_to_drop}")
        
        # Eliminar duplicados si existen
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            self.df = self.df.drop_duplicates()
            self.log(f" Duplicados eliminados: {duplicates}")
        else:
            self.log(f" No se encontraron duplicados")
        
        # Eliminar filas con valores faltantes (si existen)
        missing_rows = self.df.isnull().any(axis=1).sum()
        if missing_rows > 0:
            self.df = self.df.dropna()
            self.log(f" Filas con valores faltantes eliminadas: {missing_rows}")
        else:
            self.log(f" No hay filas con valores faltantes")
        
        # Verificar que 'diagnosis' existe
        if 'diagnosis' not in self.df.columns:
            self.log(" Columna 'diagnosis' no encontrada en los datos")
            raise ValueError("La columna 'diagnosis' es requerida")
        
        self.log(f"\n Datos después de limpieza:")
        self.log(f"   - Forma original: {original_shape}")
        self.log(f"   - Forma final: {self.df.shape}")
        self.log("")
        
    def encode_labels(self):
        """
        Codifica la variable objetivo (diagnosis) a valores numéricos.
        M (Maligno) -> 1
        B (Benigno) -> 0
        """
        self.log("="*70)
        self.log("CODIFICACIÓN DE ETIQUETAS")
        self.log("="*70)
        
        if self.df is None:
            self.log(" No hay datos cargados.")
            return
        
        # Crear codificador de etiquetas
        self.label_encoder = LabelEncoder()
        
        # Codificar diagnosis
        self.df['diagnosis_encoded'] = self.label_encoder.fit_transform(self.df['diagnosis'])
        
        # Mostrar mapeo
        self.log(" Codificación de diagnóstico:")
        for i, label in enumerate(self.label_encoder.classes_):
            diagnosis_name = "Benigno" if label == 'B' else "Maligno"
            self.log(f"   - {label} ({diagnosis_name}) -> {i}")
        
        # Guardar el codificador
        encoder_path = 'models/scalers/label_encoder.pkl'
        joblib.dump(self.label_encoder, encoder_path)
        self.log(f" Label Encoder guardado en: {encoder_path}")
        self.log("")
        
    def split_data(self, test_size=0.2, random_state=42):
        """
        Divide los datos en conjuntos de entrenamiento y prueba.
        
        Args:
            test_size (float): Proporción de datos para prueba (default: 0.2)
            random_state (int): Semilla para reproducibilidad (default: 42)
        """
        self.log("="*70)
        self.log("DIVISIÓN DE DATOS")
        self.log("="*70)
        
        if self.df is None:
            self.log(" No hay datos cargados.")
            return
        
        # Separar características (X) y variable objetivo (y)
        # Excluir columnas no numéricas y la columna objetivo original
        exclude_cols = ['diagnosis', 'diagnosis_encoded', 'id', 'Unnamed: 32']
        
        self.feature_names = [col for col in self.df.columns 
                              if col not in exclude_cols and 
                              self.df[col].dtype in [np.float64, np.int64]]
        
        X = self.df[self.feature_names].values
        y = self.df['diagnosis_encoded'].values
        
        self.log(f" Características seleccionadas: {len(self.feature_names)}")
        self.log(f"   Forma de X: {X.shape}")
        self.log(f"   Forma de y: {y.shape}")
        
        # Dividir datos
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.log(f"\n División completada:")
        self.log(f"   - Conjunto de entrenamiento: {self.X_train.shape[0]} muestras ({(1-test_size)*100:.0f}%)")
        self.log(f"   - Conjunto de prueba: {self.X_test.shape[0]} muestras ({test_size*100:.0f}%)")
        
        # Verificar distribución de clases
        train_dist = np.bincount(self.y_train)
        test_dist = np.bincount(self.y_test)
        
        self.log(f"\n Distribución de clases en entrenamiento:")
        self.log(f"   - Benigno (0): {train_dist[0]} ({train_dist[0]/len(self.y_train)*100:.2f}%)")
        self.log(f"   - Maligno (1): {train_dist[1]} ({train_dist[1]/len(self.y_train)*100:.2f}%)")
        
        self.log(f"\n Distribución de clases en prueba:")
        self.log(f"   - Benigno (0): {test_dist[0]} ({test_dist[0]/len(self.y_test)*100:.2f}%)")
        self.log(f"   - Maligno (1): {test_dist[1]} ({test_dist[1]/len(self.y_test)*100:.2f}%)")
        self.log("")
        
    def scale_features(self):
        """
        Estandariza las características usando StandardScaler.
        Ajusta el scaler solo con datos de entrenamiento y transforma ambos conjuntos.
        """
        self.log("="*70)
        self.log("ESCALADO DE CARACTERÍSTICAS")
        self.log("="*70)
        
        if self.X_train is None:
            self.log(" Los datos no han sido divididos. Ejecute split_data() primero.")
            return
        
        # Crear y ajustar el scaler
        self.scaler = StandardScaler()
        
        # Ajustar solo con datos de entrenamiento
        self.scaler.fit(self.X_train)
        
        # Transformar ambos conjuntos
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        self.log(" Características estandarizadas (StandardScaler)")
        self.log(f"   - Media de entrenamiento: ~0.0")
        self.log(f"   - Desviación estándar de entrenamiento: ~1.0")
        
        # Verificar estadísticas
        train_mean = np.mean(self.X_train, axis=0)
        train_std = np.std(self.X_train, axis=0)
        
        self.log(f"\n Estadísticas después del escalado:")
        self.log(f"   - Media de características (train): min={train_mean.min():.6f}, max={train_mean.max():.6f}")
        self.log(f"   - Desv. est. de características (train): min={train_std.min():.6f}, max={train_std.max():.6f}")
        
        # Guardar el scaler
        scaler_path = 'models/scalers/standard_scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        self.log(f"\n Scaler guardado en: {scaler_path}")
        self.log("")
        
    def save_preprocessed_data(self):
        """
        Guarda los datos preprocesados en archivos para uso posterior.
        """
        self.log("="*70)
        self.log("GUARDADO DE DATOS PREPROCESADOS")
        self.log("="*70)
        
        if self.X_train is None:
            self.log(" No hay datos preprocesados para guardar.")
            return
        
        try:
            # Guardar arrays de numpy
            np.save('data/processed/X_train.npy', self.X_train)
            np.save('data/processed/X_test.npy', self.X_test)
            np.save('data/processed/y_train.npy', self.y_train)
            np.save('data/processed/y_test.npy', self.y_test)
            
            self.log(" Arrays guardados:")
            self.log(f"   - data/processed/X_train.npy")
            self.log(f"   - data/processed/X_test.npy")
            self.log(f"   - data/processed/y_train.npy")
            self.log(f"   - data/processed/y_test.npy")
            
            # Guardar nombres de características
            feature_info = {
                'feature_names': self.feature_names,
                'n_features': len(self.feature_names),
                'n_train_samples': self.X_train.shape[0],
                'n_test_samples': self.X_test.shape[0],
                'preprocessing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open('data/processed/feature_info.json', 'w') as f:
                json.dump(feature_info, f, indent=4)
            
            self.log(f" Información de características guardada en: data/processed/feature_info.json")
            
            # Guardar resumen en CSV para inspección
            train_df = pd.DataFrame(self.X_train, columns=self.feature_names)
            train_df['diagnosis'] = self.y_train
            train_df.to_csv('data/processed/train_data.csv', index=False)
            
            test_df = pd.DataFrame(self.X_test, columns=self.feature_names)
            test_df['diagnosis'] = self.y_test
            test_df.to_csv('data/processed/test_data.csv', index=False)
            
            self.log(f" Datos en formato CSV guardados:")
            self.log(f"   - data/processed/train_data.csv")
            self.log(f"   - data/processed/test_data.csv")
            self.log("")
            
        except Exception as e:
            self.log(f" Error al guardar datos: {str(e)}")
            raise
    
    def generate_preprocessing_report(self):
        """
        Genera un reporte completo del preprocesamiento.
        """
        self.log("="*70)
        self.log("REPORTE DE PREPROCESAMIENTO")
        self.log("="*70)
        
        report = {
            "fecha_procesamiento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "datos_originales": {
                "total_registros": len(self.df) if self.df is not None else 0,
                "total_caracteristicas": len(self.df.columns) if self.df is not None else 0
            },
            "datos_procesados": {
                "caracteristicas_seleccionadas": len(self.feature_names) if self.feature_names else 0,
                "nombres_caracteristicas": self.feature_names if self.feature_names else [],
                "entrenamiento": {
                    "total_muestras": self.X_train.shape[0] if self.X_train is not None else 0,
                    "clase_0_benigno": int(np.sum(self.y_train == 0)) if self.y_train is not None else 0,
                    "clase_1_maligno": int(np.sum(self.y_train == 1)) if self.y_train is not None else 0
                },
                "prueba": {
                    "total_muestras": self.X_test.shape[0] if self.X_test is not None else 0,
                    "clase_0_benigno": int(np.sum(self.y_test == 0)) if self.y_test is not None else 0,
                    "clase_1_maligno": int(np.sum(self.y_test == 1)) if self.y_test is not None else 0
                }
            },
            "transformaciones": {
                "escalado": "StandardScaler",
                "codificacion": "LabelEncoder (B->0, M->1)"
            },
            "archivos_generados": [
                "data/processed/X_train.npy",
                "data/processed/X_test.npy",
                "data/processed/y_train.npy",
                "data/processed/y_test.npy",
                "data/processed/train_data.csv",
                "data/processed/test_data.csv",
                "data/processed/feature_info.json",
                "models/scalers/standard_scaler.pkl",
                "models/scalers/label_encoder.pkl"
            ]
        }
        
        # Guardar reporte
        report_path = 'data/processed/preprocessing_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        self.log(f" Reporte de preprocesamiento guardado en: {report_path}")
        
        # Mostrar resumen
        self.log(f"\n RESUMEN DEL PREPROCESAMIENTO:")
        self.log(f"   - Características procesadas: {report['datos_procesados']['caracteristicas_seleccionadas']}")
        self.log(f"   - Muestras de entrenamiento: {report['datos_procesados']['entrenamiento']['total_muestras']}")
        self.log(f"   - Muestras de prueba: {report['datos_procesados']['prueba']['total_muestras']}")
        self.log(f"   - Método de escalado: {report['transformaciones']['escalado']}")
        self.log(f"   - Archivos generados: {len(report['archivos_generados'])}")
        self.log("")
        
        return report
    
    def run_full_pipeline(self, test_size=0.2, random_state=42):
        """
        Ejecuta el pipeline completo de preprocesamiento.
        
        Args:
            test_size (float): Proporción de datos para prueba
            random_state (int): Semilla para reproducibilidad
        """
        self.log("\n" + "="*70)
        self.log("INICIANDO PIPELINE DE PREPROCESAMIENTO")
        self.log("="*70 + "\n")
        
        try:
            # 1. Cargar datos
            self.connect_and_load_data()
            
            # 2. Análisis exploratorio
            self.explore_data()
            
            # 3. Limpieza de datos
            self.clean_data()
            
            # 4. Codificación de etiquetas
            self.encode_labels()
            
            # 5. División de datos
            self.split_data(test_size=test_size, random_state=random_state)
            
            # 6. Escalado de características
            self.scale_features()
            
            # 7. Guardar datos preprocesados
            self.save_preprocessed_data()
            
            # 8. Generar reporte
            report = self.generate_preprocessing_report()
            
            self.log("="*70)
            self.log(" PIPELINE DE PREPROCESAMIENTO COMPLETADO EXITOSAMENTE")
            self.log("="*70)
            
            return report
            
        except Exception as e:
            self.log(f"\n❌ Error en el pipeline de preprocesamiento: {str(e)}")
            raise


def main():
    """
    Función principal para ejecutar el preprocesamiento.
    """
    print("\n" + "="*70)
    print("SCRIPT DE PREPROCESAMIENTO DE DATOS")
    print("Breast Cancer Diagnosis - MLP vs SVM")
    print("="*70 + "\n")
    
    # Crear instancia del preprocesador
    preprocessor = DataPreprocessor()
    
    # Ejecutar pipeline completo
    report = preprocessor.run_full_pipeline(test_size=0.2, random_state=42)
    
    print("\n" + "="*70)
    print("PREPROCESAMIENTO FINALIZADO")
    print("="*70)
    print("\nLos datos están listos para el entrenamiento de modelos.")
    print("Archivos generados en:")
    print("  - data/processed/")
    print("  - models/scalers/")
    print("\n")


if __name__ == "__main__":
    main()
