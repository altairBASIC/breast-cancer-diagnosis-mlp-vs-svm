"""
Script para descargar el dataset Breast Cancer Wisconsin (Diagnostic)
y generar un resumen estadístico.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_dataset(save_path: str = '../data/breast_cancer.csv') -> pd.DataFrame:
    """
    Descarga el dataset Breast Cancer Wisconsin desde sklearn.
    
    Args:
        save_path: Ruta donde guardar el archivo CSV
        
    Returns:
        DataFrame con el dataset
    """
    logger.info("Descargando dataset Breast Cancer Wisconsin...")
    
    try:
        from sklearn.datasets import load_breast_cancer
        
        # Cargar dataset desde sklearn
        data = load_breast_cancer()
        
        # Crear DataFrame
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['diagnosis'] = data.target
        
        # Mapear 0=malignant, 1=benign para que coincida con el dataset original
        # En el dataset original: M=Malignant, B=Benign
        df['diagnosis'] = df['diagnosis'].map({0: 'M', 1: 'B'})
        
        # Guardar CSV
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, save_path)
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        df.to_csv(csv_path, index=False)
        logger.info(f"Dataset guardado en: {csv_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error al descargar dataset: {e}")
        raise


def analyze_dataset(df: pd.DataFrame, summary_path: str = '../data/data_summary.txt') -> None:
    """
    Genera un análisis estadístico completo del dataset.
    
    Args:
        df: DataFrame con el dataset
        summary_path: Ruta donde guardar el resumen
    """
    logger.info("Generando análisis estadístico del dataset...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_file = os.path.join(script_dir, summary_path)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RESUMEN ESTADÍSTICO - BREAST CANCER WISCONSIN (DIAGNOSTIC) DATASET\n")
        f.write("="*80 + "\n")
        f.write(f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 1. Información general del dataset
        f.write("-"*80 + "\n")
        f.write("1. INFORMACIÓN GENERAL DEL DATASET\n")
        f.write("-"*80 + "\n")
        f.write(f"Número total de registros: {df.shape[0]}\n")
        f.write(f"Número total de características: {df.shape[1] - 1}\n")  # -1 por la columna diagnosis
        f.write(f"Dimensión del dataset: {df.shape}\n\n")
        
        # 2. Nombres de columnas
        f.write("-"*80 + "\n")
        f.write("2. NOMBRES DE COLUMNAS\n")
        f.write("-"*80 + "\n")
        feature_cols = [col for col in df.columns if col != 'diagnosis']
        for i, col in enumerate(feature_cols, 1):
            f.write(f"  {i:2d}. {col}\n")
        f.write(f"  {len(feature_cols)+1:2d}. diagnosis (variable objetivo)\n\n")
        
        # 3. Tipos de datos
        f.write("-"*80 + "\n")
        f.write("3. TIPOS DE DATOS\n")
        f.write("-"*80 + "\n")
        f.write(str(df.dtypes) + "\n\n")
        
        # 4. Valores faltantes
        f.write("-"*80 + "\n")
        f.write("4. VALORES FALTANTES\n")
        f.write("-"*80 + "\n")
        missing = df.isnull().sum()
        total_missing = missing.sum()
        f.write(f"Total de valores faltantes: {total_missing}\n")
        if total_missing > 0:
            f.write("\nValores faltantes por columna:\n")
            f.write(str(missing[missing > 0]) + "\n\n")
        else:
            f.write("No se encontraron valores faltantes en el dataset\n\n")
        
        # 5. Distribución de clases
        f.write("-"*80 + "\n")
        f.write("5. DISTRIBUCIÓN DE CLASES (DIAGNÓSTICO)\n")
        f.write("-"*80 + "\n")
        class_counts = df['diagnosis'].value_counts()
        class_pct = df['diagnosis'].value_counts(normalize=True) * 100
        f.write(f"Benign (B):    {class_counts.get('B', 0):3d} muestras ({class_pct.get('B', 0):5.2f}%)\n")
        f.write(f"Malignant (M): {class_counts.get('M', 0):3d} muestras ({class_pct.get('M', 0):5.2f}%)\n")
        f.write(f"Total:         {class_counts.sum():3d} muestras\n\n")
        
        # 6. Estadísticas descriptivas
        f.write("-"*80 + "\n")
        f.write("6. ESTADÍSTICAS DESCRIPTIVAS (CARACTERÍSTICAS NUMÉRICAS)\n")
        f.write("-"*80 + "\n")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        desc = df[numeric_cols].describe()
        f.write(str(desc) + "\n\n")
        
        # 7. Rangos de valores
        f.write("-"*80 + "\n")
        f.write("7. RANGOS DE VALORES (MIN - MAX)\n")
        f.write("-"*80 + "\n")
        for col in feature_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            f.write(f"  {col}:\n")
            f.write(f"    Min: {min_val:.4f}, Max: {max_val:.4f}\n")
        f.write("\n")
        
        # 8. Observaciones y notas
        f.write("-"*80 + "\n")
        f.write("8. OBSERVACIONES Y NOTAS\n")
        f.write("-"*80 + "\n")
        f.write("  - El dataset contiene 30 características numéricas calculadas a partir\n")
        f.write("    de imágenes digitalizadas de aspiraciones con aguja fina (FNA) de\n")
        f.write("    masas mamarias.\n\n")
        f.write("  - Las características describen propiedades de los núcleos celulares\n")
        f.write("    presentes en las imágenes.\n\n")
        f.write("  - Para cada característica se calculan tres valores:\n")
        f.write("    * mean: media de los valores\n")
        f.write("    * error: error estándar\n")
        f.write("    * worst: media de los tres valores más grandes\n\n")
        f.write("  - Las 10 características base son:\n")
        f.write("    1. radius (radio)\n")
        f.write("    2. texture (textura)\n")
        f.write("    3. perimeter (perímetro)\n")
        f.write("    4. area (área)\n")
        f.write("    5. smoothness (suavidad)\n")
        f.write("    6. compactness (compacidad)\n")
        f.write("    7. concavity (concavidad)\n")
        f.write("    8. concave points (puntos cóncavos)\n")
        f.write("    9. symmetry (simetría)\n")
        f.write("    10. fractal dimension (dimensión fractal)\n\n")
        
        f.write("  - El dataset está balanceado con aproximadamente 63% de casos benignos\n")
        f.write("    y 37% de casos malignos.\n\n")
        
        f.write("  - No se detectaron valores faltantes, lo que indica que el dataset\n")
        f.write("    está completo y listo para su uso.\n\n")
        
        f.write("  - Las características tienen diferentes escalas, por lo que será\n")
        f.write("    necesario aplicar normalización/estandarización antes del modelado.\n\n")
        
        f.write("="*80 + "\n")
    
    logger.info(f"Resumen estadístico guardado en: {summary_file}")
    
    # Mostrar resumen en consola
    print("\n" + "="*80)
    print("RESUMEN RÁPIDO DEL DATASET")
    print("="*80)
    print(f"Registros: {df.shape[0]}")
    print(f"Características: {df.shape[1] - 1}")
    print(f"Valores faltantes: {df.isnull().sum().sum()}")
    print("\nDistribución de clases:")
    print(df['diagnosis'].value_counts())
    print("="*80 + "\n")


def main():
    """Función principal."""
    print("="*80)
    print("DESCARGA Y ANÁLISIS DEL DATASET - BREAST CANCER WISCONSIN")
    print("="*80)
    print()
    
    try:
        # Descargar dataset
        df = download_dataset()
        print()
        
        # Generar análisis
        analyze_dataset(df)
        print()
        
        print("="*80)
        print("Proceso completado exitosamente")
        print("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error en el proceso: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
