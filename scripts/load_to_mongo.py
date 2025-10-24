"""
Script para cargar el dataset Breast Cancer Wisconsin (Diagnostic) en MongoDB.
Crea la colección 'patients_records' y verifica la correcta inserción de los datos.
"""

import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from datetime import datetime
import sys


def log_message(message, log_file="setup_log.txt"):
    """Registra un mensaje en el archivo de log y lo imprime en consola."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")


def connect_to_mongo(host="localhost", port=27017, timeout=5000):
    """
    Establece conexión con MongoDB.
    
    Args:
        host (str): Host de MongoDB (default: localhost)
        port (int): Puerto de MongoDB (default: 27017)
        timeout (int): Timeout de conexión en milisegundos (default: 5000)
    
    Returns:
        MongoClient: Cliente de MongoDB si la conexión es exitosa, None en caso contrario
    """
    log_message("="*70)
    log_message("CONEXION A MONGODB")
    log_message("="*70)
    
    try:
        # Crear cliente de MongoDB
        connection_string = f"mongodb://{host}:{port}/"
        client = MongoClient(connection_string, serverSelectionTimeoutMS=timeout)
        
        # Verificar la conexión
        client.admin.command('ping')
        log_message(f"[OK] Conexion establecida con MongoDB en {host}:{port}")
        
        # Obtener información del servidor
        server_info = client.server_info()
        log_message(f"[OK] MongoDB version: {server_info['version']}")
        log_message("")
        
        return client
        
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        log_message(f"[ERROR] No se pudo conectar a MongoDB: {str(e)}")
        log_message("[INFO] Asegurese de que MongoDB este instalado y ejecutandose")
        log_message("[INFO] Para iniciar MongoDB: mongod")
        log_message("")
        return None
    except Exception as e:
        log_message(f"[ERROR] Error inesperado al conectar: {str(e)}")
        log_message("")
        return None


def load_dataset_to_mongo(csv_path, client, db_name="breast_cancer_db", collection_name="patients_records"):
    """
    Carga el dataset desde un archivo CSV a MongoDB.
    
    Args:
        csv_path (str): Ruta al archivo CSV con el dataset
        client (MongoClient): Cliente de MongoDB
        db_name (str): Nombre de la base de datos
        collection_name (str): Nombre de la colección
    
    Returns:
        tuple: (número de documentos insertados, DataFrame cargado)
    """
    log_message("="*70)
    log_message("CARGA DE DATOS A MONGODB")
    log_message("="*70)
    
    if client is None:
        log_message("[ERROR] No hay conexion activa con MongoDB")
        log_message("")
        return 0, None
    
    try:
        # Cargar el dataset
        log_message(f"[INFO] Cargando dataset desde: {csv_path}")
        df = pd.read_csv(csv_path)
        log_message(f"[OK] Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")
        
        # Seleccionar base de datos
        db = client[db_name]
        log_message(f"[OK] Base de datos '{db_name}' seleccionada")
        
        # Verificar si la colección ya existe
        if collection_name in db.list_collection_names():
            log_message(f"[WARNING] La coleccion '{collection_name}' ya existe")
            response = input("Desea eliminar la coleccion existente y crear una nueva? (s/n): ")
            if response.lower() == 's':
                db[collection_name].drop()
                log_message(f"[OK] Coleccion '{collection_name}' eliminada")
            else:
                log_message("[INFO] Operacion cancelada por el usuario")
                return 0, df
        
        # Seleccionar colección
        collection = db[collection_name]
        log_message(f"[OK] Coleccion '{collection_name}' creada/seleccionada")
        
        # Convertir DataFrame a lista de diccionarios
        log_message("[INFO] Preparando documentos para insercion...")
        records = df.to_dict('records')
        
        # Agregar metadatos a cada documento
        for record in records:
            record['inserted_at'] = datetime.now()
            record['source'] = 'UCI Machine Learning Repository'
        
        log_message(f"[INFO] Insertando {len(records)} documentos en MongoDB...")
        
        # Insertar documentos
        result = collection.insert_many(records)
        inserted_count = len(result.inserted_ids)
        
        log_message(f"[OK] {inserted_count} documentos insertados exitosamente")
        log_message("")
        
        return inserted_count, df
        
    except FileNotFoundError:
        log_message(f"[ERROR] Archivo no encontrado: {csv_path}")
        log_message("[INFO] Ejecute primero el script download_dataset.py")
        log_message("")
        return 0, None
    except Exception as e:
        log_message(f"[ERROR] Error al cargar datos: {str(e)}")
        log_message("")
        return 0, None


def verify_inserted_documents(client, db_name="breast_cancer_db", collection_name="patients_records", expected_count=None):
    """
    Verifica que los documentos se hayan insertado correctamente en MongoDB.
    
    Args:
        client (MongoClient): Cliente de MongoDB
        db_name (str): Nombre de la base de datos
        collection_name (str): Nombre de la colección
        expected_count (int): Número esperado de documentos
    
    Returns:
        bool: True si la verificación es exitosa, False en caso contrario
    """
    log_message("="*70)
    log_message("VERIFICACION DE DOCUMENTOS INSERTADOS")
    log_message("="*70)
    
    if client is None:
        log_message("[ERROR] No hay conexion activa con MongoDB")
        log_message("")
        return False
    
    try:
        # Seleccionar base de datos y colección
        db = client[db_name]
        collection = db[collection_name]
        
        # Contar documentos
        document_count = collection.count_documents({})
        log_message(f"[INFO] Numero de documentos en la coleccion: {document_count}")
        
        # Verificar contra el número esperado
        if expected_count is not None:
            if document_count == expected_count:
                log_message(f"[OK] Verificacion exitosa: {document_count} == {expected_count}")
            else:
                log_message(f"[ERROR] Discrepancia encontrada: {document_count} != {expected_count}")
                return False
        
        # Verificar distribución de diagnósticos
        malignant_count = collection.count_documents({"diagnosis": "M"})
        benign_count = collection.count_documents({"diagnosis": "B"})
        
        log_message(f"[INFO] Casos Malignos (M): {malignant_count}")
        log_message(f"[INFO] Casos Benignos (B): {benign_count}")
        log_message(f"[INFO] Total: {malignant_count + benign_count}")
        
        # Mostrar un documento de ejemplo
        log_message("[INFO] Ejemplo de documento insertado:")
        sample_doc = collection.find_one()
        if sample_doc:
            # Eliminar _id para mejor legibilidad
            sample_doc.pop('_id', None)
            log_message(f"[INFO] ID: {sample_doc.get('id', 'N/A')}")
            log_message(f"[INFO] Diagnostico: {sample_doc.get('diagnosis', 'N/A')}")
            log_message(f"[INFO] Radio medio: {sample_doc.get('radius_mean', 'N/A')}")
        
        log_message("")
        log_message("[OK] Verificacion completada exitosamente")
        log_message("")
        
        return True
        
    except Exception as e:
        log_message(f"[ERROR] Error durante la verificacion: {str(e)}")
        log_message("")
        return False


def validate_collection_exists(client, db_name="breast_cancer_db", collection_name="patients_records"):
    """
    Valida que la colección exista en la base de datos.
    
    Args:
        client (MongoClient): Cliente de MongoDB
        db_name (str): Nombre de la base de datos
        collection_name (str): Nombre de la colección
    
    Returns:
        bool: True si la colección existe, False en caso contrario
    """
    log_message("="*70)
    log_message("VALIDACION DE COLECCION")
    log_message("="*70)
    
    if client is None:
        log_message("[ERROR] No hay conexion activa con MongoDB")
        log_message("")
        return False
    
    try:
        db = client[db_name]
        collections = db.list_collection_names()
        
        if collection_name in collections:
            log_message(f"[OK] La coleccion '{collection_name}' existe en la base de datos '{db_name}'")
            log_message(f"[INFO] Colecciones disponibles: {', '.join(collections)}")
            log_message("")
            return True
        else:
            log_message(f"[ERROR] La coleccion '{collection_name}' NO existe en la base de datos '{db_name}'")
            log_message(f"[INFO] Colecciones disponibles: {', '.join(collections)}")
            log_message("")
            return False
            
    except Exception as e:
        log_message(f"[ERROR] Error durante la validacion: {str(e)}")
        log_message("")
        return False


def main():
    """Función principal del script."""
    log_message("")
    log_message("="*70)
    log_message("INICIO DE CARGA DE DATOS EN MONGODB")
    log_message("="*70)
    log_message("")
    
    # Ruta al archivo CSV
    csv_path = "data/breast_cancer.csv"
    
    # Conectar a MongoDB
    client = connect_to_mongo()
    
    if client is None:
        log_message("[ERROR] No se pudo establecer conexion con MongoDB. Proceso abortado.")
        log_message("")
        sys.exit(1)
    
    try:
        # Cargar datos a MongoDB
        inserted_count, df = load_dataset_to_mongo(csv_path, client)
        
        if inserted_count == 0:
            log_message("[ERROR] No se insertaron documentos. Proceso abortado.")
            return
        
        # Verificar documentos insertados
        expected_count = len(df) if df is not None else None
        verification_ok = verify_inserted_documents(client, expected_count=expected_count)
        
        # Validar que la colección existe
        collection_exists = validate_collection_exists(client)
        
        # Resumen final
        log_message("="*70)
        log_message("RESUMEN FINAL")
        log_message("="*70)
        
        if verification_ok and collection_exists:
            log_message("[OK] Proceso completado exitosamente")
            log_message(f"[OK] Base de datos: breast_cancer_db")
            log_message(f"[OK] Coleccion: patients_records")
            log_message(f"[OK] Documentos insertados: {inserted_count}")
            log_message("[OK] Sistema listo para entrenar modelos")
        else:
            log_message("[WARNING] El proceso se completo con advertencias")
            log_message("[INFO] Revise el log para mas detalles")
        
        log_message("")
        log_message("="*70)
        log_message("FIN DEL PROCESO")
        log_message("="*70)
        
    finally:
        # Cerrar conexión
        client.close()
        log_message("[INFO] Conexion con MongoDB cerrada")


if __name__ == "__main__":
    main()
