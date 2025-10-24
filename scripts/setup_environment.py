"""
Script para configurar y verificar el entorno de desarrollo.
Verifica las versiones de librerías, conexión con MongoDB y crea la base de datos.
"""

import sys
import logging
from datetime import datetime
from typing import Dict, Tuple

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_library_versions() -> Dict[str, str]:
    """
    Comprueba las versiones de las librerías instaladas.
    
    Returns:
        Dict con el nombre de la librería y su versión
    """
    logger.info("Verificando versiones de librerías...")
    
    versions = {}
    libraries = ['pandas', 'numpy', 'pymongo', 'sklearn', 'streamlit', 'joblib']
    
    for lib in libraries:
        try:
            if lib == 'sklearn':
                import sklearn
                versions['scikit-learn'] = sklearn.__version__
            else:
                module = __import__(lib)
                versions[lib] = module.__version__
            logger.info(f"{lib}: {versions.get(lib, versions.get('scikit-learn'))}")
        except ImportError as e:
            logger.error(f"{lib}: NO INSTALADO - {e}")
            versions[lib] = "NO INSTALADO"
    
    return versions


def verify_mongodb_connection(host: str = 'localhost', port: int = 27017) -> Tuple[bool, str]:
    """
    Verifica la conexión con MongoDB.
    
    Args:
        host: Dirección del servidor MongoDB
        port: Puerto de conexión
        
    Returns:
        Tupla (éxito, mensaje)
    """
    logger.info(f"Verificando conexión con MongoDB en {host}:{port}...")
    
    try:
        from pymongo import MongoClient
        from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
        
        # Intentar conexión con timeout corto
        client = MongoClient(
            host=host,
            port=port,
            serverSelectionTimeoutMS=5000
        )
        
        # Verificar que el servidor está disponible
        client.admin.command('ping')

        logger.info("Conexión exitosa con MongoDB")
        client.close()
        return True, "Conexión exitosa"
        
    except ImportError:
        logger.error("pymongo no está instalado")
        return False, "pymongo no está instalado"
    except Exception as e:
        logger.error(f"Error de conexión: {e}")
        return False, f"Error de conexión: {e}"


def create_database_and_collection(
    db_name: str = 'breast_cancer_db',
    collection_name: str = 'patients_records',
    host: str = 'localhost',
    port: int = 27017
) -> Tuple[bool, str]:
    """
    Crea la base de datos y colección en MongoDB.
    
    Args:
        db_name: Nombre de la base de datos
        collection_name: Nombre de la colección
        host: Dirección del servidor MongoDB
        port: Puerto de conexión
        
    Returns:
        Tupla (éxito, mensaje)
    """
    logger.info(f"Creando base de datos '{db_name}' y colección '{collection_name}'...")
    
    try:
        from pymongo import MongoClient
        
        client = MongoClient(host=host, port=port, serverSelectionTimeoutMS=5000)
        
        # Crear/acceder a la base de datos
        db = client[db_name]
        
        # Crear/acceder a la colección
        collection = db[collection_name]
        
        # Insertar documento de prueba
        test_document = {
            'test': True,
            'message': 'Documento de prueba para verificar la conexión',
            'timestamp': datetime.now()
        }
        
        result = collection.insert_one(test_document)
        logger.info(f" Documento de prueba insertado con ID: {result.inserted_id}")
        
        # Verificar la inserción
        found = collection.find_one({'_id': result.inserted_id})
        if found:
            logger.info(" Verificación exitosa: documento recuperado correctamente")
            
            # Eliminar documento de prueba
            collection.delete_one({'_id': result.inserted_id})
            logger.info(" Documento de prueba eliminado")
        
        # Listar bases de datos y colecciones
        databases = client.list_database_names()
        logger.info(f"Bases de datos disponibles: {databases}")
        
        collections = db.list_collection_names()
        logger.info(f"Colecciones en '{db_name}': {collections}")
        
        client.close()
        return True, f"Base de datos '{db_name}' y colección '{collection_name}' creadas correctamente"
        
    except Exception as e:
        logger.error(f" Error al crear base de datos: {e}")
        return False, f"Error: {e}"


def write_setup_log(
    versions: Dict[str, str],
    mongo_status: Tuple[bool, str],
    db_status: Tuple[bool, str],
    log_file: str = '../setup_log.txt'
) -> None:
    """
    Escribe un log con los resultados de la configuración.
    
    Args:
        versions: Diccionario con versiones de librerías
        mongo_status: Estado de la conexión MongoDB
        db_status: Estado de la creación de BD
        log_file: Ruta del archivo de log
    """
    import os
    
    # Construir ruta absoluta
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, log_file)
    
    logger.info(f"Escribiendo log en {log_path}...")
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("LOG DE CONFIGURACIÓN DEL ENTORNO\n")
        f.write("="*80 + "\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("1. VERSIONES DE LIBRERÍAS\n")
        f.write("-"*80 + "\n")
        for lib, version in versions.items():
            f.write(f"  - {lib}: {version}\n")
        f.write("\n")
        
        f.write("-"*80 + "\n")
        f.write("2. VERIFICACIÓN DE MONGODB\n")
        f.write("-"*80 + "\n")
        f.write(f"  Estado: {'EXITOSO' if mongo_status[0] else 'FALLIDO'}\n")
        f.write(f"  Mensaje: {mongo_status[1]}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("3. CREACIÓN DE BASE DE DATOS\n")
        f.write("-"*80 + "\n")
        f.write(f"  Estado: {'EXITOSO' if db_status[0] else 'FALLIDO'}\n")
        f.write(f"  Mensaje: {db_status[1]}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("4. RESULTADO GENERAL\n")
        f.write("-"*80 + "\n")
        
        all_libs_ok = all(v != "NO INSTALADO" for v in versions.values())
        overall_success = all_libs_ok and mongo_status[0] and db_status[0]
        
        if overall_success:
            f.write("  CONFIGURACIÓN COMPLETADA EXITOSAMENTE\n")
        else:
            f.write("  CONFIGURACIÓN COMPLETADA CON ERRORES\n")
            if not all_libs_ok:
                f.write("    - Algunas librerías no están instaladas\n")
            if not mongo_status[0]:
                f.write("    - Error en la conexión con MongoDB\n")
            if not db_status[0]:
                f.write("    - Error al crear la base de datos\n")
        
        f.write("\n" + "="*80 + "\n")
    
    logger.info(f"Log guardado exitosamente en {log_path}")


def main():
    """Función principal que ejecuta todas las verificaciones."""
    print("="*80)
    print("CONFIGURACIÓN DEL ENTORNO - BREAST CANCER DIAGNOSIS PROJECT")
    print("="*80)
    print()
    
    # 1. Verificar versiones de librerías
    versions = check_library_versions()
    print()
    
    # 2. Verificar conexión con MongoDB
    mongo_status = verify_mongodb_connection()
    print()
    
    # 3. Crear base de datos y colección
    if mongo_status[0]:
        db_status = create_database_and_collection()
    else:
        db_status = (False, "No se intentó crear la BD debido a error de conexión")
        logger.warning("Omitiendo creación de BD debido a error de conexión")
    print()
    
    # 4. Escribir log
    write_setup_log(versions, mongo_status, db_status)
    print()
    
    # Resumen final
    print("="*80)
    print("RESUMEN")
    print("="*80)
    all_libs_ok = all(v != "NO INSTALADO" for v in versions.values())
    
    if all_libs_ok and mongo_status[0] and db_status[0]:
        print("Configuración completada exitosamente")
        print("El entorno está listo para trabajar")
        return 0
    else:
        print("Configuración completada con errores")
        print("\nAcciones requeridas:")
        if not all_libs_ok:
            print("  1. Instalar las librerías faltantes: pip install -r requirements.txt")
        if not mongo_status[0]:
            print("  2. Verificar que MongoDB esté instalado y ejecutándose")
            print("     - Windows: net start MongoDB")
            print("     - Docker: docker run -d -p 27017:27017 --name mongodb mongo:latest")
        return 1


if __name__ == "__main__":
    sys.exit(main())
