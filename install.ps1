# Script de Instalación Automatizada
# Proyecto: Breast Cancer Diagnosis - MLP vs SVM
# Ejecutar este script para configurar todo el entorno

Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host "INSTALACIÓN AUTOMATIZADA - BREAST CANCER DIAGNOSIS PROJECT" -ForegroundColor Cyan
Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host ""

# Verificar ubicación
$projectRoot = "c:\Users\ignac\vscode projects\breast-cancer-diagnosis-mlp-vs-svm"
if (-Not (Test-Path $projectRoot)) {
    Write-Host "[ERROR] No se encuentra el directorio del proyecto" -ForegroundColor Red
    exit 1
}

Set-Location $projectRoot
Write-Host "[INFO] Directorio del proyecto: $projectRoot" -ForegroundColor Yellow
Write-Host ""

# Paso 1: Crear entorno virtual
Write-Host "PASO 1: Creando entorno virtual..." -ForegroundColor Green
if (Test-Path "venv") {
    Write-Host "[INFO] Entorno virtual ya existe" -ForegroundColor Yellow
} else {
    python -m venv venv
    if ($?) {
        Write-Host "[OK] Entorno virtual creado" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] No se pudo crear el entorno virtual" -ForegroundColor Red
        exit 1
    }
}
Write-Host ""

# Paso 2: Activar entorno virtual
Write-Host "PASO 2: Activando entorno virtual..." -ForegroundColor Green
& ".\venv\Scripts\Activate.ps1"
Write-Host "[OK] Entorno virtual activado" -ForegroundColor Green
Write-Host ""

# Paso 3: Actualizar pip
Write-Host "PASO 3: Actualizando pip..." -ForegroundColor Green
python -m pip install --upgrade pip --quiet
Write-Host "[OK] pip actualizado" -ForegroundColor Green
Write-Host ""

# Paso 4: Instalar dependencias
Write-Host "PASO 4: Instalando dependencias..." -ForegroundColor Green
Write-Host "[INFO] Esto puede tomar varios minutos..." -ForegroundColor Yellow
python -m pip install -r requirements.txt
if ($?) {
    Write-Host "[OK] Todas las dependencias instaladas" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Error al instalar dependencias" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Paso 5: Descargar dataset
Write-Host "PASO 5: Descargando dataset..." -ForegroundColor Green
python scripts/download_dataset.py
if ($?) {
    Write-Host "[OK] Dataset descargado" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Error al descargar dataset" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Paso 6: Verificar MongoDB
Write-Host "PASO 6: Verificando MongoDB..." -ForegroundColor Green
$mongoRunning = $false

# Intentar conectar a MongoDB
try {
    $null = mongosh --eval "db.version()" --quiet 2>&1
    if ($?) {
        Write-Host "[OK] MongoDB está ejecutándose" -ForegroundColor Green
        $mongoRunning = $true
    }
} catch {
    Write-Host "[WARNING] MongoDB no está ejecutándose" -ForegroundColor Yellow
}

if (-Not $mongoRunning) {
    Write-Host ""
    Write-Host "MongoDB no está disponible. Opciones:" -ForegroundColor Yellow
    Write-Host "  1. Instalar MongoDB Community Server" -ForegroundColor Yellow
    Write-Host "  2. Ejecutar MongoDB con Docker:" -ForegroundColor Yellow
    Write-Host "     docker run -d -p 27017:27017 --name mongodb mongo:latest" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Después de instalar MongoDB, ejecuta:" -ForegroundColor Yellow
    Write-Host "     python scripts/setup_environment.py" -ForegroundColor Cyan
    Write-Host "     python scripts/load_to_mongo.py" -ForegroundColor Cyan
    Write-Host ""
}
Write-Host ""

# Paso 7: Ejecutar configuración (si MongoDB está disponible)
if ($mongoRunning) {
    Write-Host "PASO 7: Configurando entorno y base de datos..." -ForegroundColor Green
    python scripts/setup_environment.py
    
    if ($?) {
        Write-Host "[OK] Entorno configurado" -ForegroundColor Green
        
        Write-Host ""
        Write-Host "PASO 8: Cargando datos en MongoDB..." -ForegroundColor Green
        python scripts/load_to_mongo.py
        
        if ($?) {
            Write-Host "[OK] Datos cargados en MongoDB" -ForegroundColor Green
        } else {
            Write-Host "[WARNING] Problemas al cargar datos" -ForegroundColor Yellow
        }
    }
}

# Resumen Final
Write-Host ""
Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host "RESUMEN DE LA INSTALACIÓN" -ForegroundColor Cyan
Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Componentes instalados:" -ForegroundColor Green
Write-Host "  [OK] Entorno virtual Python" -ForegroundColor Green
Write-Host "  [OK] Dependencias (pandas, numpy, scikit-learn, etc.)" -ForegroundColor Green
Write-Host "  [OK] Dataset descargado" -ForegroundColor Green

if ($mongoRunning) {
    Write-Host "  [OK] MongoDB configurado y datos cargados" -ForegroundColor Green
} else {
    Write-Host "  [PENDIENTE] MongoDB - Requiere instalación manual" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Archivos generados:" -ForegroundColor Green
Write-Host "  - data/breast_cancer.csv" -ForegroundColor White
Write-Host "  - data/data_summary.txt" -ForegroundColor White
if (Test-Path "setup_log.txt") {
    Write-Host "  - setup_log.txt" -ForegroundColor White
}
Write-Host ""

Write-Host "Próximos pasos:" -ForegroundColor Yellow
if (-Not $mongoRunning) {
    Write-Host "  1. Instalar MongoDB" -ForegroundColor White
    Write-Host "  2. Ejecutar: python scripts/setup_environment.py" -ForegroundColor White
    Write-Host "  3. Ejecutar: python scripts/load_to_mongo.py" -ForegroundColor White
    Write-Host "  4. Comenzar con el análisis exploratorio (Semana 2)" -ForegroundColor White
} else {
    Write-Host "  1. Revisar setup_log.txt para detalles" -ForegroundColor White
    Write-Host "  2. Comenzar con el análisis exploratorio (Semana 2)" -ForegroundColor White
}
Write-Host ""

Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host "INSTALACIÓN COMPLETADA" -ForegroundColor Cyan
Write-Host "=================================================================================" -ForegroundColor Cyan
