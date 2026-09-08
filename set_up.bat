@echo off
REM ---------------------------------------------------------------
REM Configuracion del entorno de desarrollo del proyecto.
REM Crea el entorno virtual e instala las dependencias declaradas.
REM Uso: doble clic sobre el archivo, o ejecutar set_up.bat
REM ---------------------------------------------------------------

echo.
echo === Configuracion del entorno del proyecto ===
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no esta instalado o no esta en el PATH.
    echo Instalalo desde https://www.python.org/downloads/
    pause
    exit /b 1
)

if exist .venv (
    echo El entorno virtual .venv ya existe. Se omite su creacion.
) else (
    echo Creando entorno virtual...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: no se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
)

echo Actualizando pip...
call .venv\Scripts\python.exe -m pip install --upgrade pip --quiet

echo Instalando dependencias desde requirements.txt...
call .venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: fallo la instalacion de dependencias.
    pause
    exit /b 1
)

echo.
echo === Entorno listo ===
echo.
echo Para activarlo:            .venv\Scripts\activate
echo Para ejecutar el pipeline: python mlops_pipeline\src\ft_engineering.py
echo.
pause