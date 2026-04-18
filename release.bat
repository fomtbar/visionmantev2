@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

:: ═══════════════════════════════════════════════════════════════════
::  VisionMante v2 — Release con un clic
::
::  Qué hace este script:
::    1. Activa el entorno virtual env1
::    2. git pull  (baja los últimos cambios)
::    3. pip install  (actualiza dependencias si cambiaron)
::    4. PyInstaller  (genera el ejecutable)
::    5. Copia config\ data\ logs\  al lado del .exe
::    6. Comprime todo en  release\VisionMante_vX.Y.Z_YYYYMMDD.zip
::
::  Para distribuir: llevar el .zip a la PC de trabajo y descomprimir.
::  No requiere Python instalado en destino.
:: ═══════════════════════════════════════════════════════════════════

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "ENV=%ROOT%\env1"
set "SPEC=%ROOT%\visionmante.spec"
set "DIST=%ROOT%\dist\VisionMante"
set "RELEASE_DIR=%ROOT%\release"

:: ── Cabecera ──────────────────────────────────────────────────────
echo.
echo  ╔═══════════════════════════════════════════╗
echo  ║   VisionMante v2  —  Release Builder      ║
echo  ╚═══════════════════════════════════════════╝
echo.

:: ── Verificar que estamos en la raíz del proyecto ─────────────────
if not exist "%SPEC%" (
    echo  [ERROR] No se encuentra visionmante.spec
    echo          Ejecuta este script desde la raiz del proyecto.
    pause & exit /b 1
)

:: ── Verificar entorno virtual ─────────────────────────────────────
if not exist "%ENV%\Scripts\activate.bat" (
    echo  [ERROR] Entorno virtual 'env1' no encontrado en:
    echo          %ENV%
    pause & exit /b 1
)

call "%ENV%\Scripts\activate.bat"
echo  [OK] Entorno virtual: env1

:: ─────────────────────────────────────────────────────────────────
:: PASO 1 — git pull
:: ─────────────────────────────────────────────────────────────────
echo.
echo  [1/5] Actualizando codigo desde git...
git pull
if errorlevel 1 (
    echo.
    echo  [AVISO] git pull retorno un error.
    echo          Si no hay conexion, el build continuara con el codigo local.
    echo.
)

:: Capturar version del ultimo commit para el nombre del zip
for /f "tokens=*" %%i in ('git describe --tags --always 2^>nul') do set GIT_VER=%%i
if "!GIT_VER!"=="" set GIT_VER=local

for /f "tokens=*" %%i in ('git log -1 --format^=%%h 2^>nul') do set GIT_HASH=%%i
if "!GIT_HASH!"=="" set GIT_HASH=0000000

:: Fecha para el nombre del zip
for /f "tokens=1-3 delims=/ " %%a in ("%date%") do (
    set "DD=%%a"
    set "MM=%%b"
    set "YYYY=%%c"
)
:: Formato YYYYMMDD (ajustar según locale si es necesario)
for /f "tokens=2 delims==" %%i in ('wmic os get localdatetime /value 2^>nul') do set "DT=%%i"
if not "!DT!"=="" (
    set "FECHA=!DT:~0,8!"
) else (
    set "FECHA=%YYYY%%MM%%DD%"
)

set "ZIP_NAME=VisionMante_!GIT_VER!_!FECHA!.zip"

echo  Commit: !GIT_HASH!   Version: !GIT_VER!   Fecha: !FECHA!

:: ─────────────────────────────────────────────────────────────────
:: PASO 2 — Actualizar dependencias
:: ─────────────────────────────────────────────────────────────────
echo.
echo  [2/5] Verificando dependencias (pip install -r requirements.txt)...
pip install -r "%ROOT%\requirements.txt" --quiet --no-warn-script-location
if errorlevel 1 (
    echo  [ERROR] Fallo al instalar dependencias.
    pause & exit /b 1
)
echo  [OK] Dependencias al dia

:: Verificar/instalar PyInstaller
pyinstaller --version >nul 2>&1
if errorlevel 1 (
    echo  [INFO] Instalando PyInstaller...
    pip install pyinstaller --quiet
)

:: ─────────────────────────────────────────────────────────────────
:: PASO 3 — Compilar ejecutable
:: ─────────────────────────────────────────────────────────────────
echo.
echo  [3/5] Compilando ejecutable con PyInstaller...
echo        (esto puede tardar 1-3 minutos)
echo.

pyinstaller "%SPEC%" --clean --noconfirm
if errorlevel 1 (
    echo.
    echo  [ERROR] Fallo en PyInstaller. Revisa los mensajes anteriores.
    pause & exit /b 1
)
echo.
echo  [OK] Compilacion completada

:: ─────────────────────────────────────────────────────────────────
:: PASO 4 — Copiar assets junto al ejecutable
:: ─────────────────────────────────────────────────────────────────
echo.
echo  [4/5] Copiando assets al directorio de salida...

:: config\ — configuración de la app (editable por el usuario)
if exist "%ROOT%\config" (
    xcopy /E /I /Y /Q "%ROOT%\config" "%DIST%\config" >nul
    echo        config\   copiado
)

:: data\ — imágenes de referencia y modelos
if exist "%ROOT%\data" (
    xcopy /E /I /Y /Q "%ROOT%\data" "%DIST%\data" >nul
    echo        data\     copiado
)

:: models\ — modelos YOLO/ONNX
if exist "%ROOT%\models" (
    xcopy /E /I /Y /Q "%ROOT%\models" "%DIST%\models" >nul
    echo        models\   copiado
)

:: logs\ — carpeta vacía para que la app pueda escribir logs
if not exist "%DIST%\logs" mkdir "%DIST%\logs"
echo        logs\     creado

:: snap7.dll — PLC Siemens (opcional)
if exist "%ROOT%\snap7.dll" (
    copy /Y "%ROOT%\snap7.dll" "%DIST%\snap7.dll" >nul
    echo        snap7.dll copiado
) else (
    echo        snap7.dll no encontrado ^(PLC Siemens^) — omitido
)

:: ─────────────────────────────────────────────────────────────────
:: PASO 5 — Comprimir para distribución
:: ─────────────────────────────────────────────────────────────────
echo.
echo  [5/5] Comprimiendo para distribucion...

if not exist "%RELEASE_DIR%" mkdir "%RELEASE_DIR%"

:: Borrar zip anterior del mismo nombre si existe
if exist "%RELEASE_DIR%\%ZIP_NAME%" del /F /Q "%RELEASE_DIR%\%ZIP_NAME%"

powershell -NoProfile -Command ^
  "Compress-Archive -Path '%DIST%' -DestinationPath '%RELEASE_DIR%\%ZIP_NAME%' -Force"

if errorlevel 1 (
    echo  [ERROR] Fallo al comprimir. Verifica que PowerShell este disponible.
    pause & exit /b 1
)

:: Tamaño del zip
for %%F in ("%RELEASE_DIR%\%ZIP_NAME%") do set ZIP_SIZE=%%~zF
set /a ZIP_MB=!ZIP_SIZE! / 1048576

:: ─────────────────────────────────────────────────────────────────
:: LISTO
:: ─────────────────────────────────────────────────────────────────
echo.
echo  ╔═══════════════════════════════════════════════════════════╗
echo  ║   Build completado!                                       ║
echo  ╠═══════════════════════════════════════════════════════════╣
echo  ║                                                           ║
echo  ║   Ejecutable:  dist\VisionMante\VisionMante.exe           ║
echo  ║   ZIP listo:   release\!ZIP_NAME!
echo  ║   Tamanio:     ~!ZIP_MB! MB                                        ║
echo  ║                                                           ║
echo  ║   Para instalar en la PC de trabajo:                      ║
echo  ║     1. Copiar el .zip                                     ║
echo  ║     2. Descomprimir                                       ║
echo  ║     3. Ejecutar VisionMante\VisionMante.exe               ║
echo  ║                                                           ║
echo  ╚═══════════════════════════════════════════════════════════╝
echo.

:: Abrir la carpeta release en el Explorador
explorer "%RELEASE_DIR%"

endlocal
pause
