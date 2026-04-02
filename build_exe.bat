@echo off
setlocal enabledelayedexpansion
echo.
echo  ============================================
echo   VisionMante v2 -- Build ejecutable Windows
echo  ============================================
echo.

:: ── Verificar entorno virtual ─────────────────────────────────────────────
if not exist "env1\Scripts\activate.bat" (
    echo [ERROR] Entorno virtual 'env1' no encontrado.
    echo         Ejecuta este script desde la raiz del proyecto.
    pause & exit /b 1
)

call env1\Scripts\activate.bat

:: ── Verificar PyInstaller ─────────────────────────────────────────────────
pyinstaller --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] Instalando PyInstaller...
    pip install pyinstaller --quiet
)

:: ── Compilar ──────────────────────────────────────────────────────────────
echo [1/3] Compilando con PyInstaller...
pyinstaller visionmante.spec --clean --noconfirm
if errorlevel 1 (
    echo.
    echo [ERROR] Fallo en PyInstaller. Revisa los mensajes anteriores.
    pause & exit /b 1
)

:: ── Copiar config y data (deben quedar junto al .exe, son escribibles) ───
echo [2/3] Copiando config\ y data\ al directorio de salida...

set DEST=dist\VisionMante

if exist "config" (
    xcopy /E /I /Y "config" "%DEST%\config" >nul
    echo       config\  copiado
)

if exist "data" (
    xcopy /E /I /Y "data" "%DEST%\data" >nul
    echo       data\    copiado
)

:: ── Crear directorio de logs vacío ───────────────────────────────────────
if not exist "%DEST%\logs" mkdir "%DEST%\logs"

:: ── Snap7 DLL (PLC Siemens) ───────────────────────────────────────────────
echo [3/3] Verificando snap7.dll...
if exist "snap7.dll" (
    copy /Y "snap7.dll" "%DEST%\snap7.dll" >nul
    echo       snap7.dll encontrado y copiado
) else (
    echo.
    echo  [AVISO] snap7.dll NO encontrado en la raiz del proyecto.
    echo          Si usas PLC Siemens S7, descarga snap7.dll desde:
    echo          https://snap7.sourceforge.net/  ^(x64 para Windows^)
    echo          y colócala en:  %DEST%\snap7.dll
    echo.
)

:: ── Listo ─────────────────────────────────────────────────────────────────
echo.
echo  ============================================
echo   Build completado!
echo   Ejecutable en:  %DEST%\VisionMante.exe
echo  ============================================
echo.
echo  Para distribuir, copia la carpeta completa:
echo    %DEST%\
echo  a la PC de destino. No requiere Python instalado.
echo.
pause
endlocal
