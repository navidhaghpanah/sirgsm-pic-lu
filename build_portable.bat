@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================================
echo   Sirgsm Pic Lu - Portable Windows EXE Builder
echo   Usage:
echo     build_portable.bat            ^(Recommended fast folder + ZIP^)
echo     build_portable.bat --onefile  ^(Single portable .exe file^)
echo ============================================================
echo.

set "SIRGSM_ONEFILE=0"
if /I "%~1"=="--onefile" (
    set "SIRGSM_ONEFILE=1"
    echo [Mode] Single-file EXE ^(SIRGSM_ONEFILE=1^)
) else (
    echo [Mode] Fast Portable Folder + ZIP ^(SIRGSM_ONEFILE=0^)
)
echo.

set "VENV_DIR=.venv-build"
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [1/5] Creating isolated build virtual environment ^(%VENV_DIR%^)...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Make sure Python 3.10-3.12 x64 is installed.
        pause
        exit /b 1
    )
)

set "PY=%VENV_DIR%\Scripts\python.exe"
set "PIP=%VENV_DIR%\Scripts\pip.exe"
set "PYI=%VENV_DIR%\Scripts\pyinstaller.exe"

echo [2/5] Installing CPU-only PyTorch 2.8.0 ^(keeps EXE small and avoids WinError 1114^)...
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail

"%PIP%" install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :fail

echo [3/5] Installing application dependencies and PyInstaller...
"%PIP%" install -r requirements.txt "pyinstaller>=6.10.0"
if errorlevel 1 goto :fail

echo [4/5] Building Windows executable with PyInstaller...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"

"%PYI%" --noconfirm --clean "Sirgsm-Pic-Lu.spec"
if errorlevel 1 goto :fail

echo [5/5] Running offline self-test on compiled executable...
if "%SIRGSM_ONEFILE%"=="1" (
    set "EXE_PATH=dist\Sirgsm-Pic-Lu.exe"
    "!EXE_PATH!" --self-test
    if errorlevel 1 goto :fail
    echo.
    echo ============================================================
    echo   SUCCESS! Single-file portable EXE is ready:
    echo   !EXE_PATH!
    echo ============================================================
) else (
    set "EXE_PATH=dist\Sirgsm-Pic-Lu\Sirgsm-Pic-Lu.exe"
    "!EXE_PATH!" --self-test
    if errorlevel 1 goto :fail

    copy /Y "README.md" "dist\Sirgsm-Pic-Lu\README.md" >nul
    echo Creating portable ZIP archive...
    where tar >nul 2>nul
    if !errorlevel! equ 0 (
        tar -a -c -f "dist\Sirgsm-Pic-Lu-Windows-Portable.zip" -C "dist" "Sirgsm-Pic-Lu"
    ) else (
        powershell -NoProfile -Command "Compress-Archive -Path 'dist\Sirgsm-Pic-Lu' -DestinationPath 'dist\Sirgsm-Pic-Lu-Windows-Portable.zip' -Force"
    )
    if errorlevel 1 goto :fail

    echo.
    echo ============================================================
    echo   SUCCESS! Portable build is ready:
    echo   EXE : !EXE_PATH!
    echo   ZIP : dist\Sirgsm-Pic-Lu-Windows-Portable.zip
    echo ============================================================
)

pause
exit /b 0

:fail
echo.
echo [ERROR] Build failed! Please check the error output above.
pause
exit /b 1
