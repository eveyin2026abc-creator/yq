@echo off
setlocal
set "PUBLIC_DIR=%~dp0public"
set "BIND=%~1"
set "PORT=%~2"
if "%BIND%"=="" set "BIND=0.0.0.0"
if "%PORT%"=="" set "PORT=8766"

if not exist "%PUBLIC_DIR%\index.html" goto missing_public

where py >nul 2>nul
if not errorlevel 1 goto use_py

where python3 >nul 2>nul
if not errorlevel 1 goto use_python3

where python >nul 2>nul
if not errorlevel 1 goto use_python

echo Python 3 was not found. Install Python 3 and try again.
exit /b 1

:use_py
echo Serving on %BIND%:%PORT%
py -3 -m http.server %PORT% --bind "%BIND%" --directory "%PUBLIC_DIR%"
exit /b %errorlevel%

:use_python3
echo Serving on %BIND%:%PORT%
python3 -m http.server %PORT% --bind "%BIND%" --directory "%PUBLIC_DIR%"
exit /b %errorlevel%

:use_python
echo Serving on %BIND%:%PORT%
python -m http.server %PORT% --bind "%BIND%" --directory "%PUBLIC_DIR%"
exit /b %errorlevel%

:missing_public
echo public\index.html was not found. Pull the latest snapshot first.
exit /b 1
