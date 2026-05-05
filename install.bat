@echo off
REM Windows CMD one-command installer.
where py >nul 2>nul
if %ERRORLEVEL%==0 (
  py -3 "%~dp0install.py" %*
  exit /b %ERRORLEVEL%
)
where python >nul 2>nul
if %ERRORLEVEL%==0 (
  python "%~dp0install.py" %*
  exit /b %ERRORLEVEL%
)
echo Python 3.10+ is required. Install Python from python.org, then rerun install.bat
exit /b 1
