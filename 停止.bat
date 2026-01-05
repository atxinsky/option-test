@echo off
chcp 65001 >nul
echo ========================================
echo   停止期权量化系统
echo ========================================
echo.

cd /d "%~dp0"

docker-compose down

echo.
echo 已停止
pause
