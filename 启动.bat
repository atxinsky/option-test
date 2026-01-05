@echo off
chcp 65001 >nul
echo ========================================
echo   中国股指期权量化系统
echo ========================================
echo.
echo 正在启动Docker容器...
echo.

cd /d "%~dp0"

docker-compose up -d --build

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo   启动成功!
    echo   访问地址: http://localhost:8503
    echo ========================================
    echo.
    echo 按任意键打开浏览器...
    pause >nul
    start http://localhost:8503
) else (
    echo.
    echo 启动失败，请检查Docker是否正常运行
    pause
)
