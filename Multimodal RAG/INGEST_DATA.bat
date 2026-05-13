@echo off
title Fullstack Multimodal RAG - Data Ingestion
color 0B
cls

echo ============================================================
echo       Fullstack Multimodal RAG - Ingest PDF Data
echo ============================================================
echo.
echo This will:
echo  - Parse the RAG paper PDF (text + images + tables)
echo  - Generate descriptions using Gemini
echo  - Create embeddings using Ollama (nomic-embed-text)
echo  - Index everything into OpenSearch
echo.
echo WARNING: This takes 5-20 minutes depending on your machine.
echo.

REM Check PDF exists
if not exist "files\2312.10997v5.pdf" (
    echo [ERROR] PDF not found at: files\2312.10997v5.pdf
    echo.
    echo Please download the RAG survey paper from:
    echo   https://arxiv.org/pdf/2312.10997
    echo and save it as:  files\2312.10997v5.pdf
    echo.
    pause
    exit /b 1
)
echo [OK] PDF found.

REM Check .env has real API key
findstr /c:"YOUR_GEMINI_API_KEY_HERE" .env >nul 2>&1
if not errorlevel 1 (
    echo.
    echo [ERROR] Gemini API key not configured.
    echo Please run START_HERE.bat first to set up your API key.
    pause
    exit /b 1
)
echo [OK] API key configured.

REM Activate venv
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo Please run START_HERE.bat first to set up dependencies.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat

REM Check OpenSearch is running
echo Checking OpenSearch connection...
python -c "from helper import get_opensearch_client; get_opensearch_client('localhost', 9200)" 2>nul
if errorlevel 1 (
    echo [ERROR] Cannot connect to OpenSearch.
    echo Make sure Docker Desktop is running and containers are up.
    echo Run: docker compose up -d
    pause
    exit /b 1
)
echo [OK] OpenSearch is reachable.
echo.

echo Starting ingestion...
echo.
python ingestion.py

if errorlevel 1 (
    echo.
    echo [ERROR] Ingestion failed. Check the output above for details.
) else (
    echo.
    echo ============================================================
    echo  Ingestion complete! You can now run START_HERE.bat
    echo ============================================================
)

pause
