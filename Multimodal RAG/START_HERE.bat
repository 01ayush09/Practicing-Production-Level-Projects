@echo off
setlocal enabledelayedexpansion
title Fullstack Multimodal RAG - Setup & Launch
color 0A
cls

echo ============================================================
echo       Fullstack Multimodal RAG - Windows Launcher
echo ============================================================
echo.

REM ── Step 0: Check Python ─────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
echo [OK] Python found.

REM ── Step 1: Check Docker ─────────────────────────────────────
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker not found!
    echo Please install Docker Desktop from https://www.docker.com/products/docker-desktop/
    echo After installing, start Docker Desktop and re-run this script.
    pause
    exit /b 1
)
echo [OK] Docker found.

REM ── Step 2: Check API Key ────────────────────────────────────
findstr /c:"YOUR_GEMINI_API_KEY_HERE" .env >nul 2>&1
if not errorlevel 1 (
    echo.
    echo ============================================================
    echo  ACTION REQUIRED: Enter your Gemini API Key
    echo ============================================================
    echo  Get a FREE key at: https://aistudio.google.com/apikey
    echo ============================================================
    echo.
    set /p GEMINI_KEY="Paste your Gemini API key here: "
    if "!GEMINI_KEY!"=="" (
        echo [ERROR] No API key entered. Exiting.
        pause
        exit /b 1
    )
    REM Write the key to .env
    echo GEMINI_API_KEY=!GEMINI_KEY!> .env
    echo [OK] API key saved to .env
) else (
    echo [OK] Gemini API key already configured in .env
)
echo.

REM ── Step 3: Start OpenSearch via Docker Compose ──────────────
echo [1/4] Starting OpenSearch (Docker)...
docker compose up -d
if errorlevel 1 (
    echo [ERROR] Failed to start Docker containers.
    echo Make sure Docker Desktop is running, then try again.
    pause
    exit /b 1
)
echo [OK] OpenSearch is starting up...
echo      (Give it ~20 seconds to fully initialize)
echo.
timeout /t 20 /nobreak >nul

REM ── Step 4: Create virtual environment if needed ─────────────
echo [2/4] Setting up Python virtual environment...
if not exist ".venv\Scripts\activate.bat" (
    python -m venv .venv
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)
echo.

REM ── Step 5: Install dependencies ─────────────────────────────
echo [3/4] Installing Python dependencies (first run may take a few minutes)...
call .venv\Scripts\activate.bat
pip install -q -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    echo Try running: pip install -r requirements.txt
    pause
    exit /b 1
)
echo [OK] All dependencies installed.
echo.

REM ── Step 6: Check if Ollama is available (optional) ──────────
ollama --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] Ollama not found - the "Ollama" model option in the UI will not work.
    echo        Only Gemini will be available. To enable Ollama, install it from:
    echo        https://ollama.com  then run: ollama pull deepseek-r1:1.5b
    echo        and: ollama pull nomic-embed-text
) else (
    echo [OK] Ollama found.
    REM Pull required models in background if not already present
    echo [INFO] Ensuring Ollama models are available...
    start /b ollama pull nomic-embed-text
    start /b ollama pull deepseek-r1:1.5b
)
echo.

REM ── Step 7: Check if data has been ingested ──────────────────
echo [4/4] Checking ingestion status...
if not exist "files\2312.10997v5.pdf" (
    echo.
    echo ============================================================
    echo  BEFORE USING THE APP: Ingest your PDF first!
    echo ============================================================
    echo  1. Download the RAG paper PDF from:
    echo     https://arxiv.org/pdf/2312.10997
    echo  2. Save it as:  files\2312.10997v5.pdf
    echo  3. Run INGEST_DATA.bat to process and index the PDF
    echo  Then come back and run START_HERE.bat again.
    echo ============================================================
    echo.
    set /p CHOICE="Press ENTER to launch the app anyway, or Ctrl+C to exit: "
)

REM ── Step 8: Launch the Gradio App ────────────────────────────
echo.
echo ============================================================
echo  Launching the RAG Q^&A App...
echo  Open your browser at: http://localhost:7860
echo ============================================================
echo.
python app.py

pause
