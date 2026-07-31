# setup.ps1
# One-command setup + run for Windows PowerShell. No prior Python/Node
# environment assumed beyond having Python 3.10+ and Node 18+ installed.
#
# What it does, in order:
#   1. Creates a local virtual environment (.venv) so nothing is installed globally.
#   2. Installs Python dependencies into it.
#   3. Regenerates the two real projects' current-status snapshot.
#   4. Regenerates the synthetic 4-week trend-demo dataset and its deck.
#   5. Installs the deck-builder's Node dependency and rebuilds both decks.
#
# Run from the project root:  .\setup.ps1
# (If PowerShell blocks the script: Set-ExecutionPolicy -Scope Process Bypass)

$ErrorActionPreference = "Stop"

Write-Host "== 1/5 Creating virtual environment (.venv) ==" -ForegroundColor Cyan
python -m venv .venv
.\.venv\Scripts\Activate.ps1

Write-Host "== 2/5 Installing Python dependencies ==" -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "== 3/5 Running the real UniSan/Titan snapshot ==" -ForegroundColor Cyan
Push-Location agent
python run_current_projects.py
Pop-Location

Write-Host "== 4/5 Running the synthetic 4-week trend-demo dataset ==" -ForegroundColor Cyan
Push-Location agent
python run_trend_demo.py
Pop-Location

Write-Host "== 5/5 Building both executive decks ==" -ForegroundColor Cyan
Push-Location monthly_synthesis
python synthesize.py
python synthesize.py --demo
npm install
node build_deck.js monthly_summary.json Monthly_Executive_Presentation.pptx
node build_deck.js demo_summary.json Trend_Demo_Presentation.pptx
Pop-Location

Write-Host ""
Write-Host "Done. Key outputs:" -ForegroundColor Green
Write-Host "  outputs/weekly/week1/            - real client weekly reports"
Write-Host "  outputs/weekly_demo/week1-4/      - synthetic trend-demo weekly reports"
Write-Host "  monthly_synthesis/Monthly_Executive_Presentation.pptx  - real snapshot deck"
Write-Host "  monthly_synthesis/Trend_Demo_Presentation.pptx         - trend-demo deck"
Write-Host ""
Write-Host "Optional: to also try the live dashboard, run:" -ForegroundColor Yellow
Write-Host "  pip install -r requirements-dashboard.txt; cd dashboard; python app.py"
Write-Host "  then open http://127.0.0.1:5000"
