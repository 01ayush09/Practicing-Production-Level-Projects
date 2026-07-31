#!/usr/bin/env bash
# setup.sh — One-command setup + run for macOS/Linux. Mirrors setup.ps1.
# Run from the project root:  bash setup.sh
set -euo pipefail

echo "== 1/5 Creating virtual environment (.venv) =="
python3 -m venv .venv
source .venv/bin/activate

echo "== 2/5 Installing Python dependencies =="
pip install -r requirements.txt

echo "== 3/5 Running the real UniSan/Titan snapshot =="
(cd agent && python3 run_current_projects.py)

echo "== 4/5 Running the synthetic 4-week trend-demo dataset =="
(cd agent && python3 run_trend_demo.py)

echo "== 5/5 Building both executive decks =="
(
  cd monthly_synthesis
  python3 synthesize.py
  python3 synthesize.py --demo
  npm install
  node build_deck.js monthly_summary.json Monthly_Executive_Presentation.pptx
  node build_deck.js demo_summary.json Trend_Demo_Presentation.pptx
)

cat <<'EOF'

Done. Key outputs:
  outputs/weekly/week1/                                  - real client weekly reports
  outputs/weekly_demo/week1-4/                            - synthetic trend-demo weekly reports
  monthly_synthesis/Monthly_Executive_Presentation.pptx    - real snapshot deck
  monthly_synthesis/Trend_Demo_Presentation.pptx           - trend-demo deck

Optional: to also try the live dashboard, run:
  pip install -r requirements-dashboard.txt && cd dashboard && python3 app.py
  then open http://127.0.0.1:5000
EOF
