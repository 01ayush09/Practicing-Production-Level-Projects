# Project Health Reporting Agent

An automated agent that ingests messy, real-world project data and produces:

1. **Weekly RAG (Red/Amber/Green) status reports** with transparent, plain-English reasoning per project, and
2. **Monthly executive presentations** that synthesize trends and cross-project patterns across an entire portfolio.

It ships with two real client engagements (converted from raw Excel exports) for a genuine end-to-end demo, a synthetic multi-week dataset purpose-built to exercise trend detection, a bonus live web dashboard, and an optional weekly scheduler — all runnable **completely offline**, with no API key or internet access required.

---

## Table of Contents

- [Architecture](#architecture)
- [Key Design Decisions](#key-design-decisions)
- [Project Structure](#project-structure)
- [Getting Started on Windows](#getting-started-on-windows)
- [Running Individual Components](#running-individual-components)
- [The RAG Scoring Methodology](#the-rag-scoring-methodology)
- [What Data Is Included](#what-data-is-included)
- [Scaling Considerations](#scaling-considerations)
- [Known Limitations](#known-limitations)

---

## Architecture

The system is organized as a layered pipeline: raw source files flow up through deterministic ingestion and scoring, into per-project weekly reports, and finally into a portfolio-level monthly synthesis. A live dashboard and an optional scheduler sit alongside the core pipeline rather than inside it.

![Weekly and Monthly Data Processing Pipeline](assets/architecture-diagram.png)

| Layer | Component | Responsibility |
|---|---|---|
| 1 | **Data Sources** | Raw Excel exports (`data/raw_xlsx/`) and derived CSVs (`data/project_plans/`) are the system's only inputs. |
| 2 | **Ingestion & Scoring Pipeline** | Python scripts (`agent/`) parse tolerant CSV input, compute deterministic schedule/budget/milestone/blocker metrics, and apply a weighted RAG score with escalation overrides. |
| 3 | **Report Assembly** | Each scored project produces a paired `.md` (human-readable) and `.json` (machine-readable) weekly report (`outputs/weekly/`). |
| 4 | **Monthly Synthesis** | `monthly_synthesis/synthesize.py` aggregates the *JSON* reports across weeks and projects to detect trends, recurring blockers, and sentiment/status correlations, then `build_deck.js` renders an executive `.pptx`. |
| 5 | **Live Dashboard** | A Flask app (`dashboard/app.py`) polls the source CSV/Excel files every ~4 seconds and re-scores in memory when a file changes — a lightweight "live" view that never touches the official weekly outputs. |
| 6 | **Scheduler (Optional)** | `agent/scheduler.py` automates the weekly pipeline run (APScheduler), with commented alternatives for cron / Windows Task Scheduler / GitHub Actions. |

---

## Key Design Decisions

**1. Deterministic scoring, LLM narrative — never the other way around.**
RAG status is computed by transparent, auditable rules (`agent/rag_engine.py`, documented in full in `RAG_METHODOLOGY.md`) — not by asking a language model "is this project healthy?" A scoring engine a stakeholder can't audit isn't trustworthy enough to hand to a client. The LLM's role is narrower and better suited to it: classifying free-text sentiment and translating already-computed reasons into plain English narrative.

**2. Pluggable LLM backend, not a hard dependency.**
`agent/reasoning.py` calls the Claude API when `ANTHROPIC_API_KEY` is set, and otherwise falls back to a deterministic, keyword-based heuristic. This means the entire pipeline — ingestion through the final deck — runs end-to-end with zero external dependencies for review, demos, or CI, and produces qualitatively better prose the moment a key is added. Every sample output shipped in this project was generated in offline fallback mode.

**3. Ingestion is tolerant by design, not by accident.**
Real PM data arrives with renamed columns, inconsistent date formats, and free text where a number was expected. `agent/ingest.py` uses column-alias matching and a best-effort parser that returns `None` on failure rather than throwing. Critically, missing data is never silently treated as "fine" — any dimension that can't be computed is scored **Amber** with an explicit "insufficient data" flag, both in the scoring engine and in a dedicated Data Quality Notes section of every report.

**4. Escalation overrides sit on top of the weighted score, not instead of it.**
A weighted average alone can hide a single catastrophic signal (e.g., a critical blocker open for three weeks) inside an otherwise fine-looking average. The engine computes the weighted score first, then applies override rules — a critical/aged blocker, two or more overdue milestones, or a compounding Red-plus-negative-sentiment pattern — that can force the overall status regardless of the average. Any single Red dimension also caps the overall status at Amber, so the tool can never report "Green" next to text that says "significant slippage."

**5. Monthly synthesis aggregates structured JSON, not markdown.**
Every weekly report is saved as both `.md` and `.json`. `monthly_synthesis/synthesize.py` reads only the JSON, so trend analysis (RAG trajectory, recurring blockers, sentiment-vs-status correlation) is computed from real week-over-week data rather than having a language model re-read several markdown files and guess at trends.

**6. A clearly-labeled synthetic dataset exercises trend detection honestly.**
The two real client engagements included are each a single point-in-time snapshot, not a week-over-week series — so on their own they can't demonstrate multi-week trend detection or cross-project pattern-finding. Rather than fabricate that inside the real client data, a separate, explicitly synthetic 3-project × 4-week dataset (`data/project_plans_demo/`) was built to exercise that logic end-to-end, with its own pipeline and its own clearly-labeled deck, so it's never mistaken for real client output.

---

## Project Structure

```
project-health-agent/
├── data/
│   ├── raw_xlsx/                  Original client exports (bundled for reproducibility)
│   ├── project_plans/             Derived CSVs for the two real engagements
│   └── project_plans_demo/        Synthetic 3-project x 4-week dataset for trend detection
├── agent/
│   ├── ingest.py                  Tolerant CSV parsing, column-alias matching
│   ├── metrics.py                 Deterministic schedule/budget/milestone/blocker calculations
│   ├── rag_engine.py              Weighted scoring + escalation overrides (the methodology, in code)
│   ├── reasoning.py               Sentiment classification + narrative generation (Claude API or fallback)
│   ├── report_generator.py        Assembles the JSON + Markdown report
│   ├── run_weekly.py              CLI entrypoint for a single project
│   ├── run_current_projects.py    Runs both real projects' current snapshot
│   ├── run_trend_demo.py          Runs the synthetic 4-week trend-demo dataset
│   └── scheduler.py               Optional weekly automation (APScheduler)
├── outputs/
│   ├── weekly/week1/              Real-project weekly reports (JSON + Markdown)
│   └── weekly_demo/week1-4/       Trend-demo weekly reports
├── monthly_synthesis/
│   ├── synthesize.py              Aggregates weekly JSON into cross-project trend data
│   ├── build_deck.js              Renders the executive .pptx from the synthesized JSON
│   ├── monthly_summary.json       Real-client monthly synthesis
│   ├── demo_summary.json          Trend-demo monthly synthesis
│   └── *.pptx                     Generated executive decks
├── dashboard/
│   ├── app.py                     Flask live dashboard
│   ├── templates/ , static/       Frontend assets
├── convert_real_data.py           Converts raw Excel exports into the CSV shape ingest.py expects
├── RAG_METHODOLOGY.md             The one-page scoring framework
├── requirements.txt               Core Python dependencies
├── requirements-dashboard.txt     Dashboard-only dependency (Flask)
├── setup.ps1                      One-command setup + run for Windows
└── setup.sh                       One-command setup + run for macOS/Linux
```

---

## Getting Started on Windows

### Prerequisites

Install these before proceeding (both add themselves to `PATH` automatically if you leave the default installer options checked):

| Tool | Minimum Version | Download |
|---|---|---|
| Python | 3.10+ | https://www.python.org/downloads/windows/ |
| Node.js | 18+ | https://nodejs.org/en/download |

Verify both are on your `PATH` by opening **PowerShell** and running:

```powershell
python --version
node --version
```

### Option A — One-command setup (recommended)

1. Unzip the project and open **PowerShell** in the project's root folder (the folder containing `setup.ps1`).
2. If PowerShell blocks script execution, allow it for this session only:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```
3. Run the setup script:
   ```powershell
   .\setup.ps1
   ```

This single command will:

1. Create a local virtual environment (`.venv`) — nothing is installed globally.
2. Install all Python dependencies into it.
3. Regenerate the real UniSan/Titan weekly snapshot.
4. Regenerate the synthetic 4-week trend-demo dataset.
5. Build both executive `.pptx` decks.

When it finishes, you'll see a summary listing exactly where each output landed:

```
outputs/weekly/week1/                                  - real client weekly reports
outputs/weekly_demo/week1-4/                            - synthetic trend-demo weekly reports
monthly_synthesis/Monthly_Executive_Presentation.pptx   - real snapshot deck
monthly_synthesis/Trend_Demo_Presentation.pptx          - trend-demo deck
```

> **Fastest path of all:** the finished `.pptx` decks and `.md`/`.json` weekly reports are already included pre-generated in the zip, so you can open them directly without running anything if you'd simply like to review the output.

### Option B — Manual step-by-step (for debugging or inspection)

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the weekly agent for a single real project
cd agent
python run_weekly.py `
  --tasks ../data/project_plans/unisan_tasks_current.csv `
  --update ../data/project_plans/unisan_current_update.csv `
  --project "UniSan S2P Implementation" `
  --as-of 2026-07-02 `
  --out ../outputs/weekly/week1

# 4. Or run both real projects at once
python run_current_projects.py

# 5. Generate the synthetic 4-week trend-demo dataset
python run_trend_demo.py
cd ..

# 6. Build the executive decks
cd monthly_synthesis
python synthesize.py                 # aggregates outputs/weekly/* into monthly_summary.json
python synthesize.py --demo          # aggregates outputs/weekly_demo/* into demo_summary.json
npm install                          # installs pptxgenjs
node build_deck.js monthly_summary.json Monthly_Executive_Presentation.pptx
node build_deck.js demo_summary.json Trend_Demo_Presentation.pptx
cd ..
```

---

## Running Individual Components

### Enable live LLM reasoning (optional)

Without an API key, the agent uses the deterministic offline fallback described above — the RAG logic is identical either way, only the prose narrative changes. To turn on live Claude reasoning, set the key before running any command above:

```powershell
$env:ANTHROPIC_API_KEY = "sk-..."
```

### Run on a weekly schedule (bonus)

```powershell
pip install APScheduler
python agent\scheduler.py
```

For production use, cron / Windows Task Scheduler / GitHub Actions are recommended over a long-lived Python process — see the commented alternatives at the bottom of `scheduler.py`.

### Run the live dashboard (bonus)

```powershell
pip install -r requirements-dashboard.txt
cd dashboard
python app.py
```

Then open **http://127.0.0.1:5000** in a browser.

The dashboard polls the source CSV/Excel files every ~4 seconds and re-scores in memory whenever it detects a change — no restart required to see an edit reflected. It intentionally never writes to `outputs/weekly/`, so leaving it open in a browser tab cannot overwrite an official dated report. Because it evaluates dates against **today's real date** rather than a fixed as-of date, its live status can legitimately drift from the static reports in `outputs/weekly/week1/` over time — that's expected, and is the point of a live view.

---

## The RAG Scoring Methodology

Full detail lives in `RAG_METHODOLOGY.md`; the short version:

| Dimension | Weight | What it measures |
|---|---|---|
| Schedule | 25% | % of tasks/milestones completed on time; critical-path slippage |
| Budget | 20% | Actual vs. planned spend-to-date (burn ratio) |
| Milestones | 25% | Overdue and at-risk milestones |
| Blockers | 20% | Count, severity, and age of open blockers |
| Sentiment | 10% | Tone of PM/stakeholder free-text commentary, trended week-over-week |

Each dimension is scored independently (Red / Amber / Green), weighted and averaged into an overall status, and then subjected to escalation overrides — any Red dimension caps the overall status at Amber; a critical or long-aged blocker, two or more overdue milestones, or a compounding Red-plus-negative-sentiment pattern can force the status to Red outright.

---

## What Data Is Included

This project runs on two real engagements, converted from raw Smartsheet-style Gantt exports by `convert_real_data.py`, plus one clearly-labeled synthetic dataset:

- **UniSan S2P Implementation** (382 tasks) — currently **Amber**: a critical milestone is overdue, and budget/stakeholder-commentary data was absent from the source export, so those dimensions are flagged Amber for follow-up rather than guessed at.
- **Titan (Outokumpu) S2P Implementation** (490 tasks) — currently **Red**: two phase-level milestones are overdue, triggering the two-or-more-overdue-milestones escalation.
- **Synthetic trend-demo dataset** (`data/project_plans_demo/`) — three fictional projects over four weeks, purpose-built to exercise trend detection: one declining, one recovering, one stable, with a shared vendor dependency and a sentiment dip that precedes a status change. `generate_demo_data.py` is the fully readable source of every number in it.

Both real engagements are single point-in-time snapshots rather than a week-over-week series, so their monthly deck will correctly report "held steady" — trend detection is what the synthetic dataset above is for. As real weekly snapshots accumulate, dropping each new week's CSVs into `data/project_plans/` and a new `outputs/weekly/week{N}/` folder lets `monthly_synthesis/synthesize.py` pick up the real trend automatically.

---

## Scaling Considerations

The deterministic scoring engine (`ingest.py` → `metrics.py` → `rag_engine.py`) is pandas and arithmetic — easily thousands of projects per minute, not a bottleneck at any realistic portfolio size. The real cost and latency live in `reasoning.py`'s LLM calls.

The highest-leverage optimization already implemented: `generate_narrative()` skips the live LLM call entirely for a project that is clean Green (no escalation overrides), returning the same structured template used in offline mode — a healthy project doesn't need a nuanced paragraph explaining why it's healthy. Each report's `narrative_source` field records whether that specific report used a live call or the template, keeping the optimization auditable rather than hidden.

Two changes worth making before scaling to a much larger portfolio (not yet implemented, since the current sample doesn't need them):
1. Replace `scheduler.py`'s sequential loop with a bounded async/thread pool so remaining LLM calls for multiple projects go out concurrently, respecting the API tier's rate limit.
2. Cache sentiment classification when a PM's notes are unchanged week-over-week, avoiding a wasted call on identical text.

---

## Known Limitations

- Budget scoring uses a simple burn ratio, not full Earned Value Management (SPI/CPI with EV) — straightforward to extend if task-level cost data becomes available.
- Blocker aging currently relies on a manually supplied "critical" tag rather than an automatically tracked open-since date, since a blocker string isn't attached to the week it was first raised. The natural fix is to track each blocker's first-seen week once further week-over-week data exists.
- The offline sentiment classifier is a keyword heuristic; with an API key set, this becomes a real LLM classification and is meaningfully better at nuance (sarcasm, mixed signals, hedged language).
- Sample data is CSV; `ingest.py`'s column-alias-matching approach generalizes to Smartsheet/Jira/MS Project exports, or can be extended with an API-based ingestion module.

---

## A Note on the Included Client Data

This deliverable bundles two real client engagements' project data for reproducibility. It is intended to be reviewed locally (via the zip and the setup scripts above) rather than hosted on a public URL, since doing so would expose real project names and status publicly. Everything required to run and review the project works fully offline — no API key, account, or internet access needed.
