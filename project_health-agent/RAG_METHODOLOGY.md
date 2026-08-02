# Project Health RAG Methodology

## Purpose
Give leadership a consistent, defensible way to know a project's status without
waiting on a PM to write it up — and make the *reasoning* behind the color as
important as the color itself.

## The Five Signals

| Dimension | What we measure | Data source |
|---|---|---|
| **Schedule** | % of tasks/milestones completed on or before their planned date; days of slippage on the critical path | Task list: planned vs. actual dates, % complete |
| **Budget** | Actual spend vs. planned spend *at this point in the timeline*, normalized by % complete (a lightweight cost-performance ratio) | Weekly budget actuals vs. total/planned budget |
| **Milestones** | Overdue milestones; milestones due in the next 2 weeks flagged at risk | Task list milestone flags |
| **Blockers** | Count and severity of open blockers/risks, and how long they've been open | PM free-text blocker log |
| **Stakeholder sentiment** | Tone of the PM's/client's narrative comments (positive / neutral / negative), and whether it's trending down | PM free-text weekly notes |

Each dimension is scored **0 = Red, 1 = Amber, 2 = Green** independently first,
so the agent can show *which specific thing* is driving the overall color —
not just a single opaque number.

## Scoring Thresholds (per dimension)

- **Schedule**: Green ≤5% of tasks late / no critical milestone slipped; Amber 5–20% late or a non-critical milestone slipped; Red >20% late or a critical milestone slipped.
- **Budget**: Using a simple burn ratio = (actual spend / planned spend-to-date). Green 0.9–1.1; Amber 1.1–1.3 or 0.7–0.9 (underspend can mean stalled work); Red >1.3 or <0.7.
- **Milestones**: Green = none overdue; Amber = 1 overdue or ≥1 at-risk in next 2 weeks; Red = ≥2 overdue.
- **Blockers**: Green = 0 open blockers; Amber = 1–2 open, none >14 days old; Red = any blocker open >14 days or flagged "critical."
- **Sentiment**: LLM classifies PM/stakeholder free text into positive/neutral/negative and compares to the prior week. Green = positive/stable; Amber = neutral or a single negative mention; Red = negative and worsening.

## Rolling Up to One Overall RAG

1. **Weighted score**: Schedule 25%, Budget 20%, Milestones 25%, Blockers 20%, Sentiment 10%. Weighted average of the 0/1/2 scores maps to: ≥1.5 → Green, 0.8–1.5 → Amber, <0.8 → Red.
2. **Override rules (escalation logic)** — these trump the weighted score, because some things are simply disqualifying regardless of the average:
   - **Any single dimension scoring Red caps the overall status at Amber**, even if the weighted average clears the Green threshold. A weighted average can land in Green territory while one dimension's reason literally says "significant slippage" — reporting Green next to that text would destroy trust in the tool faster than being overly cautious would. This is a floor, not an escalation to Red by itself.
   - Any blocker explicitly tagged **"critical"** or open >21 days → overall **Red**.
   - ≥2 overdue milestones → overall **Red**, even if other dimensions are green.
   - A **second, independent** dimension scoring Red **and** sentiment also scoring Red → overall **Red** (compounding risk). Sentiment scoring Red on its own is not enough to trigger this — it caps at Amber via the rule above, same as any other lone Red dimension. Compounding requires two distinct bad signals reinforcing each other, not one dimension read twice.
3. **Never silently default to Green.** If a dimension can't be computed (see below), it is scored **Amber** and explicitly flagged as "insufficient data" in the output — missing data is a risk signal, not a clean bill of health.

## Known limitation
Blocker scoring currently uses **count and a "critical" keyword tag**, not per-blocker age — a blocker string isn't attached to the date it was first opened, so "open >14/21 days" (implied above) isn't yet computed automatically; it's a manual PM-supplied tag today. The natural fix is to track each blocker's first-seen week once week-over-week data exists (see the trend-demo dataset), and is called out here rather than silently overstated.

## Assumptions About the Data
- Project plans arrive as two files per project: a **task list** (planned/actual dates, % complete, milestone flag) and a **weekly PM update** (budget actuals, free-text blockers, free-text stakeholder notes). This mirrors a typical Smartsheet/MPP export + PM narrative.
- Dates, budget figures, and status text will be inconsistently formatted (dates as text, blockers as a paragraph instead of a list, missing owners) — the ingestion layer is built to tolerate this rather than fail.
- "Stakeholder sentiment" is inferred from whatever free text is available (PM notes, client email excerpts pasted in); if none is provided, sentiment defaults to Amber/insufficient-data rather than being ignored.
- Budget burn is a simplified proxy (actual/planned-to-date), not full Earned Value Management (SPI/CPI with EV) — appropriate for weekly PM-level reporting; can be upgraded to true EVM if task-level cost data is available later.
- RAG is a **decision-support signal**, not a replacement for PM judgment — every report includes the plain-English "why," and a PM can annotate/override with justification (logged, not silently discarded).
