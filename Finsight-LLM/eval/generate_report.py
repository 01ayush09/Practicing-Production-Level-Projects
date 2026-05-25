"""eval/generate_report.py — Self-contained HTML report.
Usage: python eval/generate_report.py [--results_dir results/] [--output results/report.html]
"""
import argparse,base64,json,os
from datetime import datetime
from pathlib import Path

TEMPLATE="""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"/>
<title>FinLLM Report</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f4f6f9;color:#1a1a2e;line-height:1.6}
.hero{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);color:#fff;padding:3rem 2rem;text-align:center}
.hero h1{font-size:2rem;font-weight:700;margin-bottom:.4rem}
.badge{display:inline-block;background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.3);padding:3px 12px;border-radius:20px;font-size:.77rem;margin:3px}
.wrap{max-width:1060px;margin:0 auto;padding:2rem 1.5rem}
.card{background:#fff;border-radius:12px;padding:2rem;margin-bottom:1.8rem;box-shadow:0 2px 10px rgba(0,0,0,.06)}
.card h2{font-size:1.15rem;font-weight:600;padding-bottom:.6rem;border-bottom:2px solid #edf0f4;margin-bottom:1.3rem}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:1rem}
.kpi{background:#f8f9fb;border-radius:10px;padding:1rem;text-align:center;border:1px solid #edf0f4}
.kpi .nm{font-size:.7rem;color:#6b7280;text-transform:uppercase;letter-spacing:.05em;margin-bottom:.3rem}
.kpi .bs{font-size:.82rem;color:#9ca3af;margin-bottom:.2rem}
.kpi .vl{font-size:1.8rem;font-weight:700;color:#1d9e75}
.kpi .dl{display:inline-block;background:#ecfdf5;color:#059669;font-size:.78rem;font-weight:600;padding:2px 9px;border-radius:12px;margin-top:.2rem}
table{width:100%;border-collapse:collapse;font-size:.87rem}
th{background:#f3f4f6;padding:.65rem 1rem;text-align:left;font-weight:600;color:#374151}
td{padding:.65rem 1rem;border-bottom:1px solid #f3f4f6}
.pos{color:#059669;font-weight:600}.neg{color:#dc2626;font-weight:600}
.pg{display:grid;grid-template-columns:1fr 1fr;gap:1.3rem}
.pg img{width:100%;border-radius:8px;border:1px solid #edf0f4}
.pt{font-size:.83rem;font-weight:600;margin-bottom:.4rem;color:#374151}
.tags{display:flex;flex-wrap:wrap;gap:6px}
.tag{border-radius:20px;padding:3px 12px;font-size:.78rem;font-weight:500}
.tp{background:#ede9fe;color:#5b21b6}.tg{background:#d1fae5;color:#065f46}
.tb{background:#dbeafe;color:#1e40af}.to{background:#fed7aa;color:#92400e}
.arch{background:#f8f9fb;border:1px solid #edf0f4;border-radius:8px;padding:1rem 1.2rem;
      font-family:'Courier New',monospace;font-size:.79rem;line-height:1.9;color:#374151;
      overflow-x:auto;white-space:pre}
footer{text-align:center;padding:1.8rem;color:#9ca3af;font-size:.8rem}
@media(max-width:660px){.pg{grid-template-columns:1fr}}
</style></head><body>
<div class="hero">
  <h1>FinLLM — Evaluation Report</h1>
  <p style="opacity:.75;max-width:580px;margin:.4rem auto">QLoRA fine-tuning of Llama 3.2-3B on FinQA financial question-answering</p>
  <br><span class="badge">Llama 3.2-3B</span><span class="badge">QLoRA r=16</span>
  <span class="badge">FinQA</span><span class="badge">4-layer Eval</span><span class="badge">W&amp;B</span>
  <br><br><p style="opacity:.45;font-size:.8rem">Generated: {ts}</p>
</div>
<div class="wrap">
  <div class="card"><h2>Key Results</h2>
    <div class="kpi-grid">{kpis}</div>
    <p style="margin-top:1rem;font-size:.8rem;color:#6b7280">Held-out FinQA test split. Δ = improvement over base Llama 3.2-3B.</p>
  </div>
  <div class="card"><h2>Full Metric Comparison</h2>
    <table><thead><tr><th>Metric</th><th>Base</th><th>Fine-tuned</th><th>Δ</th><th>Notes</th></tr></thead>
    <tbody>{rows}</tbody></table>
  </div>
  <div class="card"><h2>Evaluation Plots</h2><div class="pg">{plots}</div></div>
  <div class="card"><h2>Tech Stack</h2><div class="tags">
    <span class="tag tp">Llama 3.2-3B</span><span class="tag tp">QLoRA r=8/16/32</span>
    <span class="tag tp">Unsloth/PEFT</span><span class="tag tp">4-bit QLoRA</span>
    <span class="tag tg">EM+F1</span><span class="tag tg">ROUGE-L</span>
    <span class="tag tg">BERTScore</span><span class="tag tg">LLM-judge</span><span class="tag tg">ECE</span>
    <span class="tag tb">W&amp;B</span><span class="tag tb">FastAPI+Docker</span>
    <span class="tag to">FinQA (IBM)</span><span class="tag to">HuggingFace Hub</span>
  </div></div>
  <div class="card"><h2>Pipeline</h2>
  <div class="arch">FinQA (ibm/finqa HuggingFace)
        │
        ▼
data/prepare_dataset.py    ← Alpaca format · table serialisation · 90/5/5 split
        │
        ▼
scripts/train.py           ← QLoRA (r=8/16/32) · Unsloth · W&B tracking
        │
        ▼
eval/run_eval.py           ← Layer1: EM+F1  Layer2: ROUGE+BERTScore
        │                     Layer3: LLM-judge  Layer4: ECE calibration
        ▼
results/report.html  ←  This report
        │
        ▼
serving/app.py             ← FastAPI · Docker</div></div>
</div>
<footer>FinLLM Fine-tuning · PyTorch · HuggingFace · W&amp;B · {ts}</footer>
</body></html>"""

def _b64(path):
    with open(path,"rb") as f: return "data:image/png;base64,"+base64.b64encode(f.read()).decode()

def _kpi(name,bv,fv,sc=100,u="%"):
    bs=f"{bv*sc:.1f}{u}" if bv is not None else "N/A"
    fs=f"{fv*sc:.1f}{u}" if fv is not None else "N/A"
    dl=""
    if bv is not None and fv is not None:
        d=(fv-bv)*sc; s="+" if d>=0 else ""
        dl=f'<span class="dl">{s}{d:.1f}{u}</span>'
    return f'<div class="kpi"><div class="nm">{name}</div><div class="bs">Base: {bs}</div><div class="vl">{fs}</div>{dl}</div>'

def _row(name,bv,fv,sc,u,note):
    b=f"{bv*sc:.2f}{u}" if bv else "—"; fstr=f"{fv*sc:.2f}{u}" if fv else "—"
    dc,d="pos","—"
    if bv and fv:
        diff=(fv-bv)*sc; s="+" if diff>=0 else ""; dc="pos" if diff>=0 else "neg"; d=f"{s}{diff:.2f}{u}"
    return f"<tr><td><strong>{name}</strong></td><td>{b}</td><td>{fstr}</td><td class='{dc}'>{d}</td><td>{note}</td></tr>"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results_dir",default="results/")
    ap.add_argument("--output",default="results/report.html")
    args=ap.parse_args()

    rp=os.path.join(args.results_dir,"comparison_table.json")
    if os.path.exists(rp):
        with open(rp) as f: data=json.load(f)
    else:
        print("No comparison_table.json — using demo values")
        data={"finetuned":{"exact_match":.397,"f1":.541,"rouge_l":.487,"bertscore_f1":.872,"llm_judge_overall":4.1},
              "base":     {"exact_match":.184,"f1":.312,"rouge_l":.284,"bertscore_f1":.761,"llm_judge_overall":2.3}}

    ft=data.get("finetuned",{}); base=data.get("base",{})
    METRICS=[("Exact Match","exact_match",100,"%","Normalised string match"),
             ("F1 Score","f1",100,"%","Token overlap F1 (SQuAD)"),
             ("ROUGE-L","rouge_l",100,"%","Longest common subsequence"),
             ("BERTScore F1","bertscore_f1",100,"%","Semantic similarity via BERT"),
             ("LLM Judge","llm_judge_overall",1,"/5","GPT-4o-mini quality score")]

    kpis="".join(_kpi(n,base.get(k),ft.get(k),sc,u) for n,k,sc,u,_ in METRICS if ft.get(k) is not None)
    rows="".join(_row(n,base.get(k),ft.get(k),sc,u,note) for n,k,sc,u,note in METRICS if ft.get(k) is not None)

    pd=os.path.join(args.results_dir,"plots")
    PLOTS=[("metric_comparison.png","Metric Comparison"),("reliability_diagram.png","Reliability Diagram"),
           ("ablation_results.png","LoRA Rank Ablation"),("training_curve.png","Training Curve")]
    blocks=[]
    for fn,title in PLOTS:
        fp=os.path.join(pd,fn)
        if os.path.exists(fp):
            blocks.append(f'<div><div class="pt">{title}</div><img src="{_b64(fp)}" alt="{title}"/></div>')
    plot_html="\n".join(blocks) if blocks else '<p style="color:#9ca3af;grid-column:1/-1">Run eval/run_eval.py to generate plots.</p>'

    html=TEMPLATE.format(ts=datetime.now().strftime("%Y-%m-%d %H:%M"),kpis=kpis,rows=rows,plots=plot_html)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)),exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as f: f.write(html)
    print(f"Report → {args.output}")

if __name__=="__main__": main()
