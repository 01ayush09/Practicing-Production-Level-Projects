"""
data/prepare_dataset.py
Downloads FinQA, formats to Alpaca, writes train/val/test JSONL.
Usage: python data/prepare_dataset.py [--max_samples N]
"""
import argparse, json, os, random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Given a question about a company's "
    "financial performance and relevant context from financial reports, provide "
    "an accurate, step-by-step answer. For numerical questions, show your "
    "reasoning and calculations clearly."
)

def build_context(ex):
    parts = []
    pre = ex.get("pre_text", [])
    if pre: parts.append("Financial report excerpt:\n" + "\n".join(pre[:2]))
    table = ex.get("table", [])
    if table:
        rows = []
        for row in table:
            if isinstance(row, list): rows.append(" | ".join(str(c) for c in row))
            elif isinstance(row, str): rows.append(row)
        parts.append("Financial table:\n" + "\n".join(rows))
    post = ex.get("post_text", [])
    if post: parts.append("\n".join(post[:2]))
    return "\n\n".join(parts) if parts else "No additional context provided."

def build_answer(ex):
    answer = str(ex.get("answer","")).strip()
    steps  = ex.get("steps", [])
    if steps and isinstance(steps, list):
        lines = []
        for i, s in enumerate(steps, 1):
            if isinstance(s, dict):
                lines.append(f"Step {i}: {s.get('op','')}({', '.join(str(a) for a in s.get('args',[]))}) = {s.get('res','')}")
            elif isinstance(s, str):
                lines.append(f"Step {i}: {s}")
        if lines:
            return "\n".join(lines) + f"\n\nFinal Answer: {answer}"
    return f"Answer: {answer}"

def format_alpaca(ex):
    q = str(ex.get("question","")).strip()
    a = str(ex.get("answer","")).strip()
    if not q or not a: return None
    return {
        "instruction": f"Based on the following financial information, answer the question.\n\nQuestion: {q}",
        "input":    build_context(ex),
        "output":   build_answer(ex),
        "id":       ex.get("id",""),
        "question": q,
        "answer":   a,
    }

def build_full_prompt(ex):
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{ex['instruction']}\n\nContext:\n{ex['input']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{ex['output']}<|eot_id|>"
    )

def _synthetic_fallback():
    from datasets import Dataset, DatasetDict
    raw = [
        {"id":"s001","question":"What was the percentage change in net revenue from 2020 to 2021?",
         "pre_text":["Strong growth in fiscal 2021."],
         "table":[["","2021","2020"],["Net Revenue","$12.4B","$10.2B"],["Net Income","$2.3B","$1.8B"]],
         "post_text":["Management cited digital adoption."],
         "answer":"21.57%",
         "steps":[{"op":"subtract","args":["12.4","10.2"],"res":"2.2"},
                  {"op":"divide","args":["2.2","10.2"],"res":"0.2157"},
                  {"op":"multiply","args":["0.2157","100"],"res":"21.57"}]},
        {"id":"s002","question":"By how much did net income rise from 2020 to 2021?",
         "pre_text":["Fiscal 2021 showed strong profitability."],
         "table":[["","2021","2020"],["Net Income","$2.3B","$1.8B"]],
         "post_text":[],"answer":"$0.5B",
         "steps":[{"op":"subtract","args":["2.3","1.8"],"res":"0.5"}]},
        {"id":"s003","question":"What is the net income margin for 2021?",
         "pre_text":["Profit margins remained healthy."],
         "table":[["","2021"],["Net Revenue","$12.4B"],["Net Income","$2.3B"]],
         "post_text":[],"answer":"18.55%",
         "steps":[{"op":"divide","args":["2.3","12.4"],"res":"0.1855"},
                  {"op":"multiply","args":["0.1855","100"],"res":"18.55"}]},
    ]
    train = raw * 30; random.shuffle(train)
    return DatasetDict({"train":Dataset.from_list(train),
                        "validation":Dataset.from_list(raw*5),
                        "test":Dataset.from_list(raw*5)})

def load_and_process(max_samples=None):
    print("Loading FinQA …")
    try:
        ds = load_dataset("ibm/finqa", trust_remote_code=True)
        print(f"  Splits: { {k:len(v) for k,v in ds.items()} }")
    except Exception as e:
        print(f"  Fallback ({e})")
        ds = _synthetic_fallback()
    splits = {}
    for name in ("train","validation","test"):
        if name not in ds: continue
        data = ds[name]
        if max_samples and len(data) > max_samples:
            data = data.select(random.sample(range(len(data)), max_samples))
        processed = [f for ex in tqdm(data, desc=f"  {name}") if (f:=format_alpaca(ex))]
        splits[name] = processed
        print(f"  {name}: {len(processed)}")
    return splits

def write_jsonl(examples, path):
    with open(path,"w",encoding="utf-8") as f:
        for ex in examples: f.write(json.dumps(ex,ensure_ascii=False)+"\n")
    print(f"  Saved → {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir",  default="data/processed")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    splits = load_and_process(args.max_samples)
    for src, dst in [("train","train.jsonl"),("validation","val.jsonl"),("test","test.jsonl")]:
        if src in splits:
            write_jsonl(splits[src], os.path.join(args.output_dir, dst))
    stats = {s:len(e) for s,e in splits.items()}
    with open(os.path.join(args.output_dir,"dataset_stats.json"),"w") as f:
        json.dump(stats,f,indent=2)
    print(f"\nDone. Stats: {stats}")

if __name__ == "__main__":
    main()
