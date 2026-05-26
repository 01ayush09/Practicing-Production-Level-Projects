"""scripts/ablation_sweep.py — LoRA rank ablation. Usage: python scripts/ablation_sweep.py [--quick]"""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

def run_one(rank,args):
    run_dir=os.path.join(args.output_dir,f"r{rank}")
    cmd=[sys.executable,str(ROOT/"scripts"/"train.py"),
         "--lora_r",str(rank),"--lora_alpha",str(rank*2),"--output_dir",run_dir]
    if args.data_dir:
        cmd+=["--train_file",os.path.join(args.data_dir,"train.jsonl"),
              "--val_file",os.path.join(args.data_dir,"val.jsonl")]
    if args.quick: cmd+=["--num_epochs","1","--max_samples","200"]
    print(f"\n{'='*55}\nAblation r={rank}\n{'='*55}")
    result=subprocess.run(cmd,text=True)
    if result.returncode!=0:
        return {"lora_r":rank,"status":"failed","run_dir":run_dir}
    meta_p=os.path.join(run_dir,"training_metadata.json")
    if os.path.exists(meta_p):
        with open(meta_p) as f: meta=json.load(f)
        meta["run_dir"]=run_dir; meta["status"]="success"; return meta
    return {"lora_r":rank,"run_dir":run_dir,"status":"success"}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir",   default="data/processed")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--ranks",nargs="+",type=int,default=[8,16,32])
    ap.add_argument("--quick",action="store_true")
    args=ap.parse_args()
    os.makedirs(args.output_dir,exist_ok=True)
    results=[run_one(r,args) for r in args.ranks]
    sp=os.path.join(args.output_dir,"ablation_summary.json")
    with open(sp,"w") as f: json.dump(results,f,indent=2)
    print(f"\nSummary → {sp}")
    print(f"\n{'LoRA r':>8} | {'Val Loss':>10} | {'Status':>8}")
    print("-"*35)
    for r in results:
        vl=f"{r['best_val_loss']:.4f}" if "best_val_loss" in r else "—"
        print(f"{r.get('lora_r','?'):>8} | {vl:>10} | {r.get('status','?'):>8}")
    ok=[r for r in results if r.get("status")=="success" and "best_val_loss" in r]
    if ok:
        best=min(ok,key=lambda r:r["best_val_loss"])
        print(f"\nBest: r={best['lora_r']} (val_loss={best['best_val_loss']:.4f})")

if __name__=="__main__": main()
