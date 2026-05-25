"""eval/run_eval.py — 4-layer evaluation harness.
Usage: python eval/run_eval.py --model_dir outputs/r16 --data_dir data/processed [--base_model auto] [--max_eval_samples 50]
"""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from data.dataset_utils import build_prompt,load_jsonl
from eval.metrics import compute_bertscore,compute_calibration_data,compute_em_f1,compute_rouge_l,llm_as_judge
from eval.calibration import plot_metric_comparison,plot_reliability_diagram,plot_ablation_results

def load_for_inference(model_dir,is_base=False):
    from transformers import AutoTokenizer
    if is_base:
        meta_p=os.path.join(model_dir,"training_metadata.json")
        name="unsloth/llama-3.2-3b-instruct"
        if os.path.exists(meta_p):
            with open(meta_p) as f: name=json.load(f).get("model_name",name)
        print(f"\nLoading BASE: {name}")
        tok=AutoTokenizer.from_pretrained(name,trust_remote_code=True)
        from transformers import AutoModelForCausalLM
        model=AutoModelForCausalLM.from_pretrained(name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",trust_remote_code=True)
    else:
        print(f"\nLoading FINE-TUNED: {model_dir}")
        tok=AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
        try:
            from peft import AutoPeftModelForCausalLM
            model=AutoPeftModelForCausalLM.from_pretrained(model_dir,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu")
        except Exception:
            from transformers import AutoModelForCausalLM
            model=AutoModelForCausalLM.from_pretrained(model_dir,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu")
    tok.pad_token=tok.eos_token; tok.padding_side="left"
    model.eval(); return model,tok

@torch.no_grad()
def generate_predictions(model,tokenizer,examples,batch_size=4,max_new_tokens=256):
    preds,confs=[],[]
    device=next(model.parameters()).device
    for i in tqdm(range(0,len(examples),batch_size),desc="  Generating"):
        batch=examples[i:i+batch_size]
        prompts=[build_prompt(ex,include_output=False) for ex in batch]
        inp=tokenizer(prompts,return_tensors="pt",padding=True,truncation=True,max_length=1800).to(device)
        out=model.generate(**inp,max_new_tokens=max_new_tokens,do_sample=False,
                           pad_token_id=tokenizer.eos_token_id,
                           return_dict_in_generate=True,output_scores=True)
        pl=inp["input_ids"].shape[1]
        for seq in out.sequences:
            preds.append(tokenizer.decode(seq[pl:],skip_special_tokens=True).strip())
        if out.scores:
            for j in range(len(batch)):
                probs=[torch.softmax(s[j],dim=-1).max().item() for s in out.scores if j<s.shape[0]]
                confs.append(float(np.mean(probs)) if probs else 0.5)
    while len(confs)<len(preds): confs.append(0.5)
    return preds,confs

def extract_answer(text):
    for m in ("Final Answer:","final answer:","Answer:","answer:"):
        if m in text: return text.split(m)[-1].strip().split("\n")[0].strip()
    lines=[l.strip() for l in text.split("\n") if l.strip()]
    return lines[-1] if lines else text.strip()

def run_evaluation(model,tokenizer,examples,label,output_dir,openai_api_key=None,max_eval_samples=None,batch_size=4):
    import random
    if max_eval_samples and len(examples)>max_eval_samples:
        random.seed(42); examples=random.sample(examples,max_eval_samples)
    print(f"\n{'='*55}\nEvaluating: {label} ({len(examples)} examples)\n{'='*55}")
    preds_raw,confs=generate_predictions(model,tokenizer,examples,batch_size=batch_size)
    preds=[extract_answer(p) for p in preds_raw]
    refs=[ex["answer"] for ex in examples]
    qs  =[ex["question"] for ex in examples]
    res={"label":label,"n_examples":len(examples)}

    print("\n[Layer 1] EM + F1…")
    l1=compute_em_f1(preds,refs); res.update(l1)
    print(f"  EM={l1['exact_match']*100:.2f}%  F1={l1['f1']*100:.2f}%")

    print("\n[Layer 2] ROUGE-L + BERTScore…")
    try: l2r=compute_rouge_l(preds_raw,refs); res.update(l2r); print(f"  ROUGE-L={l2r['rouge_l']:.4f}")
    except Exception as e: print(f"  ROUGE-L failed: {e}")
    try: l2b=compute_bertscore(preds_raw,refs); res.update(l2b); print(f"  BERTScore={l2b['bertscore_f1']:.4f}")
    except Exception as e: print(f"  BERTScore failed: {e}")

    print("\n[Layer 3] LLM-as-judge…")
    l3=llm_as_judge(preds,refs,qs,api_key=openai_api_key,max_samples=50); res.update(l3)
    if l3.get("llm_judge_overall"): print(f"  Overall={l3['llm_judge_overall']:.2f}/5")

    print("\n[Layer 4] Calibration…")
    calib=compute_calibration_data(confs[:len(l1["em_scores"])],l1["em_scores"])
    res["calibration"]=calib; print(f"  ECE={calib['ece']:.4f}")

    os.makedirs(output_dir,exist_ok=True)
    tag=label.replace(" ","_").lower()
    with open(os.path.join(output_dir,f"predictions_{tag}.jsonl"),"w",encoding="utf-8") as f:
        for ex,raw,clean,ref,conf,em,f1v in zip(examples,preds_raw,preds,refs,
            confs[:len(preds)],l1["em_scores"],l1["f1_scores"]):
            f.write(json.dumps({"question":ex.get("question",""),"reference":ref,
                "prediction_raw":raw,"prediction_clean":clean,
                "confidence":round(conf,4),"exact_match":em,"f1":round(f1v,4)},ensure_ascii=False)+"\n")

    summary={k:v for k,v in res.items() if not isinstance(v,list) and k!="llm_judge_raw"}
    return summary,res

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_dir",required=True)
    ap.add_argument("--base_model",default=None,
                    help="Pass 'auto' to run base model comparison using name from metadata")
    ap.add_argument("--data_dir",default="data/processed")
    ap.add_argument("--output_dir",default="results/")
    ap.add_argument("--openai_api_key",default=os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--max_eval_samples",type=int,default=None)
    ap.add_argument("--batch_size",type=int,default=4)
    args=ap.parse_args()

    plots_dir=os.path.join(args.output_dir,"plots")
    os.makedirs(plots_dir,exist_ok=True)
    test_ex=load_jsonl(os.path.join(args.data_dir,"test.jsonl"))
    print(f"Test examples: {len(test_ex)}")
    all_res={}

    ft_m,ft_t=load_for_inference(args.model_dir,is_base=False)
    ft_sum,ft_full=run_evaluation(ft_m,ft_t,test_ex,"Fine-tuned",args.output_dir,
                                   args.openai_api_key,args.max_eval_samples,args.batch_size)
    all_res["finetuned"]=ft_sum
    del ft_m
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    base_sum=base_full=None
    if args.base_model is not None:
        try:
            bm,bt=load_for_inference(args.model_dir,is_base=True)
            base_sum,base_full=run_evaluation(bm,bt,test_ex,"Base",args.output_dir,
                                               args.openai_api_key,args.max_eval_samples,args.batch_size)
            all_res["base"]=base_sum
            del bm
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as e: print(f"\nBase eval skipped: {e}")

    with open(os.path.join(args.output_dir,"comparison_table.json"),"w") as f:
        json.dump(all_res,f,indent=2)
    print(f"\nResults → {args.output_dir}/comparison_table.json")

    print("\nGenerating plots…")
    if base_sum and ft_sum:
        plot_metric_comparison(base_sum,ft_sum,os.path.join(plots_dir,"metric_comparison.png"))
    if base_full and ft_full:
        plot_reliability_diagram(base_full["calibration"],ft_full["calibration"],
                                  os.path.join(plots_dir,"reliability_diagram.png"))

    print(f"\n{'='*60}")
    print(f"{'METRIC':<26}{'BASE':>10}{'FINE-TUNED':>13}{'Δ':>10}")
    print("-"*60)
    for name,key,sc,unit in [("Exact Match (%)","exact_match",100,"%"),
        ("F1 Score (%)","f1",100,"%"),("ROUGE-L (%)","rouge_l",100,"%"),
        ("BERTScore F1 (%)","bertscore_f1",100,"%"),("LLM Judge (1-5)","llm_judge_overall",1,"/5")]:
        fv=ft_sum.get(key)
        if fv is None: continue
        bv=base_sum.get(key) if base_sum else None
        fs=f"{fv*sc:.2f}{unit}"; bs=f"{bv*sc:.2f}{unit}" if bv else "—"
        ds=f"+{(fv-bv)*sc:.2f}{unit}" if bv else "—"
        print(f"{name:<26}{bs:>10}{fs:>13}{ds:>10}")
    print("="*60)

if __name__=="__main__": main()
