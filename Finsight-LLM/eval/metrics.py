"""eval/metrics.py — 4-layer evaluation metrics."""
import json, re, string
from collections import Counter
from typing import Optional
import numpy as np

def normalize_answer(text):
    text = str(text).lower().strip()
    text = re.sub(r"\b(a|an|the)\b"," ",text)
    text = text.replace("$","").replace("£","").replace("€","").replace("%","")
    text = re.sub(r"\b(billion)\b","000000000",text)
    text = re.sub(r"\b(million)\b","000000",text)
    text = re.sub(r"\b(thousand)\b","000",text)
    text = re.sub(r"(?<=\d)b\b","000000000",text)
    text = re.sub(r"(?<=\d)m\b","000000",text)
    text = re.sub(r"(?<=\d)k\b","000",text)
    text = text.replace(",","")
    text = text.translate(str.maketrans("","",string.punctuation))
    return " ".join(text.split())

def exact_match(pred, ref):
    return float(normalize_answer(pred) == normalize_answer(ref))

def token_f1(pred, ref):
    pt = normalize_answer(pred).split()
    gt = normalize_answer(ref).split()
    if not pt and not gt: return 1.0
    if not pt or not gt: return 0.0
    common = Counter(pt) & Counter(gt)
    nc = sum(common.values())
    if nc == 0: return 0.0
    p = nc/len(pt); r = nc/len(gt)
    return (2*p*r)/(p+r)

def compute_em_f1(predictions, references):
    assert len(predictions)==len(references)
    em = [exact_match(p,r) for p,r in zip(predictions,references)]
    f1 = [token_f1(p,r)    for p,r in zip(predictions,references)]
    return {"exact_match":float(np.mean(em)),"f1":float(np.mean(f1)),
            "em_scores":em,"f1_scores":f1}

def compute_rouge_l(predictions, references):
    try: from rouge_score import rouge_scorer
    except ImportError: raise ImportError("pip install rouge-score")
    sc = rouge_scorer.RougeScorer(["rougeL"],use_stemmer=True)
    scores = [sc.score(r,p)["rougeL"].fmeasure for p,r in zip(predictions,references)]
    return {"rouge_l":float(np.mean(scores)),"rouge_l_scores":scores}

def compute_bertscore(predictions, references,
                       model_type="distilbert-base-uncased", batch_size=32):
    try: from bert_score import score as _bs
    except ImportError: raise ImportError("pip install bert-score")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    P,R,F1 = _bs(predictions,references,model_type=model_type,
                  batch_size=batch_size,device=device,verbose=False)
    f1l = F1.tolist()
    return {"bertscore_f1":float(np.mean(f1l)),
            "bertscore_precision":float(P.mean().item()),
            "bertscore_recall":float(R.mean().item()),
            "bertscore_f1_scores":f1l}

_JUDGE = """\
You are a senior financial analyst evaluating an AI answer.
Question: {question}
Reference: {reference}
Model answer: {prediction}
Score on 3 criteria (1-5 each):
1. Accuracy — is the numerical answer correct?
2. Reasoning — are calculation steps clear?
3. Completeness — is the answer fully explained?
Respond ONLY with JSON: {{"accuracy":X,"reasoning":X,"completeness":X,"overall":X,"comment":"..."}}
overall = mean of three scores rounded to 1 decimal."""

def llm_as_judge(predictions, references, questions,
                  api_key=None, model="gpt-4o-mini", max_samples=100):
    empty = {"llm_judge_overall":None,"llm_judge_accuracy":None,
             "llm_judge_reasoning":None,"llm_judge_completeness":None,"llm_judge_n_evaluated":0}
    if not api_key:
        print("  [LLM-judge] No API key — skipping"); return empty
    try: from openai import OpenAI
    except ImportError: print("  [LLM-judge] openai not installed"); return empty
    client = OpenAI(api_key=api_key)
    n = min(len(predictions), max_samples)
    if n < len(predictions):
        import random; idx = random.sample(range(len(predictions)),n)
        predictions=[predictions[i] for i in idx]
        references=[references[i] for i in idx]
        questions=[questions[i] for i in idx]
    from tqdm import tqdm
    results=[]
    for pred,ref,q in tqdm(zip(predictions,references,questions),total=n,desc="LLM-judge"):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":_JUDGE.format(question=q,reference=ref,prediction=pred)}],
                temperature=0.0,max_tokens=200,response_format={"type":"json_object"})
            results.append(json.loads(r.choices[0].message.content))
        except Exception as e:
            print(f"  judge err: {e}")
    if not results: return empty
    avg = lambda k: float(np.mean([r[k] for r in results if k in r]))
    return {"llm_judge_overall":avg("overall"),"llm_judge_accuracy":avg("accuracy"),
            "llm_judge_reasoning":avg("reasoning"),"llm_judge_completeness":avg("completeness"),
            "llm_judge_n_evaluated":len(results),"llm_judge_raw":results}

def compute_calibration_data(confidences, correct, n_bins=10):
    confs = np.clip(np.array(confidences,dtype=float),0,1)
    corr  = np.array(correct,dtype=float)
    n     = min(len(confs),len(corr))
    confs,corr = confs[:n],corr[:n]
    edges = np.linspace(0,1,n_bins+1)
    ba,bc,bn = [],[],[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask = (confs>lo)&(confs<=hi)
        cnt  = int(mask.sum())
        ba.append(float(corr[mask].mean()) if cnt>0 else 0.0)
        bc.append(float(confs[mask].mean()) if cnt>0 else float((lo+hi)/2))
        bn.append(cnt)
    ece = float(np.sum([(cnt/n)*abs(a-c) for a,c,cnt in zip(ba,bc,bn)]))
    return {"ece":ece,"bin_accuracies":ba,"bin_confidences":bc,"bin_counts":bn,"n_bins":n_bins}
