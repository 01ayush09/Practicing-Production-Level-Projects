"""eval/calibration.py — All evaluation plots."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS={"base":"#E24B4A","finetuned":"#1D9E75","accent":"#7F77DD","amber":"#D85A30","blue":"#378ADD"}

def _save(fig,path,dpi=150):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=dpi,bbox_inches="tight"); plt.close(fig)
    print(f"  Plot → {path}")

def plot_reliability_diagram(calib_base,calib_finetuned,output_path):
    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    fig.suptitle("Reliability Diagram — Base vs Fine-tuned",fontsize=14,fontweight="bold",y=1.01)
    for ax,calib,label,color in zip(axes,[calib_base,calib_finetuned],
        ["Base model (Llama 3.2-3B)","Fine-tuned (QLoRA r=16)"],[COLORS["base"],COLORS["finetuned"]]):
        bc=np.array(calib["bin_confidences"]); ba=np.array(calib["bin_accuracies"])
        bn=np.array(calib["bin_counts"]); ece=calib["ece"]
        ax.plot([0,1],[0,1],"k--",lw=1.2,alpha=0.5,label="Perfect")
        ax.fill_between(bc,bc,ba,alpha=0.18,color=color)
        ax.bar(bc,ba,width=0.07,alpha=0.75,color=color,label=f"Accuracy|ECE={ece:.3f}")
        for c,a,n in zip(bc,ba,bn):
            if n>0: ax.text(c,a+0.025,str(n),ha="center",fontsize=7,color="#555")
        ax.set_xlim(0,1); ax.set_ylim(0,1.05)
        ax.set_xlabel("Confidence",fontsize=11); ax.set_ylabel("Accuracy",fontsize=11)
        ax.set_title(f"{label}\nECE={ece:.4f}",fontsize=11)
        ax.legend(loc="upper left",fontsize=9); ax.grid(True,alpha=0.25)
    plt.tight_layout(); _save(fig,output_path)

def plot_metric_comparison(results_base,results_ft,output_path):
    defs=[("exact_match","Exact Match",100),("f1","F1 Score",100),
          ("rouge_l","ROUGE-L",100),("bertscore_f1","BERTScore F1",100)]
    labels=[d[1] for d in defs]
    bvals=[results_base.get(d[0],0)*d[2] for d in defs]
    fvals=[results_ft.get(d[0],0)*d[2]   for d in defs]
    x=np.arange(len(labels)); w=0.34
    fig,ax=plt.subplots(figsize=(10,6))
    b1=ax.bar(x-w/2,bvals,w,label="Base",   color=COLORS["base"],     alpha=0.82)
    b2=ax.bar(x+w/2,fvals,w,label="Fine-tuned r=16",color=COLORS["finetuned"],alpha=0.82)
    for bar in b1:
        h=bar.get_height(); ax.text(bar.get_x()+bar.get_width()/2,h+0.6,f"{h:.1f}%",ha="center",va="bottom",fontsize=9)
    for bar in b2:
        h=bar.get_height(); ax.text(bar.get_x()+bar.get_width()/2,h+0.6,f"{h:.1f}%",ha="center",va="bottom",fontsize=9,fontweight="bold")
    for i,(bv,fv) in enumerate(zip(bvals,fvals)):
        d=fv-bv; s="+" if d>=0 else ""
        ax.annotate(f"{s}{d:.1f}%",(x[i]+w/2,fv+3.5),ha="center",va="bottom",fontsize=9,color="#065f46",fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=11)
    ax.set_ylabel("Score (%)",fontsize=12); ax.set_ylim(0,112)
    ax.set_title("Base vs Fine-tuned — Metric Comparison",fontsize=13,fontweight="bold")
    ax.legend(fontsize=11); ax.grid(True,axis="y",alpha=0.25)
    plt.tight_layout(); _save(fig,output_path)

def plot_ablation_results(ablation_data,output_path):
    if not ablation_data: return
    ranks=sorted({d["lora_r"] for d in ablation_data if "lora_r" in d})
    if not ranks: return
    defs=[("exact_match","Exact Match",COLORS["accent"]),("f1","F1 Score",COLORS["finetuned"]),
          ("rouge_l","ROUGE-L",COLORS["amber"]),("bertscore_f1","BERTScore F1",COLORS["blue"])]
    fig,ax=plt.subplots(figsize=(9,5.5))
    for key,label,color in defs:
        vals=[next((d for d in ablation_data if d.get("lora_r")==r),{}).get(key,0)*100 for r in ranks]
        ax.plot(ranks,vals,"o-",label=label,color=color,lw=2,markersize=8)
        for r,v in zip(ranks,vals):
            ax.annotate(f"{v:.1f}%",(r,v),textcoords="offset points",xytext=(0,9),ha="center",fontsize=8.5)
    ax.set_xticks(ranks); ax.set_xticklabels([f"r={r}" for r in ranks],fontsize=11)
    ax.set_ylabel("Score (%)",fontsize=12); ax.set_ylim(0,100)
    ax.set_title("LoRA Rank Ablation",fontsize=13,fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True,alpha=0.25)
    plt.tight_layout(); _save(fig,output_path)

def plot_training_curve(loss_data,output_path):
    if not loss_data: return
    steps=[d.get("step",d.get("_step",i)) for i,d in enumerate(loss_data)]
    tl=[d.get("train_loss",d.get("train/loss",0)) for d in loss_data]
    vl=[d.get("val_loss",d.get("val/loss",None)) for d in loss_data]
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(steps,tl,lw=1.6,color=COLORS["accent"],label="Train loss")
    vs=[s for s,v in zip(steps,vl) if v is not None]
    vv=[v for v in vl if v is not None]
    if vs: ax.plot(vs,vv,"o-",lw=1.6,markersize=5,color=COLORS["amber"],label="Val loss")
    ax.set_xlabel("Step",fontsize=12); ax.set_ylabel("Loss",fontsize=12)
    ax.set_title("Training Curve",fontsize=13,fontweight="bold")
    ax.legend(fontsize=11); ax.grid(True,alpha=0.25)
    plt.tight_layout(); _save(fig,output_path)
