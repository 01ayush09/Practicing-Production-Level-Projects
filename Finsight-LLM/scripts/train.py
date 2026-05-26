"""scripts/train.py — QLoRA fine-tuning. Usage: python scripts/train.py [--lora_r 16]"""
import argparse,json,os,sys
from functools import partial
from pathlib import Path
import torch, yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.dataset_utils import FinQADataset, get_collate_fn, load_jsonl

def load_config():
    with open(ROOT/"configs"/"train_config.yaml") as f: return yaml.safe_load(f)

def load_model_and_tokenizer(args, cfg):
    model_cfg = cfg["model"]
    try:
        from unsloth import FastLanguageModel
        print("Using Unsloth ✓")
        model,tok = FastLanguageModel.from_pretrained(
            model_name=args.model_name, max_seq_length=model_cfg["max_seq_length"],
            load_in_4bit=model_cfg["load_in_4bit"], dtype=None)
        model = FastLanguageModel.get_peft_model(
            model, r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=cfg["lora"]["dropout"], target_modules=cfg["lora"]["target_modules"],
            bias=cfg["lora"]["bias"], use_gradient_checkpointing="unsloth",
            random_state=cfg["training"]["seed"])
    except ImportError:
        print("Unsloth not found — using standard PEFT")
        from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
        from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
        tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        tok.pad_token = tok.eos_token; tok.padding_side = "right"
        bnb = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_use_double_quant=True,
              bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16) \
              if model_cfg["load_in_4bit"] and torch.cuda.is_available() else None
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, quantization_config=bnb,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True)
        if bnb: model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=cfg["lora"]["dropout"], target_modules=cfg["lora"]["target_modules"],
            bias=cfg["lora"]["bias"], task_type=cfg["lora"]["task_type"]))
        model.print_trainable_parameters()
    return model, tok

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); total,n=0.0,0
    for batch in loader:
        batch={k:v.to(device) for k,v in batch.items()}
        total+=model(**batch).loss.item(); n+=1
    model.train(); return total/max(n,1)

def train(args, cfg):
    try:
        import wandb
        wandb.init(project=cfg.get("wandb",{}).get("project","finllm-finetune"),
                   name=f"r{args.lora_r}_lr{args.learning_rate}",
                   config=vars(args), tags=cfg.get("wandb",{}).get("tags",[]))
        use_wb=True
    except: use_wb=False

    model,tok = load_model_and_tokenizer(args,cfg)
    device = next(model.parameters()).device
    t = cfg["training"]

    train_raw = load_jsonl(args.train_file or cfg["data"]["train_file"])
    val_raw   = load_jsonl(args.val_file   or cfg["data"]["val_file"])
    if args.max_samples:
        import random; random.seed(t["seed"])
        train_raw = random.sample(train_raw, min(args.max_samples,len(train_raw)))
        val_raw   = val_raw[:max(len(val_raw)//5,10)]
    print(f"Train:{len(train_raw)} Val:{len(val_raw)}")

    pad_id = tok.pad_token_id or tok.eos_token_id
    from torch.utils.data import DataLoader
    train_dl = DataLoader(FinQADataset(train_raw,tok,cfg["model"]["max_seq_length"]),
                          batch_size=args.batch_size, shuffle=True,
                          collate_fn=get_collate_fn(pad_id), num_workers=0)
    val_dl   = DataLoader(FinQADataset(val_raw,tok,cfg["model"]["max_seq_length"]),
                          batch_size=args.eval_batch_size, shuffle=False,
                          collate_fn=get_collate_fn(pad_id), num_workers=0)

    from transformers import get_cosine_schedule_with_warmup
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=args.learning_rate, weight_decay=t["weight_decay"])
    total_steps  = (len(train_dl)//args.gradient_accumulation_steps)*args.num_epochs
    warmup_steps = int(total_steps*t["warmup_ratio"])
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val, gs = float("inf"), 0
    for epoch in range(args.num_epochs):
        model.train(); epoch_loss,nb=0.0,0; opt.zero_grad()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for step,batch in enumerate(pbar):
            batch = {k:v.to(device) for k,v in batch.items()}
            loss  = model(**batch).loss / args.gradient_accumulation_steps
            loss.backward()
            if (step+1)%args.gradient_accumulation_steps==0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step(); sched.step(); opt.zero_grad(); gs+=1
                rl = loss.item()*args.gradient_accumulation_steps
                pbar.set_postfix(loss=f"{rl:.4f}")
                if use_wb and gs%t["logging_steps"]==0:
                    import wandb; wandb.log({"train/loss":rl,"train/lr":sched.get_last_lr()[0],"step":gs})
                if gs%t["eval_steps"]==0:
                    vl = evaluate(model,val_dl,device)
                    print(f"\n  [step {gs}] val_loss={vl:.4f}")
                    if use_wb: import wandb; wandb.log({"val/loss":vl,"step":gs})
                    if vl<best_val:
                        best_val=vl; bd=os.path.join(args.output_dir,"best")
                        model.save_pretrained(bd); tok.save_pretrained(bd)
                        print(f"  ✓ New best → {bd}")
            epoch_loss+=loss.item()*args.gradient_accumulation_steps; nb+=1
        print(f"Epoch {epoch+1} avg loss: {epoch_loss/nb:.4f}")

    model.save_pretrained(args.output_dir); tok.save_pretrained(args.output_dir)
    meta = {"model_name":args.model_name,"lora_r":args.lora_r,"lora_alpha":args.lora_alpha,
            "learning_rate":args.learning_rate,"num_epochs":args.num_epochs,
            "batch_size":args.batch_size,"best_val_loss":best_val,"global_steps":gs}
    with open(os.path.join(args.output_dir,"training_metadata.json"),"w") as f:
        json.dump(meta,f,indent=2)
    print(f"\nDone. Best val loss: {best_val:.4f} | Saved: {args.output_dir}")
    if use_wb: import wandb; wandb.finish()
    return meta

def parse_args():
    cfg=load_config(); t=cfg["training"]
    p=argparse.ArgumentParser()
    p.add_argument("--model_name",  default=cfg["model"]["name"])
    p.add_argument("--train_file",  default=None)
    p.add_argument("--val_file",    default=None)
    p.add_argument("--output_dir",  default=cfg["output"]["dir"])
    p.add_argument("--lora_r",      type=int,   default=cfg["lora"]["r"])
    p.add_argument("--lora_alpha",  type=int,   default=cfg["lora"]["alpha"])
    p.add_argument("--num_epochs",  type=int,   default=t["num_epochs"])
    p.add_argument("--batch_size",  type=int,   default=t["per_device_train_batch_size"])
    p.add_argument("--eval_batch_size",type=int,default=t["per_device_eval_batch_size"])
    p.add_argument("--learning_rate",type=float,default=t["learning_rate"])
    p.add_argument("--gradient_accumulation_steps",type=int,default=t["gradient_accumulation_steps"])
    p.add_argument("--max_samples", type=int,   default=None)
    return p.parse_args(), cfg

if __name__=="__main__":
    args,cfg=parse_args(); train(args,cfg)
