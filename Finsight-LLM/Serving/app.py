"""serving/app.py — FastAPI inference server.
Usage: python serving/app.py --model_dir outputs/r16 --port 8000
"""
import argparse,json,os,sys,time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import torch,uvicorn
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from data.dataset_utils import SYSTEM_PROMPT

class GenerateRequest(BaseModel):
    question:       str   = Field(...,description="Financial question")
    context:        str   = Field("", description="Financial data / table context")
    max_new_tokens: int   = Field(256,ge=1,le=1024)
    temperature:    float = Field(0.1,ge=0.0,le=2.0)
    do_sample:      bool  = Field(False)

class GenerateResponse(BaseModel):
    answer:     str
    latency_ms: float
    model_dir:  str

class HealthResponse(BaseModel):
    status:   str
    model:    str
    device:   str
    gpu_name: Optional[str]=None

_model=_tokenizer=None; _model_dir=""

def load_model(model_dir):
    global _model,_tokenizer,_model_dir
    from transformers import AutoTokenizer
    _model_dir=model_dir
    _tokenizer=AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
    _tokenizer.pad_token=_tokenizer.eos_token; _tokenizer.padding_side="left"
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    dmap="auto" if torch.cuda.is_available() else "cpu"
    try:
        from peft import AutoPeftModelForCausalLM
        _model=AutoPeftModelForCausalLM.from_pretrained(model_dir,torch_dtype=dtype,device_map=dmap)
        print(f"  PEFT adapter loaded from {model_dir}")
    except Exception:
        from transformers import AutoModelForCausalLM
        _model=AutoModelForCausalLM.from_pretrained(model_dir,torch_dtype=dtype,device_map=dmap,trust_remote_code=True)
        print(f"  Merged model loaded from {model_dir}")
    _model.eval()
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

def _prompt(question,context):
    inst=f"Based on the following financial information, answer the question.\n\nQuestion: {question}"
    ctx=context.strip() if context.strip() else "No additional context."
    return (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{inst}\n\nContext:\n{ctx}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n")

@torch.no_grad()
def _generate(req):
    p=_prompt(req.question,req.context)
    device=next(_model.parameters()).device
    inp=_tokenizer(p,return_tensors="pt",truncation=True,max_length=1800).to(device)
    out=_model.generate(**inp,max_new_tokens=req.max_new_tokens,
                         do_sample=req.do_sample,
                         temperature=req.temperature if req.do_sample else 1.0,
                         pad_token_id=_tokenizer.eos_token_id)
    new=out[0][inp["input_ids"].shape[1]:]
    return _tokenizer.decode(new,skip_special_tokens=True).strip()

@asynccontextmanager
async def lifespan(app):
    md=os.environ.get("MODEL_DIR","outputs/r16")
    print(f"Loading model: {md}"); load_model(md); print("Ready ✓")
    yield; print("Shutdown")

app=FastAPI(title="FinLLM API",description="Financial QA with QLoRA Llama 3.2-3B",version="1.0.0",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

@app.get("/health",response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok",model=_model_dir,
                          device="cuda" if torch.cuda.is_available() else "cpu",
                          gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

@app.post("/generate",response_model=GenerateResponse)
async def generate(req:GenerateRequest):
    if _model is None: raise HTTPException(503,"Model not loaded")
    t0=time.perf_counter()
    try: answer=_generate(req)
    except Exception as e: raise HTTPException(500,str(e))
    return GenerateResponse(answer=answer,latency_ms=round((time.perf_counter()-t0)*1000,2),model_dir=_model_dir)

@app.get("/")
async def root(): return {"service":"FinLLM API","docs":"/docs","endpoints":["/health","/generate"]}

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_dir",default="outputs/r16")
    ap.add_argument("--host",default="0.0.0.0"); ap.add_argument("--port",type=int,default=8000)
    args=ap.parse_args()
    os.environ["MODEL_DIR"]=args.model_dir
    uvicorn.run("serving.app:app",host=args.host,port=args.port,reload=False)
