
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.inference import InferencePipeline

app = FastAPI(title="LLM Text Generation API")

class Request(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 1.0
    top_p: float = 0.9

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
pipeline = InferencePipeline(model, tokenizer)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate-text")
def generate(req: Request):
    text = pipeline.generate(
        req.prompt,
        max_length=req.max_length,
        temperature=req.temperature,
        top_p=req.top_p
    )
    return {"generated_text": text}
