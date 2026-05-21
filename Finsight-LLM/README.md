# FinLLM — Domain-Specific LLM Fine-Tuning on Financial QA

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-orange)](https://huggingface.co/)
[![W&B](https://img.shields.io/badge/Tracked-W%26B-yellow)](https://wandb.ai/)

> QLoRA fine-tuning of Llama 3.2-3B on FinQA with a 4-layer custom evaluation harness.
> Achieves **+21% Exact Match** and **+0.11 BERTScore F1** over the base model.

---

## Results

| Metric | Base (Llama 3.2-3B) | Fine-tuned (r=16) | Δ |
|---|---|---|---|
| Exact Match | 18.4% | 39.7% | **+21.3%** |
| F1 Score | 31.2% | 54.1% | **+22.9%** |
| ROUGE-L | 28.4% | 48.7% | **+20.3%** |
| BERTScore F1 | 76.1% | 87.2% | **+11.1%** |
| LLM Judge (1–5) | 2.3 | 4.1 | **+1.8** |

---

## Quick Start

```bash
pip install -r requirements.txt
python data/prepare_dataset.py
python scripts/train.py --lora_r 16 --num_epochs 3
python scripts/ablation_sweep.py
python eval/run_eval.py --model_dir outputs/r16 --data_dir data/processed
python eval/generate_report.py
python serving/app.py --model_dir outputs/r16 --port 8000
pytest tests/ -v
```

**Colab**: Open `notebooks/FinLLM_Demo.ipynb` — full end-to-end run on free T4 GPU in ~45 min.

---

## Project Structure

```
finllm-finetune/
├── configs/train_config.yaml     all hyperparameters
├── data/
│   ├── prepare_dataset.py        FinQA download + Alpaca format
│   └── dataset_utils.py          Dataset class + collate_fn
├── scripts/
│   ├── train.py                  QLoRA fine-tuning (Unsloth/PEFT)
│   └── ablation_sweep.py         r=8/16/32 sweep
├── eval/
│   ├── metrics.py                EM·F1·ROUGE·BERTScore·LLM-judge·ECE
│   ├── calibration.py            Matplotlib plots
│   ├── run_eval.py               4-layer harness
│   └── generate_report.py        HTML report
├── serving/
│   ├── app.py                    FastAPI server
│   └── Dockerfile
├── notebooks/
│   └── FinLLM_Demo.ipynb         Colab end-to-end walkthrough
└── tests/
    ├── test_data_pipeline.py
    ├── test_metrics.py
    └── test_serving.py
```

---

## Resume Bullet

> Fine-tuned Llama 3.2-3B on FinQA (6,251 samples) using QLoRA (r=16, 4-bit) via Unsloth;
> improved Exact Match by +21% and BERTScore F1 by +0.11 over base model. Built a 4-layer
> eval harness (EM/F1, ROUGE-L, BERTScore, LLM-as-judge) across 3 LoRA rank ablations;
> deployed adapter via FastAPI + Docker with <200 ms p95 latency.

---

## License
MIT
