# DPO-TruthfulLLM: Scientific Factuality Alignment via Direct Preference Optimization

Production-grade Direct Preference Optimization (DPO) pipeline using Unsloth and Hugging Face TRL to align open-weight language models (Llama 3, Qwen 2.5) for strict truthfulness, physical consistency, and out-of-context refusal in chemical/engineering applications.

## Key Features
- **Zero-Hallucination Objective:** Trains models to output explicit refusal tokens when queried on ungrounded chemical reactions or physical values outside known thermodynamic bounds.
- **QLoRA Parameter Efficiency:** Fine-tunes 4-bit quantized base models on custom paired preference datasets (grounded literature responses vs. plausibly hallucinated answers).
- **Evaluation Harness:** Benchmark evaluation on TruthfulQA and SciFact scientific claims.

## Quickstart
```bash
git clone https://github.com/dhrubchatterjee16-lang/DPO-TruthfulLLM.git
cd DPO-TruthfulLLM
pip install -r requirements.txt
python train_dpo.py --config config/qwen_dpo.yaml
```
