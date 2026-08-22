import torch
from unsloth import FastLanguageModel

max_seq_length = 4096
load_in_4bit = True

print("⌛ Loading Qwen 2.5 7B in 4-bit mode...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit,
)

# Configure LoRA Adapters for DPO Training
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)
print("✅ Model and LoRA Adapters ready!")
