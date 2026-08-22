from datasets import load_dataset
from unsloth import PatchDPOTrainer
from trl import DPOTrainer, DPOConfig

# 1. Apply Unsloth Memory-Efficient DPO Patch
PatchDPOTrainer()

# 2. Load DPO Dataset
dataset = load_dataset("json", data_files="dpo_dataset.jsonl", split="train")

# 3. Configure DPO Hyperparameters
dpo_config = DPOConfig(
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    warmup_ratio = 0.1,
    max_steps = 30, # 30 steps for Colab test run
    learning_rate = 5e-6, # DPO requires smaller learning rate
    beta = 0.1, # DPO temperature preference penalty
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 1,
    optim = "adamw_8bit",
    output_dir = "dpo_outputs",
)

# 4. Initialize DPOTrainer
trainer = DPOTrainer(
    model = model,
    ref_model = None, # Managed in VRAM by Unsloth patch
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = dpo_config,
)

print("⚡ Starting DPO Training Loop...")
trainer.train()
print("🎉 DPO Preference Alignment Training Complete!")

# Save Aligned LoRA Model
model.save_pretrained("dpo_aligned_lora")
tokenizer.save_pretrained("dpo_aligned_lora")
print("💾 Saved DPO Aligned LoRA Model to 'dpo_aligned_lora'!")
