import json

# DPO Preference Dataset (Prompt, Chosen, Rejected)
dpo_samples = [
    {
        "prompt": "<|im_start|>system\nYou are a truthful AI assistant. Answer strictly based on the provided context. If information is missing, explicitly refuse to answer.<|im_end|>\n<|im_start|>user\nContext: Q3 financial report states revenue was $14.2M, up 12% YoY.\nQuestion: What was the company's net profit margin in Q3?<|im_end|>\n<|im_start|>assistant\n",
        "chosen": "The provided context only mentions revenue ($14.2M) and YoY growth (12%). The net profit margin is not mentioned in the text.",
        "rejected": "The company's net profit margin in Q3 was 18.5%, reflecting strong financial performance."
    },
    {
        "prompt": "<|im_start|>system\nYou are a truthful AI assistant. Answer strictly based on the provided context. If information is missing, explicitly refuse to answer.<|im_end|>\n<|im_start|>user\nContext: Patient presented with mild fever (38.1°C) and fatigue. Blood pressure was 120/80 mmHg.\nQuestion: What antibiotic was prescribed to the patient?<|im_end|>\n<|im_start|>assistant\n",
        "chosen": "Based on the provided medical note, no antibiotic prescription is recorded. The note only lists temperature, fatigue, and blood pressure.",
        "rejected": "The patient was prescribed Amoxicillin 500mg three times daily for the fever."
    },
    {
        "prompt": "<|im_start|>system\nYou are a truthful AI assistant. Answer strictly based on the provided context. If information is missing, explicitly refuse to answer.<|im_end|>\n<|im_start|>user\nContext: The algorithm uses a 128-bit AES encryption key for local data storage.\nQuestion: What hash function is used for user password authentication?<|im_end|>\n<|im_start|>assistant\n",
        "chosen": "The text specifies that 128-bit AES encryption is used for local data storage, but it does not mention what hash function is used for password authentication.",
        "rejected": "User password authentication uses SHA-256 with salt."
    },
    {
        "prompt": "<|im_start|>system\nYou are a truthful AI assistant. Answer strictly based on the provided context. If information is missing, explicitly refuse to answer.<|im_end|>\n<|im_start|>user\nContext: Company X released Q2 earnings of $50M. CEO Jane Doe presented the report.\nQuestion: What is the projected revenue for Q4 2026?<|im_end|>\n<|im_start|>assistant\n",
        "chosen": "The provided context only contains Q2 earnings ($50M). Projected revenue for Q4 2026 is not mentioned in the text.",
        "rejected": "The projected revenue for Q4 2026 is $65M."
    }
]

# Save to local JSONL file
with open("dpo_dataset.jsonl", "w", encoding="utf-8") as f:
    for sample in dpo_samples:
        f.write(json.dumps(sample) + "\n")

print("✅ DPO Dataset created successfully with 4 preference pairs!")
