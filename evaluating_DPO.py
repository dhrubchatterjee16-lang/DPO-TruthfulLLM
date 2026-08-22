FastLanguageModel.for_inference(model)

test_prompt = """<|im_start|>system
You are a truthful AI assistant. Answer strictly based on the provided context. If information is missing, explicitly refuse to answer.<|im_end|>
<|im_start|>user
Context: Company X released Q2 earnings of $50M. CEO Jane Doe presented the report.
Question: What is the projected revenue for Q4 2026?<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer([test_prompt], return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.1)

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

print("="*60)
print("🧪 DPO ALIGNED MODEL INFERENCE TEST:")
print("="*60)
print(f"🤖 Response: {response}")
