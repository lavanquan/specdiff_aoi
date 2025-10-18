# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Efficient-Large-Model/Fast_dLLM_1.5B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# %%
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
inputs = tokenizer([text], return_tensors="pt").to(model.device)  # 29 tokens

# Fast-dLLM v2 parallel decoding
gen_ids = model.generate(
    inputs["input_ids"],
    tokenizer=tokenizer,
    max_new_tokens=512,
    small_block_size=8,
    threshold=0.9,
)

print(f"gen_ids {gen_ids}")  # 137 tokens

response = tokenizer.decode(
    gen_ids[0][inputs["input_ids"].shape[1]:], 
    skip_special_tokens=True
)
print(response)

# %%
