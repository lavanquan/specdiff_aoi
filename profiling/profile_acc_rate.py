# %%
import os
import time
import torch
import openai
import transformers
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
# transformers.logging.set_verbosity_info()
transformers.logging.set_verbosity_error()


system_prompt = """
Solve the following math problem efficiently and clearly. Please reason step by step, 
separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \\boxed{{}}.
Problem: {problem}
"""

dataset_name = "aime"

if dataset_name == "aime":
    dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]
elif dataset_name == "math":
    dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
elif dataset_name == "gpqa":
    if os.getenv("HF_HUB_OFFLINE", "0") == "1":
        dataset = load_from_disk("/scratch/gpfs/rp2773/hf_cache/datasets/gpqa")
    else:    
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
else:
    raise NotImplementedError
    
# %%
draft_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# target_model_name = "Qwen/Qwen2.5-32B-Instruct"
target_model_name = "Qwen/Qwen2.5-7B-Instruct"
dllm_name = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"

draft_model = AutoModelForCausalLM.from_pretrained(
    draft_model_name,
    torch_dtype="auto",
    device_map="auto"
)
draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
target_model = AutoModelForCausalLM.from_pretrained(
    target_model_name,
    torch_dtype="auto",
    device_map="auto"
)
target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
# NOTE(ruipan): maybe they should use the same tokenizer?
dllm = AutoModelForCausalLM.from_pretrained(
    dllm_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
dllm_tokenizer = AutoTokenizer.from_pretrained(dllm_name, trust_remote_code=True)


# %%
def get_target_token_ids(model, tokenizer, messages):
    """Get the target series of token IDs for the given messages.
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    num_input_tokens = model_inputs.input_ids.shape[1]
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        # use greedy decoding, not sampling
        do_sample=False,
        # temperature=1.0,
        # top_p=1.0,
        # top_k=0.0,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    return generated_ids[0].tolist(), model_inputs


def get_next_n_tokens(model, tokenizer, orig_model_inputs, token_ids_so_far, n):
    """Get the next n tokens from the model given the token IDs so far.
    """
    new_tokens = torch.tensor(token_ids_so_far, device=orig_model_inputs['input_ids'].device, dtype=torch.long).unsqueeze(0)
    new_mask = torch.ones_like(new_tokens, dtype=torch.long)  # attention mask = 1 for new tokens

    # Append along the sequence dimension (dim=1)
    new_model_inputs = {
        'input_ids': torch.cat([orig_model_inputs['input_ids'], new_tokens], dim=1),
        'attention_mask': torch.cat([orig_model_inputs['attention_mask'], new_mask], dim=1)
    }

    generated_ids = model.generate(
        **new_model_inputs,
        max_new_tokens=n,
        # use greedy decoding, not sampling
        do_sample=False,
        # temperature=1.0,
        # top_p=1.0,
        # top_k=0.0,
    )
    generated_ids = generated_ids[0][len(new_model_inputs["input_ids"][0]):]
    
    return generated_ids.tolist()




# %%
total_accepted_tokens = 0
total_rejected_tokens = 0

for problem_id in tqdm(range(30)):
    if dataset_name == "aime":
        problem = dataset["problem"][problem_id]
        options = None
    elif dataset_name == "math":
        problem = dataset["problem"][problem_id]
        options = None
    elif dataset_name == "gpqa":
        problem = dataset["Question"][problem_id]
        options = {
            "A": dataset["Correct Answer"][problem_id],
            "B": dataset["Incorrect Answer 1"][problem_id],
            "C": dataset["Incorrect Answer 2"][problem_id],
            "D": dataset["Incorrect Answer 3"][problem_id],
        }
    
    messages = [
        {"role": "user", "content": system_prompt.format(problem=problem)},
    ]
    
    target_ids, orig_model_inputs = get_target_token_ids(target_model, target_tokenizer, messages)
    
    print(f"len(target_ids) = {len(target_ids)}")

    n = 5  # number of speculative tokens proposed each time
    accepted_tokens = 0
    rejected_tokens = 0
    current_token_ids = []  # prefix tokens generated so far

    # Start speculative decoding loop
    while len(current_token_ids) < len(target_ids):
        # Get next n speculative tokens from draft model
        draft_proposal = get_next_n_tokens(draft_model, draft_tokenizer, orig_model_inputs, current_token_ids, n=n)

        # The corresponding slice of ground-truth target tokens
        target_slice = target_ids[len(current_token_ids): len(current_token_ids) + n]

        # Compare draft proposal with target tokens one by one
        for draft_tok, target_tok in zip(draft_proposal, target_slice):
            if draft_tok == target_tok:
                accepted_tokens += 1
                current_token_ids.append(draft_tok)
            else:
                rejected_tokens += 1
                # replace with correct target token, sync with target model
                current_token_ids.append(target_tok)
                # print(f"Rejection, current length: {len(current_token_ids)}, draft_tok {draft_tok}, target_tok {target_tok}")
                # print(f"draft token decoded: {draft_tokenizer.decode(draft_tok)}")
                # print(f"target token decoded: {target_tokenizer.decode(target_tok)}")
                break  # speculative generation diverged; go back to draft proposal step
            
                # FIXME(ruipan): math, question 1, len(target_ids) = 235, strange mismatch at len 148

        # If we’ve already matched the full target sequence, stop
        if len(current_token_ids) >= len(target_ids):
            break

    # Compute token acceptance rate
    acceptance_rate = accepted_tokens / (accepted_tokens + rejected_tokens)
    print(f"Acceptance rate: {acceptance_rate:.3f}")
    print(f"Accepted: {accepted_tokens}, Rejected: {rejected_tokens}, Total: {accepted_tokens + rejected_tokens}")
    total_accepted_tokens += accepted_tokens
    total_rejected_tokens += rejected_tokens

overall_acceptance_rate = total_accepted_tokens / (total_accepted_tokens + total_rejected_tokens)
print(f"Overall acceptance rate: {overall_acceptance_rate:.3f}")

# %%
