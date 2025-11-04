# %%
import os
import sys
import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk

# Set seeds and logging for reproducibility
transformers.logging.set_verbosity_error()
transformers.set_seed(42)

class Colors:
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def is_notebook():
    """Detect if running inside Jupyter or IPython kernel."""
    try:
        from IPython import get_ipython
        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except Exception:
        return False

def is_interactive():
    return is_notebook() or sys.stdout.isatty()

system_prompt = """
Solve the following math problem efficiently and clearly. Please reason step by step, 
separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \\boxed{{}}.
Problem: {problem}
"""

dataset_name = "math"

if dataset_name == "aime":
    dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]
elif dataset_name == "math":
    dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
elif dataset_name == "gpqa":
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
else:
    raise NotImplementedError

# %%
draft_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
target_model_name = "Qwen/Qwen2.5-32B-Instruct"
# target_model_name = "Qwen/Qwen2.5-7B-Instruct"
dllm_name = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"

target_model = AutoModelForCausalLM.from_pretrained(
    target_model_name,
    torch_dtype="auto",
    device_map="auto"
)
target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
draft_model = AutoModelForCausalLM.from_pretrained(
    draft_model_name,
    torch_dtype="auto",
    device_map="auto"
)
# draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
draft_tokenizer = target_tokenizer  # assume they use the same tokenizer
# NOTE(ruipan): maybe they should use the same tokenizer?
dllm = AutoModelForCausalLM.from_pretrained(
    dllm_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
dllm_tokenizer = AutoTokenizer.from_pretrained(dllm_name, trust_remote_code=True)
# --- Helper Functions ---

# %%
def get_full_target_generation(model, tokenizer, messages):
    """Generates the ground-truth token sequence using the target model's standard .generate() method."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    print(f"Number of prompt tokens: {model_inputs.input_ids.shape[1]}")
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=16,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        top_k=0.0,
    )
    return generated_ids[0][model_inputs.input_ids.shape[1]:].tolist(), model_inputs

def get_speculative_tokens(model, tokenizer, orig_inputs, prefix_ids, n):
    """Generates n speculative tokens from the draft model given the current context."""
    if not prefix_ids:
        # Use the original prompt if the prefix is empty
        input_ids = orig_inputs['input_ids']
    else:
        prefix_tensor = torch.tensor([prefix_ids], device=model.device, dtype=torch.long)
        input_ids = torch.cat([orig_inputs['input_ids'], prefix_tensor], dim=1)

    generated_ids = model.generate(
        input_ids,
        max_new_tokens=n,
        do_sample=False,
        # pad_token_id=tokenizer.eos_token_id,
        temperature=0.0,
        top_p=1.0,
        top_k=0.0,
    )
    return generated_ids[0][input_ids.shape[1]:].tolist()

def get_target_next_token(model, tokenizer, orig_inputs, prefix_ids):
    """
    Gets the single next token from the target model given the full prefix.
    This is inefficient but guarantees correctness by avoiding state management.
    """
    if not prefix_ids:
        input_ids = orig_inputs['input_ids']
    else:
        prefix_tensor = torch.tensor([prefix_ids], device=model.device, dtype=torch.long)
        input_ids = torch.cat([orig_inputs['input_ids'], prefix_tensor], dim=1)
        
    generated_ids = model.generate(
        input_ids,
        max_new_tokens=1,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        top_k=0.0,
    )
    return generated_ids[0][-1].item()

# --- Main Logic ---
for problem_id in range(1):
    problem = dataset["problem"][problem_id]
    messages = [{"role": "user", "content": system_prompt.format(problem=problem)}]

    # 1. Generate the ground-truth answer for verification
    target_ids, orig_model_inputs = get_full_target_generation(target_model, target_tokenizer, messages)
    print(f"Target (vanilla) generation length: {len(target_ids)} tokens")

    # 2. Begin the speculative decoding loop
    current_token_ids = []
    accepted_tokens = 0
    rejected_tokens = 0
    num_speculation_rounds = 0
    n = 5 # Number of tokens to speculate each round

    max_len = len(target_ids)
    if is_interactive():
        progress_bar = tqdm(total=max_len, desc=f"Speculative Generation (Problem {problem_id})")

    while len(current_token_ids) < max_len:
        num_speculation_rounds += 1

        # A. PROPOSE: Get n speculative tokens from the draft model
        draft_proposal = get_speculative_tokens(draft_model, target_tokenizer, orig_model_inputs, current_token_ids, n)
        print(f"\nSpeculation round {num_speculation_rounds}, draft_proposal: {draft_proposal}")

        if not draft_proposal:
            print(f"{Colors.RED}Warning: draft model failed to propose any tokens{Colors.RESET}")
            # If draft model fails to propose, advance with one token from the target model
            next_token = get_target_next_token(target_model, target_tokenizer, orig_model_inputs, current_token_ids)
            current_token_ids.append(next_token)
            if is_interactive(): progress_bar.update(1)
        else:
            # B. VERIFY: Check the proposal token by token against the target model
            accepted_in_round = 0
            for i, draft_tok in enumerate(draft_proposal):
                # Get the ground-truth next token based on the confirmed prefix
                target_next_tok = get_target_next_token(target_model, target_tokenizer, orig_model_inputs, current_token_ids)
                print(f"Speculation round {num_speculation_rounds}, draft token {draft_tok}, target token {target_next_tok}")

                if draft_tok == target_next_tok:
                    # Correct guess: accept the token and continue verifying
                    current_token_ids.append(draft_tok)
                    accepted_in_round += 1
                else:
                    # Incorrect guess: accept the target's token and end this round
                    current_token_ids.append(target_next_tok)
                    rejected_tokens += len(draft_proposal) - i
                    break # Stop verifying this proposal
            
            accepted_tokens += accepted_in_round
            # Update progress bar by the number of tokens we actually added in this round
            if is_interactive(): progress_bar.update(accepted_in_round + (1 if accepted_in_round < len(draft_proposal) else 0))

        # Check for early exit if an EOS token was generated
        if target_tokenizer.eos_token_id in current_token_ids:
            break

    if is_interactive():
        progress_bar.close()

    # --- Verification and Stats ---
    # Trim the speculative output to the length of the target for a fair comparison
    final_spec_ids = current_token_ids[:len(target_ids)]

    print(f"\n{Colors.BOLD}--- Verification ---{Colors.RESET}")
    print(f"Final speculative output length: {len(final_spec_ids)}")
    if final_spec_ids == target_ids:
        print(f"{Colors.GREEN}Success: Speculative decoding output matches vanilla decoding output.{Colors.RESET}")
    else:
        print(f"{Colors.RED}Warning: Mismatch between output of speculative and vanilla decoding!{Colors.RESET}")

    denom = (accepted_tokens + rejected_tokens)
    acceptance_rate = accepted_tokens / denom if denom > 0 else 0.0

    print(f"\n{Colors.BOLD}--- Statistics ---{Colors.RESET}")
    print(f"{Colors.YELLOW}Problem {problem_id} draft token acceptance rate: {acceptance_rate:.3f}{Colors.RESET}")
    print(f"Accepted: {accepted_tokens}, Rejected: {rejected_tokens}, Total Drafted: {denom}")
    print(f"Number of speculation rounds: {num_speculation_rounds}")

# %%
print(target_ids)
print(target_tokenizer.decode(target_ids))
# %%
print(current_token_ids)
print(target_tokenizer.decode(current_token_ids))
# %%
