# %%
import os
import sys
import time
import torch
import openai
import argparse
import transformers
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
# transformers.logging.set_verbosity_info()
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
        shell = get_ipython().__class__.__name__
        return shell in ('ZMQInteractiveShell',)  # Jupyter/IPython
    except Exception:
        return False

def is_interactive():
    return is_notebook() or sys.stdout.isatty()
    
system_prompt = """
Solve the following math problem efficiently and clearly. Please reason step by step, 
separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \\boxed{{}}.
Problem: {problem}
"""

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


# %%
def get_target_token_ids(model, tokenizer, messages, target_len):
    """Get the target series of token IDs for the given messages.
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    num_input_tokens = model_inputs.input_ids.shape[1]
    print(f"num_input_tokens {num_input_tokens}, first eight tokens: {model_inputs.input_ids[0, :8].tolist()}")
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=target_len,  # was 512 in vanilla sd experiments
        # use greedy decoding, not sampling
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        top_k=0.0,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    return generated_ids[0].tolist(), model_inputs


def get_next_n_tokens_ar(model, orig_model_inputs, token_ids_so_far, n):
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
        temperature=0.0,
        top_p=1.0,
        top_k=0.0,
    )
    generated_ids = generated_ids[0][len(new_model_inputs["input_ids"][0]):]
    
    return generated_ids.tolist()



def get_next_n_tokens_dllm(dllm, orig_model_inputs, token_ids_so_far, n, output_seqlen=512, small_block_size=8, threshold=0.9):
    """Get the next n tokens from the model given the token IDs so far.
    """
    num_tokens_in_prompt = orig_model_inputs.input_ids.shape[1]
    new_tokens = torch.tensor(token_ids_so_far, device=orig_model_inputs['input_ids'].device, dtype=torch.long).unsqueeze(0)
    new_mask = torch.ones_like(new_tokens, dtype=torch.long)  # attention mask = 1 for new tokens

    # Append along the sequence dimension (dim=1)
    new_model_inputs = {
        'input_ids': torch.cat([orig_model_inputs['input_ids'], new_tokens], dim=1),
        'attention_mask': torch.cat([orig_model_inputs['attention_mask'], new_mask], dim=1)
    }

    generated_ids = dllm.generate(
        # **new_model_inputs,
        new_model_inputs["input_ids"],
        # max_new_tokens=output_seqlen,
        max_new_tokens=32,  # NOTE(ruipan): setting this to 8 will not lead to new tokens hmm
        small_block_size=small_block_size,
        threshold=threshold,
        # use greedy decoding, not sampling
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        top_k=0.0,
    )
    
    full_output_seqlen = generated_ids.shape[1]
    assert full_output_seqlen > num_tokens_in_prompt + len(token_ids_so_far), f"full_output_seqlen {full_output_seqlen}, num_tokens_in_prompt {num_tokens_in_prompt}, len(token_ids_so_far) {len(token_ids_so_far)}"
    generated_ids = generated_ids[0][len(new_model_inputs["input_ids"][0]):]
    generated_ids = generated_ids.tolist()[:n]  # only take the next n tokens
    
    return generated_ids


# %%
parser = argparse.ArgumentParser(description="Profiles the acceptance rate of speculative decoding within a single query.")
parser.add_argument("--dataset_name", type=str, choices=["aime", "math", "gpqa"], default="math",
                    help="Dataset")
parser.add_argument("--model_type", type=str, choices=["ar", "dllm"], default="ar",
                    help="Model type")
parser.add_argument("--num_questions", type=int, default=1,
                    help="Number of questions to run profiling on")
parser.add_argument("--target_len", type=int, default=512,
                    help="Target model generation length")
args, _ = parser.parse_known_args()


# %%
if args.dataset_name == "aime":
    args.dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]
elif args.dataset_name == "math":
    args.dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
elif args.dataset_name == "gpqa":
    if os.getenv("HF_HUB_OFFLINE", "0") == "1":
        args.dataset = load_from_disk("/scratch/gpfs/rp2773/hf_cache/datasets/gpqa")
    else:    
        args.dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
else:
    raise NotImplementedError

total_accepted_tokens = 0
total_drafted_tokens = 0
for problem_id in range(args.num_questions):
    if args.dataset_name == "aime":
        problem = args.dataset["problem"][problem_id]
        options = None
    elif args.dataset_name == "math":
        problem = args.dataset["problem"][problem_id]
        options = None
    elif args.dataset_name == "gpqa":
        problem = args.dataset["Question"][problem_id]
        options = {
            "A": args.dataset["Correct Answer"][problem_id],
            "B": args.dataset["Incorrect Answer 1"][problem_id],
            "C": args.dataset["Incorrect Answer 2"][problem_id],
            "D": args.dataset["Incorrect Answer 3"][problem_id],
        }

    messages = [
        {"role": "user", "content": system_prompt.format(problem=problem)},
    ]

    target_ids, orig_model_inputs = get_target_token_ids(target_model, target_tokenizer, messages, target_len=args.target_len)
    print(f"Target (vanilla) generation length: {len(target_ids)} tokens")
    # print(f"Target token IDs: {target_ids}")

    current_token_ids = []  # prefix tokens accepted so far
    accepted_tokens = 0
    rejected_tokens = 0
    n= 5  # speculative length proposed by draft model each round
    num_speculation_rounds = 0

    if is_interactive():
        inner_bar = tqdm(total=args.target_len, miniters=25,
                         desc=f"Generation (Problem {problem_id})",
                         position=1, leave=True, dynamic_ncols=False, file=sys.stdout)


    # Main speculative loop: propose n tokens from draft, verify them live with target model
    while len(current_token_ids) < args.target_len:
        num_speculation_rounds += 1

        # A. PROPOSE: Get next n speculative tokens from draft model based on current accepted prefix
        if args.model_type == "ar":
            draft_proposal = get_next_n_tokens_ar(draft_model, orig_model_inputs, current_token_ids, n=n)
        elif args.model_type == "dllm":
            draft_proposal = get_next_n_tokens_dllm(dllm, orig_model_inputs, current_token_ids, n=n)
        else:
            raise NotImplementedError
        
        if not draft_proposal: # Stop if the draft model has nothing to say
            print(f"{Colors.RED}Warning: Draft model returned no tokens{Colors.RESET}")
            break

        # print(f"\nSpeculation round {num_speculation_rounds}, current length: {len(current_token_ids)}")
        # print(f"draft_proposal {draft_proposal}")
        # decoded_proposal = target_tokenizer.decode(draft_proposal, skip_special_tokens=True)
        # print(f"draft_proposal (decoded): {decoded_proposal}")

        # B. Verify proposed tokens
        prefix_len = len(current_token_ids)
        combined_ids = current_token_ids + draft_proposal
        
        verify_input_tensor = torch.tensor([combined_ids], device=target_model.device)
        full_input_ids = torch.cat([orig_model_inputs['input_ids'], verify_input_tensor], dim=1)

        with torch.no_grad():
            outputs = target_model(input_ids=full_input_ids)
            
            # The logits for the draft tokens start after the prompt and the accepted prefix.
            # The logit at sequence position 't' is the prediction for the token at 't+1'.
            # So we need the logits from the token *before* the first draft token up to the one *before* the last.
            start_index = orig_model_inputs['input_ids'].shape[1] + prefix_len - 1
            end_index = start_index + len(draft_proposal)
            verify_logits = outputs.logits[0, start_index:end_index]
            
            
        # C. ACCEPT/REJECT
        accepted_len = 0
        for i in range(len(draft_proposal)):
            # The target's prediction for position `i` is the argmax of the logits at position `i`
            target_pred = torch.argmax(verify_logits[i, :], dim=-1).item()
            # print(f"  Verifying draft token at position {i}: {draft_proposal[i]}. Target would have picked {target_pred}.")

            if draft_proposal[i] == target_pred:
                accepted_len += 1
            else:
                # Mismatch found. The correct token is the target's prediction.
                final_token = target_pred
                break
        else:
            # All draft tokens were accepted. Get a "bonus" token from the final logit.
            final_token_logits = outputs.logits[0, -1, :]
            final_token = torch.argmax(final_token_logits, dim=-1).item()
            # print(f"  All draft tokens accepted! Bonus token is {final_token}.")
            
        # D. UPDATE
        tokens_to_append = draft_proposal[:accepted_len] + [final_token]
        current_token_ids.extend(tokens_to_append)
        
        accepted_tokens += accepted_len
        rejected_tokens += len(draft_proposal) - accepted_len
        
        if is_interactive():
            inner_bar.update(len(tokens_to_append))

        if target_tokenizer.eos_token_id in tokens_to_append:
            break
        
    if is_interactive():
        inner_bar.close()

    print(f"\n{Colors.BOLD}--- [Problem {problem_id}] Verification ---{Colors.RESET}")
    final_spec_ids = current_token_ids[:len(target_ids)]
    # print(f"Target (vanilla) IDs: {target_ids}")
    # print(f"Final speculative IDs:  {final_spec_ids}")
    if final_spec_ids != target_ids:
        print(f"{Colors.RED}[Problem {problem_id}] Warning: Mismatch between speculative and vanilla decoding!{Colors.RESET}")
    else:
        print(f"{Colors.GREEN}[Problem {problem_id}] Success: Speculative decoding output exactly matches vanilla decoding output.{Colors.RESET}")

    # Compute token acceptance rate for this problem
    drafted_tokens = num_speculation_rounds * n
    acceptance_rate = accepted_tokens / drafted_tokens if drafted_tokens > 0 else 0.0

    print(f"\n{Colors.BOLD}--- [Problem {problem_id}] Statistics ---{Colors.RESET}")
    print(f"{Colors.YELLOW}[Problem {problem_id}] Acceptance rate: {acceptance_rate:.3f}{Colors.RESET}")
    print(f"[Problem {problem_id}] Accepted: {accepted_tokens}, Rejected: {rejected_tokens}, Total drafted: {drafted_tokens}")
    # print(f"[Problem {problem_id}] Number of speculation rounds: {num_speculation_rounds}")
    print(f"[Problem {problem_id}] Final output: {len(current_token_ids)} tokens")

    total_accepted_tokens += accepted_tokens
    total_drafted_tokens += drafted_tokens
    
    # print(f"target_ids: {target_ids}")
    # print(f"target_ids decoded: {target_tokenizer.decode(target_ids)}")
    # print(f"current_token_ids: {current_token_ids}")
    # print(f"current_token_ids decoded: {target_tokenizer.decode(current_token_ids)}")

# Overall acceptance rate across problems
overall_acceptance_rate = total_accepted_tokens / total_drafted_tokens
print(f"{Colors.CYAN}Overall acceptance rate: {overall_acceptance_rate:.3f}{Colors.RESET}")

# %%
