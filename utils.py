# %%
import torch

def calculate_spec_decoding_speedup(alpha, gamma, c):
    """Calculate the speculative decoding speedup.

    Reference: Theorem 3.8 in https://arxiv.org/pdf/2211.17192
    
    Args:
        alpha (float): Avg per-token acceptance rate, between 0 and 1.
        gamma (int): The number of drafted tokens.
        c (float): The drafter-to-target per-token latency ratio.
    """
    numerator = 1 - alpha ** (gamma + 1)
    denominator = (1 - alpha) * (c * gamma + 1)
    speedup = numerator / denominator
    return speedup

def check_prefill_output_equivalence(output1, output2, round_idx):
    if not torch.equal(output1.logits, output2.logits):
        print(f"[Round {round_idx}] Logits are not equal!")
    output1_kvs = output1.past_key_values.to_legacy_cache()
    output2_kvs = output2.past_key_values.to_legacy_cache()
    
    for layer_idx, (layer_kvs1, layer_kvs2) in enumerate(zip(output1_kvs, output2_kvs)):
        for kv_idx, (kv1, kv2) in enumerate(zip(layer_kvs1, layer_kvs2)):
            if not torch.equal(kv1, kv2):
                print(f"[Round {round_idx}] Past key values are not equal at layer {layer_idx}, kv {kv_idx}!")

def check_prefill_output_list_equivalence(output1, output2):
    for round_idx in range(min(len(output1), len(output2))):
        o1 = output1[round_idx]
        o2 = output2[round_idx]
        check_prefill_output_equivalence(o1, o2, round_idx)
# %%
