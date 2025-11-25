# %%
import re
from collections import defaultdict

# --- Regex patterns (unchanged) ---
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
RUNNING_RE = re.compile(r"Running drafter:\s*(\S+)")
PROB_DRAFTER_BRACKET_RE = re.compile(r"\[Problem\s+(\d+),\s*([^\]]+)\]")
PROB_ONLY_RE = re.compile(r"\[Problem\s+(\d+)\]")
ACCEPT_RE = re.compile(r"Acceptance rate:\s*([\d.]+)%")
FWD_RE = re.compile(r"Avg fwd passes/round:\s*([\d.]+)")
SPEED_RE = re.compile(r"Speedup:\s*([\d.]+)x")

def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)

def parse_log(filename):
    """Parse the log into {problem_id: {drafter: [accept, speed, fwd]}}"""
    data = defaultdict(dict)
    cur_prob = None
    cur_drafter = None

    with open(filename, "r", encoding="utf-8") as f:
        for raw in f:
            line = strip_ansi(raw.rstrip("\n"))

            m_run = RUNNING_RE.search(line)
            if m_run:
                m_prob = PROB_ONLY_RE.search(line)
                if m_prob:
                    cur_prob = int(m_prob.group(1))
                cur_drafter = m_run.group(1)
                continue

            m_pd = PROB_DRAFTER_BRACKET_RE.search(line)
            if m_pd:
                cur_prob = int(m_pd.group(1))
                cur_drafter = m_pd.group(2)

            if cur_prob is None or cur_drafter is None:
                continue

            a = ACCEPT_RE.search(line)
            s = SPEED_RE.search(line)
            f = FWD_RE.search(line)

            if a:
                data[cur_prob].setdefault(cur_drafter, [None, None, None])[0] = float(a.group(1))
            if s:
                data[cur_prob].setdefault(cur_drafter, [None, None, None])[1] = float(s.group(1))
            if f:
                data[cur_prob].setdefault(cur_drafter, [None, None, None])[2] = float(f.group(1))
    return data


# ----------------------------------------------------------------------
# NEW: extract per-problem stats and compute cross-problem averages
# ----------------------------------------------------------------------

AR_NAME = "ar_None_sf_None_None"
SF_NAME = "dllm_0.05_sf_None_None"

def analyze(data):
    """Print per-problem summaries and compute averages."""

    # Accumulate averages for each config
    sums = defaultdict(float)
    counts = defaultdict(int)

    # For oracle metric: best config per problem
    oracle_best_spds = []
    oracle_best_cfgs = {}

    print("=== Per-Problem Results ===")
    for pid in sorted(data.keys()):
        drafter_data = data[pid]

        ar_spd = drafter_data.get(AR_NAME, [None, None, None])[1]
        sf_spd = drafter_data.get(SF_NAME, [None, None, None])[1]

        if ar_spd is None or sf_spd is None:
            print(f"[Problem {pid}] Missing AR or SF config — skipping.")
            continue

        # record AR + SF sums
        sums[AR_NAME] += ar_spd
        counts[AR_NAME] += 1

        sums[SF_NAME] += sf_spd
        counts[SF_NAME] += 1

        # best OTHER config (exclude AR and SF)
        best_other_name = None
        best_other_spd = -1

        # best OVERALL config (oracle)
        best_overall_name = None
        best_overall_spd = -1

        for name, (_, spd, _) in drafter_data.items():
            if spd is None:
                continue

            # global accumulation
            sums[name] += spd
            counts[name] += 1

            # best *other* config (exclude AR + SF)
            if name not in (AR_NAME, SF_NAME):
                if spd > best_other_spd:
                    best_other_spd = spd
                    best_other_name = name

            # best overall config (oracle)
            if spd > best_overall_spd:
                best_overall_spd = spd
                best_overall_name = name

        # record oracle for averaging later
        oracle_best_spds.append(best_overall_spd)
        oracle_best_cfgs[pid] = (best_overall_name, best_overall_spd)

        diff = best_other_spd - sf_spd

        print(f"[Problem {pid}] AR speedup = {ar_spd:.3f}x")
        print(f"[Problem {pid}] SF (dllm_0.05_sf_None_None) speedup = {sf_spd:.3f}x")
        print(f"[Problem {pid}] Best OTHER config = {best_other_name} ({best_other_spd:.3f}x), "
              f"win over SF = {diff:.3f}x (absolute difference); relative win = {100 * (best_other_spd / sf_spd - 1):.2f}%")
        print(f"[Problem {pid}] ORACLE best config = {best_overall_name} ({best_overall_spd:.3f}x)\n")

    # ------------------------------------------------------------------
    # Global averages
    # ------------------------------------------------------------------
    avg = {k: (sums[k] / counts[k]) for k in sums.keys() if counts[k] > 0}

    # Best config overall (single config best average)
    best_global_cfg = max(avg.items(), key=lambda x: x[1])

    # Oracle average (best config per problem, varying)
    oracle_avg = sum(oracle_best_spds) / len(oracle_best_spds) if oracle_best_spds else 0.0

    print("=== Global Averages (Across All Problems) ===")
    print(f"AR average speedup: {avg[AR_NAME]:.3f}x")
    print(f"SF (dllm_0.05_sf_None_None) average speedup: {avg[SF_NAME]:.3f}x")
    print(f"Best single global config: {best_global_cfg[0]} with {best_global_cfg[1]:.3f}x")
    print(f"ORACLE average (best config per problem): {oracle_avg:.3f}x\n")

    return avg



# %%
# Example usage:
# data = parse_log("/scratch/gpfs/RAVIAN/rp2773/data/diffspec/logs/2025_11_24_22_22_math.ansi")
# data = parse_log("/scratch/gpfs/RAVIAN/rp2773/data/diffspec/logs/2025_11_24_22_06_aime.ansi")

# data = parse_log("/scratch/gpfs/RAVIAN/rp2773/data/diffspec/logs/2025_11_24_23_39_math.ansi")
data = parse_log("/scratch/gpfs/RAVIAN/rp2773/data/diffspec/logs/2025_11_24_23_40_aime.ansi")




analyze(data)
