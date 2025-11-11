# %%
import re
import sys
from collections import defaultdict

# Patterns
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")  # strip ANSI escapes
RUNNING_RE = re.compile(r"Running drafter:\s*(\S+)")
PROB_DRAFTER_BRACKET_RE = re.compile(r"\[Problem\s+(\d+),\s*([^\]]+)\]")
PROB_ONLY_RE = re.compile(r"\[Problem\s+(\d+)\]")
ACCEPT_RE = re.compile(r"Acceptance rate:\s*([\d.]+)%")
FWD_RE = re.compile(r"Avg fwd passes/round:\s*([\d.]+)")
SPEED_RE = re.compile(r"Speedup:\s*([\d.]+)x")

# Desired column order
ORDER = ["ar_None", "dllm_0.9", "dllm_0.7", "dllm_0.5", "dllm_0.3", "dllm_0.1", "dllm_0.01"]

def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)

def parse_log(filename):
    data = defaultdict(dict)  # problem_id -> drafter -> [accept, speed, fwd]
    cur_prob = None
    cur_drafter = None

    with open(filename, "r", encoding="utf-8") as f:
        for raw in f:
            line = strip_ansi(raw.rstrip("\n"))

            # If this is a "Running drafter" line, update context
            # Example: [2025-11-07 01:13:39] [INFO] === [Problem 0] Running drafter: dllm_0.9 ===
            m_run = RUNNING_RE.search(line)
            if m_run:
                # try to capture problem id on same line
                m_prob = PROB_ONLY_RE.search(line)
                if m_prob:
                    cur_prob = int(m_prob.group(1))
                cur_drafter = m_run.group(1)
                # Normalize common names: if it's "ar_None" or "ar", keep ar_None
                # (we don't change here; the ORDER list determines expected names)
                continue

            # Try to parse lines that include [Problem X, drafter]
            m_pd = PROB_DRAFTER_BRACKET_RE.search(line)
            if m_pd:
                prob = int(m_pd.group(1))
                drafter = m_pd.group(2)
                cur_prob = prob
                cur_drafter = drafter

            # If no explicit [Problem X, drafter], but we have context from Running drafter, use that
            if cur_prob is None or cur_drafter is None:
                # nothing to attach stats to
                continue

            # Extract stats if they appear in this line
            a = ACCEPT_RE.search(line)
            s = SPEED_RE.search(line)
            f = FWD_RE.search(line)

            if a:
                val = float(a.group(1))
                data[cur_prob].setdefault(cur_drafter, [None, None, None])[0] = val
            if s:
                val = float(s.group(1))
                data[cur_prob].setdefault(cur_drafter, [None, None, None])[1] = val
            if f:
                val = float(f.group(1))
                data[cur_prob].setdefault(cur_drafter, [None, None, None])[2] = val

    return data

def print_table(data):
    headers = ["Problem"] + ORDER
    print("\t".join(headers))

    # Collect numeric sums/counts for averages
    sums = {drafter: [0.0, 0.0, 0.0] for drafter in ORDER}
    counts = {drafter: [0, 0, 0] for drafter in ORDER}

    for pid in sorted(data.keys()):
        row = [str(pid)]
        for drafter in ORDER:
            if drafter in data[pid]:
                acc, spd, fwd = data[pid][drafter]

                # Track averages (only count valid values)
                if acc is not None:
                    sums[drafter][0] += acc
                    counts[drafter][0] += 1
                if spd is not None:
                    sums[drafter][1] += spd
                    counts[drafter][1] += 1
                if fwd is not None:
                    sums[drafter][2] += fwd
                    counts[drafter][2] += 1

                acc_s = f"{acc:.1f}%" if acc is not None else "NA"
                spd_s = f"{spd:.2f}x" if spd is not None else "NA"
                fwd_s = f"{fwd:.2f}" if fwd is not None else "NA"
                row.append(f"({acc_s}, {spd_s}, {fwd_s})")
            else:
                row.append("-")
        print("\t".join(row))

    # Compute and print average row
    avg_row = ["Average"]
    for drafter in ORDER:
        a_sum, s_sum, f_sum = sums[drafter]
        a_cnt, s_cnt, f_cnt = counts[drafter]
        acc_avg = a_sum / a_cnt if a_cnt else None
        spd_avg = s_sum / s_cnt if s_cnt else None
        fwd_avg = f_sum / f_cnt if f_cnt else None

        acc_s = f"{acc_avg:.1f}%" if acc_avg is not None else "NA"
        spd_s = f"{spd_avg:.2f}x" if spd_avg is not None else "NA"
        fwd_s = f"{fwd_avg:.2f}" if fwd_avg is not None else "NA"
        avg_row.append(f"({acc_s}, {spd_s}, {fwd_s})")

    print("\t".join(avg_row))


# %%
# filename = "/data2/ruipan/diffspec/logs/2025_11_07_01_12_aime.ansi"
# filename = "/data2/ruipan/diffspec/logs/2025_11_07_14_19_aime.ansi"
filename = "/data2/ruipan/diffspec/logs/2025_11_10_22_38_aime.ansi"
data = parse_log(filename)
print_table(data)

# %%
