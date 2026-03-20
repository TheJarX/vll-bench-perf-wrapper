import json, glob, os, sys, re
from collections import defaultdict

results_dir = sys.argv[1] if len(sys.argv) > 1 else "./performance_results"
output_file = sys.argv[2] if len(sys.argv) > 2 else "benchmark_results.txt"

# Redirect stdout to the output file
results = []
for f in sorted(glob.glob(f"./{results_dir}/*_c*_run*-*.json")):
    mtime = os.path.getmtime(f)
    try:
        with open(f) as fh:
            data = json.load(fh)
        if data.get("completed", 0) > 0:
            results.append((mtime, f, data))
    except Exception:
        pass

if not results:
    print("No results found.")
    sys.exit(0)

results.sort(key=lambda x: x[0], reverse=True)

groups = defaultdict(list)
for _, fpath, data in results:
    label = data.get("label", "")
    m = re.match(r"^(.+)_c(\d+)_run\d+$", label)
    if not m:
        continue
    model_name = m.group(1)
    concurrency = int(m.group(2))
    groups[(model_name, concurrency)].append(data)

def stats(values):
    n = len(values)
    avg = sum(values) / n
    if n > 1:
        std = (sum((v - avg) ** 2 for v in values) / (n - 1)) ** 0.5
        cv = (std / avg * 100) if avg != 0 else 0
    else:
        std, cv = 0.0, 0.0
    return avg, std, cv

METRICS = [
    ("output_throughput",  "Throughput (tok/s)"),
    ("mean_ttft_ms",       "Mean TTFT (ms)"),
    ("median_ttft_ms",     "Median TTFT (ms)"),
    ("p99_ttft_ms",        "P99 TTFT (ms)"),
    ("mean_tpot_ms",       "Mean TPOT (ms)"),
    ("mean_itl_ms",        "Mean ITL (ms)"),
]

sorted_keys = sorted(groups.keys(), key=lambda k: (k[0], k[1]))

sys.stdout = open(output_file, 'w')
print("\n" + "=" * 100)
print("BENCHMARK SUMMARY -- Per Model, Per Concurrency (averaged across runs)")
print("=" * 100)

current_model = None
for (model_name, concurrency) in sorted_keys:
    runs_data = groups[(model_name, concurrency)]
    if model_name != current_model:
        if current_model is not None:
            print()
        current_model = model_name
        print(f"\n{'=' * 100}")
        print(f"  Model: {model_name}")
        print(f"{'=' * 100}")
        print(f"  {'Concurrency':<14} {'Metric':<22} {'Avg':>10} {'Std':>10} {'CV%':>8}   {'Individual runs'}")
        print(f"  {'-'*14} {'-'*22} {'-'*10} {'-'*10} {'-'*8}   {'-'*30}")

    first_metric = True
    for key, label in METRICS:
        values = [r[key] for r in runs_data if key in r]
        if not values:
            continue
        avg, std, cv = stats(values)
        vals_str = ", ".join(f"{v:.1f}" for v in values)
        conc_col = str(concurrency) if first_metric else ""
        print(f"  {conc_col:<14} {label:<22} {avg:>10.1f} {std:>10.1f} {cv:>7.1f}%   [{vals_str}]")
        first_metric = False
    print()

print("=" * 100)

models = sorted(set(k[0] for k in sorted_keys))
concurrencies = sorted(set(k[1] for k in sorted_keys))
if len(models) == 2:
    m1, m2 = models
    print(f"\n{'=' * 100}")
    print(f"COMPARISON: {m1} vs {m2}")
    print(f"{'=' * 100}")
    print(f"  {'Concurrency':<14} {'Metric':<22} {m1:>16} {m2:>16} {'Diff':>10}")
    print(f"  {'-'*14} {'-'*22} {'-'*16} {'-'*16} {'-'*10}")

    for conc in concurrencies:
        r1 = groups.get((m1, conc), [])
        r2 = groups.get((m2, conc), [])
        if not r1 or not r2:
            continue
        first_metric = True
        for key, label in METRICS:
            v1 = [r[key] for r in r1 if key in r]
            v2 = [r[key] for r in r2 if key in r]
            if not v1 or not v2:
                continue
            a1 = sum(v1) / len(v1)
            a2 = sum(v2) / len(v2)
            if a1 != 0:
                diff_pct = (a2 - a1) / abs(a1) * 100
                diff_str = f"{diff_pct:+.1f}%"
            else:
                diff_str = "N/A"
            conc_col = str(conc) if first_metric else ""
            print(f"  {conc_col:<14} {label:<22} {a1:>16.1f} {a2:>16.1f} {diff_str:>10}")
            first_metric = False
        print()

print("=" * 100)
print()

sys.stdout.close()