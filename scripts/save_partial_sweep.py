"""تجميع نتائج الـ 11 تجربة المكتملة (r=8 seed=2 مفقود بسبب الإجهاض)"""
import json
import math
from pathlib import Path

LOG = Path("/home/abodojana301zx/Documents/بنيان +/Wameed/results/stage3_compression_run.log")
OUT = Path("/home/abodojana301zx/Documents/بنيان +/Wameed/results/stage3_compression_summary.json")

# قُرأت من الـ log يدوياً
rows = [
    {"experiment": "r1", "seed": 0, "compress_ratio": 1, "perplexity": 26.2981, "params": 122115, "seconds": 93, "peak_ram_mb": 332},
    {"experiment": "r1", "seed": 1, "compress_ratio": 1, "perplexity": 26.0240, "params": 122115, "seconds": 98, "peak_ram_mb": 332},
    {"experiment": "r1", "seed": 2, "compress_ratio": 1, "perplexity": 30.7728, "params": 122115, "seconds": 77, "peak_ram_mb": 333},
    {"experiment": "r2", "seed": 0, "compress_ratio": 2, "perplexity": 23.8291, "params": 124259, "seconds": 78, "peak_ram_mb": 335},
    {"experiment": "r2", "seed": 1, "compress_ratio": 2, "perplexity": 23.8890, "params": 124259, "seconds": 112, "peak_ram_mb": 335},
    {"experiment": "r2", "seed": 2, "compress_ratio": 2, "perplexity": 26.7536, "params": 124259, "seconds": 108, "peak_ram_mb": 334},
    {"experiment": "r4", "seed": 0, "compress_ratio": 4, "perplexity": 29.0113, "params": 120115, "seconds": 127, "peak_ram_mb": 335},
    {"experiment": "r4", "seed": 1, "compress_ratio": 4, "perplexity": 27.9360, "params": 120115, "seconds": 115, "peak_ram_mb": 334},
    {"experiment": "r4", "seed": 2, "compress_ratio": 4, "perplexity": 28.9014, "params": 120115, "seconds": 94, "peak_ram_mb": 334},
    {"experiment": "r8", "seed": 0, "compress_ratio": 8, "perplexity": 30.2464, "params": 118043, "seconds": 71, "peak_ram_mb": 334},
    {"experiment": "r8", "seed": 1, "compress_ratio": 8, "perplexity": 24.8460, "params": 118043, "seconds": 89, "peak_ram_mb": 335},
]

# r=8 seed 2 مفقود بسبب إجهاض المستخدم — مُوثّق بصدق


def mean_std(vals):
    m = sum(vals) / len(vals)
    s = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
    return m, s


by_ratio = {}
for r in rows:
    by_ratio.setdefault(r["compress_ratio"], []).append(r)

baseline_ppl, _ = mean_std([r["perplexity"] for r in by_ratio[1]])
baseline_ram, _ = mean_std([r["peak_ram_mb"] for r in by_ratio[1]])

table = []
for ratio in sorted(by_ratio.keys()):
    rs = by_ratio[ratio]
    ppl_m, ppl_s = mean_std([r["perplexity"] for r in rs])
    ram_m, _ = mean_std([r["peak_ram_mb"] for r in rs])
    table.append({
        "compress_ratio": ratio,
        "n_seeds": len(rs),
        "params": rs[0]["params"],
        "ppl_mean": round(ppl_m, 4),
        "ppl_std": round(ppl_s, 4),
        "ram_mean_mb": round(ram_m, 1),
        "ppl_increase_pct": round((ppl_m - baseline_ppl) / baseline_ppl * 100, 3),
        "ram_saving_pct": round((baseline_ram - ram_m) / baseline_ram * 100, 3),
    })

out = {
    "config": {
        "steps": 250,
        "eval_batches": 8,
        "batch_size": 8,
        "seq_len": 128,
        "data": "fallback_local إجباري",
        "device": "CPU i7-4510U",
        "note": "r=8 seed=2 مفقود (إجهاض المستخدم بعد 11/12 تجربة)",
    },
    "rows": rows,
    "sweep_table": table,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"محفوظ: {OUT}")
print()
print(f"{'r':>3} | {'PPL':>14} | {'ΔPPL %':>8} | {'RAM':>7} | {'توفير RAM %':>12} | seeds")
for row in table:
    print(f"{row['compress_ratio']:>3} | {row['ppl_mean']:>7.2f} ± {row['ppl_std']:.2f} | {row['ppl_increase_pct']:>+7.2f}% | {row['ram_mean_mb']:>6.1f} | {row['ram_saving_pct']:>+11.2f}% | {row['n_seeds']}")
