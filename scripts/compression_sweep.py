"""
سكربت المرحلة 3 — مهمة 3.1: قياس مقايضة الضغط × الجودة.

يدرّب 4 نسب ضغط (1, 2, 4, 8) × 3 بذور = 12 تجربة على WikiText-2.
كل تجربة بنفس الإعداد (seq, steps, batch) لاختلاف واحد فقط: نسبة الضغط.

المقاييس:
  - PPL: كلما انخفضت كان أفضل.
  - ذروة الرامة: كلما انخفضت كان أفضل (التوفير).
  - زيادة PPL%: مقارنة بـ r=1 (baseline).

معيار القبول (نقطة فحص 3، فرع الضغط):
  ضغط 4:1 يوفّر ≥30% رامة مقابل زيادة PPL ≤5% (متوسط 3 بذور).

التشغيل:
  python scripts/compression_sweep.py --mode smoke   # اختبار سريع
  python scripts/compression_sweep.py --mode sweep   # التجربة الكاملة
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psutil
import torch
import torch.nn.functional as F

from wameed.engine.ssm import TinySSM, count_parameters
from wameed.utils.tracker import log_experiment


FALLBACK_TEXT = (
    "وميض مشروع صغير لاختبار نماذج خفيفة على أجهزة ضعيفة. "
    "The quick brown fox jumps over the lazy dog. "
    "Python is a wonderful language for machine learning experiments. "
) * 400


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def encode_bytes(text: str) -> torch.Tensor:
    data = list(text.encode("utf-8", errors="ignore"))
    data.append(256)
    return torch.tensor(data, dtype=torch.long)


def load_data(max_train_chars: int, max_valid_chars: int, force_fallback: bool = False):
    if force_fallback:
        return (
            encode_bytes(FALLBACK_TEXT[:max_train_chars]),
            encode_bytes(FALLBACK_TEXT[-max_valid_chars:]),
            "fallback_local إجباري",
        )
    try:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
        return (
            encode_bytes("\n".join(ds["train"]["text"])[:max_train_chars]),
            encode_bytes("\n".join(ds["validation"]["text"])[:max_valid_chars]),
            "WikiText-2",
        )
    except Exception as err:
        return (
            encode_bytes(FALLBACK_TEXT[:max_train_chars]),
            encode_bytes(FALLBACK_TEXT[-max_valid_chars:]),
            f"fallback_local بسبب: {type(err).__name__}",
        )


def make_batch(data, batch_size, seq_len):
    starts = torch.randint(0, data.numel() - seq_len - 1, (batch_size,))
    x = torch.stack([data[s:s + seq_len] for s in starts])
    y = torch.stack([data[s + 1:s + seq_len + 1] for s in starts])
    return x, y


def make_scheduler(optimizer, warmup, total_steps):
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, valid_data, batch_size, seq_len, eval_batches):
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = make_batch(valid_data, batch_size, seq_len)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))
        losses.append(loss.item())
    return math.exp(sum(losses) / len(losses))


def run_one(args, compress_ratio: int, seed: int) -> dict:
    set_seed(seed)
    proc = psutil.Process(os.getpid())
    print(f"    [r={compress_ratio} بذرة {seed}] تحميل البيانات...", flush=True)

    train_data, valid_data, data_source = load_data(
        args.max_train_chars, args.max_valid_chars, args.force_fallback
    )

    model = TinySSM(
        vocab_size=257,
        embed_dim=args.embed_dim,
        state_dim=args.state_dim,
        num_layers=args.num_layers,
        use_importance=True,
        use_memory=True,
        mem_k=args.mem_k,
        mem_K=args.mem_K,
        mem_threshold3=args.threshold3,
        mem_compress_ratio=compress_ratio,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = make_scheduler(optimizer, args.warmup, args.steps)

    params_count = count_parameters(model)
    peak_ram = proc.memory_info().rss / 1024**2
    start = time.perf_counter()

    print(
        f"  ▶ [r={compress_ratio}:1] بذرة {seed} | {args.steps} خطوة | "
        f"seq={args.seq_len} | معاملات={params_count} | بيانات={data_source[:20]}",
        flush=True,
    )

    model.train()
    for step in range(args.steps):
        x, y = make_batch(train_data, args.batch_size, args.seq_len)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        peak_ram = max(peak_ram, proc.memory_info().rss / 1024**2)

        if step % max(1, args.steps // 5) == 0 or step == args.steps - 1:
            print(f"    خطوة {step:5d} | خسارة={loss.item():.4f}")

    ppl = evaluate(model, valid_data, args.batch_size, args.seq_len, args.eval_batches)
    seconds = time.perf_counter() - start

    params_dict = {
        "stage": 3,
        "task": "3.1_compression_sweep",
        "data_source": data_source,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "embed_dim": args.embed_dim,
        "state_dim": args.state_dim,
        "num_layers": args.num_layers,
        "lr": args.lr,
        "mem_k": args.mem_k,
        "mem_K": args.mem_K,
        "threshold3": args.threshold3,
        "compress_ratio": compress_ratio,
        "params_count": params_count,
    }
    log_experiment(
        experiment=f"compression_r{compress_ratio}",
        seed=seed,
        params=params_dict,
        metric_name="perplexity",
        metric_value=ppl,
        seconds=seconds,
        peak_ram_mb=peak_ram,
        csv_path=args.csv_path,
    )
    print(f"    ✓ PPL={ppl:.4f} | رامة={peak_ram:.0f}م.ب | {seconds:.0f}ث")

    return {
        "experiment": f"r{compress_ratio}",
        "seed": seed,
        "compress_ratio": compress_ratio,
        "perplexity": ppl,
        "params": params_count,
        "seconds": seconds,
        "peak_ram_mb": peak_ram,
    }


def mean_std(vals):
    m = sum(vals) / len(vals)
    s = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
    return m, s


def build_sweep_table(rows: list[dict]) -> list[dict]:
    by_ratio = {}
    for r in rows:
        by_ratio.setdefault(r["compress_ratio"], []).append(r)

    baseline = by_ratio.get(1, [])
    base_ppl = mean_std([r["perplexity"] for r in baseline])[0] if baseline else None
    base_ram = mean_std([r["peak_ram_mb"] for r in baseline])[0] if baseline else None
    base_params = baseline[0]["params"] if baseline else None

    table = []
    for ratio in sorted(by_ratio.keys()):
        rs = by_ratio[ratio]
        ppl_m, ppl_s = mean_std([r["perplexity"] for r in rs])
        ram_m, _ = mean_std([r["peak_ram_mb"] for r in rs])
        ppl_inc = ((ppl_m - base_ppl) / base_ppl * 100) if base_ppl else 0.0
        ram_save = ((base_ram - ram_m) / base_ram * 100) if base_ram else 0.0
        table.append({
            "compress_ratio": ratio,
            "n_seeds": len(rs),
            "params": rs[0]["params"],
            "ppl_mean": round(ppl_m, 4),
            "ppl_std": round(ppl_s, 4),
            "ram_mean_mb": round(ram_m, 1),
            "ppl_increase_pct": round(ppl_inc, 3),
            "ram_saving_pct": round(ram_save, 3),
        })
    return table


def print_sweep_report(table: list[dict], rows: list[dict]) -> None:
    print("\n" + "=" * 64)
    print("📊 جدول مقايضة الضغط (المهمة 3.1)")
    print("=" * 64)
    print(f"{'r':>3} | {'PPL mean±std':>16} | {'ΔPPL %':>8} | "
          f"{'رامة م.ب':>9} | {'توفير رامة %':>13} | {'معاملات':>10}")
    print("-" * 64)
    for row in table:
        print(
            f"{row['compress_ratio']:>3} | "
            f"{row['ppl_mean']:>7.3f} ± {row['ppl_std']:.3f} | "
            f"{row['ppl_increase_pct']:>+7.2f}% | "
            f"{row['ram_mean_mb']:>8.1f} | "
            f"{row['ram_saving_pct']:>+12.2f}% | "
            f"{row['params']:>10d}"
        )

    # نقطة فحص 3 (فرع الضغط)
    r4 = next((r for r in table if r["compress_ratio"] == 4), None)
    print("\n🏁 فحص نقطة 3 (فرع الضغط):")
    if r4:
        ok = r4["ram_saving_pct"] >= 30 and r4["ppl_increase_pct"] <= 5
        print(
            f"   r=4:1 ⇒ توفير رامة {r4['ram_saving_pct']:.2f}% "
            f"(≥30%؟ {'✅' if r4['ram_saving_pct']>=30 else '❌'}) | "
            f"زيادة PPL {r4['ppl_increase_pct']:+.2f}% "
            f"(≤5%؟ {'✅' if r4['ppl_increase_pct']<=5 else '❌'}) | "
            f"الحكم: {'✅ نجح' if ok else '❌ لم ينجح'}"
        )
    print("=" * 64)


def parse_args():
    p = argparse.ArgumentParser(description="مهمة 3.1: scan نسب الضغط")
    p.add_argument("--mode", choices=["smoke", "one", "sweep"], default="smoke")
    p.add_argument("--compress-ratio", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--eval-batches", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--state-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--mem-k", type=int, default=16)
    p.add_argument("--mem-K", type=int, default=128)
    p.add_argument("--threshold3", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--max-train-chars", type=int, default=300_000)
    p.add_argument("--max-valid-chars", type=int, default=50_000)
    p.add_argument("--force-fallback", action="store_true")
    p.add_argument("--csv-path", default="results/experiments.csv")
    p.add_argument("--summary-path", default="results/stage3_compression_summary.json")
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "smoke":
        print("🔍 وضع smoke — 6 نسب ضغط تجريبية سريعة (10 خطوات)")
        args.steps = 10
        args.eval_batches = 2
        args.warmup = 2
        rows = []
        for r in (1, 2, 4, 8):
            for seed in (0,):
                rows.append(run_one(args, r, seed))
    elif args.mode == "one":
        rows = [run_one(args, args.compress_ratio, args.seed)]
    else:
        rows = []
        for r in (1, 2, 4, 8):
            for seed in (0, 1, 2):
                rows.append(run_one(args, r, seed))

    table = build_sweep_table(rows)
    print_sweep_report(table, rows)

    out = {"rows": rows, "sweep_table": table}
    path = Path(args.summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 محفوظ في: {path}")


if __name__ == "__main__":
    main()
