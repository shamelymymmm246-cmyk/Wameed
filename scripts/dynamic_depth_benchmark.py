"""
سكربت المرحلة 3 — مهمة 3.2 + 3.3: العمق الديناميكي.

يدرّب نموذجاً مع رؤوس الخروج ثم يقيس:
1. PPL مع/بدون خروج مبكر.
2. متوسط الطبقات المستخدمة على توكنات سهلة vs صعبة.
3. تسريع الزمن من الخروج المبكر.

معيار القبول (نقطة فحص 3، فرع العمق الديناميكي):
  تخطّي ≥20% من الحساب (avg_layers ≤ 0.8 × num_layers) مع زيادة PPL ≤5%.

التشغيل:
  python scripts/dynamic_depth_benchmark.py --mode smoke   # سريع
  python scripts/dynamic_depth_benchmark.py --mode sweep   # 2 و 4 طبقات
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psutil
import torch
import torch.nn.functional as F

from wameed.engine.ssm import TinySSM
from wameed.engine.dynamic_depth import DynamicDepthSSM, measure_avg_layers
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


def compute_token_ranks(train_data: torch.Tensor) -> dict[int, int]:
    """يرجع قاموس token→rank حسب التردّد (0=الأكثر شيوعاً)."""
    counts = Counter(train_data.tolist())
    sorted_tokens = sorted(counts.items(), key=lambda x: -x[1])
    return {tok: rank for rank, (tok, _) in enumerate(sorted_tokens)}


@torch.no_grad()
def evaluate_ppl(model: DynamicDepthSSM, valid_data, batch_size, seq_len, eval_batches,
                 use_dd: bool = False) -> float:
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = make_batch(valid_data, batch_size, seq_len)
        logits, _ = model(x, use_dynamic_depth=use_dd)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))
        losses.append(loss.item())
    return math.exp(sum(losses) / len(losses))


@torch.no_grad()
def measure_easy_vs_hard(
    model: DynamicDepthSSM,
    valid_data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    token_ranks: dict[int, int],
    exit_threshold: float,
    n_batches: int = 5,
) -> dict:
    """
    يقيس متوسط الطبقات على توكنات سهلة (rank<100) vs صعبة (rank>1000).
    """
    model.eval()
    easy_ranks_threshold = 100
    hard_ranks_threshold = 1000

    layers_easy: list[float] = []
    layers_hard: list[float] = []
    correct_easy: list[int] = []
    correct_hard: list[int] = []

    for _ in range(n_batches):
        x, y = make_batch(valid_data, batch_size, seq_len)
        logits, _, layers_used = model(x, use_dynamic_depth=True, return_layers_used=True)
        preds = logits.argmax(-1)  # (batch, seq_len)
        for b in range(batch_size):
            for t in range(seq_len):
                tok = int(x[b, t].item())
                rank = token_ranks.get(tok, 9999)
                if rank < easy_ranks_threshold:
                    layers_easy.append(int(layers_used[b, t].item()))
                    correct_easy.append(int(preds[b, t].item() == y[b, t].item()))
                elif rank > hard_ranks_threshold:
                    layers_hard.append(int(layers_used[b, t].item()))
                    correct_hard.append(int(preds[b, t].item() == y[b, t].item()))

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "n_easy": len(layers_easy),
        "n_hard": len(layers_hard),
        "mean_layers_easy": mean(layers_easy),
        "mean_layers_hard": mean(layers_hard),
        "correct_easy_pct": 100.0 * mean(correct_easy),
        "correct_hard_pct": 100.0 * mean(correct_hard),
        "exit_threshold": exit_threshold,
    }


def train_and_benchmark(
    args, num_layers: int, seed: int, exit_threshold: float
) -> dict:
    set_seed(seed)
    proc = psutil.Process(os.getpid())
    print(f"    [L={num_layers} بذرة {seed} thr={exit_threshold}] تحميل البيانات...", flush=True)

    train_data, valid_data, data_source = load_data(
        args.max_train_chars, args.max_valid_chars, args.force_fallback
    )
    token_ranks = compute_token_ranks(train_data)

    base = TinySSM(
        vocab_size=257,
        embed_dim=args.embed_dim,
        state_dim=args.state_dim,
        num_layers=num_layers,
        use_importance=True,
        use_memory=args.use_memory,
        mem_k=args.mem_k,
        mem_K=args.mem_K,
        mem_threshold3=args.threshold3,
    )
    model = DynamicDepthSSM(base, exit_threshold=exit_threshold, exit_loss_weight=0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = make_scheduler(optimizer, args.warmup, args.steps)

    params_count = model.count_parameters()
    peak_ram = proc.memory_info().rss / 1024**2
    start = time.perf_counter()

    print(
        f"  ▶ [L={num_layers} بذرة {seed} thr={exit_threshold}] "
        f"{args.steps} خطوة | seq={args.seq_len} | params={params_count}",
        flush=True,
    )

    # ====== تدريب ======
    model.train()
    for step in range(args.steps):
        x, y = make_batch(train_data, args.batch_size, args.seq_len)
        _, loss = model(x, targets=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        peak_ram = max(peak_ram, proc.memory_info().rss / 1024**2)

        if step % max(1, args.steps // 5) == 0 or step == args.steps - 1:
            print(f"    خطوة {step:5d} | loss={loss.item():.4f}")

    # ====== تقييم ======
    ppl_full = evaluate_ppl(model, valid_data, args.batch_size, args.seq_len, args.eval_batches, use_dd=False)
    ppl_dd = evaluate_ppl(model, valid_data, args.batch_size, args.seq_len, args.eval_batches, use_dd=True)

    # ====== قياس الطبقات على سهل/صعب ======
    eh = measure_easy_vs_hard(
        model, valid_data, args.batch_size, args.seq_len,
        token_ranks, exit_threshold, n_batches=3,
    )

    # ====== قياس تسريع الزمن ======
    times_full, times_dd = [], []
    model.eval()
    with torch.no_grad():
        for _ in range(args.time_batches):
            x = torch.randint(0, 257, (args.batch_size, args.seq_len))
            t0 = time.perf_counter()
            _ = model(x, use_dynamic_depth=False)
            times_full.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            _ = model(x, use_dynamic_depth=True)
            times_dd.append(time.perf_counter() - t0)

    mean_time_full = sum(times_full) / len(times_full)
    mean_time_dd = sum(times_dd) / len(times_dd)
    speedup = mean_time_full / mean_time_dd if mean_time_dd > 0 else 0.0
    total_seconds = time.perf_counter() - start

    ppl_inc_pct = (ppl_dd - ppl_full) / ppl_full * 100

    # تسجيل في CSV
    log_experiment(
        experiment=f"dyndepth_L{num_layers}_thr{exit_threshold}",
        seed=seed,
        params={
            "stage": 3, "task": "3.2_3.3_dyndepth",
            "data_source": data_source, "steps": args.steps,
            "batch_size": args.batch_size, "seq_len": args.seq_len,
            "num_layers": num_layers, "lr": args.lr,
            "exit_threshold": exit_threshold,
            "params_count": params_count,
        },
        metric_name="ppl_full",
        metric_value=ppl_full,
        seconds=total_seconds, peak_ram_mb=peak_ram,
        csv_path=args.csv_path,
    )
    log_experiment(
        experiment=f"dyndepth_L{num_layers}_thr{exit_threshold}",
        seed=seed,
        params={
            "stage": 3, "task": "3.2_3.3_dyndepth",
            "data_source": data_source, "steps": args.steps,
            "batch_size": args.batch_size, "seq_len": args.seq_len,
            "num_layers": num_layers, "lr": args.lr,
            "exit_threshold": exit_threshold,
            "params_count": params_count,
        },
        metric_name="ppl_dd",
        metric_value=ppl_dd,
        seconds=total_seconds, peak_ram_mb=peak_ram,
        csv_path=args.csv_path,
    )

    print(
        f"    ✓ PPL_full={ppl_full:.3f} | PPL_dd={ppl_dd:.3f} ({ppl_inc_pct:+.2f}%) | "
        f"avg_layers easy={eh['mean_layers_easy']:.2f} hard={eh['mean_layers_hard']:.2f} | "
        f"speedup={speedup:.2f}x | RAM={peak_ram:.0f}م.ب"
    )

    return {
        "num_layers": num_layers, "seed": seed, "exit_threshold": exit_threshold,
        "ppl_full": ppl_full, "ppl_dd": ppl_dd, "ppl_inc_pct": ppl_inc_pct,
        "mean_layers_easy": eh["mean_layers_easy"],
        "mean_layers_hard": eh["mean_layers_hard"],
        "n_easy": eh["n_easy"], "n_hard": eh["n_hard"],
        "correct_easy_pct": eh["correct_easy_pct"],
        "correct_hard_pct": eh["correct_hard_pct"],
        "mean_time_full": mean_time_full,
        "mean_time_dd": mean_time_dd,
        "speedup": speedup,
        "params": params_count,
        "peak_ram_mb": peak_ram,
        "seconds": total_seconds,
    }


def build_summary_table(rows: list[dict]) -> list[dict]:
    """تجميع حسب (num_layers, exit_threshold)."""
    by_key: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["num_layers"], r["exit_threshold"])
        by_key.setdefault(key, []).append(r)

    table = []
    for (nl, thr), rs in sorted(by_key.items()):
        m_ppl_full = sum(r["ppl_full"] for r in rs) / len(rs)
        m_ppl_dd = sum(r["ppl_dd"] for r in rs) / len(rs)
        m_inc = (m_ppl_dd - m_ppl_full) / m_ppl_full * 100
        m_easy = sum(r["mean_layers_easy"] for r in rs) / len(rs)
        m_hard = sum(r["mean_layers_hard"] for r in rs) / len(rs)
        m_speed = sum(r["speedup"] for r in rs) / len(rs)
        m_save_pct = (1 - m_easy / nl) * 100  # تقريب: نسبة التوفير الإجمالية
        table.append({
            "num_layers": nl, "exit_threshold": thr, "n_seeds": len(rs),
            "ppl_full_mean": round(m_ppl_full, 3),
            "ppl_dd_mean": round(m_ppl_dd, 3),
            "ppl_increase_pct": round(m_inc, 3),
            "mean_layers_easy": round(m_easy, 2),
            "mean_layers_hard": round(m_hard, 2),
            "compute_saving_pct_easy": round(m_save_pct, 2),
            "speedup": round(m_speed, 3),
        })
    return table


def parse_args():
    p = argparse.ArgumentParser(description="مهمة 3.2/3.3: العمق الديناميكي")
    p.add_argument("--mode", choices=["smoke", "sweep"], default="smoke")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--eval-batches", type=int, default=5)
    p.add_argument("--time-batches", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--state-dim", type=int, default=64)
    p.add_argument("--mem-k", type=int, default=16)
    p.add_argument("--mem-K", type=int, default=128)
    p.add_argument("--threshold3", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--use-memory", action="store_true", default=True)
    p.add_argument("--max-train-chars", type=int, default=200_000)
    p.add_argument("--max-valid-chars", type=int, default=50_000)
    p.add_argument("--force-fallback", action="store_true")
    p.add_argument("--csv-path", default="results/experiments.csv")
    p.add_argument("--summary-path", default="results/stage3_dyndepth_summary.json")
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "smoke":
        # اختبار سريع: طبقة واحدة لكل من 2 و 4
        print("🔍 وضع smoke — طبقة 2 و 4، 30 خطوة، 1 بذرة")
        args.steps = 30
        args.eval_batches = 2
        args.time_batches = 2
        rows = []
        for nl in (2, 4):
            for thr in (0.7,):
                rows.append(train_and_benchmark(args, nl, seed=0, exit_threshold=thr))
    else:
        rows = []
        # طبقتان: هدف متوقع ≤1.6 طبقة (≥20% توفير)
        for nl in (2, 4):
            for thr in (0.7, 0.85, 0.95):
                for seed in (0, 1, 2):
                    rows.append(train_and_benchmark(args, nl, seed=seed, exit_threshold=thr))

    table = build_summary_table(rows)
    print("\n" + "=" * 84)
    print("📊 جدول نتائج العمق الديناميكي (المهمة 3.2/3.3)")
    print("=" * 84)
    print(f"{'L':>2} | {'thr':>5} | {'PPL_full':>8} | {'PPL_dd':>8} | {'ΔPPL%':>7} | "
          f"{'L_easy':>6} | {'L_hard':>6} | {'توفير %':>9} | {'speedup':>8}")
    print("-" * 84)
    for row in table:
        print(
            f"{row['num_layers']:>2} | {row['exit_threshold']:>5.2f} | "
            f"{row['ppl_full_mean']:>8.3f} | {row['ppl_dd_mean']:>8.3f} | "
            f"{row['ppl_increase_pct']:>+6.2f}% | {row['mean_layers_easy']:>6.2f} | "
            f"{row['mean_layers_hard']:>6.2f} | {row['compute_saving_pct_easy']:>+8.2f}% | "
            f"{row['speedup']:>7.2f}x"
        )
    print("=" * 84)

    # فحص نقطة 3 (فرع العمق الديناميكي)
    print("\n🏁 فحص نقطة 3 (فرع العمق الديناميكي):")
    for row in table:
        if row["num_layers"] >= 4:
            saving_ok = row["compute_saving_pct_easy"] >= 20
            ppl_ok = row["ppl_increase_pct"] <= 5
            ok = saving_ok and ppl_ok
            print(
                f"   L={row['num_layers']} thr={row['exit_threshold']}: "
                f"توفير {row['compute_saving_pct_easy']:.1f}% (≥20%؟ {'✅' if saving_ok else '❌'}) | "
                f"ΔPPL {row['ppl_increase_pct']:+.2f}% (≤5%؟ {'✅' if ppl_ok else '❌'}) | "
                f"الحكم: {'✅ نجح' if ok else '❌ لم ينجح'}"
            )

    out = {"rows": rows, "summary_table": table}
    path = Path(args.summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 محفوظ في: {path}")


if __name__ == "__main__":
    main()
