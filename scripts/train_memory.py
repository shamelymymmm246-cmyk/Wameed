"""
سكربت المرحلة 2 — الذاكرة الهرمية

يُدرّب نموذجَين (بذاكرة / بلا ذاكرة) على WikiText-2 ويقيس:
1. PPL القياسية (standard perplexity)
2. PPL المتأخرة (late_ppl) — خسارة على النصف الثاني من التسلسل
   الفرضية: النموذج بالذاكرة يستفيد من السياق المبكر لتحسين توقّعاته المتأخرة.
3. تحسّن المسبار (probe_improvement) — تحقّق من تذكّر البداية في النصوص الطويلة.

التشغيل:
  # اختبار سريع (smoke test):
  python scripts/train_memory.py --mode smoke

  # تجارب كاملة (6 تجارب على الجهاز):
  python scripts/train_memory.py --mode six

  # تجارب على Colab (تسلسل أطول):
  python scripts/train_memory.py --mode six --seq-len 256 --steps 5000
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
from scipy import stats as scipy_stats

from wameed.engine.ssm import TinySSM, count_parameters
from wameed.utils.tracker import log_experiment


# نص احتياطي يُستخدم عند تعذّر تحميل WikiText-2.
FALLBACK_TEXT = """
وميض مشروع صغير لاختبار نماذج خفيفة على أجهزة ضعيفة. نكرر هذه الجملة حتى
توجد بيانات كافية لاختبار المسار البرمجي عندما يتعذر تحميل WikiText-2.
The quick brown fox jumps over the lazy dog. This fallback text repeats many times.
Python is a wonderful language for machine learning experiments and research.
""" * 400


def set_seed(seed: int) -> None:
    """يثبّت العشوائية لضمان قابلية التكرار."""
    random.seed(seed)
    torch.manual_seed(seed)


def encode_bytes(text: str) -> torch.Tensor:
    """يحوّل النص إلى توكنات بايت (0–256)، 256 = نهاية مستند."""
    data = list(text.encode("utf-8", errors="ignore"))
    data.append(256)
    return torch.tensor(data, dtype=torch.long)


def load_data(
    max_train_chars: int,
    max_valid_chars: int,
    force_fallback: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """يحمل WikiText-2؛ يرجع إلى نص محلي عند الفشل."""
    if force_fallback:
        src = "fallback_local إجباري"
        return encode_bytes(FALLBACK_TEXT[:max_train_chars]), encode_bytes(FALLBACK_TEXT[-max_valid_chars:]), src
    try:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
        train_text = "\n".join(ds["train"]["text"])[:max_train_chars]
        valid_text = "\n".join(ds["validation"]["text"])[:max_valid_chars]
        return encode_bytes(train_text), encode_bytes(valid_text), "WikiText-2"
    except Exception as err:
        src = f"fallback_local بسبب: {type(err).__name__}"
        return encode_bytes(FALLBACK_TEXT[:max_train_chars]), encode_bytes(FALLBACK_TEXT[-max_valid_chars:]), src


def make_batch(
    data: torch.Tensor, batch_size: int, seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """يأخذ مقاطع عشوائية متجاورة من البيانات."""
    if data.numel() <= seq_len + 1:
        raise ValueError("البيانات أقصر من طول التسلسل.")
    starts = torch.randint(0, data.numel() - seq_len - 1, (batch_size,))
    x = torch.stack([data[s: s + seq_len] for s in starts])
    y = torch.stack([data[s + 1: s + seq_len + 1] for s in starts])
    return x, y


def current_ram_mb(proc: psutil.Process) -> float:
    return proc.memory_info().rss / 1024**2


def make_scheduler(optimizer, warmup: int, total_steps: int):
    """تسخين خطي ثم تناقص جيبي (cosine)."""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_lambda)


def grad_norm(model: TinySSM) -> float:
    """معيار التدرّج الكلي قبل القص."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


@torch.no_grad()
def evaluate_full(
    model: TinySSM,
    valid_data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
) -> tuple[float, float]:
    """
    يحسب PPL القياسية و PPL المتأخرة.

    PPL القياسية: متوسط الخسارة على كامل التسلسل.
    PPL المتأخرة: متوسط الخسارة على النصف الثاني فقط.
    الفرضية: النموذج بالذاكرة يستفيد أكثر في النصف الثاني
    لأنه يتذكّر السياق من النصف الأول.
    """
    model.eval()
    losses_all = []
    losses_late = []
    half = seq_len // 2  # بداية النصف الثاني.

    for _ in range(eval_batches):
        x, y = make_batch(valid_data, batch_size, seq_len)
        logits = model(x)  # (batch, seq_len, vocab)

        # PPL القياسية — على كامل التسلسل.
        loss_all = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))
        losses_all.append(loss_all.item())

        # PPL المتأخرة — على التوكنات من النصف فصاعداً.
        logits_late = logits[:, half:, :]          # (batch, half, vocab)
        y_late = y[:, half:]                        # (batch, half)
        loss_late = F.cross_entropy(logits_late.reshape(-1, model.vocab_size), y_late.reshape(-1))
        losses_late.append(loss_late.item())

    avg_loss_all  = sum(losses_all) / len(losses_all)
    avg_loss_late = sum(losses_late) / len(losses_late)
    return math.exp(avg_loss_all), math.exp(avg_loss_late)


@torch.no_grad()
def probe_test(
    model: TinySSM,
    seq_len: int,
    n_probes: int = 20,
    key_byte: int = 1,
) -> float:
    """
    اختبار المسبار: هل يتذكّر النموذج بداية التسلسل الطويل؟

    الفكرة:
    - نبني تسلسلاً: [key, filler..., filler_last]
    - الهدف عند الموضع الأخير هو key (التوكن الحارس).
    - نقيس الخسارة عند هذا الموضع مع وجود key في البداية مقارنةً بغيابه.
    - إذا كان النموذج يتذكّر، فالخسارة ستكون أقل حين key في البداية.

    المخرج: probe_improvement = (خسارة بلا key − خسارة بـ key) / خسارة بلا key × 100
    """
    model.eval()

    # نختار byte حشو مختلف عن key و 0.
    other_byte = (key_byte + 1) % 256
    if other_byte == 0:
        other_byte = 2

    improvements = []
    for _ in range(n_probes):
        # حشو عشوائي (أرقام 10–200 لتجنّب bytes الخاصة).
        filler_len = seq_len - 2
        filler = torch.randint(10, 200, (1, filler_len))

        # تسلسل مع key في البداية: [key, filler..., key]
        # x: [key, filler[0..n-2]], y: [filler[0..n-2], key]
        x_with = torch.cat([
            torch.tensor([[key_byte]]),
            filler[:, :-1]
        ], dim=1)  # (1, seq_len-1)
        y_with = torch.cat([
            filler[:, :-1],
            torch.tensor([[key_byte]])
        ], dim=1)  # (1, seq_len-1) — الهدف الأخير = key

        # تسلسل بلا key في البداية: [other, filler..., key]
        x_no = torch.cat([
            torch.tensor([[other_byte]]),
            filler[:, :-1]
        ], dim=1)

        # نحسب الخسارة عند الموضع الأخير فقط (توقّع key).
        logits_with = model(x_with)  # (1, seq_len-1, vocab)
        logits_no   = model(x_no)

        target_last = y_with[:, -1]  # = key_byte

        loss_with = F.cross_entropy(logits_with[:, -1, :], target_last).item()
        loss_no   = F.cross_entropy(logits_no[:, -1, :],   target_last).item()

        if loss_no > 1e-8:
            improvement = (loss_no - loss_with) / loss_no * 100.0
            improvements.append(improvement)

    return float(sum(improvements) / len(improvements)) if improvements else 0.0


def run_one(
    args: argparse.Namespace,
    use_memory: bool,
    seed: int,
) -> dict:
    """يدرّب إعداداً واحداً ويرجع قاموس أرقامه."""
    set_seed(seed)
    proc = psutil.Process(os.getpid())

    train_data, valid_data, data_source = load_data(
        args.max_train_chars,
        args.max_valid_chars,
        force_fallback=args.force_fallback,
    )

    model = TinySSM(
        vocab_size=257,
        embed_dim=args.embed_dim,
        state_dim=args.state_dim,
        num_layers=args.num_layers,
        use_importance=True,     # دائماً نستخدم وسم الأهمية من المرحلة 1.
        use_memory=use_memory,
        mem_k=args.mem_k,
        mem_K=args.mem_big_k,
        mem_threshold3=args.threshold3,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay
    )
    scheduler = make_scheduler(optimizer, args.warmup, args.steps)
    params_count = count_parameters(model)
    peak_ram = current_ram_mb(proc)
    start = time.perf_counter()

    tag = "بذاكرة" if use_memory else "بلا ذاكرة"
    print(f"\n  ▶ [{tag}] بذرة {seed} | {args.steps} خطوة | seq={args.seq_len} | {data_source[:24]}")

    # ======= حلقة التدريب =======
    model.train()
    for step in range(args.steps):
        x, y = make_batch(train_data, args.batch_size, args.seq_len)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = grad_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        peak_ram = max(peak_ram, current_ram_mb(proc))

        if step % max(1, args.steps // 5) == 0 or step == args.steps - 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"    خطوة {step:5d} | خسارة={loss.item():.4f} | تدرّج={gnorm:.4f} | lr={lr_now:.2e}")

    # ======= التقييم =======
    ppl, late_ppl = evaluate_full(
        model, valid_data, args.batch_size, args.seq_len, args.eval_batches
    )
    probe_imp = probe_test(model, seq_len=min(args.seq_len, 128), n_probes=30)
    seconds = time.perf_counter() - start

    experiment_name = "memory" if use_memory else "no_memory"
    params_dict = {
        "stage": 2,
        "data_source": data_source,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "embed_dim": args.embed_dim,
        "state_dim": args.state_dim,
        "num_layers": args.num_layers,
        "lr": args.lr,
        "use_importance": True,
        "use_memory": use_memory,
        "mem_k": args.mem_k,
        "mem_K": args.mem_big_k,
        "threshold3": args.threshold3,
        "params_count": params_count,
    }

    # نسجّل كل مقياس في سطر منفصل في الـ CSV.
    for metric_name, metric_value in [("perplexity", ppl), ("late_ppl", late_ppl), ("probe_improvement", probe_imp)]:
        log_experiment(
            experiment=experiment_name,
            seed=seed,
            params=params_dict,
            metric_name=metric_name,
            metric_value=metric_value,
            seconds=seconds,
            peak_ram_mb=peak_ram,
            csv_path=args.csv_path,
        )

    print(f"    ✓ PPL={ppl:.4f} | late_PPL={late_ppl:.4f} | probe={probe_imp:.2f}% | رامة={peak_ram:.0f}م.ب")

    return {
        "experiment": experiment_name,
        "seed": seed,
        "perplexity": ppl,
        "late_ppl": late_ppl,
        "probe_improvement": probe_imp,
        "params": params_count,
        "seconds": seconds,
        "peak_ram_mb": peak_ram,
        "data_source": data_source,
    }


def statistical_summary(rows: list[dict]) -> dict:
    """يحسب المتوسطات والتحليل الإحصائي t-test."""
    no_mem_ppl   = [r["perplexity"]      for r in rows if r["experiment"] == "no_memory"]
    mem_ppl      = [r["perplexity"]      for r in rows if r["experiment"] == "memory"]
    no_mem_late  = [r["late_ppl"]        for r in rows if r["experiment"] == "no_memory"]
    mem_late     = [r["late_ppl"]        for r in rows if r["experiment"] == "memory"]
    no_mem_probe = [r["probe_improvement"] for r in rows if r["experiment"] == "no_memory"]
    mem_probe    = [r["probe_improvement"] for r in rows if r["experiment"] == "memory"]

    def mean_std(vals):
        m = sum(vals) / len(vals)
        s = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
        return m, s

    def safe_ttest(a, b):
        if len(a) < 2 or len(b) < 2:
            return float("nan"), float("nan")
        t, p = scipy_stats.ttest_ind(a, b)
        return float(t), float(p)

    ppl_m_no, ppl_s_no   = mean_std(no_mem_ppl)
    ppl_m_w,  ppl_s_w    = mean_std(mem_ppl)
    late_m_no, late_s_no = mean_std(no_mem_late)
    late_m_w,  late_s_w  = mean_std(mem_late)
    probe_m_no, _        = mean_std(no_mem_probe)
    probe_m_w, _         = mean_std(mem_probe)

    _, p_ppl   = safe_ttest(no_mem_ppl,  mem_ppl)
    _, p_late  = safe_ttest(no_mem_late, mem_late)
    _, p_probe = safe_ttest(no_mem_probe, mem_probe)

    ppl_improve   = (ppl_m_no - ppl_m_w) / ppl_m_no * 100
    late_improve  = (late_m_no - late_m_w) / late_m_no * 100
    probe_improve = probe_m_w - probe_m_no

    return {
        "ppl_no_memory":    round(ppl_m_no, 4),
        "ppl_no_std":       round(ppl_s_no, 4),
        "ppl_memory":       round(ppl_m_w, 4),
        "ppl_memory_std":   round(ppl_s_w, 4),
        "ppl_improvement_pct": round(ppl_improve, 3),
        "ppl_p_value":      round(p_ppl, 4) if not math.isnan(p_ppl) else "nan",
        "late_ppl_no_memory":   round(late_m_no, 4),
        "late_ppl_no_std":      round(late_s_no, 4),
        "late_ppl_memory":      round(late_m_w, 4),
        "late_ppl_memory_std":  round(late_s_w, 4),
        "late_ppl_improvement_pct": round(late_improve, 3),
        "late_ppl_p_value":     round(p_late, 4) if not math.isnan(p_late) else "nan",
        "probe_no_memory":  round(probe_m_no, 3),
        "probe_memory":     round(probe_m_w, 3),
        "probe_delta":      round(probe_improve, 3),
        "probe_p_value":    round(p_probe, 4) if not math.isnan(p_probe) else "nan",
    }


def print_4line_report(summary: dict, rows: list[dict]) -> None:
    """يطبع تقرير الأربعة أسطر المطلوب بعد كل مجموعة تجارب."""
    print("\n" + "=" * 60)
    print("📋 تقرير المرحلة 2 — الذاكرة الهرمية")
    print("=" * 60)
    print(f"1️⃣  ماذا عملت: دربّت {len(rows)} تجربة ({len([r for r in rows if r['experiment']=='no_memory'])} بلا ذاكرة + {len([r for r in rows if r['experiment']=='memory'])} بذاكرة) على WikiText-2")
    print(f"2️⃣  الأمر: python scripts/train_memory.py --mode six --seq-len {rows[0].get('seq_len', '?') if rows else '?'}")
    print(f"3️⃣  النتائج:")
    print(f"     PPL     — بلا ذاكرة: {summary['ppl_no_memory']:.4f}±{summary['ppl_no_std']:.4f}  |  بذاكرة: {summary['ppl_memory']:.4f}±{summary['ppl_memory_std']:.4f}  |  تحسّن: {summary['ppl_improvement_pct']:.2f}%  p={summary['ppl_p_value']}")
    print(f"     late_PPL— بلا ذاكرة: {summary['late_ppl_no_memory']:.4f}±{summary['late_ppl_no_std']:.4f}  |  بذاكرة: {summary['late_ppl_memory']:.4f}±{summary['late_ppl_memory_std']:.4f}  |  تحسّن: {summary['late_ppl_improvement_pct']:.2f}%  p={summary['late_ppl_p_value']}")
    print(f"     probe   — بلا ذاكرة: {summary['probe_no_memory']:.2f}%  |  بذاكرة: {summary['probe_memory']:.2f}%  |  فرق: {summary['probe_delta']:.2f}ن.م")
    print(f"4️⃣  المهمة التالية: {'نقطة الفحص 2 اجتازت — انتقل للمرحلة 3' if summary['late_ppl_improvement_pct'] >= 2.0 else 'لم تجتز نقطة الفحص — راجع التوثيق وناقش مع أبو دجانة'}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="تدريب الذاكرة الهرمية — وميض المرحلة 2")
    p.add_argument("--mode", choices=["smoke", "one", "six"], default="smoke",
                   help="smoke=8خطوات سريعة | one=تجربة واحدة | six=3بذور×2إعداد")
    p.add_argument("--use-memory", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--eval-batches", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--state-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--mem-k", type=int, default=16, help="تردّد تحديث M2")
    p.add_argument("--mem-big-k", type=int, default=256, help="تردّد تحديث M4")
    p.add_argument("--threshold3", type=float, default=0.6,
                   help="عتبة أهمية تحديث M3؛ خفّضها لتفعيل الطبقة الدلالية")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--max-train-chars", type=int, default=300_000)
    p.add_argument("--max-valid-chars", type=int, default=50_000)
    p.add_argument("--force-fallback", action="store_true")
    p.add_argument("--csv-path", default="results/experiments.csv")
    p.add_argument("--summary-path", default="results/stage2_summary.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "smoke":
        # اختبار سريع: 8 خطوات فقط للتأكد من أن الكود يعمل.
        print("🔍 وضع smoke test — 8 خطوات للتحقّق من الكود")
        args.steps = 8
        args.eval_batches = 2
        args.batch_size = 4
        args.seq_len = 64
        args.warmup = 2
        rows = [
            run_one(args, use_memory=False, seed=0),
            run_one(args, use_memory=True,  seed=0),
        ]
    elif args.mode == "one":
        rows = [run_one(args, use_memory=args.use_memory, seed=args.seed)]
    else:  # six
        # 3 بذور × 2 إعداد = 6 تجارب.
        rows = []
        for use_memory in (False, True):
            for seed in (0, 1, 2):
                rows.append(run_one(args, use_memory=use_memory, seed=seed))

    summary = statistical_summary(rows)
    print_4line_report(summary, rows)

    result = {"rows": rows, "summary": summary}
    path = Path(args.summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 النتائج مُحفظة في: {path}")


if __name__ == "__main__":
    main()
