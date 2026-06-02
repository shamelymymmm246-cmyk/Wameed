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


FALLBACK_TEXT = """
وميض مشروع صغير لاختبار نماذج خفيفة على أجهزة ضعيفة. نكرر هذه الجملة حتى
توجد بيانات كافية لاختبار المسار البرمجي عندما يتعذر تحميل WikiText-2.
The quick local fallback exists only as a smoke test, not as a scientific result.
""" * 200


def set_seed(seed: int) -> None:
    """يثبت العشوائية حتى تكون المقارنة قابلة للتكرار."""
    random.seed(seed)
    torch.manual_seed(seed)


def encode_bytes(text: str) -> torch.Tensor:
    """يحوّل النص إلى أرقام بايت بسيطة؛ المفردات هنا 257 قيمة."""
    data = list(text.encode("utf-8", errors="ignore"))
    data.append(256)
    return torch.tensor(data, dtype=torch.long)


def load_wikitext_or_fallback(
    max_train_chars: int,
    max_valid_chars: int,
    force_fallback: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """يحمل WikiText-2، وإن فشل يستخدم نصاً محلياً صغيراً للاختبار."""
    if force_fallback:
        train_text = FALLBACK_TEXT[:max_train_chars]
        valid_text = FALLBACK_TEXT[-max_valid_chars:]
        return encode_bytes(train_text), encode_bytes(valid_text), "fallback_local إجباري لاختبار الجهاز"

    try:
        from datasets import load_dataset

        # المعرّف الحديث على HF Hub؛ المعرّف القديم "wikitext" لم يعد مدعوماً في datasets 4.x.
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
        train_text = "\n".join(dataset["train"]["text"])[:max_train_chars]
        valid_text = "\n".join(dataset["validation"]["text"])[:max_valid_chars]
        source = "WikiText-2"
    except Exception as error:
        train_text = FALLBACK_TEXT[:max_train_chars]
        valid_text = FALLBACK_TEXT[-max_valid_chars:]
        source = f"fallback_local بسبب: {type(error).__name__}: {error}"

    return encode_bytes(train_text), encode_bytes(valid_text), source


def make_batch(data: torch.Tensor, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """ينشئ دفعة عشوائية من مقاطع متجاورة للتدريب أو التقييم."""
    if data.numel() <= seq_len + 1:
        raise ValueError("البيانات أقصر من طول التسلسل المطلوب.")
    starts = torch.randint(0, data.numel() - seq_len - 1, (batch_size,))
    x = torch.stack([data[start : start + seq_len] for start in starts])
    y = torch.stack([data[start + 1 : start + seq_len + 1] for start in starts])
    return x, y


def current_ram_mb(process: psutil.Process) -> float:
    """يرجع الذاكرة الحالية للمعالجة بالميجابايت."""
    return process.memory_info().rss / (1024 * 1024)


def make_scheduler(optimizer: torch.optim.Optimizer, warmup: int, total_steps: int):
    """جدول معدل التعلّم: تسخين خطي ثم تناقص جيبي (cosine)."""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)  # ارتفاع خطي حتى نهاية التسخين.
        progress = (step - warmup) / max(1, total_steps - warmup)  # نسبة التقدم بعد التسخين.
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))  # تناقص جيبي نحو الصفر.

    return LambdaLR(optimizer, lr_lambda)


def grad_global_norm(model: TinySSM) -> float:
    """معيار التدرّج الكلي قبل القص؛ مؤشر صحة التعلّم."""
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += parameter.grad.data.norm(2).item() ** 2
    return total ** 0.5


@torch.no_grad()
def evaluate(
    model: TinySSM,
    valid_data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    collect_importance: bool = False,
) -> tuple[float, dict[str, float] | None]:
    """يحسب متوسط خسارة التنبؤ على بيانات التحقق، ويجمع إحصاءات الوسم إن طُلب."""
    model.eval()
    losses = []
    score_chunks = []
    for _ in range(eval_batches):
        x, y = make_batch(valid_data, batch_size, seq_len)
        if collect_importance:
            out = model(x, return_importance=True)
            logits = out.logits
            if out.importance_scores is not None:
                score_chunks.append(out.importance_scores.reshape(-1))
        else:
            logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))
        losses.append(loss.item())
    importance_stats = None
    if collect_importance and score_chunks:
        all_scores = torch.cat(score_chunks)
        importance_stats = {
            "mean": float(all_scores.mean().item()),
            "min": float(all_scores.min().item()),
            "max": float(all_scores.max().item()),
            "std": float(all_scores.std().item()),
        }
    return sum(losses) / len(losses), importance_stats


def run_one(args: argparse.Namespace, use_importance: bool, seed: int) -> dict[str, float | int | str | bool]:
    """يدرّب إعداداً واحداً ثم يرجع أرقامه."""
    set_seed(seed)
    process = psutil.Process(os.getpid())
    train_data, valid_data, data_source = load_wikitext_or_fallback(
        args.max_train_chars,
        args.max_valid_chars,
        force_fallback=args.force_fallback,
    )

    model = TinySSM(
        vocab_size=257,
        embed_dim=args.embed_dim,
        state_dim=args.state_dim,
        num_layers=args.num_layers,
        use_importance=use_importance,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay
    )
    scheduler = make_scheduler(optimizer, args.warmup, args.steps)
    params = count_parameters(model)
    peak_ram = current_ram_mb(process)
    start_time = time.perf_counter()
    last_importance_mean = None
    tag = "بوسم" if use_importance else "بلا وسم"
    print(f"  ▶ تدريب [{tag}] بذرة {seed} | {args.steps} خطوة | مصدر: {data_source[:24]}")

    model.train()
    for step in range(args.steps):
        x, y = make_batch(train_data, args.batch_size, args.seq_len)
        result = model(x, return_importance=use_importance)
        logits = result.logits if use_importance else result
        loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))
        if use_importance and args.sparsity_lambda > 0 and result.importance_scores is not None:
            loss = loss + args.sparsity_lambda * result.importance_scores.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = grad_global_norm(model)  # معيار التدرّج قبل القص.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        peak_ram = max(peak_ram, current_ram_mb(process))
        if use_importance and result.importance_scores is not None:
            last_importance_mean = float(result.importance_scores.mean().item())
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"    خطوة {step:5d} | خسارة={loss.item():.4f} | تدرّج={gnorm:.4f} | lr={lr_now:.2e}")

    valid_loss, importance_stats = evaluate(
        model, valid_data, args.batch_size, args.seq_len, args.eval_batches,
        collect_importance=use_importance,
    )
    ppl = math.exp(valid_loss)
    seconds = time.perf_counter() - start_time
    experiment = "importance" if use_importance else "base"
    if args.sparsity_lambda > 0 and use_importance:
        experiment = f"{experiment}_sparsity_{args.sparsity_lambda:g}"

    params_dict = {
        "stage": 1,
        "data_source": data_source,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "embed_dim": args.embed_dim,
        "state_dim": args.state_dim,
        "num_layers": args.num_layers,
        "lr": args.lr,
        "use_importance": use_importance,
        "sparsity_lambda": args.sparsity_lambda if use_importance else 0.0,
    }
    log_experiment(
        experiment=experiment,
        seed=seed,
        params=params_dict,
        metric_name="perplexity",
        metric_value=ppl,
        seconds=seconds,
        peak_ram_mb=peak_ram,
        csv_path=args.csv_path,
    )

    return {
        "experiment": experiment,
        "seed": seed,
        "perplexity": ppl,
        "valid_loss": valid_loss,
        "params": params,
        "seconds": seconds,
        "peak_ram_mb": peak_ram,
        "data_source": data_source,
        "last_importance_mean": last_importance_mean,
        "importance_stats": importance_stats,
    }


def summarize(rows: list[dict[str, float | int | str | bool]]) -> dict[str, object]:
    """يلخص النتائج حسب اسم التجربة."""
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["experiment"]), []).append(float(row["perplexity"]))

    summary = {}
    for name, values in grouped.items():
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        summary[name] = {"values": values, "mean": mean, "std": math.sqrt(variance)}

    if "base" in summary and "importance" in summary:
        base_mean = summary["base"]["mean"]
        tag_mean = summary["importance"]["mean"]
        summary["improvement_pct"] = (base_mean - tag_mean) / base_mean * 100
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="تدريب مقارنة وسم الأهمية في وميض.")
    parser.add_argument("--mode", choices=["one", "six"], default="six")
    parser.add_argument("--use-importance", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--sparsity-lambda", type=float, default=0.0)
    parser.add_argument("--max-train-chars", type=int, default=200_000)
    parser.add_argument("--max-valid-chars", type=int, default=30_000)
    parser.add_argument("--force-fallback", action="store_true")
    parser.add_argument("--csv-path", default="results/experiments.csv")
    parser.add_argument("--summary-path", default="results/stage1_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "one":
        rows = [run_one(args, use_importance=args.use_importance, seed=args.seed)]
    else:
        rows = []
        for use_importance in (False, True):
            for seed in (0, 1, 2):
                rows.append(run_one(args, use_importance=use_importance, seed=seed))

    summary = {"rows": rows, "summary": summarize(rows)}
    path = Path(args.summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
