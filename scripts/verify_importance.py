#!/usr/bin/env python3
"""فحص معيار القبول للمهمة 1.2 — طبقة وسم الأهمية.

يتحقق من أربعة أمور بالأرقام:
1) درجات الأهمية s_t كلها داخل [0,1] (يطبع min/max/mean).
2) زيادة عدد المعاملات بسبب الوسم < 1%.
3) معيار التدرّج عند خطوة مبكرة بين 0.01 و 10.
4) فرق الرامة بين النسختين لا يتجاوز 300 م.ب.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psutil
import torch
import torch.nn.functional as F

from wameed.engine.ssm import TinySSM, count_parameters

# معاملات جهازه (انظر جدول المرجع في الخطة).
VOCAB, EMBED, STATE, LAYERS = 257, 128, 64, 2
BATCH, SEQ = 8, 64


def ram_mb() -> float:
    """رامة هذه المعالجة بالميجابايت."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def main() -> None:
    torch.manual_seed(0)
    print("=" * 56)
    print("فحص المهمة 1.2 — طبقة وسم الأهمية")
    print("=" * 56)

    # --- المعيار 2: عدد المعاملات ---
    base = TinySSM(VOCAB, EMBED, STATE, LAYERS, use_importance=False)
    ram_after_base = ram_mb()
    imp = TinySSM(VOCAB, EMBED, STATE, LAYERS, use_importance=True)
    ram_after_imp = ram_mb()

    p_base = count_parameters(base)
    p_imp = count_parameters(imp)
    growth = (p_imp - p_base) / p_base * 100
    print(f"\n[2] المعاملات: بلا وسم={p_base:,} | بوسم={p_imp:,}")
    print(f"    الزيادة={growth:.4f}%  (المطلوب < 1%)  -> "
          + ("نجح" if growth < 1.0 else "فشل"))

    # --- المعيار 1: درجات الأهمية داخل [0,1] ---
    tokens = torch.randint(0, VOCAB, (BATCH, SEQ))
    out = imp(tokens, return_importance=True)
    s = out.importance_scores  # (batch, layers, seq)
    s_min, s_max, s_mean = s.min().item(), s.max().item(), s.mean().item()
    in_range = bool(s_min >= 0.0 and s_max <= 1.0)
    print(f"\n[1] درجات s_t: min={s_min:.4f} max={s_max:.4f} mean={s_mean:.4f}")
    print(f"    داخل [0,1]؟ {'نعم' if in_range else 'لا'}  -> "
          + ("نجح" if in_range else "فشل"))

    # --- المعيار 3: معيار التدرّج عند خطوة تدريب واحدة ---
    optimizer = torch.optim.AdamW(imp.parameters(), lr=3e-4)
    y = torch.randint(0, VOCAB, (BATCH, SEQ))
    logits = imp(tokens)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    total_norm = 0.0
    for p in imp.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    grad_ok = 0.01 <= total_norm <= 10.0
    print(f"\n[3] معيار التدرّج (خطوة أولى)={total_norm:.4f}  (المطلوب 0.01–10)  -> "
          + ("نجح" if grad_ok else "فشل"))

    # --- المعيار 4: فرق الرامة ---
    diff_ram = ram_after_imp - ram_after_base
    ram_ok = diff_ram <= 300.0
    print(f"\n[4] فرق الرامة بسبب الوسم={diff_ram:.2f} م.ب  (المطلوب ≤ 300)  -> "
          + ("نجح" if ram_ok else "فشل"))
    print(f"    رامة المعالجة الحالية={ram_mb():.1f} م.ب")

    all_ok = (growth < 1.0) and in_range and grad_ok and ram_ok
    print("\n" + "=" * 56)
    print("النتيجة الكلية للمهمة 1.2: " + ("✅ نجحت" if all_ok else "❌ فشلت"))
    print("=" * 56)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
