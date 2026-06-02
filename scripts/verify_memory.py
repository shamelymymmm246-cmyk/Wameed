"""
سكربت تحقّق المرحلة 2 — الذاكرة الهرمية (مهمتا 2.1 و 2.2)

يتحقّق بالأرقام من:
  (2.1) جداول تحديث الطبقات الأربع عبر 100 خطوة + الرامة الإضافية.
  (2.2) ربط الذاكرة بنموذج SSM: نجاح التمرير، شكل المخرج، زيادة المعاملات،
        تدفّق التدرّجات إلى معاملات الذاكرة.
  (تشخيص) كم طبقة ذاكرة "حيّة" فعلاً على تسلسلات بأطوال مختلفة.

التشغيل:
  python scripts/verify_memory.py
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

from wameed.engine.memory import HierarchicalMemory
from wameed.engine.ssm import TinySSM, count_parameters


def check_2_1() -> None:
    """مهمة 2.1: محاكاة 100 خطوة والتحقّق من عدّادات التحديث + الرامة."""
    print("=" * 56)
    print("(2.1) اختبار جداول التحديث — 100 خطوة")
    print("=" * 56)

    proc = psutil.Process(os.getpid())
    ram_before = proc.memory_info().rss / 1024**2

    B, D = 2, 64
    # نستخدم عتبة منخفضة (0.3) لرؤية تحديثات M3 بوضوح في الاختبار.
    mem = HierarchicalMemory(state_dim=D, k=16, K=256, threshold3=0.3)
    mem.init_memory(B, device="cpu")
    for _ in range(100):
        mem.update(torch.randn(B, D), torch.rand(B, 1))
    mem.print_update_counts()

    ram_after = proc.memory_info().rss / 1024**2
    extra = ram_after - ram_before
    print(f"  الرامة الإضافية للذاكرة: {extra:.1f} م.ب")

    # معايير القبول
    ok = (
        mem.step_count == 100
        and 5 <= mem.m2_updates <= 7
        and mem.m3_updates > 0
        and mem.m4_updates == 0
        and extra < 300
    )
    print(f"  معيار القبول 2.1: {'✅ نجح' if ok else '❌ فشل'}")


def check_2_2() -> None:
    """مهمة 2.2: ربط الذاكرة بالنموذج — شكل، معاملات، تدرّجات."""
    print("\n" + "=" * 56)
    print("(2.2) ربط الذاكرة بنموذج المرحلة 1")
    print("=" * 56)

    V = 257
    base = TinySSM(V, 128, 64, 2, use_importance=True, use_memory=False)
    mem = TinySSM(V, 128, 64, 2, use_importance=True, use_memory=True)

    pb, pm = count_parameters(base), count_parameters(mem)
    inc = (pm - pb) / pb * 100
    print(f"  معاملات بلا ذاكرة: {pb} | بذاكرة: {pm} | الزيادة: {inc:.2f}%")

    x = torch.randint(0, V, (4, 64))
    out = mem(x)
    shape_ok = tuple(out.shape) == (4, 64, V)
    print(f"  شكل المخرج: {tuple(out.shape)} -> {'✅' if shape_ok else '❌'}")

    # تدفّق التدرّجات إلى معاملات الذاكرة
    y = torch.randint(0, V, (4, 64))
    loss = F.cross_entropy(out.reshape(-1, V), y.reshape(-1))
    loss.backward()
    g = sum(
        p.grad.norm().item()
        for n, p in mem.named_parameters()
        if n.startswith("memory.") and p.grad is not None
    )
    print(f"  مجموع نورم تدرّجات الذاكرة: {g:.4f} -> {'✅ تتدفّق' if g > 0 else '❌ صفر'}")


def check_live_layers() -> None:
    """تشخيص: كم طبقة ذاكرة تنشط فعلاً حسب طول التسلسل والعتبة."""
    print("\n" + "=" * 56)
    print("(تشخيص) الطبقات النشطة حسب الطول والعتبة")
    print("=" * 56)
    for seq, thr, big_k in [(128, 0.6, 256), (256, 0.6, 256), (256, 0.1, 128), (512, 0.1, 256)]:
        m = TinySSM(257, 128, 64, 2, use_importance=True, use_memory=True,
                    mem_threshold3=thr, mem_K=big_k)
        _ = m(torch.randint(0, 257, (2, seq)))
        print(f"  seq={seq:3d} thr3={thr} K={big_k}: "
              f"M2={m.memory.m2_updates} M3={m.memory.m3_updates} M4={m.memory.m4_updates}")
    print("  ملاحظة: مع العتبة الافتراضية 0.6 لا تنشط M3 أبداً لأن الأهمية المتعلّمة ≈0.12.")


if __name__ == "__main__":
    torch.manual_seed(0)
    check_2_1()
    check_2_2()
    check_live_layers()
