#!/usr/bin/env python3
"""المهمة 1.4 — التحليل الإحصائي لنتائج وسم الأهمية.

يقرأ ملخّص التجارب (stage1_summary.json) الناتج عن train_importance.py،
ويحسب: المتوسط والانحراف لكل إعداد، نسبة التحسّن، واختبار t لعيّنتين،
ثم يطبع قرار نقطة الفحص 1 (تحسّن ≥ 1% ويُفضَّل p < 0.1).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def load_rows(summary_path: Path) -> list[dict]:
    """يحمّل صفوف التجارب من ملف الملخّص."""
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return data["rows"]


def main() -> None:
    summary_path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/stage1_summary.json")
    rows = load_rows(summary_path)

    base = np.array([r["perplexity"] for r in rows if r["experiment"] == "base"], dtype=float)
    imp = np.array([r["perplexity"] for r in rows if r["experiment"] == "importance"], dtype=float)

    if base.size == 0 or imp.size == 0:
        print("❌ لم أجد صفوف base و importance معاً في الملخّص.")
        sys.exit(1)

    print("=" * 56)
    print("📊 نتائج المرحلة 1 — وسم الأهمية (perplexity، الأقل أفضل)")
    print("=" * 56)
    print(f"مصدر البيانات: {rows[0].get('data_source', 'غير معروف')}")
    print(f"\nبلا وسم: قيم={np.round(base, 2).tolist()}")
    print(f"         متوسط={base.mean():.2f} ± {base.std(ddof=0):.2f}")
    print(f"بوسم:    قيم={np.round(imp, 2).tolist()}")
    print(f"         متوسط={imp.mean():.2f} ± {imp.std(ddof=0):.2f}")

    improvement = (base.mean() - imp.mean()) / base.mean() * 100
    print(f"\nنسبة التحسّن: {improvement:.2f}%  (المطلوب ≥ 1%)")

    # اختبار t لعيّنتين مستقلتين.
    t_stat, p_val = stats.ttest_ind(base, imp)
    print(f"اختبار t: t={t_stat:.3f} | p={p_val:.4f}  (المقبول < 0.1)")

    # إحصاءات الوسم أثناء التقييم (من أول صف importance).
    imp_rows = [r for r in rows if r["experiment"] == "importance"]
    st = next((r.get("importance_stats") for r in imp_rows if r.get("importance_stats")), None)
    if st:
        print(f"\nإحصاءات s_t أثناء التقييم: mean={st['mean']:.3f} "
              f"min={st['min']:.3f} max={st['max']:.3f} std={st['std']:.3f}")
        spread = st["max"] - st["min"]
        print(f"  مدى الانتشار (max-min)={spread:.3f} -> "
              + ("توزّعت بشكل جيد" if spread > 0.05 else "شبه ثابتة (قد تحتاج sparsity)"))

    print("\n" + "=" * 56)
    if improvement >= 1.0 and p_val < 0.1:
        print("✅ قرار نقطة الفحص 1: الوسم يُحسِّن بشكل معنوي → انتقل للمرحلة 2")
    elif improvement >= 1.0 and p_val >= 0.1:
        print("⚠️ تحسّن ≥ 1% لكن p ≥ 0.1 (غير مؤكّد إحصائياً) → زِد الخطوات/البذور")
    else:
        print("❌ التحسّن < 1% → طبّق الخطط البديلة (sparsity / تبسيط)")
    print("=" * 56)


if __name__ == "__main__":
    main()
