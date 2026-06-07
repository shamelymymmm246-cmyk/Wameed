"""توثيق مبدئي: عدد المعاملات وحجوم الذاكرة لكل نسبة ضغط."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch

from wameed.engine.ssm import TinySSM, count_parameters


def main():
    results = []
    base_params = None
    for r in (1, 2, 4, 8):
        m = TinySSM(
            257, 128, 64, 2,
            use_importance=True, use_memory=True,
            mem_compress_ratio=r, mem_threshold3=0.1, mem_K=128,
        )
        p = count_parameters(m)
        if base_params is None:
            base_params = p
        # تخزين M3 و M4 لكل token لكل batch
        m3_storage = m.memory.m3_dim
        m4_storage = m.memory.m4_dim
        # storage per batch (fixed per-sequence, independent of seq_len)
        storage_per_batch = m3_storage + m4_storage
        results.append({
            "compress_ratio": r,
            "total_params": p,
            "param_delta_vs_r1": p - base_params,
            "m3_dim": m3_storage,
            "m4_dim": m4_storage,
            "memory_storage_per_batch": storage_per_batch,
            "storage_saving_vs_r1_pct": (1 - storage_per_batch / (64 + 32)) * 100,
        })

    for row in results:
        print(json.dumps(row, ensure_ascii=False, indent=2))

    out_path = Path("results/stage3_param_table.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
