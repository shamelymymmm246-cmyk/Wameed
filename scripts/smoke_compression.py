"""smoke test لميزة الضغط في الذاكرة الهرمية"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from wameed.engine.ssm import TinySSM, count_parameters


def main():
    print("=== smoke test: ضغط M3/M4 بأربع نسب ===")
    for r in (1, 2, 4, 8):
        m = TinySSM(
            257, 128, 64, 2,
            use_importance=True, use_memory=True,
            mem_compress_ratio=r, mem_threshold3=0.1, mem_K=128,
        )
        p = count_parameters(m)
        out = m(torch.randint(0, 257, (2, 256)))
        loss = out.sum()
        loss.backward()
        g = sum(
            pp.grad.norm().item()
            for n, pp in m.named_parameters()
            if n.startswith("memory.") and pp.grad is not None
        )
        print(
            f"  r={r}: params={p} | m3_dim={m.memory.m3_dim} | "
            f"m4_dim={m.memory.m4_dim} | shape={tuple(out.shape)} | "
            f"mem_grad={g:.4f}"
        )

    print("\n=== smoke test: عدّادات التحديث (r=4, seq=256, thr3=0.1, K=128) ===")
    m = TinySSM(
        257, 128, 64, 2,
        use_importance=True, use_memory=True,
        mem_compress_ratio=4, mem_threshold3=0.1, mem_K=128,
    )
    _ = m(torch.randint(0, 257, (2, 256)))
    m.memory.print_update_counts()


if __name__ == "__main__":
    main()
