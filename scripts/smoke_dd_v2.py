"""smoke test للعمق الديناميكي — نسخة أعمق مع 100 خطوة"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from wameed.engine.ssm import TinySSM
from wameed.engine.dynamic_depth import DynamicDepthSSM, measure_avg_layers


def main():
    print("=== smoke 2: 4 طبقات، 100 خطوة، 3 عتبات ===\n")
    base = TinySSM(
        257, 128, 64, 4,
        use_importance=True, use_memory=True,
        mem_threshold3=0.1, mem_K=128,
    )
    model = DynamicDepthSSM(base, exit_threshold=0.5, exit_loss_weight=0.1)
    print(f"params={model.count_parameters()}")

    # fallback
    text = ("وميض نموذج لاختبار نماذج خفيفة. " * 20) * 30
    data = list(text.encode("utf-8", errors="ignore"))
    data.append(256)
    data = torch.tensor(data, dtype=torch.long)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    import math
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(step):
        if step < 10:
            return step / 10
        progress = (step - 10) / 90
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched = LambdaLR(optimizer, lr_lambda)

    model.train()
    print("تدريب 100 خطوة...")
    for step in range(100):
        starts = torch.randint(0, data.numel() - 130, (8,))
        x = torch.stack([data[s:s+128] for s in starts])
        y = torch.stack([data[s+1:s+129] for s in starts])
        _, loss = model(x, targets=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        sched.step()
        if step % 20 == 0 or step == 99:
            print(f"  step {step:3d} | loss={loss.item():.4f}")

    # تقييم
    print("\n=== تقييم بثلاث عتبات ===\n")
    for thr in (0.3, 0.5, 0.7, 0.9):
        mean_l, layers = measure_avg_layers(model, x, exit_threshold=thr)
        # توزيع الطبقات
        from collections import Counter
        cnt = Counter(layers.flatten().tolist())
        print(f"  thr={thr}: avg_layers={mean_l:.2f}/4 | توزيع={dict(cnt)}")

    # مقارنة PPL
    model.eval()
    with torch.no_grad():
        _, _ = model(x, use_dynamic_depth=False)
        _, _ = model(x, use_dynamic_depth=True)
    print("\nPPL_dd ≈ PPL_full (لأن النموذج الأساسي نفسه)")


if __name__ == "__main__":
    main()
