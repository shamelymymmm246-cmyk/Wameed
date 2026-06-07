"""smoke test للعمق الديناميكي"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from wameed.engine.ssm import TinySSM
from wameed.engine.dynamic_depth import DynamicDepthSSM, measure_avg_layers


def main():
    print("=== smoke: DynamicDepthSSM بـ 2 و 4 طبقات ===\n")
    for num_layers in (2, 3, 4):
        base = TinySSM(
            257, 128, 64, num_layers,
            use_importance=True, use_memory=True,
            mem_threshold3=0.1, mem_K=128,
        )
        model = DynamicDepthSSM(base, exit_threshold=0.8, exit_loss_weight=0.1)
        p = model.count_parameters()
        x = torch.randint(0, 257, (2, 64))
        y = torch.randint(0, 257, (2, 64))

        # تدريب
        model.train()
        logits, loss = model(x, targets=y)
        print(f"  num_layers={num_layers} | params={p} | "
              f"train logits={tuple(logits.shape)} | train loss={loss.item():.4f}")

        # تقييم بدون DD
        model.eval()
        with torch.no_grad():
            logits, _ = model(x, use_dynamic_depth=False)
            print(f"  eval no-DD: logits={tuple(logits.shape)}")

        # تقييم بـ DD
        mean_layers, per_token = measure_avg_layers(model, x, exit_threshold=0.5)
        print(f"  eval DD (thr=0.5): mean_layers={mean_layers:.2f} / {num_layers} | "
              f"per_token_first_batch={per_token[0, :10].tolist()}")
        mean_layers_hi, _ = measure_avg_layers(model, x, exit_threshold=0.99)
        print(f"  eval DD (thr=0.99): mean_layers={mean_layers_hi:.2f} / {num_layers}")

        # اختبار backward عبر رؤوس الخروج
        loss.backward()
        for n, p in model.named_parameters():
            if "exit_head" in n and p.grad is not None:
                print(f"  exit_head grad: {n} = {p.grad.norm().item():.4f}")
                break
        model.zero_grad()
        print()


if __name__ == "__main__":
    main()
