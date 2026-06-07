"""تشخيص: هل رؤوس الثقة تتعلم فعلاً؟"""
import sys
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.optim.lr_scheduler import LambdaLR

from wameed.engine.ssm import TinySSM
from wameed.engine.dynamic_depth import DynamicDepthSSM


def main():
    base = TinySSM(257, 128, 64, 4, use_importance=True, use_memory=True,
                   mem_threshold3=0.1, mem_K=128)
    model = DynamicDepthSSM(base, exit_threshold=0.5, exit_loss_weight=0.1)

    text = ("وميض نموذج لاختبار نماذج خفيفة. " * 20) * 30
    data = list(text.encode("utf-8", errors="ignore"))
    data.append(256)
    data = torch.tensor(data, dtype=torch.long)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    def lr_lambda(step):
        if step < 10:
            return step / 10
        return 0.5 * (1.0 + math.cos(math.pi * (step - 10) / 90))
    sched = LambdaLR(optimizer, lr_lambda)

    model.train()
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

    # فحص قيم الثقة المُتعلّمة
    print("=== قيم exit_heads بعد التدريب ===\n")
    for i, head in enumerate(model.exit_heads):
        weight_norm = head.weight.norm().item()
        bias_val = head.bias.item()
        sigmoid_bias = torch.sigmoid(torch.tensor(bias_val)).item()
        print(f"  exit_head[{i}]: |W|={weight_norm:.4f} | bias={bias_val:+.4f} "
              f"| sigmoid(bias)={sigmoid_bias:.4f}")

    # قياس الثقة على دفعة جديدة
    print("\n=== قياس الثقة على دفعة جديدة ===")
    model.eval()
    with torch.no_grad():
        starts = torch.randint(0, data.numel() - 130, (4,))
        x = torch.stack([data[s:s+128] for s in starts])
        # نستخرج الثقة لكل طبقة
        for i in range(model.num_layers - 1):
            h_at_i = model._process_one_step(
                0, model.base.embedding(x),
                [torch.zeros(4, 64) for _ in range(4)],
                [[] for _ in range(4)],
            )[2][i]
            conf = torch.sigmoid(model.exit_heads[i](h_at_i)).squeeze(-1)
            print(f"  Layer {i+1}: mean_conf={conf.mean().item():.4f} | "
                  f"max={conf.max().item():.4f} | min={conf.min().item():.4f} | "
                  f"std={conf.std().item():.4f}")

    # فحص بنية الـ auxiliary loss
    print("\n=== اختبار: هل الإشارة 'was_correct' تعطي 1 للتوكنات السهلة؟ ===")
    with torch.no_grad():
        starts = torch.randint(0, data.numel() - 130, (4,))
        x = torch.stack([data[s:s+128] for s in starts])
        y = torch.stack([data[s+1:s+129] for s in starts])
        # تشغيل الطبقات الأولى
        current_input, last_importance, intermediate_h = model._process_one_step(
            0, model.base.embedding(x),
            [torch.zeros(4, 64) for _ in range(4)],
            [[] for _ in range(4)],
        )
        # للطبقة الأولى
        early_logits = model.early_exit_projs[0](intermediate_h[0])
        was_correct = (early_logits.argmax(-1) == y[:, 0]).float()
        print(f"  Layer 1 predictions: was_correct mean={was_correct.mean().item():.3f}, "
              f"any correct={was_correct.any().item()}, "
              f"all correct={was_correct.all().item()}")


if __name__ == "__main__":
    main()
