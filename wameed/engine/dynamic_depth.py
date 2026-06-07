"""
العمق الديناميكي (المرحلة 3، الإضافة (ج)).

DynamicDepthSSM: يلفّ TinySSM ويضيف:
- رأس ثقة بعد كل طبقة وسيطة.
- إسقاط تنبؤ مبكر (لكل طبقة وسيطة) لحساب خسارة مساعدة.
- منطق خروج مبكر أثناء التقييم: إذا الثقة > العتبة، نتوقف عن المرور بالطبقات التالية.

الفكرة:
- أثناء التدريب: نمرّ بكل الطبقات لحساب الخسارة الرئيسية + خسائر الخروج المساعدة.
- أثناء التقييم: قد نخرج مبكراً على التوكنات السهلة.

ملاحظة: حسب الخطة البديل (د)، النماذج ذات الطبقتين على الجهاز المحدود
لا تستفيد كثيراً. نضيف المنطق ونقيس بصدق.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from wameed.engine.ssm import TinySSM, count_parameters


class DynamicDepthSSM(nn.Module):
    """
    نموذج SSM بعمق ديناميكي (خروج مبكر قائم على الثقة).

    الاستخدام:
        base = TinySSM(vocab, ..., num_layers=2, use_importance=True)
        model = DynamicDepthSSM(base, exit_threshold=0.8, exit_loss_weight=0.1)
        logits, loss, layers_used = model(x, targets=y, use_dynamic_depth=False)
    """

    def __init__(
        self,
        base: TinySSM,
        exit_threshold: float = 0.8,
        exit_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if base.num_layers < 2:
            raise ValueError("العمق الديناميكي يحتاج ≥ 2 طبقات.")

        self.base = base
        self.num_layers = base.num_layers
        self.state_dim = base.state_dim
        self.vocab_size = base.vocab_size
        self.exit_threshold = exit_threshold
        self.exit_loss_weight = exit_loss_weight

        # رأس ثقة واحد لكل طبقة وسيطة (الطبقة الأخيرة دائماً تُستخدم).
        self.exit_heads = nn.ModuleList([
            nn.Linear(base.state_dim, 1) for _ in range(base.num_layers - 1)
        ])
        # إسقاط تنبؤ مبكر لكل طبقة وسيطة (لحساب الخسارة المساعدة).
        self.early_exit_projs = nn.ModuleList([
            nn.Linear(base.state_dim, base.vocab_size) for _ in range(base.num_layers - 1)
        ])

        # تهيئة رؤوس الثقة بانحياز متحفّظ (≈0.27 بعد sigmoid)
        # يمنع الخروج المبكر جداً في بداية التدريب.
        for head in self.exit_heads:
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, -1.0)

    def count_parameters(self) -> int:
        """عدد المعاملات الكلي."""
        return count_parameters(self)

    def _process_one_step(
        self,
        t: int,
        x_embed: torch.Tensor,
        states: list[torch.Tensor],
        all_importance_by_layer: list[list[torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        """
        يعالج خطوة زمنية واحدة عبر كل الطبقات.
        يُعيد (h_final, last_importance, intermediate_h's).
        intermediate_h's: قائمة h_t لكل طبقة وسيطة (لحساب رؤوس الخروج).
        """
        current_input = x_embed[:, t, :]
        last_importance: torch.Tensor | None = None
        intermediate_h: list[torch.Tensor] = []

        for layer_index in range(self.num_layers):
            input_part = self.base.input_projections[layer_index](current_input)
            state_part = self.base.state_projections[layer_index](states[layer_index])

            if self.base.use_importance:
                importance = torch.sigmoid(
                    self.base.importance_heads[layer_index](states[layer_index])
                )
                state_part = importance * state_part
                all_importance_by_layer[layer_index].append(importance.squeeze(-1))
                last_importance = importance

            h_t = torch.tanh(state_part + input_part)
            states[layer_index] = h_t
            current_input = h_t

            if layer_index < self.num_layers - 1:
                intermediate_h.append(h_t)

        return current_input, last_importance, intermediate_h

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        use_dynamic_depth: bool = False,
        return_layers_used: bool = False,
    ):
        """
        تمرير أمامي مع خرج ديناميكي.

        Args:
            tokens: (batch, seq_len).
            targets: (batch, seq_len) — للتدريب فقط.
            use_dynamic_depth: True أثناء التقييم لتفعيل الخروج المبكر.
            return_layers_used: True لإرجاع عدد الطبقات لكل توكن.

        Returns:
            logits: (batch, seq_len, vocab).
            loss: scalar (إذا targets موجود).
            layers_used: (batch, seq_len) int tensor (إذا return_layers_used=True).
        """
        if tokens.ndim != 2:
            raise ValueError("tokens يجب أن يكون (batch, seq_len).")

        batch_size, seq_len = tokens.shape
        device = tokens.device
        x_embed = self.base.embedding(tokens)

        states = [
            torch.zeros(batch_size, self.state_dim, device=device, dtype=x_embed.dtype)
            for _ in range(self.num_layers)
        ]
        if self.base.use_memory:
            self.base.memory.init_memory(batch_size, device)

        all_logits: list[torch.Tensor] = []
        all_layers_used: list[torch.Tensor] = []  # واحد لكل خطوة زمنية
        auxiliary_losses: list[torch.Tensor] = []
        all_importance_by_layer = [[] for _ in range(self.num_layers)]

        for t in range(seq_len):
            current_input, last_importance, intermediate_h = self._process_one_step(
                t, x_embed, states, all_importance_by_layer
            )

            # ====== منطق الخروج المبكر (تقييم فقط) — لكل توكن على حدة ======
            layers_used_step = torch.full(
                (batch_size,), self.num_layers, device=device, dtype=torch.long
            )
            if use_dynamic_depth and not self.training:
                # كل توكن له قراره. إذا خرج → يتجمّد عند هذه الطبقة.
                # التوكنات التي لم تخرج تستمر للطبقات التالية.
                for i, h in enumerate(intermediate_h):
                    conf = torch.sigmoid(self.exit_heads[i](h)).squeeze(-1)
                    exit_now = conf > self.exit_threshold  # (batch,)
                    if exit_now.any():
                        # استبدل current_input للتوكنات الخارجة بـ h
                        mask = exit_now.unsqueeze(-1)  # (batch, 1)
                        current_input = torch.where(mask, h, current_input)
                        # سجّل عدد الطبقات لهذه التوكنات (i+1)
                        new_count = torch.full_like(layers_used_step, i + 1)
                        layers_used_step = torch.where(exit_now, new_count, layers_used_step)

            # الذاكرة (إن فُعّلت) — تُحدَّث بأفضل تمثيل متاح.
            if self.base.use_memory:
                self.base.memory.update(current_input, last_importance)
                h_out = self.base.memory.read(current_input)
            else:
                h_out = current_input

            all_layers_used.append(layers_used_step)

            logits_t = self.base.output(h_out)
            all_logits.append(logits_t)

            # ====== خسارة الخروج المساعدة (تدريب فقط) ======
            if targets is not None and self.training:
                tgt_t = targets[:, t]  # (batch,)
                for i, h in enumerate(intermediate_h):
                    early_logits = self.early_exit_projs[i](h)  # (batch, vocab)
                    was_correct = (early_logits.argmax(-1) == tgt_t).float()
                    pred_conf = torch.sigmoid(self.exit_heads[i](h)).squeeze(-1)
                    # BCE على ثقة الرأس vs إشارة "هل كان التنبؤ المبكر صحيحاً"
                    exit_loss = F.binary_cross_entropy(
                        pred_conf, was_correct.detach(), reduction="mean"
                    )
                    auxiliary_losses.append(exit_loss)

        logits = torch.stack(all_logits, dim=1)  # (batch, seq_len, vocab)
        loss = None
        if targets is not None:
            main_loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size), targets.reshape(-1)
            )
            if auxiliary_losses:
                aux_total = torch.stack(auxiliary_losses).mean()
                loss = main_loss + self.exit_loss_weight * aux_total
            else:
                loss = main_loss

        if return_layers_used:
            layers_tensor = torch.stack(all_layers_used, dim=1)  # (batch, seq_len)
            return logits, loss, layers_tensor
        return logits, loss


def measure_avg_layers(
    model: DynamicDepthSSM,
    tokens: torch.Tensor,
    exit_threshold: float | None = None,
) -> tuple[float, torch.Tensor]:
    """
    يحسب متوسط الطبقات المستخدمة عبر التوكنات في دفعة.
    يُعيد (mean_layers, per_token_layers).
    """
    if exit_threshold is not None:
        old = model.exit_threshold
        model.exit_threshold = exit_threshold
    try:
        with torch.no_grad():
            model.eval()
            _, _, layers_used = model(tokens, use_dynamic_depth=True, return_layers_used=True)
    finally:
        if exit_threshold is not None:
            model.exit_threshold = old
    return layers_used.float().mean().item(), layers_used
