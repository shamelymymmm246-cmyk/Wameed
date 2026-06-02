from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from wameed.engine.memory import HierarchicalMemory


@dataclass
class TinySSMOutput:
    """ناتج النموذج عند طلب درجات الأهمية صراحة."""

    logits: torch.Tensor               # توقعات النموذج لكل توكن.
    importance_scores: torch.Tensor | None  # درجات الأهمية إن كان الوسم مفعلاً.


class TinySSM(nn.Module):
    """
    نموذج SSM صغير جداً للتجارب على المعالج فقط.
    يدعم ثلاثة أوضاع:
    1. أساسي (use_importance=False, use_memory=False)
    2. وسم الأهمية (use_importance=True)  — إضافة المرحلة 1
    3. الذاكرة الهرمية (use_memory=True) — إضافة المرحلة 2 (يوصى بتفعيل use_importance أيضاً)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        state_dim: int = 64,
        num_layers: int = 1,
        use_importance: bool = False,
        use_memory: bool = False,
        mem_k: int = 16,
        mem_K: int = 256,
        mem_threshold3: float = 0.6,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers يجب أن يكون 1 أو أكثر.")

        self.vocab_size = vocab_size    # حجم المفردات: عدد التوكنات الممكنة.
        self.embed_dim = embed_dim      # بُعد التضمين: حجم متجه كل توكن.
        self.state_dim = state_dim      # بُعد الحالة: حجم الذاكرة المتدحرجة.
        self.num_layers = num_layers    # عدد طبقات SSM المتتابعة.
        self.use_importance = use_importance  # يفعّل بوابة وسم الأهمية من المرحلة 1.
        self.use_memory = use_memory          # يفعّل الذاكرة الهرمية من المرحلة 2.

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # يحوّل أرقام التوكنات إلى متجهات.
        self.input_projections = nn.ModuleList()   # طبقات B: تُدخل x_t إلى مساحة الحالة.
        self.state_projections = nn.ModuleList()   # طبقات A: تُحدّث الحالة القديمة h_{t-1}.
        self.importance_heads = nn.ModuleList()    # رؤوس صغيرة لحساب s_t من الحالة.

        for layer_index in range(num_layers):
            # الطبقة الأولى تقرأ التضمين، والباقي تقرأ حالة الطبقة السابقة.
            layer_input_dim = embed_dim if layer_index == 0 else state_dim
            self.input_projections.append(nn.Linear(layer_input_dim, state_dim))
            self.state_projections.append(nn.Linear(state_dim, state_dim, bias=False))
            if use_importance:
                self.importance_heads.append(nn.Linear(state_dim, 1))

        # رأس الخرج: يُسقط آخر حالة (أو خرج الذاكرة) على المفردات.
        self.output = nn.Linear(state_dim, vocab_size)

        # وحدة الذاكرة الهرمية — تُربط تلقائياً بالنموذج كـ nn.Module فرعي.
        # نمرّر k و K و threshold3 لتمكين ضبطها من التجارب (الخطة البديلة ب).
        if use_memory:
            self.memory = HierarchicalMemory(
                state_dim=state_dim,
                k=mem_k,
                K=mem_K,
                threshold3=mem_threshold3,
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """يضبط الأوزان بقيم صغيرة حتى لا تنفجر الحالة في البداية."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for input_projection in self.input_projections:
            nn.init.xavier_uniform_(input_projection.weight)
            nn.init.zeros_(input_projection.bias)
        for state_projection in self.state_projections:
            nn.init.orthogonal_(state_projection.weight)
        for importance_head in self.importance_heads:
            nn.init.zeros_(importance_head.weight)
            # انحياز -2 يعطي s≈0.12 ويمنع تجميد الحالة منذ البداية.
            nn.init.constant_(importance_head.bias, -2.0)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        return_importance: bool = False,
    ) -> torch.Tensor | TinySSMOutput:
        """
        يمرر دفعة توكنات ويعيد logits بالشكل (batch, seq_len, vocab).

        بنية المعالجة:
        - الحلقة الخارجية على الزمن (خطوة بخطوة) — ضروري لتراكم الذاكرة.
        - الحلقة الداخلية على الطبقات — نفس المنطق السابق.
        - بعد آخر طبقة لكل خطوة: تحديث الذاكرة ثم القراءة منها.
        """
        if tokens.ndim != 2:
            raise ValueError("tokens يجب أن يكون بالشكل (batch, seq_len).")

        batch_size, seq_len = tokens.shape
        device = tokens.device

        x_embed = self.embedding(tokens)  # (batch, seq_len, embed_dim)

        # تهيئة حالات الطبقات بالأصفار لكل تسلسل جديد.
        states: list[torch.Tensor] = [
            torch.zeros(batch_size, self.state_dim, device=device, dtype=x_embed.dtype)
            for _ in range(self.num_layers)
        ]

        # تهيئة الذاكرة الهرمية إن كانت مفعّلة.
        if self.use_memory:
            self.memory.init_memory(batch_size, device)

        all_h_out: list[torch.Tensor] = []
        # قوائم لجمع درجات الأهمية لكل طبقة عبر الزمن.
        all_importance_by_layer: list[list[torch.Tensor]] = [
            [] for _ in range(self.num_layers)
        ]

        for t in range(seq_len):
            # المدخل للطبقة الأولى هو تضمين التوكن الحالي.
            current_input = x_embed[:, t, :]  # (batch, embed_dim)

            last_importance: torch.Tensor | None = None  # درجة الأهمية من آخر طبقة.

            for layer_index in range(self.num_layers):
                input_part = self.input_projections[layer_index](current_input)  # B·x_t
                state_part = self.state_projections[layer_index](states[layer_index])  # A·h_{t-1}

                if self.use_importance:
                    # s_t: درجة أهمية مشتقة من الحالة السابقة — تعمل كبوابة على A·h.
                    importance = torch.sigmoid(
                        self.importance_heads[layer_index](states[layer_index])
                    )
                    state_part = importance * state_part
                    all_importance_by_layer[layer_index].append(importance.squeeze(-1))
                    # نحتفظ بأهمية آخر طبقة لتمريرها إلى وحدة الذاكرة.
                    last_importance = importance

                h_t = torch.tanh(state_part + input_part)  # h_t = tanh(s·A·h + B·x)
                states[layer_index] = h_t
                current_input = h_t  # خرج هذه الطبقة يصبح مدخل الطبقة التالية.

            # current_input الآن = خرج آخر طبقة SSM لهذه الخطوة الزمنية.
            if self.use_memory:
                # نكتب h_t في الذاكرة (مع درجة الأهمية إن وُجدت).
                self.memory.update(current_input, last_importance)
                # نقرأ من الذاكرة للحصول على تمثيل أغنى بالسياق التاريخي.
                h_out = self.memory.read(current_input)
            else:
                h_out = current_input

            all_h_out.append(h_out)

        # نجمع خرج كل الخطوات في مصفوفة واحدة.
        output_tensor = torch.stack(all_h_out, dim=1)  # (batch, seq_len, state_dim)
        logits = self.output(output_tensor)             # (batch, seq_len, vocab_size)

        if return_importance:
            if all_importance_by_layer[0]:
                # نبني مصفوفة الأهمية: (batch, num_layers, seq_len).
                scores_per_layer = [
                    torch.stack(layer_scores, dim=1)
                    for layer_scores in all_importance_by_layer
                ]
                scores = torch.stack(scores_per_layer, dim=1)
            else:
                scores = None
            return TinySSMOutput(logits=logits, importance_scores=scores)

        return logits


def count_parameters(model: nn.Module) -> int:
    """يرجع عدد المعاملات القابلة للتعلّم في النموذج."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
