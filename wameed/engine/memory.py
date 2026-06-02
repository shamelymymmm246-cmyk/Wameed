"""
وحدة الذاكرة الهرمية لوميض
أربع طبقات: قصيرة (M1) → عاملة (M2) → دلالية (M3) → مجرّدة (M4)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HierarchicalMemory(nn.Module):
    """
    ذاكرة هرمية من أربع طبقات تُضاف فوق نموذج SSM.

    المبدأ: كل طبقة أبطأ تحديثاً وأكثر تجريداً من التي تحتها،
    مثل دماغ يحتفظ بالتفاصيل لحظياً وبالخلاصة طويلاً.
    """

    def __init__(
        self,
        state_dim: int,
        k: int = 16,
        K: int = 256,
        threshold3: float = 0.6,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.k = k              # تردّد تحديث M2: كل k خطوة.
        self.K = K              # تردّد تحديث M4: كل K خطوة.
        self.thr3 = threshold3  # عتبة الأهمية لتحديث M3.

        d = state_dim

        # --- بوابات الكتابة (Write Gates) ---
        # تتحكّم في كمية المعلومات الجديدة التي تدخل كل طبقة.
        self.gate2 = nn.Linear(d, d)
        self.gate3 = nn.Linear(d, d)
        self.gate4 = nn.Linear(d, d // 2)

        # --- طبقات التلخيص (Summary Layers) ---
        # تضغط M2/M3 قبل تخزينها في الطبقة التالية.
        self.summary3 = nn.Linear(d, d)
        self.summary4 = nn.Linear(d, d // 2)

        # --- طبقات الإسقاط للقراءة (Read Projections) ---
        # توحّد أبعاد M1..M4 ليتمكّن النموذج من الدمج.
        self.proj1 = nn.Linear(d, d)
        self.proj2 = nn.Linear(d, d)
        self.proj3 = nn.Linear(d, d)
        self.proj4 = nn.Linear(d // 2, d)

        # طبقة الدمج النهائي: تأخذ [h, r1, r2, r3, r4] وتُعيد متجهاً بحجم D.
        self.fusion = nn.Linear(5 * d, d)

        # تهيئة بوابات الكتابة بتحيّز إيجابي لضمان الكتابة الفعلية منذ البداية.
        # sigmoid(1.0) ≈ 0.73 — يسمح بدخول ~73% من المعلومات الجديدة.
        nn.init.constant_(self.gate2.bias, 1.0)
        nn.init.constant_(self.gate3.bias, 0.5)

        # الحالة الداخلية — تُهيَّأ بـ init_memory() قبل كل تسلسل.
        self.M1: torch.Tensor | None = None  # قصيرة المدى.
        self.M2: torch.Tensor | None = None  # العاملة.
        self.M3: torch.Tensor | None = None  # الدلالية.
        self.M4: torch.Tensor | None = None  # المجرّدة.

        # عدّادات التتبّع — للتحقّق من جداول التحديث.
        self.step_count = 0
        self.m2_updates = 0
        self.m3_updates = 0
        self.m4_updates = 0

        # بفرات (Buffers) لحساب المتوسطات المتحركة.
        self._m1_buffer: list[torch.Tensor] = []
        self._imp_buffer: list[torch.Tensor] = []

    def init_memory(self, batch_size: int, device: torch.device) -> None:
        """تُصفّر الذاكرة الأربع في بداية كل تسلسل جديد."""
        d = self.state_dim
        self.M1 = torch.zeros(batch_size, d, device=device)
        self.M2 = torch.zeros(batch_size, d, device=device)
        self.M3 = torch.zeros(batch_size, d, device=device)
        self.M4 = torch.zeros(batch_size, d // 2, device=device)
        self.step_count = 0
        self._m1_buffer.clear()
        self._imp_buffer.clear()

    def update(self, h_t: torch.Tensor, importance_t: torch.Tensor | None = None) -> None:
        """
        يُحدّث الطبقات الأربع حسب جداول التحديث.

        h_t:          (batch, state_dim) — الحالة الحالية للطبقة الأخيرة.
        importance_t: (batch, 1) — درجة الأهمية من وسم المرحلة 1 (اختياري).

        ملاحظة: نستخدم .detach() على M1 لقطع BPTT الطويل عبر التاريخ.
        هذا تصميم مقصود وليس خطأ — يمنع انفجار التدرّجات على النصوص الطويلة.
        """
        if self.M1 is None:
            raise RuntimeError("يجب استدعاء init_memory() قبل update().")

        self.step_count += 1

        # --- الطبقة 1 (M1): كل خطوة — تخزّن الحالة الفورية ---
        self.M1 = h_t.detach()
        self._m1_buffer.append(self.M1.clone())
        if len(self._m1_buffer) > self.k:
            self._m1_buffer.pop(0)

        # --- الطبقة 2 (M2): كل k خطوة — متوسط نافذة الـ k الأخيرة ---
        if self.step_count % self.k == 0 and self._m1_buffer:
            self.m2_updates += 1
            summary = torch.stack(self._m1_buffer, dim=0).mean(dim=0)
            g2 = torch.sigmoid(self.gate2(self.M1))
            self.M2 = (1.0 - g2) * self.M2 + g2 * summary

        # --- الطبقة 3 (M3): عند ارتفاع الأهمية فوق العتبة ---
        if importance_t is not None:
            self._imp_buffer.append(importance_t.detach().squeeze(-1))
            if len(self._imp_buffer) > self.k:
                self._imp_buffer.pop(0)

            avg_imp = torch.stack(self._imp_buffer, dim=0).mean(dim=0).mean()
            if avg_imp.item() > self.thr3:
                self.m3_updates += 1
                g3 = torch.sigmoid(self.gate3(self.M2))
                self.M3 = (1.0 - g3) * self.M3 + g3 * torch.tanh(self.summary3(self.M2))

        # --- الطبقة 4 (M4): كل K خطوة — خلاصة الخلاصة ---
        if self.step_count % self.K == 0:
            self.m4_updates += 1
            g4 = torch.sigmoid(self.gate4(self.M3))
            self.M4 = (1.0 - g4) * self.M4 + g4 * torch.tanh(self.summary4(self.M3))

    def read(self, h_t: torch.Tensor) -> torch.Tensor:
        """
        يُنتج تمثيلاً مدمجاً من h_t والطبقات الأربع.

        المدخل:  h_t  (batch, state_dim)
        المخرج:  h_out (batch, state_dim) — يحمل معلومات تاريخية
        """
        if self.M1 is None:
            raise RuntimeError("يجب استدعاء init_memory() قبل read().")

        r1 = torch.tanh(self.proj1(self.M1))
        r2 = torch.tanh(self.proj2(self.M2))
        r3 = torch.tanh(self.proj3(self.M3))
        r4 = torch.tanh(self.proj4(self.M4))

        # نسلسل h مع قراءات الطبقات الأربع ثم ندمجها بطبقة خطية.
        combined = torch.cat([h_t, r1, r2, r3, r4], dim=-1)  # (batch, 5*D)
        return torch.tanh(self.fusion(combined))

    def print_update_counts(self) -> None:
        """يطبع عدّادات التحديث للتحقّق من صحة الجداول."""
        print(f"  طبقة 1 (M1): {self.step_count} تحديث")
        print(f"  طبقة 2 (M2): {self.m2_updates} تحديث")
        print(f"  طبقة 3 (M3): {self.m3_updates} تحديث")
        print(f"  طبقة 4 (M4): {self.m4_updates} تحديث")
