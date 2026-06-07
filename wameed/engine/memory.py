"""
وحدة الذاكرة الهرمية لوميض
أربع طبقات: قصيرة (M1) → عاملة (M2) → دلالية (M3) → مجرّدة (M4)

الإضافة (المرحلة 3): خيار ضغط M3 و M4 بإسقاط خطي قابل للضبط.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HierarchicalMemory(nn.Module):
    """
    ذاكرة هرمية من أربع طبقات تُضاف فوق نموذج SSM.

    المبدأ: كل طبقة أبطأ تحديثاً وأكثر تجريداً من التي تحتها،
    مثل دماغ يحتفظ بالتفاصيل لحظياً وبالخلاصة طويلاً.

    إضافة (المرحلة 3): `compress_ratio` يضغط M3 و M4 في الذاكرة.
    - r=1: بلا ضغط (سلوك المرحلة 2 تماماً).
    - r=2: M3 من d إلى d/2، M4 من d/2 إلى d/4.
    - r=4: M3 من d إلى d/4، M4 من d/2 إلى d/8.
    - r=8: M3 من d إلى d/8، M4 من d/2 إلى d/16.
    """

    def __init__(
        self,
        state_dim: int,
        k: int = 16,
        K: int = 256,
        threshold3: float = 0.6,
        compress_ratio: int = 1,
    ) -> None:
        super().__init__()
        if compress_ratio < 1:
            raise ValueError("compress_ratio يجب أن يكون ≥ 1 (1 = بلا ضغط).")

        self.state_dim = state_dim
        self.k = k              # تردّد تحديث M2: كل k خطوة.
        self.K = K              # تردّد تحديث M4: كل K خطوة.
        self.thr3 = threshold3  # عتبة الأهمية لتحديث M3.
        self.compress_ratio = compress_ratio  # نسبة ضغط M3/M4 (1=بلا ضغط).

        d = state_dim
        r = compress_ratio

        # الأبعاد المضغوطة لـ M3 و M4.
        self.m3_dim = d // r
        self.m4_dim = (d // 2) // r

        # --- بوابات الكتابة (Write Gates) ---
        # تعمل على البُعد الأصلي (قبل الضغط) لأن الإدخال من M2 بحجم d.
        self.gate2 = nn.Linear(d, d)
        # gate3 يُنتج m3_dim ليطابق M3 المضغوط (وكذلك في r=1 يكون d = m3_dim).
        self.gate3 = nn.Linear(d, self.m3_dim)
        # gate4 يُنتج m4_dim ليطابق M4 المضغوط (في r=1: d//2 == m4_dim).
        self.gate4 = nn.Linear(d, self.m4_dim)

        # --- طبقات التلخيص (Summary Layers) ---
        # تنتج بحجم d و d//2 قبل الضغط.
        self.summary3 = nn.Linear(d, d)
        self.summary4 = nn.Linear(d, d // 2)

        # --- طبقات الإسقاط للقراءة (Read Projections) ---
        # تعمل على البُعد الأصلي (بعد فكّ الضغط).
        self.proj1 = nn.Linear(d, d)
        self.proj2 = nn.Linear(d, d)
        self.proj3 = nn.Linear(d, d)
        self.proj4 = nn.Linear(d // 2, d)

        # --- طبقات الضغط/فكّ الضغط (المرحلة 3) ---
        if r > 1:
            # ضغط بعد الـ summary: d → m3_dim  و  d//2 → m4_dim
            self.compress_m3 = nn.Linear(d, self.m3_dim)
            self.compress_m4 = nn.Linear(d // 2, self.m4_dim)
            # فكّ الضغط للقراءة: m3_dim → d  و  m4_dim → d//2
            self.decompress_m3 = nn.Linear(self.m3_dim, d)
            self.decompress_m4 = nn.Linear(self.m4_dim, d // 2)
        else:
            self.compress_m3 = None
            self.compress_m4 = None
            self.decompress_m3 = None
            self.decompress_m4 = None

        # طبقة الدمج النهائي: تأخذ [h, r1, r2, r3, r4] وتُعيد متجهاً بحجم D.
        self.fusion = nn.Linear(5 * d, d)

        # تهيئة بوابات الكتابة بتحيّز إيجابي لضمان الكتابة الفعلية منذ البداية.
        # sigmoid(1.0) ≈ 0.73 — يسمح بدخول ~73% من المعلومات الجديدة.
        nn.init.constant_(self.gate2.bias, 1.0)
        nn.init.constant_(self.gate3.bias, 0.5)

        # الحالة الداخلية — تُهيَّأ بـ init_memory() قبل كل تسلسل.
        self.M1: torch.Tensor | None = None  # قصيرة المدى (d).
        self.M2: torch.Tensor | None = None  # العاملة (d).
        self.M3: torch.Tensor | None = None  # الدلالية (m3_dim).
        self.M4: torch.Tensor | None = None  # المجرّدة (m4_dim).

        # عدّادات التتبّع — للتحقّق من جداول التحديث.
        self.step_count = 0
        self.m2_updates = 0
        self.m3_updates = 0
        self.m4_updates = 0

        # بفرات (Buffers) لحساب المتوسطات المتحركة.
        self._m1_buffer: list[torch.Tensor] = []
        self._imp_buffer: list[torch.Tensor] = []

    def _compress(self, x: torch.Tensor, layer: str) -> torch.Tensor:
        """يضغط متجهاً قبل تخزينه. layer ∈ {'m3','m4'}."""
        if self.compress_ratio == 1:
            return x
        lin = self.compress_m3 if layer == "m3" else self.compress_m4
        return torch.tanh(lin(x))

    def _decompress(self, x: torch.Tensor, layer: str) -> torch.Tensor:
        """يفكّ ضغط متجه قبل القراءة. layer ∈ {'m3','m4'}."""
        if self.compress_ratio == 1:
            return x
        lin = self.decompress_m3 if layer == "m3" else self.decompress_m4
        return torch.tanh(lin(x))

    def init_memory(self, batch_size: int, device: torch.device) -> None:
        """تُصفّر الذاكرة الأربع في بداية كل تسلسل جديد."""
        d = self.state_dim
        self.M1 = torch.zeros(batch_size, d, device=device)
        self.M2 = torch.zeros(batch_size, d, device=device)
        self.M3 = torch.zeros(batch_size, self.m3_dim, device=device)
        self.M4 = torch.zeros(batch_size, self.m4_dim, device=device)
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
                # نلخّص أولاً (d-dim) ثم نضغط (m3_dim) قبل التخزين.
                summary3_raw = torch.tanh(self.summary3(self.M2))
                summary3_comp = self._compress(summary3_raw, "m3")
                self.M3 = (1.0 - g3) * self.M3 + g3 * summary3_comp

        # --- الطبقة 4 (M4): كل K خطوة — خلاصة الخلاصة ---
        if self.step_count % self.K == 0:
            self.m4_updates += 1
            # نفكّ ضغط M3 إلى d قبل تغذية البوابة والـ summary (مُعرّفتان على d).
            m3_for_gates = self._decompress(self.M3, "m3") if self.compress_ratio > 1 else self.M3
            g4 = torch.sigmoid(self.gate4(m3_for_gates))
            summary4_raw = torch.tanh(self.summary4(m3_for_gates))
            summary4_comp = self._compress(summary4_raw, "m4")
            self.M4 = (1.0 - g4) * self.M4 + g4 * summary4_comp

    def read(self, h_t: torch.Tensor) -> torch.Tensor:
        """
        يُنتج تمثيلاً مدمجاً من h_t والطبقات الأربع.

        المدخل:  h_t  (batch, state_dim)
        المخرج:  h_out (batch, state_dim) — يحمل معلومات تاريخية
        """
        if self.M1 is None:
            raise RuntimeError("يجب استدعاء init_memory() قبل read().")

        # نفكّ ضغط M3 و M4 لتصبح بنفس حجم القراءة (d, d//2).
        if self.compress_ratio > 1:
            m3_for_read = self._decompress(self.M3, "m3")
            m4_for_read = self._decompress(self.M4, "m4")
        else:
            m3_for_read = self.M3
            m4_for_read = self.M4

        r1 = torch.tanh(self.proj1(self.M1))
        r2 = torch.tanh(self.proj2(self.M2))
        r3 = torch.tanh(self.proj3(m3_for_read))
        r4 = torch.tanh(self.proj4(m4_for_read))

        # نسلسل h مع قراءات الطبقات الأربع ثم ندمجها بطبقة خطية.
        combined = torch.cat([h_t, r1, r2, r3, r4], dim=-1)  # (batch, 5*D)
        return torch.tanh(self.fusion(combined))

    def print_update_counts(self) -> None:
        """يطبع عدّادات التحديث للتحقّق من صحة الجداول."""
        print(f"  طبقة 1 (M1): {self.step_count} تحديث")
        print(f"  طبقة 2 (M2): {self.m2_updates} تحديث")
        print(f"  طبقة 3 (M3): {self.m3_updates} تحديث | بُعد={self.m3_dim}")
        print(f"  طبقة 4 (M4): {self.m4_updates} تحديث | بُعد={self.m4_dim}")
        print(f"  نسبة الضغط: {self.compress_ratio}:1")
