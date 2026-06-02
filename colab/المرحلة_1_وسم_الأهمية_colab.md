# خلية Colab للمرحلة 1 — وسم الأهمية

انسخ هذه الخلايا إلى Google Colab (كارت T4 مجاني) بعد رفع مجلد المشروع أو
استنساخه من GitHub. هذه الأرقام تُستخدم للتأكيد على عتاد أقوى؛ وقد شُغّلت
التجربة الكاملة أيضاً محلياً على المعالج (انظر `التوثيق/توثيق_المرحلة_1.md`).

```python
# خلية 1 — البيئة
!pip install -q torch datasets pandas psutil scipy
import torch
print("PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
```

```python
# خلية 2 — إحضار الكود
# ارفع مجلد Wameed إلى Drive ثم انسخه، أو استنسخه من GitHub:
# !git clone <رابط المستودع> Wameed
# %cd Wameed
```

```python
# خلية 3 — تدريب 6 تجارب (إعدادان × 3 بذور) بإعداد Colab الكامل
from pathlib import Path
!python scripts/train_importance.py \
  --mode six \
  --steps 5000 \
  --eval-batches 30 \
  --batch-size 32 \
  --seq-len 256 \
  --embed-dim 256 \
  --state-dim 128 \
  --num-layers 4 \
  --lr 3e-4 \
  --warmup 200 \
  --weight-decay 1e-2 \
  --max-train-chars 4000000 \
  --max-valid-chars 400000 \
  --csv-path results/experiments.csv \
  --summary-path results/stage1_summary_colab.json
```

```python
# خلية 4 — التحليل الإحصائي والقرار
!python scripts/analyze_stage1.py results/stage1_summary_colab.json
```

إذا كان التحسّن أقل من 1% أو الدرجات شبه ثابتة، شغّل الخطة البديلة (أ) — هدف
التناثر (sparsity loss):

```python
!python scripts/train_importance.py \
  --mode six --steps 5000 --eval-batches 30 \
  --batch-size 32 --seq-len 256 --embed-dim 256 --state-dim 128 --num-layers 4 \
  --lr 3e-4 --warmup 200 --weight-decay 1e-2 --sparsity-lambda 0.01 \
  --csv-path results/experiments.csv \
  --summary-path results/stage1_summary_colab_sparsity.json
!python scripts/analyze_stage1.py results/stage1_summary_colab_sparsity.json
```
