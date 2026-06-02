# خلية Colab للمرحلة 2 — الذاكرة الهرمية

> الغرض: تشغيل اختبار التذكّر على **تسلسلات طويلة (512 توكن)** على كارت
> T4 المجاني. الذاكرة الهرمية — حسب التصميم — تظهر فائدتها على النصوص
> الطويلة تحديداً (انظر «الخطة البديلة أ»). جهاز أبو دجانة (معالج بطيء،
> رامة ~3.4غ) لا يحتمل حلقة زمنية بطول 512، لذلك يُجرى هذا الاختبار على Colab.
>
> على الجهاز جرى اختبار قصير عادل (seq=256) موثّق في
> `التوثيق/توثيق_المرحلة_2.md`.

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
# خلية 3 — التجربة الكاملة على تسلسل طويل (512)
# عند seq=512 تنشط الطبقات الأربع كلها:
#   M2 كل 16، M3 عند الأهمية (threshold3 منخفض)، M4 كل 256 (مرتان).
!python scripts/train_memory.py \
  --mode six \
  --steps 4000 \
  --eval-batches 20 \
  --batch-size 16 \
  --seq-len 512 \
  --embed-dim 256 \
  --state-dim 128 \
  --num-layers 4 \
  --lr 3e-4 \
  --warmup 200 \
  --weight-decay 1e-2 \
  --mem-k 16 \
  --mem-big-k 256 \
  --threshold3 0.1 \
  --max-train-chars 4000000 \
  --max-valid-chars 400000 \
  --csv-path results/experiments.csv \
  --summary-path results/stage2_summary_colab.json
```

```python
# خلية 4 — عرض الملخّص الإحصائي والقرار
import json
s = json.load(open('results/stage2_summary_colab.json'))['summary']
print("PPL      بلا ذاكرة:", s['ppl_no_memory'], "± ", s['ppl_no_std'],
      "| بذاكرة:", s['ppl_memory'], "± ", s['ppl_memory_std'],
      "| تحسّن%:", s['ppl_improvement_pct'], "| p:", s['ppl_p_value'])
print("late_PPL بلا ذاكرة:", s['late_ppl_no_memory'], "± ", s['late_ppl_no_std'],
      "| بذاكرة:", s['late_ppl_memory'], "± ", s['late_ppl_memory_std'],
      "| تحسّن%:", s['late_ppl_improvement_pct'], "| p:", s['late_ppl_p_value'])
# نقطة الفحص 2: نجاح إذا late_ppl_improvement_pct >= 2 و p < 0.1
ok = s['late_ppl_improvement_pct'] >= 2.0 and (s['late_ppl_p_value'] != 'nan' and float(s['late_ppl_p_value']) < 0.1)
print("نقطة الفحص 2:", "✅ اجتازت" if ok else "⚠️ لم تجتز")
```

```python
# خلية 5 — حفظ النتائج على Drive (اختياري)
from google.colab import drive; drive.mount('/content/drive')
import shutil
shutil.copy('results/experiments.csv', '/content/drive/MyDrive/Wameed/results/experiments.csv')
shutil.copy('results/stage2_summary_colab.json', '/content/drive/MyDrive/Wameed/results/stage2_summary_colab.json')
print("✅ النتائج مُحفظة على Drive")
```

## ملاحظة علمية مهمة
- معيار النجاح الأساسي هنا هو **late_PPL** (الخسارة على النصف الثاني من
  التسلسل): إن «تذكّر» النموذج السياق المبكر فستنخفض خسارته المتأخرة.
- مقياس `probe_improvement` الحالي يظل ≈0 لأن نموذجاً لغوياً صغيراً لا
  يتعلّم «نسخ» رمز حارس من البداية للنهاية؛ لذلك لا نعتمد عليه ونكتفي
  بـ late_PPL كدليل على التذكّر. هذا موثّق بأمانة في توثيق المرحلة.
