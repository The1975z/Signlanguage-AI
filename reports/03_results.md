# ขั้นที่ 6–7: Training & Evaluation

## โมเดลที่ทดลอง

| Model | Experiment | macro-F1 (test) | หมายเหตุ |
|---|---|---|---|
| KNN (k=5) | E2 Baseline | 0.9414 | เร็วสุด 0.0s |
| SVM (RBF, C=10) | E2 Baseline | 0.9577 | |
| Random Forest (300 trees) | E2 Baseline | 0.9613 | |
| **MLP (128→64)** | **E2 Baseline** | **0.9748** | **ดีที่สุด** |
| KNN | E4 Mirror Aug | 0.9495 | +aug ช่วย KNN เล็กน้อย |
| SVM | E4 Mirror Aug | 0.9555 | aug ไม่ช่วย SVM |
| RF | E4 Mirror Aug | 0.9627 | aug ช่วย RF เล็กน้อย |
| MLP | E4 Mirror Aug | 0.9735 | aug ไม่ช่วย MLP |

**Best model: MLP E2 Baseline** — บันทึกเป็น `models/best.joblib`

## ผลบน Test Set (1,405 ภาพ)

| Metric | ค่า |
|---|---|
| Accuracy (detected only) | 0.9749 |
| End-to-end accuracy | 0.9687 (1,361/1,405) |
| Macro-Precision | 0.9755 |
| Macro-Recall | 0.9749 |
| Macro-F1 | **0.9748** ✅ (เป้า ≥ 0.90) |

**ผ่าน** เกณฑ์ macro-F1 ≥ 0.90

## ผลบน Unseen Set (300 ภาพ จากคน 4 คนใหม่)

| Metric | ค่า |
|---|---|
| Accuracy (detected only) | 0.9130 |
| End-to-end accuracy | 0.9100 (273/300) |
| Macro-Precision | 0.9334 |
| Macro-Recall | 0.9133 |
| Macro-F1 | **0.9092** ✅ (เป้า ≥ 0.90) |

**ผ่าน** เกณฑ์ macro-F1 ≥ 0.90

## คลาสที่อ่อนบน Unseen

| ตัว | precision | recall | F1 | สาเหตุที่น่าจะเป็น |
|---|---|---|---|---|
| อ | 1.00 | 0.40 | 0.57 | predict เป็น ม หรือ ส แทน |
| ต | 0.54 | 0.70 | 0.61 | สับสนกับ ส, ม |
| น | 1.00 | 0.65 | 0.79 | recall ต่ำ — หลายภาพถูก predict เป็นคลาสอื่น |

**หมายเหตุ:** คลาสเดียวกันบน Test set ทำผลดี (ต=0.93, อ=0.98, น=0.93) → generalization gap จาก unseen มีคนใหม่ 4 คน ท่าทางอาจต่างจาก training set

## เกณฑ์ผ่านขั้นที่ 7

- [x] macro-F1 บน test ≥ 0.90 → **0.9748**
- [x] macro-F1 บน unseen ≥ 0.90 → **0.9092**
- [x] บันทึก confusion matrix (cm_test.npy, cm_unseen.npy)
- [x] unseen set ใช้แค่ครั้งเดียว (ไม่เคยใช้ระหว่าง development)
