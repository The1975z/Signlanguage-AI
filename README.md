# Thai Fingerspelling Recognition

ระบบจำแนกภาษามือไทยแบบสะกดนิ้ว 15 ตัวอักษร (ก ด ต น บ พ ฟ ม ย ร ล ว ส ห อ)
ด้วย MediaPipe Hand Landmarks + Machine Learning

| รายการ | รายละเอียด |
|---|---|
| Dataset | One-Stage-TFS (Mendeley, DOI: 10.17632/rknd3wbz42.1) |
| Feature extraction | MediaPipe HandLandmarker (Tasks API) → 63 ค่า (21 จุด × x/y/z) |
| Classifier | KNN / SVM / Random Forest / MLP (scikit-learn) |
| เป้าหมาย | macro-F1 ≥ 0.90 (Test), end-to-end accuracy ≥ 80% (Unseen) |

## โครงสร้างโปรเจกต์

```
Signlanguage-AI/
├── data/
│   ├── raw/          ← dataset (gitignored)
│   └── processed/    ← landmarks CSV
├── notebooks/        ← EDA และการวิเคราะห์
├── src/
│   ├── features.py   ← normalize (ใช้ร่วมกัน train/eval/demo)
│   ├── train.py
│   ├── evaluate.py
│   └── realtime.py
├── models/           ← model.joblib (gitignored)
├── results/          ← experiments.csv
└── reports/          ← รายงานวิชา (gitignored เก็บในเครื่องเท่านั้น)
```

## วิธีติดตั้ง

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

ดาวน์โหลดจาก Mendeley Data และแตกไฟล์ไว้ที่ `data/raw/`
ห้าม commit ภาพจาก dataset (ดู `.gitignore`)

## อ้างอิง

- Lata & Surinta (2022). ICIC Express Letters. DOI: 10.24507/icicel.16.05.529
- One-Stage-TFS Dataset. DOI: 10.17632/rknd3wbz42.1
