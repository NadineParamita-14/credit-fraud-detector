# FraudShield — Real-time Credit Card Fraud Detection

> Sistem deteksi penipuan kartu kredit berbasis Machine Learning yang dibangun end-to-end: dari eksplorasi data, training model, hingga deployment sebagai REST API dan dashboard interaktif yang bisa diakses siapapun.

![Model](https://img.shields.io/badge/Model-LightGBM-green)
![API](https://img.shields.io/badge/API-FastAPI-009688)
![Deploy](https://img.shields.io/badge/Deploy-Railway%20%2B%20Netlify-brightgreen)
![Explainability](https://img.shields.io/badge/Explainability-SHAP-orange)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

**🌐 Live Demo:** [Dashboard](https://creditfrauddetector.netlify.app/) &nbsp;|&nbsp; **📡 API Docs:** [Swagger UI](https://web-production-e88050.up.railway.app/docs)

---

## A. Tentang Project Ini

Penipuan kartu kredit menyebabkan kerugian global lebih dari **$32 miliar per tahun**. Tantangan terbesarnya bukan sekadar mendeteksi fraud — tapi melakukannya secara **otomatis, real-time, dan akurat** di tengah jutaan transaksi yang hampir semuanya sah.

Project ini membangun sistem deteksi fraud **end-to-end** yang menangani tiga tantangan utama:

- **Imbalanced data** — hanya 0.172% transaksi yang fraud (1 dari 578)
- **Model explainability** — setiap keputusan harus bisa dijelaskan, bukan sekadar "model bilang begitu"
- **Business-aware optimization** — threshold dipilih berdasarkan dampak finansial nyata, bukan hanya metrik teknis

---

## B. Arsitektur Sistem

```
┌──────────────────────────────────────────────────────┐
│                   USER / BROWSER                     │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│           NETLIFY — HTML Dashboard                     │
│  index.html │ predict.html │ batch.html │ insights.html│
└──────────────────────┬─────────────────────────────────┘
                       │ fetch() API calls
                       ▼
┌────────────────────────────────────────────────────────┐
│          RAILWAY — FastAPI Backend                     │
│  POST /predict  │  POST /predict-batch  │  GET /health │
└──────────────────────┬─────────────────────────────────┘
                       │ joblib.load()
                       ▼
┌──────────────────────────────────────────────────────┐
│            MODEL ARTIFACTS (.pkl files)              │
│  fraud_detection_model.pkl  │  robust_scaler.pkl     │
│  optimal_threshold.pkl      │  feature_columns.pkl   │
└──────────────────────────────────────────────────────┘
```

**Alur kerja singkat:**
1. User membuka dashboard di Netlify
2. User memasukkan data transaksi
3. Dashboard mengirim data ke FastAPI yang berjalan di Railway
4. FastAPI memproses data dan menjalankan model LightGBM
5. Hasil prediksi dikembalikan ke dashboard dalam < 1 detik
6. Dashboard menampilkan hasil dengan visualisasi yang mudah dibaca

---

## C. Struktur Repository

```
credit-fraud-detector/
│
├── notebook/
│   └── fraud_detection.ipynb       # Notebook utama — 6 fase pipeline
│
├── Fraud-Detection-Frontend/    # HTML Dashboard (deploy ke Netlify)
│   ├── index.html                  # Halaman Overview
│   ├── predict.html                # Single Transaction Prediction
│   ├── batch.html                  # Batch Analysis
│   ├── insights.html               # Model Insights & Explainability
│   ├── css/style.css               # Styling terpusat
│   └── js/api.js                   # Konfigurasi & fungsi API calls
│
├── visualizations/              # Semua chart hasil EDA & modeling
│   ├── amount_analysis.png
│   ├── shap_global_importance.png
│   ├── threshold_analysis.png
│   └── ... (16 visualisasi total)
│
├── models/                      # Trained model artifacts
│   ├── fraud_detection_model.pkl   # Model LightGBM hasil training
│   ├── optimal_threshold.pkl       # Threshold optimal (0.80)
│   ├── robust_scaler.pkl           # Scaler untuk normalisasi fitur
│   └── feature_columns.pkl         # Daftar 37 fitur yang digunakan
│
├── utils/
│   └── predictor.py                # Feature engineering & prediksi logic
│
├── main.py                         # FastAPI application
├── Dockerfile                      # Container configuration
├── railway.toml                    # Railway deployment config
└── requirements.txt                # Python dependencies
```

---

## D. Proses Pembuatan — 6 Fase Pipeline

### Fase 1 — Data Understanding

Dataset yang digunakan adalah **[Kaggle Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**.

| Properti | Detail |
|---|---|
| Total transaksi | 284,807 |
| Transaksi fraud | 492 (0.172%) |
| Transaksi normal | 284,315 (99.828%) |
| Periode data | ~2 hari transaksi |
| Imbalance ratio | 1 fraud per 578 transaksi normal |

**Mengapa PR-AUC, bukan Accuracy?**
Model yang selalu memprediksi "normal" tanpa berpikir sudah punya accuracy 99.83% — tapi sama sekali tidak berguna. PR-AUC lebih jujur karena fokus pada performa di kelas minoritas (fraud) yang justru paling penting.

---

### Fase 2 — Exploratory Data Analysis (EDA)

**Temuan utama:**

**1. Pola Amount (Nominal Transaksi)**
- Fraud cenderung pada nominal sangat kecil — disebut *card testing pattern*: fraudster menguji kartu dengan transaksi kecil sebelum melakukan transaksi besar
- Distribusi amount fraud lebih lebar (std dev lebih tinggi) — fraudster beroperasi di dua ujung ekstrem
- Nominal saja tidak cukup sebagai sinyal fraud

**2. Pola Temporal (Waktu)**
- Fraud rate tertinggi di jam 00:00–06:00 (dini hari)
- Saat monitoring manusia paling lemah
- Mendukung pentingnya sistem deteksi otomatis 24/7

**3. Discriminative Power Fitur V**
- Menggunakan Cohen's d + KS test untuk mengukur seberapa baik fitur membedakan fraud vs normal
- V14, V17, V12 terbukti paling diskriminatif

---

### Fase 3 — Feature Engineering

Dari 2 fitur yang bisa diengineering (Time & Amount), dibuat 9 fitur baru:

| Fitur Baru | Dibuat dari | Alasan |
|---|---|---|
| `log_amount` | Amount | Handle distribusi yang sangat skewed |
| `amount_zscore` | Amount | Seberapa "tidak biasa" nominal ini secara statistik |
| `is_small_amount` | Amount | Flag nominal ≤ €10 (card testing pattern) |
| `is_large_amount` | Amount | Flag nominal ≥ €1000 (cash out pattern) |
| `hour_sin` + `hour_cos` | Time | Cyclic encoding jam — jam 23 dan 0 berdekatan |
| `is_night` | Time | Flag jam 00:00–06:00 |
| `is_weekend` | Time | Estimasi hari weekend |
| `time_diff_log` | Time | Jarak dari transaksi sebelumnya (velocity) |
| `v_fraud_composite` | V1–V28 | Weighted composite dari top V features |

**Mengapa RobustScaler, bukan StandardScaler?**
Data keuangan sering punya transaksi ekstrem yang mendistorsi StandardScaler. RobustScaler menggunakan median & IQR — lebih robust terhadap outlier.

---

### Fase 4 — Modeling & Imbalanced Handling

| Strategi | PR-AUC | ROC-AUC | Recall | Precision |
|---|---|---|---|---|
| LR Baseline | 0.7445 | 0.9582 | 0.6633 | 0.8025 |
| LightGBM + Class Weight | 0.8713 | 0.9763 | 0.8367 | 0.9011 |
| LightGBM + SMOTE | 0.8812 | 0.9814 | 0.8469 | 0.9121 |
| **LightGBM + Combined ✅** | **0.8834** | **0.9807** | **0.8469** | **0.8925** |

**Threshold Tuning:**
Threshold dioptimasi ke **0.80** berdasarkan F1 maksimum dan simulasi net benefit finansial dimana hal ini merupakan keputusan bisnis, bukan semata-mata teknis.

---

### Fase 5 — Explainability (SHAP)

**Top features berdasarkan Mean |SHAP|:**

| Rank | Fitur | Tipe | Arah |
|---|---|---|---|
| 1 | V14 | PCA | → Fraud jika nilai rendah |
| 2 | V17 | PCA | → Fraud jika nilai rendah |
| 3 | V12 | PCA | → Fraud jika nilai rendah |
| 7 | log_amount | Amount | → Normal jika nominal tinggi |
| 9 | hour_sin | Temporal | → Fraud di jam dini hari |

---

### Fase 6 — Deployment

- **FastAPI** — REST API dengan automatic Swagger documentation
- **Docker** — containerization untuk konsistensi environment
- **Railway** — cloud hosting untuk backend API
- **Netlify** — static site hosting untuk frontend dashboard

---

## E. Panduan Penggunaan Dashboard

### 1) Overview (`index.html`)

Halaman utama yang menampilkan ringkasan keseluruhan sistem:
- Status koneksi API Railway (real-time)
- Performa model: PR-AUC, ROC-AUC, Recall, Precision
- Grafik perbandingan 4 strategi yang diuji
- Proyeksi business impact tahunan
- Ringkasan 6 fase pipeline metodologi

---

### 2) Predict (`predict.html`)

Prediksi **satu transaksi** secara real-time.

**Cara pakai:**

**Opsi 1 — Preset (Paling Mudah)**

| Preset | Kondisi | Hasil Diharapkan |
|---|---|---|
| 🟢 Typical Normal | Amount €45, V14=+0.5, jam siang | NORMAL |
| 🔴 Suspicious | Amount €1.99, V14=-8.5, dini hari | FRAUD (~98%) |
| 🟡 Borderline | Amount €120, V14=-2.1 | Dekat threshold |

**Opsi 2 — Input Manual**

```
Amount (€)  → Nominal transaksi
Time        → Detik sejak transaksi pertama
V14         → ⭐ Paling penting. Sangat negatif = sinyal fraud kuat
V17, V12    → ⭐ Fitur penting kedua & ketiga
V10–V7      → Fitur pendukung
```

**Output:**
- Verdict: FRAUD atau NORMAL dengan warna indikator
- Probability & Risk Score (0–100)
- SHAP explanation: fitur apa yang mendorong keputusan

---

### 3) Batch (`batch.html`)

Analisis **banyak transaksi sekaligus** dari file CSV.

**Cara pakai:**

**Opsi 1 — Upload CSV Sendiri**

Format CSV yang diperlukan:

```csv
V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount,Time
-1.3598,-0.0728,2.5363,1.3782,-0.3383,0.4624,0.2399,0.0987,0.3637,0.0908,-0.5516,-0.6178,-0.9913,-0.3111,1.4681,-0.4704,0.2079,0.0258,0.4040,0.2514,-0.0183,0.2778,-0.1105,0.0669,0.1285,-0.1891,0.1336,-0.0211,149.62,0
1.1919,0.2661,-0.1665,0.4482,-0.0439,0.3003,0.0836,0.2082,0.1600,0.1609,-0.1533,-0.0517,-1.3196,-0.7334,0.0598,0.2099,-0.4973,0.6880,0.1135,-0.2402,-0.0948,0.2467,-0.0833,0.0848,-0.2083,-0.5695,0.1461,-0.0645,2.69,1
```

**Aturan format CSV:**
- Baris pertama **wajib** berupa header dengan nama kolom persis
- Semua kolom **V1–V28, Amount, Time** harus ada
- Nilai V1–V28: angka desimal, boleh negatif
- Amount: angka positif (nominal dalam Euro)
- Time: angka positif (detik sejak transaksi pertama)
- Kolom lain boleh ada tapi akan diabaikan sistem
- Ukuran file maksimal: 50MB

**Opsi 2 — Sample Data**
Klik "Gunakan Sample Data" untuk generate 100 transaksi simulasi. Nilai di-generate secara acak setiap klik sehingga jumlah fraud bisa berbeda.

**Output:**
- Summary: total, fraud detected, normal, avg risk score
- Grafik probabilitas per transaksi (top 20)
- Donut chart distribusi fraud vs normal
- Tabel detail dengan filter (Semua/Fraud/Normal) dan search
- Download hasil sebagai CSV

---

### 4) Insights (`insights.html`)

Penjelasan mendalam model dalam 4 tab:

**Tab SHAP Explainability**
- Ranking fitur berdasarkan Mean |SHAP| value
- Arah pengaruh: mendorong ke FRAUD (merah) atau NORMAL (hijau)
- Breakdown fitur berdasarkan kategori (PCA, Amount, Temporal, Interaction)

**Tab Threshold Analysis**
- Grafik Precision vs Recall di berbagai threshold (0.1–0.85)
- Tabel metrik per threshold dengan highlight threshold optimal
- Penjelasan trade-off bisnis

**Tab Business Impact** *(Interaktif)*
Kalkulator dampak finansial — ubah nilai untuk melihat proyeksi:
- Kerugian per fraud (default: €500)
- Biaya per false alarm (default: €15)
- TP, FP, FN dari test set
- Scale factor (default: 182.5)

**Tab Key Decisions**
8 pertanyaan teknis beserta jawabannya, plus tech stack lengkap.

---

## F. API Reference

Base URL: `https://web-production-e88050.up.railway.app`

### `GET /health`
```json
{
  "status": "healthy",
  "model": "LightGBM",
  "threshold": 0.8
}
```

### `POST /predict`

**Request:**
```json
{
  "V14": -8.5,
  "V17": -5.2,
  "V12": -6.1,
  "Amount": 1.99,
  "Time": 5000
}
```
*Field yang tidak disebutkan default ke 0.0*

**Response:**
```json
{
  "status": "success",
  "data": {
    "prediction": 1,
    "label": "FRAUD",
    "probability": 0.9838,
    "risk_score": 98,
    "threshold_used": 0.8
  }
}
```

### `POST /predict-batch`

**Request:** `multipart/form-data` dengan field `file` berisi CSV

**Response:**
```json
{
  "status": "success",
  "summary": {
    "total": 100,
    "fraud_detected": 3,
    "normal": 97,
    "fraud_rate": 3.0
  },
  "data": [
    {
      "index": 0,
      "prediction": 0,
      "label": "NORMAL",
      "probability": 0.0012,
      "risk_score": 0
    }
  ]
}
```

---

## G. Menjalankan Secara Lokal

### Backend (FastAPI)

```bash
# Clone repo
git clone https://github.com/NadineParamita-14/credit-fraud-detector.git
cd credit-fraud-detector

# Setup virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Jalankan API
uvicorn main:app --reload --port 8000
```

API: `http://localhost:8000`
Docs: `http://localhost:8000/docs`

### Frontend

Buka `Fraud-Detection-Frontend/index.html` langsung di browser, atau gunakan Live Server di VS Code.

> **Penting:** Pastikan `js/api.js` mengarah ke URL yang benar:
> - Lokal: `const API_BASE = 'http://localhost:8000'`
> - Production: `const API_BASE = 'https://web-production-e88050.up.railway.app'`

---

## *Pelajaran yang Didapat*

**1. Imbalanced data bukan hanya masalah teknis**
Cara menanganinya harus mempertimbangkan konteks bisnis. Kombinasi SMOTE + class weight terbukti paling efektif untuk kasus ini.

**2. Threshold adalah keputusan bisnis**
Tidak ada threshold yang "benar" secara teknis. Optimal bergantung pada berapa biaya false alarm vs kerugian fraud yang ditoleransi bisnis.

**3. Metrik harus sesuai problem**
Accuracy menyesatkan. ROC-AUC bisa optimistic untuk imbalanced data. PR-AUC adalah metrik paling jujur untuk fraud detection.

**4. Explainability bukan optional di industri keuangan**
Model yang tidak bisa dijelaskan sulit diadopsi di production. Regulator dan tim risk management perlu tahu *mengapa* suatu transaksi diblokir.

**5. End-to-end pipeline lebih berharga dari model sempurna**
Model yang bisa diakses via API dan divisualisasikan di dashboard jauh lebih berguna daripada model dengan skor lebih tinggi yang hanya ada di notebook.

---

## *Pengembangan Selanjutnya*

- [ ] Real-time streaming dengan Apache Kafka
- [ ] Graph Neural Network untuk fraud ring detection
- [ ] Concept drift monitoring & automated retraining
- [ ] MLflow untuk experiment tracking
- [ ] A/B testing framework untuk model baru vs production

---

## *Referensi*

- Dal Pozzolo, A. et al. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification.* IEEE SSCI.
- Lundberg, S. & Lee, S. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
- He, H. & Garcia, E. (2009). *Learning from Imbalanced Data.* IEEE TKDE.
- [Kaggle Dataset — Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## *Author*: **Nadine Riskia Windi Paramita**





