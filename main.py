
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import io

from utils.predictor import predict_single, predict_batch, THRESHOLD

# ── App setup ────────────────────────────────────────────────
app = FastAPI(
    title       = "Fraud Detection API",
    description = "REST API untuk deteksi penipuan kartu kredit menggunakan LightGBM",
    version     = "1.0.0"
)

# ── CORS — izinkan request dari Netlify nanti ────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # nanti diganti URL Netlify
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Schema ───────────────────────────────────────────────────
class TransactionInput(BaseModel):
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0
    Amount: float
    Time: float = 50000.0

    class Config:
        json_schema_extra = {
            "example": {
                "V14": -8.5,
                "V17": -5.2,
                "V12": -6.1,
                "Amount": 1.99,
                "Time": 5000
            }
        }


# ── Endpoints ────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message"   : "Fraud Detection API is running",
        "version"   : "1.0.0",
        "threshold" : THRESHOLD,
        "endpoints" : ["/predict", "/predict-batch", "/health", "/docs"]
    }


@app.get("/health")
def health_check():
    return {
        "status"   : "healthy",
        "model"    : "LightGBM",
        "threshold": THRESHOLD
    }


@app.post("/predict")
def predict(transaction: TransactionInput):
    """
    Prediksi satu transaksi.
    Kirim data transaksi, dapat hasil prediksi + probabilitas fraud.
    """
    try:
        raw_input = transaction.model_dump()
        result    = predict_single(raw_input)
        return {
            "status" : "success",
            "data"   : result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-batch")
async def predict_batch_endpoint(file: UploadFile = File(...)):
    """
    Prediksi banyak transaksi dari file CSV.
    Upload CSV dengan kolom V1-V28, Amount, Time.
    """
    try:
        # Baca CSV
        contents = await file.read()
        df       = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Validasi kolom
        required = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code = 400,
                detail      = f"Kolom tidak lengkap: {missing}"
            )

        # Prediksi
        records = df.to_dict(orient='records')
        results = predict_batch(records)

        # Summary
        total      = len(results)
        fraud_count = sum(1 for r in results if r['prediction'] == 1)

        return {
            "status"  : "success",
            "summary" : {
                "total"          : total,
                "fraud_detected" : fraud_count,
                "normal"         : total - fraud_count,
                "fraud_rate"     : round(fraud_count / total * 100, 2)
            },
            "data": results
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))