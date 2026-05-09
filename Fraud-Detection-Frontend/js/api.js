// js/api.js
// ── Konfigurasi API Railway ───────────────────────────────────
// Ganti URL ini dengan URL Railway kamu yang sudah live
const API_BASE = 'https://web-production-e88050.up.railway.app';

// ── Single prediction ─────────────────────────────────────────
async function predictSingle(transactionData) {
  const response = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(transactionData),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${response.status}`);
  }

  const data = await response.json();
  return data.data;
}

// ── Batch prediction (upload CSV file) ───────────────────────
async function predictBatch(csvFile) {
  const formData = new FormData();
  formData.append('file', csvFile);

  const response = await fetch(`${API_BASE}/predict-batch`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${response.status}`);
  }

  return await response.json();
}

// ── Health check ──────────────────────────────────────────────
async function checkHealth() {
  const response = await fetch(`${API_BASE}/health`, {
    signal: AbortSignal.timeout(5000),
  });
  return await response.json();
}
