"""
Lightweight FastAPI server that exposes the monitoring pipeline as HTTP endpoints.
Used by n8n's HTTP Request node to trigger monitoring runs.

Start:  python src/api.py
        uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
  POST /run          Run monitoring pipeline
  GET  /alerts/latest  Return latest row from alert_log.csv
  GET  /health       Health check
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.monitoring import run_monitoring_pipeline

app = FastAPI(title="SER Monitoring API", version="1.0")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORTS_DIR  = PROJECT_ROOT / "data" / "monitoring_exports"


# ── Request / Response models ─────────────────────────────────────────────────

class RunRequest(BaseModel):
    scenario:   str   = "baseline"   # baseline | noise | speaker_split | gradual
    batch_size: int   = 50
    noise_std:  float = 1.0


class RunResponse(BaseModel):
    status:          str
    scenario:        str
    ks_drift_ratio:  float
    f1_score:        float
    baseline_f1:     float
    f1_drop:         float
    drift_alert:     bool
    perf_alert:      bool
    any_alert:       bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "message": "SER Monitoring API is running"}


@app.post("/run", response_model=RunResponse)
def run_monitoring(req: RunRequest):
    """Trigger a monitoring pipeline run and return the alert result."""
    try:
        alert = run_monitoring_pipeline(
            scenario=req.scenario,
            batch_size=req.batch_size,
            noise_std=req.noise_std,
        )
        return RunResponse(**alert)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts/latest")
def latest_alert():
    """Return the most recent row from alert_log.csv."""
    alert_file = EXPORTS_DIR / "alert_log.csv"
    if not alert_file.exists():
        raise HTTPException(status_code=404, detail="No alerts logged yet.")
    df = pd.read_csv(alert_file)
    return df.iloc[-1].to_dict()


@app.get("/alerts/all")
def all_alerts():
    """Return all rows from alert_log.csv."""
    alert_file = EXPORTS_DIR / "alert_log.csv"
    if not alert_file.exists():
        raise HTTPException(status_code=404, detail="No alerts logged yet.")
    df = pd.read_csv(alert_file)
    return df.to_dict(orient="records")


@app.get("/metrics")
def monitoring_metrics():
    """Return all rows from monitoring_log.csv."""
    log_file = EXPORTS_DIR / "monitoring_log.csv"
    if not log_file.exists():
        raise HTTPException(status_code=404, detail="No monitoring log yet.")
    df = pd.read_csv(log_file)
    return df.to_dict(orient="records")


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
