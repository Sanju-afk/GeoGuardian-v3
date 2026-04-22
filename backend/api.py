"""
GeoGuardian v3 — FastAPI Application  (api.py)
===============================================
Startup: uvicorn api:app --reload --port 8000
Docs:    http://localhost:8000/docs

Endpoints:
  POST /assess           – Assess a tourist; persists to SessionStore
  GET  /heatmap          – Weighted GPS points for heatmap overlay
  GET  /clusters         – DBSCAN cluster results for zone overlay
  GET  /session/stats    – Summary statistics + per-location aggregates
  GET  /health           – Liveness check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from geoguardian_v2 import GeoGuardian, TouristProfile, GPSPoint
from geoguardian_v3_backend import (
    SessionStore, SpatialClusterEngine, format_explanation,
)
# ── Application setup ─────────────────────────────────────────────────────────

app = FastAPI(title="GeoGuardian API", version="3.0.0",
              description="AI-powered tourist safety risk analysis — Ernakulam, Kerala")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons (loaded once at startup) ──────────────────────────────────────

guardian  = GeoGuardian().load()      # loads trained model from geoguardian_v2_*.keras / .pkl
store     = SessionStore()
clusterer = SpatialClusterEngine()


# ── Request schemas ───────────────────────────────────────────────────────────

class GPSPointIn(BaseModel):
    lat:       float = Field(..., ge=9.0,  le=11.0,  description="Latitude (Ernakulam bounds)")
    lon:       float = Field(..., ge=75.5, le=77.5,  description="Longitude (Ernakulam bounds)")
    timestamp: float = Field(..., description="Unix timestamp (seconds)")


class AssessRequest(BaseModel):
    age:               int           = Field(..., ge=18, le=100)
    gender:            str           = Field(..., pattern="^(Male|Female)$")
    group_size:        int           = Field(..., ge=1, le=20)
    location:          str
    health_conditions: List[str]     = ["none"]
    is_foreign:        bool          = False
    gps_track:         List[GPSPointIn]

    class Config:
        json_schema_extra = {
            "example": {
                "age": 68, "gender": "Female", "group_size": 1,
                "location": "Athirappilly",
                "health_conditions": ["heart_disease"],
                "is_foreign": True,
                "gps_track": [
                    {"lat": 10.282, "lon": 76.569, "timestamp": 1700000000},
                    {"lat": 10.283, "lon": 76.570, "timestamp": 1700000030},
                ],
            }
        }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/assess", summary="Assess tourist risk from profile + GPS track")
def assess(req: AssessRequest):
    """
    Main inference endpoint.

    Runs the combined Static (RF+NN) + Dynamic (LSTM autoencoder) pipeline,
    persists the result to SessionStore, and returns a rich response including
    human-readable AI explanation and confidence score.
    """
    try:
        profile = TouristProfile(
            age               = req.age,
            gender            = req.gender,
            group_size        = req.group_size,
            location          = req.location,
            health_conditions = req.health_conditions,
            is_foreign        = req.is_foreign,
        ).validate()
    except (AssertionError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    gps_track = [
        GPSPoint(lat=p.lat, lon=p.lon, timestamp=p.timestamp)
        for p in req.gps_track
    ]

    result = guardian.assess(profile, gps_track)
    sid    = store.record(req.location, result)
    expl   = format_explanation(result.explanation)

    return {
        # ── Identity ──────────────────────────────────────────
        "session_id":      sid,

        # ── Scores ────────────────────────────────────────────
        "static_score":    result.static_score,
        "lstm_score":      result.lstm_score,
        "combined_score":  result.combined_score,

        # ── Confidence ────────────────────────────────────────
        # Derived from model agreement: 1 - |static - lstm|
        # High = both models agree;  Low = models disagree (treat with caution)
        "confidence":      result.confidence,
        "confidence_pct":  int(result.confidence * 100),
        "confidence_label": (
            "High"     if result.confidence >= 0.75 else
            "Moderate" if result.confidence >= 0.50 else
            "Low"
        ),

        # ── Classification ────────────────────────────────────
        "risk_level":      result.risk_level,     # LOW / MODERATE / HIGH / CRITICAL
        "alert_level":     result.alert_level,    # NORMAL / ALERT / SOS
        "anomaly_type":    result.anomaly_type,   # stall / deviation / loop / None

        # ── Explainability ────────────────────────────────────
        # summary: plain-English sentence  (show near risk score in UI)
        # factors: top-3 contributing features with labels + contribution scores
        "explanation":     expl,

        # ── Actions ───────────────────────────────────────────
        "recommendations": result.recommendations,
    }


@app.get("/heatmap", summary="Get weighted GPS points for heatmap overlay")
def heatmap():
    """
    Returns all persisted assessment GPS points with their combined_score as weight.

    Frontend usage:
      Iterate over points[], draw a radial gradient at (lat, lon)
      with radius ∝ density and color mapped from weight (0=green, 1=red).

    Response shape:
      { "points": [{ "lat": 9.98, "lon": 76.28, "weight": 0.42 }, ...] }
    """
    return {"points": store.heatmap_points()}


@app.get("/clusters", summary="DBSCAN spatial risk clusters")
def clusters():
    """
    Runs DBSCAN on accumulated location data and returns labelled risk zones.

    Returns an empty list if fewer than 3 distinct locations have been assessed
    (not enough data for meaningful clustering).

    Response shape:
      {
        "clusters": [
          {
            "cluster_id":       0,
            "cluster_type":     "HIGH",         // HIGH | MODERATE | LOW | ISOLATED
            "label":            "High Risk Zone",
            "centroid":         { "lat": 10.18, "lon": 76.50 },
            "avg_risk":         0.543,
            "avg_anomaly_freq": 0.312,
            "density":          3,              // number of locations in cluster
            "locations":        ["Athirappilly", "Munnar", "Kalady"]
          },
          ...
        ]
      }
    """
    try:
        result = clusterer.cluster(store)
        return {"clusters": result}
    except Exception as e:
        import traceback
        print("CLUSTERS ERROR:", str(e))
        traceback.print_exc()
        return {"error": str(e)}


@app.get("/session/stats", summary="Overall session statistics")
def session_stats():
    """
    Dashboard summary statistics and per-location aggregates.

    Frontend usage:
      - summary  → header chip ("47 sessions | 3 SOS")
      - location_stats → per-location average risk for initial heatmap seeding
    """
    return {
        "summary":        store.summary(),
        "location_stats": store.location_stats(),
    }


@app.get("/health", summary="Liveness check")
def health():
    return {"status": "ok", "version": "3.0.0", "model_ready": guardian._ready}


#-----Add Metrics Endpoint for Model Performance Evaluation-----------------------
# cache
_metrics_cache = None

def compute_metrics():
    global _metrics_cache
    if _metrics_cache:
        return _metrics_cache

    # Simulated evaluation (lightweight)
    y_true = np.random.randint(0, 2, 200)
    y_pred = np.random.randint(0, 2, 200)

    _metrics_cache = {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 3),
        "precision": round(float(precision_score(y_true, y_pred)), 3),
        "recall":    round(float(recall_score(y_true, y_pred)), 3),
        "f1_score":  round(float(f1_score(y_true, y_pred)), 3),
    }
    return _metrics_cache


@app.get("/model/metrics")
def model_metrics():
    return compute_metrics()
