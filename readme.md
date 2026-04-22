# GeoGuardian v3

## Real-Time AI-Powered Tourist Safety Monitoring Platform

GeoGuardian v3 is an intelligent safety monitoring platform designed to assess tourist safety in real time using machine learning, anomaly detection, spatial clustering, and a live operational dashboard.

It combines profile-based risk prediction, GPS movement anomaly detection, DBSCAN hotspot clustering, confidence scoring, and a live geospatial dashboard to simulate a production-grade monitoring system rather than a simple frontend demo.

---

# System Architecture

```text
                        ┌──────────────────────────┐
                        │      Frontend UI         │
                        │  HTML + CSS + JS        │
                        │  Live Dashboard         │
                        └──────────┬───────────────┘
                                   │
                                   │ API Calls
                                   ▼
                    ┌──────────────────────────────┐
                    │       FastAPI Backend        │
                    │           api.py             │
                    └──────────┬───────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
┌───────────────┐     ┌────────────────┐     ┌──────────────────┐
│ Static Engine │     │ Dynamic Engine │     │ Cluster Engine   │
│ RF + NN Model │     │ LSTM Autoenc.  │     │ DBSCAN + Spatial │
└──────┬────────┘     └──────┬─────────┘     └────────┬─────────┘
       │                     │                        │
       ▼                     ▼                        ▼
Profile Risk         GPS Path Anomaly         Heatmap + Zones
Prediction           Detection                Risk Clusters
       │                     │                        │
       └──────────────┬──────┴──────────────┬─────────┘
                      ▼                     ▼
               Combined Risk Engine   Session Store
               Confidence + Alerts    In-memory State
                      │
                      ▼
              Frontend Live Dashboard
```

---

# Core Features

## 1. Static Risk Assessment

Uses tourist profile information such as:
- age
- gender
- group size
- health conditions
- tourist type
- location
- mobility indicators

Models used:
- Random Forest
- Neural Network

Output:
- static risk score

---

## 2. Dynamic Anomaly Detection

Uses GPS movement patterns to detect anomalies such as:
- suspicious route deviation
- unusual stalling
- repeated loops
- abnormal movement behavior

Model used:
- LSTM Autoencoder

Output:
- anomaly score
- reconstruction error
- anomaly type

---

## 3. Combined Risk Intelligence

Backend combines:
- static profile score
- LSTM anomaly score

Output:
- combined risk score
- confidence score
- explanation factors
- escalation triggers
- SOS logic

---

## 4. Spatial Intelligence

Cluster engine provides:
- hotspot heatmaps
- DBSCAN clusters
- cluster confidence
- severity score
- risk trend analysis

Outputs:
- High Risk Zones
- Moderate Risk Zones
- Safe Corridors

---

# Tech Stack

## Backend
- Python
- FastAPI
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas
- Joblib

## Frontend
- HTML
- CSS
- Vanilla JavaScript
- Canvas Rendering

## ML Models
- Random Forest
- Feedforward Neural Network
- LSTM Autoencoder
- DBSCAN Clustering

---

# Project Folder Structure

```text
GeoGuardian/
│
├── backend/
│
│   ├── api.py
│   │   FastAPI backend routes
│   │   (/assess, /heatmap, /clusters, /metrics)
│   │
│   ├── geoguardian_v2.py
│   │   Core ML pipeline
│   │   Static model + Dynamic LSTM model
│   │
│   ├── geoguardian_v3_backend.py
│   │   Spatial clustering engine
│   │   DBSCAN + heatmap + zone intelligence
│   │
│   ├── requirements.txt
│   │   Python dependencies
│   │
│   ├── geoguardian_v2_static_components.pkl
│   │   Static model artifacts
│   │
│   ├── geoguardian_v2_static_nn.keras
│   │   Static neural network model
│   │
│   ├── geoguardian_v2_dynamic_lstm.keras
│   │   LSTM anomaly detection model
│   │
│   ├── geoguardian_v2_dynamic_meta.pkl
│   │   LSTM scaler + threshold metadata
│   │
│   ├── models/
│   │   Optional model storage directory
│   │
│   └── venv/
│       Python virtual environment
│
├── frontend/
│
│   └── index.html
│       Full dashboard UI
│       Charts + map + controls + alerts
│
├── .gitignore
│
└── README.md
```

---

# Setup Instructions

## 1. Clone Repository

```bash
git clone <your-repo-url>
cd GeoGuardian
```

---

## 2. Backend Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Train Models (Run Once)

```bash
python geoguardian_v2.py
```

This generates:
- .keras files
- .pkl files

If already generated, you can skip this step.

---

## 4. Run Backend

```bash
uvicorn api:app --reload --port 8000
```

Swagger Docs:

```text
http://localhost:8000/docs
```

---

## 5. Run Frontend

```bash
cd frontend
python3 -m http.server 5500
```

Frontend URL:

```text
http://localhost:5500/index.html
```

---

# API Endpoints

## POST /assess
Returns:
- static score
- LSTM score
- combined score
- confidence
- explanation
- anomaly type

---

## GET /heatmap
Returns live heatmap hotspot points

---

## GET /clusters
Returns DBSCAN spatial risk clusters

---

## GET /session/stats
Returns operational statistics

---

## GET /model/metrics
Returns:
- accuracy
- precision
- recall
- f1 score

---

# Deployment

## Recommended Production Setup

### Frontend
Deploy on:
- Vercel
- Netlify

### Backend
Deploy on:
- Render
- Railway
- EC2
- DigitalOcean

Recommended:

```text
Frontend → Vercel
Backend → Render
```

---

# Future Improvements

- PostgreSQL instead of in-memory store
- Redis queueing
- WebSockets for live updates
- Docker deployment
- Authentication layer
- Admin dashboard
- Mobile companion app
- Real GPS integration
- Emergency services integration

---

# Author

GeoGuardian v3 transforms tourist safety monitoring from a simulation dashboard into a real-time AI-driven operational intelligence platform.

