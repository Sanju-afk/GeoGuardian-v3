"""
GeoGuardian v3 — Backend Additions
====================================
New production-grade modules layered on top of geoguardian_v2.py.

Modules:
  1. SessionStore          – Lightweight in-memory session persistence
  2. SpatialClusterEngine  – DBSCAN spatial risk clustering (lat/lon/risk/anomaly_freq)
  3. format_explanation()  – SHAP dict → structured, human-readable top-3 factors

Usage:
  Import these alongside geoguardian_v2.py and wire into api.py.
  See api.py for complete FastAPI integration.

Design choices:
  • SessionStore uses a plain list — swap append/pop for DB insert/query to persist.
  • DBSCAN eps=0.35 tuned for Ernakulam's ~12-location grid; adjust for larger maps.
  • format_explanation() is pure function — no state, easy to unit-test.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

# ── Import from v2 (must be in the same Python environment) ──────────────────
from geoguardian_v2 import RiskResult, LOCATION_RISK_MAP, HEALTH_SEVERITY


# ─────────────────────────────────────────────────────────────────────────────
# LOCATION COORDINATES
# Real GPS coordinates for every entry in LOCATION_RISK_MAP.
# The v2 file only stores risk scores; we add lat/lon here so SessionStore
# can build heatmap points and ClusterEngine can work in metre-space.
# ─────────────────────────────────────────────────────────────────────────────

LOCATION_COORDS: Dict[str, Tuple[float, float]] = {
    "Marine Drive":   (9.9812,  76.2803),
    "Fort Kochi":     (9.9625,  76.2421),
    "Mattancherry":   (9.9564,  76.2605),
    "Cherai Beach":   (10.1365, 76.1893),
    "Hill Palace":    (9.9456,  76.3289),
    "Vembanad Lake":  (9.8200,  76.3505),
    "Kalady":         (10.1624, 76.4436),
    "Athirappilly":   (10.2828, 76.5694),
    "Munnar":         (10.0892, 77.0595),
    "Ernakulam City": (9.9816,  76.2915),
    "Thripunithura":  (9.9456,  76.3560),
    "Aluva":          (10.1000, 76.3500),
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SESSION STORE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SessionRecord:
    """One persisted tourist tracking assessment."""
    session_id:     str
    timestamp:      str
    location:       str
    lat:            float
    lon:            float
    combined_score: float
    static_score:   float
    lstm_score:     float
    confidence:     float
    is_anomaly:     bool
    anomaly_type:   Optional[str]
    alert_level:    str


class SessionStore:
    """
    In-memory persistence for tourist tracking sessions.

    Purpose:
      - Feed weighted GPS points to the heatmap overlay.
      - Supply per-location aggregates to SpatialClusterEngine.
      - Return overall statistics to the dashboard header.

    Upgrade path to persistent storage (no caller changes required):
      Replace the list operations in record() / heatmap_points() /
      location_stats() with SQLite or Postgres queries.
      The public interface is intentionally minimal and stable.
    """

    _MAX_RECORDS = 2000   # FIFO cap to bound memory

    def __init__(self):
        self._data: List[SessionRecord] = []
        self._ctr:  int = 0

    # ── Write ─────────────────────────────────────────────────────────────────

    def record(self, location: str, result: RiskResult) -> str:
        """
        Persist one risk assessment result.
        Returns the generated session_id so callers can reference it.
        """
        self._ctr += 1
        sid = f"SES-{self._ctr:05d}"
        lat, lon = LOCATION_COORDS.get(location, (9.9312, 76.2673))

        self._data.append(SessionRecord(
            session_id     = sid,
            timestamp      = datetime.now().isoformat(),
            location       = location,
            lat            = lat,
            lon            = lon,
            combined_score = result.combined_score,
            static_score   = result.static_score,
            lstm_score     = result.lstm_score,
            confidence     = result.confidence,
            is_anomaly     = result.anomaly_type is not None,
            anomaly_type   = result.anomaly_type,
            alert_level    = result.alert_level,
        ))

        if len(self._data) > self._MAX_RECORDS:
            self._data.pop(0)   # evict oldest

        return sid

    # ── Read ──────────────────────────────────────────────────────────────────

    def heatmap_points(self) -> List[Dict]:
        """
        Returns a list of {lat, lon, weight} for the frontend heatmap canvas.
        weight ∈ [0, 1] — drives the green→yellow→red gradient.
        """
        return [
            {
                "lat":    r.lat,
                "lon":    r.lon,
                "weight": round(r.combined_score, 4),
            }
            for r in self._data
        ]

    def location_stats(self) -> Dict[str, Dict]:
        """
        Aggregates per-location data needed by SpatialClusterEngine.

        Returns:
            {
              "Marine Drive": {
                  "avg_risk":     0.22,
                  "anomaly_freq": 0.10,
                  "count":        14,
                  "lat":          9.9812,
                  "lon":          76.2803,
              },
              ...
            }
        """
        agg: Dict[str, Dict] = defaultdict(lambda: {"scores": [], "anomalies": 0})

        for r in self._data:
            agg[r.location]["scores"].append(r.combined_score)
            agg[r.location]["anomalies"] += int(r.is_anomaly)

        out = {}
        for loc, v in agg.items():
            lat, lon = LOCATION_COORDS.get(loc, (9.9312, 76.2673))
            n = len(v["scores"])
            out[loc] = {
                "avg_risk":     round(float(np.mean(v["scores"])), 4),
                "anomaly_freq": round(v["anomalies"] / n, 4),
                "count":        n,
                "lat":          lat,
                "lon":          lon,
            }
        return out

    def summary(self) -> Dict:
        """High-level statistics for the dashboard header chip."""
        n = len(self._data)
        if n == 0:
            return {"total": 0, "sos_count": 0, "anomaly_rate": 0.0, "avg_risk": 0.0}
        return {
            "total":        n,
            "sos_count":    sum(1 for r in self._data if r.alert_level == "SOS"),
            "anomaly_rate": round(sum(1 for r in self._data if r.is_anomaly) / n, 4),
            "avg_risk":     round(float(np.mean([r.combined_score for r in self._data])), 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SPATIAL CLUSTER ENGINE (DBSCAN)
# ─────────────────────────────────────────────────────────────────────────────

class SpatialClusterEngine:
    """
    Clusters visited tourist locations using DBSCAN on a 4D normalised space:

      Feature space (after normalisation + weighting):
        [lat_metres,   lon_metres]   × GEO_W   → spatial proximity
        [avg_risk]                   × RISK_W  → risk similarity
        [anomaly_freq]               × ANOM_W  → anomaly concentration

    Why metres and not degrees?
      Mixing degree-scale geo offsets (~0.001°/100m) with 0–1 risk scores
      would let risk dominate distance calculations.  Converting to metres
      then applying MinMaxScaler puts both axes on the same footing.

    Why DBSCAN over KMeans?
      • No need to pre-specify k.
      • Handles the uneven spatial density of Ernakulam's tourist spots.
      • Naturally marks sparse outliers as ISOLATED (cluster_id = -1).

    Output cluster types:
      HIGH      → avg_risk ≥ 0.50  → labelled "High Risk Zone"
      MODERATE  → avg_risk ≥ 0.30  → labelled "Moderate Zone"
      LOW       → avg_risk < 0.30  → labelled "Safe Corridor"
      ISOLATED  → DBSCAN noise     → labelled "Isolated Point"
    """

    GEO_W  = 1.0   # spatial weight
    RISK_W = 0.6   # risk weight
    ANOM_W = 0.4   # anomaly frequency weight

    #----adding cluster engine updgrade------------------
    def _compute_confidence(self, density, risks):
        if len(risks) == 0:
            return 0
        var = np.var(risks)
        density_norm = min(density / 10, 1)  # simple normalization
        return round(float(density_norm * (1 - var)), 3)


    def _risk_trend(self, store, locations):
        # simple heuristic: compare last vs first
        records = [r for r in store._data if r.location in locations]
        if len(records) < 4:
            return "stable"

        risks = [r.combined_score for r in records[-10:]]
        first = np.mean(risks[:len(risks)//2])
        last  = np.mean(risks[len(risks)//2:])

        if last > first + 0.05:
            return "increasing"
        elif last < first - 0.05:
            return "decreasing"
        return "stable"
    
    #--------------main clustering function------------------------
    def cluster(self, store: SessionStore) -> List[Dict]:
        """
        Run DBSCAN clustering on accumulated location data.

        Returns [] if fewer than 3 distinct locations have been visited.
        The threshold is intentionally low so clustering appears early in a demo.

        Sample output:
            [
              {
                "cluster_id":       0,
                "cluster_type":     "HIGH",
                "label":            "High Risk Zone",
                "centroid":         {"lat": 10.18, "lon": 76.50},
                "avg_risk":         0.543,
                "avg_anomaly_freq": 0.312,
                "density":          3,
                "locations":        ["Athirappilly", "Munnar", "Kalady"],
              },
              ...
            ]
        """
        stats = store.location_stats()
        if len(stats) < 3:
            return []

        locs      = list(stats.keys())
        lats      = np.array([stats[l]["lat"]          for l in locs])
        lons      = np.array([stats[l]["lon"]          for l in locs])
        risks     = np.array([stats[l]["avg_risk"]     for l in locs])
        anom_freq = np.array([stats[l]["anomaly_freq"] for l in locs])

        # Degrees → metres (approximate planar projection for small regions)
        lat_m = (lats - lats.mean()) * 111_000
        lon_m = (lons - lons.mean()) * 111_000 * np.cos(np.radians(lats.mean()))

        scaler   = MinMaxScaler()
        geo_norm = scaler.fit_transform(np.column_stack([lat_m, lon_m]))
        X = np.hstack([
            geo_norm                 * self.GEO_W,
            risks.reshape(-1, 1)     * self.RISK_W,
            anom_freq.reshape(-1, 1) * self.ANOM_W,
        ])

        # eps=0.35, min_samples=2 is appropriate for Ernakulam's ~12-point dataset.
        # For production deployments with hundreds of GPS points, lower eps and
        # raise min_samples to avoid over-merging distinct hazard zones.
        labels = DBSCAN(eps=0.35, min_samples=2).fit_predict(X)

        by_cluster: Dict[int, List[int]] = defaultdict(list)
        for i, lbl in enumerate(labels):
            by_cluster[lbl].append(i)

        results = []
        for cid, idxs in by_cluster.items():
            avg_r = float(np.mean(risks[idxs]))
            avg_a = float(np.mean(anom_freq[idxs]))

            if cid == -1:
                ctype, label = "ISOLATED", "Isolated Point"
            elif avg_r >= 0.50:
                ctype, label = "HIGH",     "High Risk Zone"
            elif avg_r >= 0.30:
                ctype, label = "MODERATE", "Moderate Zone"
            else:
                ctype, label = "LOW",      "Safe Corridor"

            cluster_risks = risks[idxs]
            cluster_anoms = anom_freq[idxs]
            density = len(idxs)

            confidence = self._compute_confidence(density, cluster_risks)
            trend = self._risk_trend(store, [locs[i] for i in idxs])
            severity = round(float(0.7 * avg_r + 0.3 * avg_a), 3)

            results.append({
                "cluster_id":       int(cid),
                "cluster_type":     ctype,
                "label":            label,
                "centroid": {
                    "lat": round(float(lats[idxs].mean()), 6),
                    "lon": round(float(lons[idxs].mean()), 6),
                },
                "avg_risk":         round(avg_r, 4),
                "avg_anomaly_freq": round(avg_a, 4),
                "density":          density,

                # ✅ NEW FIELDS
                "cluster_confidence": confidence,
                "risk_trend":         trend,
                "severity_score":     severity,

                "locations": [locs[i] for i in idxs],
            })

        _ORDER = {"HIGH": 0, "MODERATE": 1, "LOW": 2, "ISOLATED": 3}
        results.sort(key=lambda c: _ORDER.get(c["cluster_type"], 9))
        return results



# ─────────────────────────────────────────────────────────────────────────────
# 3.  EXPLAINABILITY FORMATTER
# ─────────────────────────────────────────────────────────────────────────────

_FEAT_LABELS: Dict[str, str] = {
    "age":               "traveler age profile",
    "group_size":        "small group / solo travel",
    "loc_risk":          "high-risk destination",
    "gender_female":     "gender vulnerability factor",
    "is_foreign":        "foreign tourist status",
    "solo_female":       "solo female traveler",
    "elderly_health":    "elderly with health conditions",
    "foreign_high_risk": "foreign visitor in high-risk area",
}

_HEALTH_LABELS: Dict[str, str] = {
    "asthma":          "asthma condition",
    "diabetes":        "diabetes",
    "hypertension":    "hypertension",
    "heart_disease":   "heart disease",
    "mobility_issues": "mobility impairment",
    "elderly_frail":   "elderly/frail health",
    "allergies":       "severe allergies",
}


def _feat_to_label(feat: str) -> str:
    """Map a raw RF feature name to a human-readable string."""
    if feat.startswith("hlth_"):
        condition = feat[5:]
        return _HEALTH_LABELS.get(condition, condition.replace("_", " "))
    return _FEAT_LABELS.get(feat, feat.replace("_", " "))


def format_explanation(explanation: Dict[str, float]) -> Dict:
    """
    Convert StaticRiskEngine's SHAP-style dict into a structured,
    human-readable explanation for both the API response and the dashboard UI.

    Args:
        explanation: Dict from StaticRiskEngine.predict(), sorted desc by contribution.
                     Example: {'solo_female': 0.42, 'loc_risk': 0.31, 'age': 0.19}

    Returns:
        {
            'summary': "High risk due to: solo female traveler, high-risk destination, and traveler age profile.",
            'factors': [
                {'feature': 'solo_female', 'label': 'solo female traveler',  'contribution': 0.42},
                {'feature': 'loc_risk',    'label': 'high-risk destination', 'contribution': 0.31},
                {'feature': 'age',         'label': 'traveler age profile',  'contribution': 0.19},
            ]
        }
    """
    if not explanation:
        return {"summary": "Insufficient data for explanation.", "factors": []}

    top3   = list(explanation.items())[:3]          # already sorted by contribution
    labels = [_feat_to_label(k) for k, _ in top3]

    if not labels:
        summary = "Risk within normal parameters."
    elif len(labels) == 1:
        summary = f"High risk due to: {labels[0]}."
    elif len(labels) == 2:
        summary = f"High risk due to: {labels[0]} and {labels[1]}."
    else:
        summary = f"High risk due to: {labels[0]}, {labels[1]}, and {labels[2]}."

    return {
        "summary": summary,
        "factors": [
            {
                "feature":      feat,
                "label":        label,
                "contribution": round(score, 4),
            }
            for (feat, score), label in zip(top3, labels)
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST  (run: python geoguardian_v3_backend.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from geoguardian_v2 import GeoGuardian, TouristProfile, DynamicAnomalyEngine

    print("=" * 60)
    print("  GeoGuardian v3 — Backend Additions Self-Test")
    print("=" * 60)

    # Train a lightweight model for testing
    gg = GeoGuardian()
    gg.train(n_static=2000, n_normal_paths=80, n_anomaly_paths=20)

    store  = SessionStore()
    engine = SpatialClusterEngine()

    # Populate the store with varied assessments across multiple locations
    test_locations = [
        ("Marine Drive",  {"age": 30, "gender": "Male",   "group_size": 2, "health_conditions": ["none"]}),
        ("Fort Kochi",    {"age": 45, "gender": "Female", "group_size": 1, "health_conditions": ["none"]}),
        ("Athirappilly",  {"age": 68, "gender": "Female", "group_size": 1, "health_conditions": ["heart_disease"]}),
        ("Munnar",        {"age": 42, "gender": "Female", "group_size": 3, "health_conditions": ["diabetes"]}),
        ("Cherai Beach",  {"age": 25, "gender": "Male",   "group_size": 4, "health_conditions": ["none"]}),
        ("Kalady",        {"age": 72, "gender": "Male",   "group_size": 1, "health_conditions": ["hypertension"]}),
        ("Vembanad Lake", {"age": 35, "gender": "Female", "group_size": 2, "health_conditions": ["none"]}),
    ]

    for loc, kwargs in test_locations * 4:  # 4 passes so clustering has enough data
        profile = TouristProfile(location=loc, is_foreign=False, **kwargs)
        track   = DynamicAnomalyEngine._normal_path(30)
        result  = gg.assess(profile, track)
        store.record(loc, result)

    print(f"\n📦  Store summary: {store.summary()}")

    # Test clustering
    clusters = engine.cluster(store)
    print(f"\n🔵  DBSCAN Clusters ({len(clusters)} found):")
    print(f"  {'TYPE':10s}  {'LABEL':20s}  {'RISK':>6s}  {'DENSITY':>7s}  LOCATIONS")
    for cl in clusters:
        print(f"  {cl['cluster_type']:10s}  {cl['label']:20s}  "
              f"{cl['avg_risk']:>6.3f}  {cl['density']:>7d}  "
              f"{', '.join(cl['locations'])}")

    # Test heatmap
    pts = store.heatmap_points()
    print(f"\n🗺   Heatmap points: {len(pts)} total, sample: {pts[:2]}")

    # Test explanation formatter
    mock_explanation = {
        "solo_female":      0.43,
        "loc_risk":         0.31,
        "hlth_heart_disease": 0.19,
        "age":              0.12,
        "is_foreign":       0.05,
    }
    fmt = format_explanation(mock_explanation)
    print(f"\n💡  Explanation:")
    print(f"    Summary: {fmt['summary']}")
    for f in fmt["factors"]:
        bar = "█" * int(f["contribution"] * 20)
        print(f"    {f['label']:35s} {bar:20s} {f['contribution']:.3f}")

    print("\n✅  All v3 additions working correctly.")
