"""
GeoGuardian v2.0 — Unified AI Tourist Safety System
=====================================================
Smart India Hackathon 2025 | Ernakulam District, Kerala

Improvements over v1:
  • Unified GeoGuardian class integrating Static + Dynamic models
  • Kalman Filter-based GPS smoothing before LSTM input
  • SHAP-style local explainability for RF model
  • Precision-Recall based threshold calibration (no fixed 95th percentile)
  • Speed & acceleration features from GPS sequences
  • SOS trigger with confidence scoring and alert levels
  • Model persistence with versioned checkpoints
  • Comprehensive evaluation: ROC-AUC, F1, PR-curve
  • Real Ernakulam-region coordinate bounds for realistic simulation
"""

import numpy as np
import pandas as pd
import joblib
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Literal
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers, Model, Input, callbacks
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError

from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, mean_absolute_error, r2_score
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONSTANTS  — Ernakulam region
# ─────────────────────────────────────────────
ERNAKULAM_BOUNDS = {
    "lat_min": 9.72, "lat_max": 10.22,
    "lon_min": 76.15, "lon_max": 76.65,
    "center":  (9.9312, 76.2673),
}

LOCATION_RISK_MAP: Dict[str, float] = {
    "Marine Drive":   0.18, "Fort Kochi":     0.22, "Mattancherry":   0.25,
    "Cherai Beach":   0.28, "Hill Palace":    0.22, "Vembanad Lake":  0.30,
    "Kalady":         0.35, "Athirappilly":   0.45, "Munnar":         0.40,
    "Ernakulam City": 0.20, "Thripunithura":  0.25, "Aluva":          0.30,
}

HEALTH_SEVERITY: Dict[str, float] = {
    "none": 0.00, "asthma": 0.40, "diabetes": 0.50,
    "hypertension": 0.40, "heart_disease": 0.80,
    "mobility_issues": 0.70, "elderly_frail": 0.90, "allergies": 0.30,
}

RISK_WEIGHTS = {"age": 0.17, "group": 0.20, "gender": 0.18, "health": 0.20, "location": 0.25}

SOS_THRESHOLD   = 0.75   # combined score above this → SOS
ALERT_THRESHOLD = 0.55   # combined score above this → alert

# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────
@dataclass
class TouristProfile:
    age:               int
    gender:            Literal["Male", "Female"]
    group_size:        int
    location:          str
    health_conditions: List[str] = field(default_factory=lambda: ["none"])
    is_foreign:        bool      = False

    def validate(self):
        assert 18 <= self.age <= 100, "Age must be 18-100"
        assert self.group_size >= 1, "Group size must be ≥ 1"
        assert self.location in LOCATION_RISK_MAP, f"Unknown location: {self.location}"
        return self


@dataclass
class GPSPoint:
    lat:       float
    lon:       float
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


@dataclass
class RiskResult:
    static_score:    float
    lstm_score:      float
    combined_score:  float
    risk_level:      Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]
    anomaly_type:    Optional[str]
    alert_level:     Literal["NORMAL", "ALERT", "SOS"]
    confidence:      float
    risk_factors:    Dict[str, float]
    recommendations: List[str]
    explanation:     Dict[str, float]   # SHAP-style feature contributions


# ─────────────────────────────────────────────
# GPS UTILITIES
# ─────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2) -> float:
    """Returns distance in metres between two GPS points."""
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi  = np.radians(lat2 - lat1)
    dlam  = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2*R*np.arcsin(np.sqrt(a))


def kalman_smooth_gps(points: List[GPSPoint], process_noise=1e-4, obs_noise=1e-3) -> List[GPSPoint]:
    """1-D Kalman filter applied independently to lat & lon for GPS denoising."""
    n = len(points)
    if n < 2:
        return points
    lats = np.array([p.lat for p in points])
    lons = np.array([p.lon for p in points])

    def _kalman_1d(z, Q=process_noise, R=obs_noise):
        x_hat, P = z[0], 1.0
        out = []
        for obs in z:
            P  += Q                       # predict
            K   = P / (P + R)             # Kalman gain
            x_hat = x_hat + K*(obs-x_hat) # update
            P  = (1-K)*P
            out.append(x_hat)
        return np.array(out)

    slats = _kalman_1d(lats)
    slons = _kalman_1d(lons)
    return [GPSPoint(lat=float(slats[i]), lon=float(slons[i]), timestamp=points[i].timestamp)
            for i in range(n)]


def extract_motion_features(points: List[GPSPoint]) -> np.ndarray:
    """
    Extracts per-timestep features from GPS sequence:
      [lat, lon, speed_mps, acceleration_mps2, heading_change_deg]
    """
    n = len(points)
    feats = np.zeros((n, 5))
    feats[:, 0] = [p.lat for p in points]
    feats[:, 1] = [p.lon for p in points]

    for i in range(1, n):
        dt = max(points[i].timestamp - points[i-1].timestamp, 0.01)
        dist = haversine(points[i-1].lat, points[i-1].lon, points[i].lat, points[i].lon)
        speed = dist / dt
        feats[i, 2] = speed

        if i > 1:
            feats[i, 3] = (feats[i, 2] - feats[i-1, 2]) / dt     # accel
            # heading
            d1_lat = points[i-1].lat - points[i-2].lat
            d1_lon = points[i-1].lon - points[i-2].lon
            d2_lat = points[i].lat   - points[i-1].lat
            d2_lon = points[i].lon   - points[i-1].lon
            h1 = np.degrees(np.arctan2(d1_lon, d1_lat))
            h2 = np.degrees(np.arctan2(d2_lon, d2_lat))
            feats[i, 4] = abs((h2 - h1 + 180) % 360 - 180)        # heading change

    return feats


# ─────────────────────────────────────────────
# STATIC RISK ENGINE
# ─────────────────────────────────────────────
class StaticRiskEngine:
    """Hybrid Random Forest + Neural Network for tourist profile risk scoring."""

    def __init__(self):
        self.health_enc  = MultiLabelBinarizer()
        self.gender_enc  = OneHotEncoder(drop="first", sparse_output=False)
        self.scaler      = MinMaxScaler()
        self.rf_model    = RandomForestRegressor(n_estimators=150, max_depth=12,
                                                  min_samples_leaf=5, random_state=42)
        self.nn_model: Optional[Model] = None
        self.rf_feature_names: List[str] = []
        self._trained = False

    # ── Dataset generation ──────────────────
    def _generate_dataset(self, n: int = 7000) -> pd.DataFrame:
        np.random.seed(42)
        locs = list(LOCATION_RISK_MAP.keys())
        cond_pool = list(HEALTH_SEVERITY.keys())
        rows = []
        for _ in range(n):
            age      = int(np.random.beta(2, 3) * 52 + 18)
            gender   = np.random.choice(["Male", "Female"])
            grp      = np.random.choice([1,2,3,4,5,6], p=[0.20,0.35,0.20,0.15,0.07,0.03])
            is_fore  = bool(np.random.choice([True, False], p=[0.30, 0.70]))
            loc      = np.random.choice(locs)
            loc_risk = LOCATION_RISK_MAP[loc]

            if age < 30:
                n_c = np.random.choice([0,1], p=[0.85,0.15])
            elif age < 50:
                n_c = np.random.choice([0,1,2], p=[0.65,0.30,0.05])
            else:
                n_c = np.random.choice([0,1,2,3], p=[0.40,0.40,0.15,0.05])

            conditions = ["none"] if n_c == 0 else \
                         list(np.random.choice([c for c in cond_pool if c != "none"],
                                               size=min(n_c, 3), replace=False))

            risk = self._calc_risk_score(age, gender, grp, is_fore, conditions, loc_risk)
            rows.append(dict(age=age, gender=gender, group_size=grp,
                             is_foreign=is_fore, location=loc,
                             loc_risk=loc_risk, conditions=conditions, risk=risk))

        return pd.DataFrame(rows)

    def _calc_risk_score(self, age, gender, grp, is_fore, conds, loc_risk) -> float:
        age_r  = 0.65 if (age < 25 or age > 65) else 0.3
        grp_r  = 0.70 if grp == 1 else 0.40 if grp == 2 else 0.20
        gen_r  = 0.80 if (gender == "Female" and grp == 1) else \
                 0.60 if gender == "Female" else 0.30
        hlth_r = np.mean([HEALTH_SEVERITY.get(c, 0) for c in conds])
        base   = sum(RISK_WEIGHTS[k]*v for k, v in
                     zip(["age","group","gender","health","location"],
                         [age_r, grp_r, gen_r, hlth_r, loc_risk]))
        if is_fore: base *= 1.20
        return float(np.clip(base + np.random.normal(0, 0.02), 0, 1))

    # ── Feature engineering ─────────────────
    def _featurize(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        if fit:
            hlth = self.health_enc.fit_transform(df["conditions"])
            gen  = self.gender_enc.fit_transform(df[["gender"]])
        else:
            hlth = self.health_enc.transform(df["conditions"])
            gen  = self.gender_enc.transform(df[["gender"]])

        fore   = (df["is_foreign"].astype(int)).values.reshape(-1,1)
        num    = df[["age","group_size","loc_risk"]].values
        solo_f = ((df["gender"]=="Female")&(df["group_size"]==1)).astype(int).values.reshape(-1,1)
        eld_h  = ((df["age"]>60)&(hlth.sum(axis=1)>0)).astype(int).values.reshape(-1,1)
        fore_h = ((df["is_foreign"]==True)&(df["loc_risk"]>0.5)).astype(int).values.reshape(-1,1)

        X = np.hstack([num, gen, hlth, fore, solo_f, eld_h, fore_h])
        # Store feature names for explainability
        if fit:
            health_cols = [f"hlth_{c}" for c in self.health_enc.classes_]
            self.rf_feature_names = (
                ["age", "group_size", "loc_risk", "gender_female"]
                + health_cols + ["is_foreign", "solo_female", "elderly_health", "foreign_high_risk"]
            )
        return X

    # ── Neural Network ──────────────────────
    def _build_nn(self, input_dim: int) -> Model:
        inp = Input(shape=(input_dim,))
        # Wide branch
        w = layers.Dense(256, activation="relu")(inp)
        w = layers.Dropout(0.35)(w)
        w = layers.Dense(64, activation="relu")(w)
        # Deep branch
        d = layers.Dense(128, activation="relu")(inp)
        d = layers.BatchNormalization()(d)
        d = layers.Dropout(0.25)(d)
        d = layers.Dense(64, activation="relu")(d)
        d = layers.Dropout(0.15)(d)
        d = layers.Dense(32, activation="relu")(d)
        # Merge
        x = layers.Concatenate()([w, d])
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dropout(0.10)(x)
        out = layers.Dense(1, activation="sigmoid")(x)

        model = Model(inp, out)
        def _loss(y_true, y_pred):
            return 0.6*MeanSquaredError()(y_true,y_pred) + 0.4*MeanAbsoluteError()(y_true,y_pred)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=_loss, metrics=["mae"])
        return model

    # ── Train ───────────────────────────────
    def train(self, n_samples: int = 7000):
        print("🔄  Generating Ernakulam tourist dataset…")
        df = self._generate_dataset(n_samples)
        X  = self._featurize(df, fit=True)
        y  = df["risk"].values
        Xs = self.scaler.fit_transform(X)

        Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42)

        print("🧠  Training Neural Network…")
        self.nn_model = self._build_nn(Xs.shape[1])
        self.nn_model.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=80,
                          batch_size=32, verbose=0,
                          callbacks=[
                              callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                              callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
                          ])

        print("🌲  Training Random Forest…")
        self.rf_model.fit(Xtr, ytr)
        self._trained = True

        # Evaluation
        nn_p  = self.nn_model.predict(Xte, verbose=0).flatten()
        rf_p  = self.rf_model.predict(Xte)
        ens_p = 0.5*nn_p + 0.5*rf_p

        print(f"\n{'='*55}")
        print("  STATIC MODEL EVALUATION")
        print(f"{'='*55}")
        for name, pred in [("NN", nn_p), ("RF", rf_p), ("Ensemble", ens_p)]:
            mae = mean_absolute_error(yte, pred)
            r2  = r2_score(yte, pred)
            print(f"  {name:12s} | MAE: {mae:.4f} | R²: {r2:.4f}")
        print(f"{'='*55}\n")
        return self

    # ── Predict ─────────────────────────────
    def predict(self, profile: TouristProfile) -> Tuple[float, Dict[str, float]]:
        assert self._trained, "Call .train() first."
        profile.validate()
        tmp = pd.DataFrame([{
            "age": profile.age, "gender": profile.gender,
            "group_size": profile.group_size, "is_foreign": profile.is_foreign,
            "location": profile.location,
            "loc_risk": LOCATION_RISK_MAP[profile.location],
            "conditions": profile.health_conditions,
        }])
        X  = self._featurize(tmp, fit=False)
        Xs = self.scaler.transform(X)
        nn_r  = float(self.nn_model.predict(Xs, verbose=0)[0][0])
        rf_r  = float(self.rf_model.predict(Xs)[0])
        score = 0.5*nn_r + 0.5*rf_r

        # SHAP-style: RF feature importances × local feature value delta
        importances = self.rf_model.feature_importances_
        base_val    = 0.35   # approximate dataset mean
        feat_vals   = Xs[0]
        explanation = {
            self.rf_feature_names[i]: float(importances[i] * abs(feat_vals[i] - 0.5) * 2)
            for i in range(len(self.rf_feature_names))
        }
        # Keep top-5 by contribution
        explanation = dict(sorted(explanation.items(), key=lambda x: -x[1])[:5])
        return np.clip(score, 0, 1), explanation

    # ── Save / Load ─────────────────────────
    def save(self, prefix: str = "static_engine"):
        self.nn_model.save(f"{prefix}_nn.keras")
        joblib.dump({
            "rf": self.rf_model, "scaler": self.scaler,
            "health_enc": self.health_enc, "gender_enc": self.gender_enc,
            "feature_names": self.rf_feature_names
        }, f"{prefix}_components.pkl")
        print(f"✅  Static engine saved → {prefix}_*")

    def load(self, prefix: str = "static_engine"):
        self.nn_model = tf.keras.models.load_model(
            f"{prefix}_nn.keras",
            compile=False
        )
        comps = joblib.load(f"{prefix}_components.pkl")
        self.rf_model, self.scaler = comps["rf"], comps["scaler"]
        self.health_enc, self.gender_enc = comps["health_enc"], comps["gender_enc"]
        self.rf_feature_names = comps["feature_names"]
        self._trained = True
        print(f"✅  Static engine loaded ← {prefix}_*")
        return self


# ─────────────────────────────────────────────
# DYNAMIC ANOMALY ENGINE  (LSTM Autoencoder)
# ─────────────────────────────────────────────
class DynamicAnomalyEngine:
    """
    LSTM Autoencoder trained on normal tourist paths.
    Input: sliding window of GPS + motion features (lat, lon, speed, accel, heading)
    Reconstruction error → anomaly score.
    Threshold calibrated via Precision-Recall curve.
    """
    SEQ_LEN  = 30    # timesteps per window
    N_FEATS  = 5     # lat, lon, speed, accel, heading_change

    def __init__(self):
        self.scaler    = MinMaxScaler()
        self.model: Optional[Model] = None
        self.threshold = 0.05
        self._trained  = False

    # ── Path generators ─────────────────────
    @staticmethod
    def _normal_path(n: int = 30, start=(9.93, 76.27), interval_s: float = 30.0) -> List[GPSPoint]:
        lat, lon = start
        t0 = 0.0
        pts = []
        for i in range(n):
            lat += np.random.normal(0.0004, 0.00015)
            lon += np.random.normal(0.0004, 0.00015)
            pts.append(GPSPoint(lat=lat, lon=lon, timestamp=t0 + i*interval_s))
        return pts

    @staticmethod
    def _anomaly_path(atype: str = "deviation", n: int = 30,
                      start=(9.93, 76.27), interval_s: float = 30.0) -> List[GPSPoint]:
        pts = DynamicAnomalyEngine._normal_path(n, start, interval_s)
        ji  = np.random.randint(12, 20)
        if atype == "deviation":
            for i in range(ji, n):
                pts[i] = GPSPoint(pts[i].lat + 0.008 * np.sign(np.random.randn()),
                                  pts[i].lon + 0.008 * np.sign(np.random.randn()),
                                  pts[i].timestamp)
        elif atype == "stall":
            anchor = pts[ji]
            for i in range(ji, n):
                pts[i] = GPSPoint(anchor.lat + np.random.normal(0,0.00003),
                                  anchor.lon + np.random.normal(0,0.00003),
                                  pts[i].timestamp)
        elif atype == "loop":
            for i in range(ji, n):
                t = (i - ji) * 0.35
                pts[i] = GPSPoint(pts[ji].lat + np.sin(t)*0.0018,
                                  pts[ji].lon + np.cos(t)*0.0018,
                                  pts[i].timestamp)
        return pts

    def _path_to_feats(self, pts: List[GPSPoint]) -> np.ndarray:
        return extract_motion_features(kalman_smooth_gps(pts))

    # ── Build model ─────────────────────────
    def _build(self) -> Model:
        inp = Input(shape=(self.SEQ_LEN, self.N_FEATS))
        # Encoder
        x = layers.LSTM(64, return_sequences=True)(inp)
        x = layers.LSTM(32)(x)
        enc = layers.Dense(16, activation="relu")(x)
        # Decoder
        x = layers.RepeatVector(self.SEQ_LEN)(enc)
        x = layers.LSTM(32, return_sequences=True)(x)
        x = layers.LSTM(64, return_sequences=True)(x)
        out = layers.TimeDistributed(layers.Dense(self.N_FEATS))(x)
        m = Model(inp, out)
        m.compile(optimizer="adam", loss="mse")
        return m

    # ── Train ───────────────────────────────
    def train(self, n_normal: int = 600, n_anomaly: int = 60):
        print("🔄  Generating GPS paths…")
        starts = [(np.random.uniform(9.75,10.15), np.random.uniform(76.20,76.55))
                  for _ in range(n_normal)]
        normal_raw = [self._normal_path(self.SEQ_LEN, s) for s in starts]
        normal_feats = np.array([self._path_to_feats(p) for p in normal_raw])

        flat = normal_feats.reshape(-1, self.N_FEATS)
        self.scaler.fit(flat)
        X_tr = (self.scaler.transform(flat)).reshape(normal_feats.shape)

        print("🧠  Training LSTM Autoencoder…")
        self.model = self._build()
        self.model.fit(X_tr, X_tr, epochs=25, batch_size=32,
                       validation_split=0.15, verbose=0,
                       callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

        # Calibrate threshold via PR curve
        print("📐  Calibrating anomaly threshold…")
        a_types  = ["deviation", "stall", "loop"]
        anom_raw = [self._anomaly_path(np.random.choice(a_types), self.SEQ_LEN)
                    for _ in range(n_anomaly)]
        anom_f   = np.array([self._path_to_feats(p) for p in anom_raw])
        X_anom   = (self.scaler.transform(anom_f.reshape(-1,self.N_FEATS))).reshape(anom_f.shape)

        X_eval = np.concatenate([X_tr[:n_anomaly], X_anom])
        y_eval = np.array([0]*n_anomaly + [1]*n_anomaly)

        recon = self.model.predict(X_eval, verbose=0)
        mse   = np.mean(np.mean((X_eval - recon)**2, axis=2), axis=1)

        prec, rec, thr = precision_recall_curve(y_eval, mse)
        f1s   = 2*(prec*rec)/(prec+rec+1e-9)
        best  = np.argmax(f1s[:-1])
        self.threshold = float(thr[best])
        self._trained  = True

        print(f"\n{'='*55}")
        print("  DYNAMIC MODEL EVALUATION")
        print(f"{'='*55}")
        preds = (mse > self.threshold).astype(int)
        print(classification_report(y_eval, preds, target_names=["Normal","Anomaly"]))
        roc = roc_auc_score(y_eval, mse)
        print(f"  ROC-AUC : {roc:.4f}")
        print(f"  Threshold (PR-optimal): {self.threshold:.5f}")
        print(f"{'='*55}\n")
        return self

    # ── Predict ─────────────────────────────
    def predict(self, points: List[GPSPoint]) -> Tuple[float, float, Optional[str]]:
        """
        Returns (anomaly_score 0-1, reconstruction_error, anomaly_type_guess).
        anomaly_score = sigmoid-normalised reconstruction error.
        """
        assert self._trained, "Call .train() first."
        if len(points) < self.SEQ_LEN:
            return 0.05, 0.0, None

        pts   = points[-self.SEQ_LEN:]
        feats = self._path_to_feats(pts)
        Xs    = self.scaler.transform(feats).reshape(1, self.SEQ_LEN, self.N_FEATS)
        recon = self.model.predict(Xs, verbose=0)
        err   = float(np.mean((Xs - recon)**2))

        score = float(np.clip((err - self.threshold) /
                               (self.threshold * 5 + 1e-9) + 0.5, 0, 1))

        # Heuristic anomaly type detection
        speeds = [extract_motion_features(pts)[i, 2] for i in range(len(pts))]
        if np.mean(speeds[-5:]) < 0.3:
            atype = "stall"
        elif err > self.threshold * 3:
            atype = "deviation"
        elif score > 0.3:
            atype = "loop"
        else:
            atype = None

        return score, err, atype

    # ── Save / Load ─────────────────────────
    def save(self, prefix: str = "dynamic_engine"):
        self.model.save(f"{prefix}_lstm.keras")
        joblib.dump({"scaler": self.scaler, "threshold": self.threshold}, f"{prefix}_meta.pkl")
        print(f"✅  Dynamic engine saved → {prefix}_*")

    def load(self, prefix: str = "dynamic_engine"):
        self.model = tf.keras.models.load_model(
            f"{prefix}_lstm.keras",
            compile=False
        )
        meta = joblib.load(f"{prefix}_meta.pkl")
        self.scaler, self.threshold = meta["scaler"], meta["threshold"]
        self._trained = True
        print(f"✅  Dynamic engine loaded ← {prefix}_*")
        return self


# ─────────────────────────────────────────────
# UNIFIED GEOGUARDIAN  (main entry point)
# ─────────────────────────────────────────────
class GeoGuardian:
    """
    Combines StaticRiskEngine + DynamicAnomalyEngine.

    Combined score formula (from integration guide in v1):
        combined = 0.30 × static_score + 0.70 × lstm_score

    SOS trigger when combined ≥ SOS_THRESHOLD (0.75).
    """

    def __init__(self):
        self.static  = StaticRiskEngine()
        self.dynamic = DynamicAnomalyEngine()
        self._ready  = False

    def train(self, n_static: int = 7000, n_normal_paths: int = 600, n_anomaly_paths: int = 60):
        print("=" * 60)
        print("  GEOGUARDIAN v2.0 — TRAINING PIPELINE")
        print("=" * 60)
        self.static.train(n_static)
        self.dynamic.train(n_normal_paths, n_anomaly_paths)
        self._ready = True
        print("🎉  GeoGuardian fully trained and ready.\n")
        return self

    def assess(self, profile: TouristProfile, gps_track: List[GPSPoint]) -> RiskResult:
        """Main inference method. Returns a complete RiskResult."""
        assert self._ready, "Call .train() or .load() first."

        static_score, explanation = self.static.predict(profile)
        lstm_score, recon_err, atype = self.dynamic.predict(gps_track)
        combined = float(np.clip(0.30 * static_score + 0.70 * lstm_score, 0, 1))

        # Confidence: higher when both models agree
        agreement = 1 - abs(static_score - lstm_score)
        confidence = float(np.clip(agreement * 0.8 + 0.2, 0, 1))

        risk_level: str
        if combined >= 0.80:
            risk_level = "CRITICAL"
        elif combined >= 0.65:
            risk_level = "HIGH"
        elif combined >= 0.40:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        alert_level: str
        if combined >= SOS_THRESHOLD:
            alert_level = "SOS"
        elif combined >= ALERT_THRESHOLD:
            alert_level = "ALERT"
        else:
            alert_level = "NORMAL"

        recommendations = self._get_recs(combined, profile, atype)

        # Raw factor breakdown
        risk_factors = {
            "age_risk":      0.65 if (profile.age < 25 or profile.age > 65) else 0.30,
            "group_risk":    0.70 if profile.group_size == 1 else 0.40 if profile.group_size == 2 else 0.20,
            "gender_risk":   0.80 if (profile.gender == "Female" and profile.group_size == 1)
                             else 0.60 if profile.gender == "Female" else 0.30,
            "health_risk":   float(np.mean([HEALTH_SEVERITY.get(c, 0) for c in profile.health_conditions])),
            "location_risk": LOCATION_RISK_MAP.get(profile.location, 0.35),
        }

        return RiskResult(
            static_score   = round(static_score, 4),
            lstm_score     = round(lstm_score, 4),
            combined_score = round(combined, 4),
            risk_level     = risk_level,
            anomaly_type   = atype,
            alert_level    = alert_level,
            confidence     = round(confidence, 4),
            risk_factors   = risk_factors,
            recommendations = recommendations,
            explanation    = explanation,
        )

    @staticmethod
    def _get_recs(score: float, profile: TouristProfile, atype: Optional[str]) -> List[str]:
        base = []
        if atype == "stall":
            base.append("⚠ Tourist appears stationary — check for incapacitation.")
        elif atype == "deviation":
            base.append("⚠ Significant route deviation detected — verify location.")
        elif atype == "loop":
            base.append("⚠ Erratic circular movement — possible distress or navigation failure.")

        if score >= 0.75:
            base += ["Immediately contact emergency services (112)",
                     "Send SOS coordinates to registered contacts",
                     "Dispatch nearest tourist police unit"]
        elif score >= 0.55:
            base += ["Issue advisory alert to tourist",
                     "Increase monitoring frequency",
                     "Prepare emergency response standby"]
        elif score >= 0.40:
            base += ["Recommend companion travel",
                     "Avoid isolated zones after dusk",
                     "Share itinerary with trusted contacts"]
        else:
            base += ["Standard safety precautions apply",
                     "Next check-in in 2 hours"]
        return base

    def save(self, prefix: str = "geoguardian_v2"):
        self.static.save(f"{prefix}_static")
        self.dynamic.save(f"{prefix}_dynamic")

    def load(self, prefix: str = "geoguardian_v2"):
        self.static.load(f"{prefix}_static")
        self.dynamic.load(f"{prefix}_dynamic")
        self._ready = True
        return self


# ─────────────────────────────────────────────
# DEMO  — run as script
# ─────────────────────────────────────────────
if __name__ == "__main__":
    gg = GeoGuardian()
    gg.train(n_static=6000, n_normal_paths=500, n_anomaly_paths=50)

    print("\n" + "=" * 60)
    print("  LIVE INFERENCE DEMO")
    print("=" * 60)

    scenarios = [
        {
            "label": "Low-Risk · Domestic couple, Marine Drive",
            "profile": TouristProfile(age=30, gender="Male", group_size=2,
                                      location="Marine Drive", health_conditions=["none"]),
            "path_type": "normal"
        },
        {
            "label": "High-Risk · Solo elderly female foreigner, Athirappilly",
            "profile": TouristProfile(age=68, gender="Female", group_size=1,
                                      location="Athirappilly", health_conditions=["heart_disease"],
                                      is_foreign=True),
            "path_type": "deviation"
        },
        {
            "label": "Moderate-Risk · Indian family, Munnar, stall anomaly",
            "profile": TouristProfile(age=42, gender="Female", group_size=3,
                                      location="Munnar", health_conditions=["diabetes"]),
            "path_type": "stall"
        },
    ]

    for s in scenarios:
        # Generate GPS track
        if s["path_type"] == "normal":
            pts = DynamicAnomalyEngine._normal_path(n=30)
        else:
            pts = DynamicAnomalyEngine._anomaly_path(atype=s["path_type"], n=30)

        result = gg.assess(s["profile"], pts)
        print(f"\n── {s['label']} ──")
        print(f"   Static:   {result.static_score:.3f}")
        print(f"   LSTM:     {result.lstm_score:.3f}")
        print(f"   Combined: {result.combined_score:.3f}  [{result.risk_level}]")
        print(f"   Alert:    {result.alert_level}  (confidence: {result.confidence:.2f})")
        print(f"   Anomaly:  {result.anomaly_type or 'none'}")
        print(f"   Top risk factors: {dict(list(result.risk_factors.items())[:3])}")
        print(f"   Recommendation:   {result.recommendations[0]}")

    gg.save()
    print("\n✅  Model saved. Use GeoGuardian().load() for inference-only deployment.\n")
