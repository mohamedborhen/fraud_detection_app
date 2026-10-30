"""
api.py — Module 6 : Service FastAPI pour la Détection de Fraude Explicable
===========================================================================
Charge les artefacts depuis artifacts/fraud_artifacts.joblib au démarrage.
Endpoints : /health  /metrics  /predict  /explain
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Compatibilité désérialisation scikit-learn (pour artefacts créés sous sklearn < 1.7)
try:
    import sklearn.compose._column_transformer
    if not hasattr(sklearn.compose._column_transformer, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
except Exception:
    pass

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Chemin des artefacts
# ---------------------------------------------------------------------------
ARTIFACTS_PATH = Path("artifacts") / "fraud_artifacts.joblib"

# ---------------------------------------------------------------------------
# Chargement des artefacts (une seule fois au démarrage)
# ---------------------------------------------------------------------------
artifacts: Dict[str, Any] = {}


def load_artifacts() -> None:
    """Charge fraud_artifacts.joblib dans le dict global `artifacts`."""
    global artifacts
    if not ARTIFACTS_PATH.exists():
        raise FileNotFoundError(f"Artefact introuvable : {ARTIFACTS_PATH}")
    artifacts = joblib.load(ARTIFACTS_PATH)
    required_keys = [
        "preprocessor", "feature_names", "num_feats", "cat_feats",
        "models", "iso_forest", "retained", "tau_star", "Ca", "k_daily",
        "master_table",
    ]
    missing = [k for k in required_keys if k not in artifacts]
    if missing:
        raise KeyError(f"Clés manquantes dans l'artefact : {missing}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bayes_correct(p: np.ndarray, beta: Optional[float]) -> np.ndarray:
    """
    Calibration de Bayes pour corriger le biais introduit par le sous-échantillonnage.
    p_cal = (beta * p) / (beta * p + (1 - p))
    Si beta est None ou 1.0 (pas de sous-échantillonnage), p reste inchangé.
    """
    if beta is None or beta == 1.0:
        return p
    return (beta * p) / (beta * p + (1.0 - p))


def get_clf(pipe: Any) -> Any:
    """
    Extrait le classifieur final d'un Pipeline scikit-learn
    (dernier step du pipe, en ignorant les transformateurs).
    """
    if hasattr(pipe, "steps"):
        return pipe.steps[-1][1]
    return pipe


def encode_transaction(row: Dict[str, Any]) -> np.ndarray:
    """
    Transforme un dictionnaire de variables brutes en matrice encodée
    via le preprocessor ajusté lors de l'entraînement.
    """
    preprocessor = artifacts["preprocessor"]
    num_feats: List[str] = artifacts["num_feats"]
    cat_feats: List[str] = artifacts["cat_feats"]

    r = dict(row)
    if "hour" in r:
        try:
            h = float(r["hour"])
            if "hour_sin" not in r:
                r["hour_sin"] = float(np.sin(2 * np.pi * h / 24.0))
            if "hour_cos" not in r:
                r["hour_cos"] = float(np.cos(2 * np.pi * h / 24.0))
        except Exception:
            pass
    if "day_of_week" not in r:
        r["day_of_week"] = 2.0

    all_feats = num_feats + cat_feats
    df = pd.DataFrame([{f: r.get(f, 0) for f in all_feats}])
    # Assurer les types corrects
    for f in num_feats:
        df[f] = pd.to_numeric(df[f], errors="coerce").fillna(0.0)
    for f in cat_feats:
        df[f] = df[f].astype(str)
    return preprocessor.transform(df)


def predict_proba_calibrated(
    X_enc: np.ndarray, model_name: str
) -> tuple[np.ndarray, float]:
    """Retourne (p_cal, beta) pour un modèle donné."""
    pipe = artifacts["models"][model_name]["pipe"]
    beta: float = artifacts["models"][model_name]["beta"]
    p_raw: np.ndarray = pipe.predict_proba(X_enc)[:, 1]
    p_cal = bayes_correct(p_raw, beta)
    return p_cal, beta


def iso_score(X_enc: np.ndarray) -> float:
    """Score d'anomalie Isolation Forest normalisé en [0, 1] (plus élevé = plus anormal)."""
    iso = artifacts["iso_forest"]
    # decision_function retourne un score : négatif = anomalie
    raw_score = iso.decision_function(X_enc)[0]
    # Normalisation : on inverse pour que 1 = très anormal
    return float(1.0 - (raw_score + 0.5).clip(0.0, 1.0))


def compute_shap_values(
    X_enc: np.ndarray, model_name: str
) -> tuple[np.ndarray, float]:
    """
    Calcule les valeurs SHAP via TreeExplainer (ou Explainer) sur le classifieur final du pipe.
    Retourne (shap_values_1d, base_value) pour la classe 1 (fraude).
    """
    pipe = artifacts["models"][model_name]["pipe"]
    clf = get_clf(pipe)

    try:
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer(X_enc)
    except Exception:
        explainer = shap.Explainer(clf, X_enc)
        shap_vals = explainer(X_enc)

    sv = shap_vals.values
    bv = shap_vals.base_values

    if sv.ndim == 3:
        # multi-output : prendre la classe 1
        sv = sv[:, :, 1]
        bv = bv[:, 1] if hasattr(bv, "__len__") and np.ndim(bv) > 1 else bv

    shap_1d = sv[0]  # première (et unique) ligne
    base = float(bv[0]) if hasattr(bv, "__len__") else float(bv)
    return shap_1d, base


def reason_codes(
    shap_1d: np.ndarray,
    row: Dict[str, Any],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Extrait les top_n variables avec la plus forte contribution SHAP absolue.
    Retourne une liste de dicts avec : variable, valeur_observee, contribution_shap.
    """
    feature_names: List[str] = artifacts["feature_names"]
    num_feats: List[str] = artifacts["num_feats"]
    cat_feats: List[str] = artifacts["cat_feats"]
    all_input_feats = num_feats + cat_feats

    # Mapping : nom encodé → valeur brute (meilleure approximation)
    abs_contrib = np.abs(shap_1d)
    sorted_idx = np.argsort(abs_contrib)[::-1][:top_n]

    codes = []
    for idx in sorted_idx:
        fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        # Trouver la variable d'entrée la plus proche (si encodée, extraire le préfixe)
        raw_val = None
        for raw_feat in all_input_feats:
            if fname.startswith(raw_feat):
                raw_val = row.get(raw_feat, "N/A")
                break
        if raw_val is None:
            raw_val = row.get(fname, "N/A")

        codes.append({
            "variable": fname,
            "valeur_observee": raw_val,
            "contribution_shap": float(shap_1d[idx]),
        })
    return codes


# ---------------------------------------------------------------------------
# Application FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="API Détection de Fraude Explicable",
    description=(
        "Service FastAPI exposant le modèle de détection de fraude entraîné, "
        "la calibration de Bayes, le coût métier et les explications SHAP. "
        "Projet Challenge 2 — ENSI II3-IA & II3-IF 2026-2027."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    load_artifacts()


# ---------------------------------------------------------------------------
# Schémas Pydantic
# ---------------------------------------------------------------------------

class TransactionInput(BaseModel):
    # Champs numériques (num_feats)
    amt: float = Field(default=250.0, description="Montant de la transaction ($)", examples=[250.0])
    city_pop: float = Field(default=50000.0, description="Population de la ville", examples=[50000.0])
    lat: float = Field(default=36.8, description="Latitude du titulaire", examples=[36.8])
    long: float = Field(default=10.2, description="Longitude du titulaire", examples=[10.2])
    merch_lat: float = Field(default=36.9, description="Latitude du marchand", examples=[36.9])
    merch_long: float = Field(default=10.1, description="Longitude du marchand", examples=[10.1])
    tx_count_1h: float = Field(default=1.0, description="Nb transactions dernière heure", examples=[1.0])
    tx_count_24h: float = Field(default=3.0, description="Nb transactions dernières 24h", examples=[3.0])
    amt_ratio_30d: float = Field(default=1.5, description="Ratio montant / moy. 30 jours", examples=[1.5])
    time_since_last_tx: float = Field(default=3600.0, description="Temps depuis dernière tx (sec)", examples=[3600.0])
    dist_to_merchant: float = Field(default=5.0, description="Distance au marchand (km)", examples=[5.0])
    speed_kmh: float = Field(default=2.5, description="Vitesse estimée (km/h)", examples=[2.5])
    hour: float = Field(default=14.0, description="Heure de la transaction", examples=[14.0])
    # Champ catégoriel (cat_feats)
    category: str = Field(default="shopping_net", description="Catégorie du marchand", examples=["shopping_net"])
    # Paramètres de scoring
    threshold: Optional[float] = Field(default=None, description="Seuil τ (défaut = tau_star)", examples=[0.0035])
    model_name: Optional[str] = Field(default=None, description="Modèle ('XGBoost', 'Random Forest', 'Régression logistique')", examples=["XGBoost"])

    class Config:
        extra = "allow"


class PredictResponse(BaseModel):
    model_name: str
    p_raw: float
    p_cal: float
    iso_anomaly_score: float
    threshold: float
    decision: str
    is_fraud: bool
    n_alerts_estimated: int
    cost_estimated: float
    beta: Optional[float] = None


class ExplainResponse(BaseModel):
    model_name: str
    p_raw: float
    p_cal: float
    iso_anomaly_score: float
    threshold: float
    decision: str
    is_fraud: bool
    base_value: float
    shap_values: List[float]
    feature_names: List[str]
    reason_codes: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Vérification de santé du service")
async def health() -> Dict[str, Any]:
    """Vérifie que les artefacts sont chargés et retourne les paramètres clés."""
    if not artifacts:
        raise HTTPException(status_code=503, detail="Artefacts non chargés.")
    return {
        "status": "ok",
        "retained_model": artifacts["retained"],
        "tau_star": float(artifacts["tau_star"]),
        "Ca": float(artifacts["Ca"]),
        "k_daily": int(artifacts["k_daily"]),
        "available_models": list(artifacts["models"].keys()),
        "n_features": len(artifacts["feature_names"]),
    }


@app.get("/metrics", summary="Tableau comparatif des modèles (master_table)")
async def metrics() -> Dict[str, Any]:
    """Retourne le contenu de master_table ainsi que la liste des modèles disponibles."""
    if not artifacts:
        raise HTTPException(status_code=503, detail="Artefacts non chargés.")
    mt: pd.DataFrame = artifacts["master_table"]
    # Remplacer NaN par None pour la sérialisation JSON
    mt_clean = mt.where(mt.notna(), other=None)
    return {
        "master_table": mt_clean.to_dict(orient="records"),
        "available_models": list(artifacts["models"].keys()),
        "retained": artifacts["retained"],
    }


@app.get("/models", summary="Alias de /metrics — liste des modèles disponibles")
async def models_endpoint() -> Dict[str, Any]:
    """Alias de /metrics pour compatibilité."""
    return await metrics()


@app.post("/predict", response_model=PredictResponse, summary="Scoring d'une transaction")
async def predict(tx: TransactionInput) -> PredictResponse:
    """
    Applique le preprocessor, calcule la probabilité calibrée, le score Isolation Forest,
    et retourne la décision ALERTE FRAUDE / NORMALE ainsi que le coût métier estimé.
    """
    if not artifacts:
        raise HTTPException(status_code=503, detail="Artefacts non chargés.")

    model_name: str = tx.model_name or artifacts["retained"]
    if model_name in ("string", "", None):
        model_name = artifacts["retained"]
    if model_name not in artifacts["models"]:
        raise HTTPException(
            status_code=422,
            detail=f"Modèle '{model_name}' inconnu. Disponibles : {list(artifacts['models'].keys())}",
        )

    threshold: float = tx.threshold if tx.threshold is not None else float(artifacts["tau_star"])
    row = tx.model_dump(exclude={"threshold", "model_name"})

    X_enc = encode_transaction(row)
    p_cal, beta = predict_proba_calibrated(X_enc, model_name)
    p_raw_val = float(artifacts["models"][model_name]["pipe"].predict_proba(X_enc)[0, 1])
    p_cal_val = float(p_cal[0])

    is_fraud = bool(p_cal_val >= threshold)
    decision = "ALERTE FRAUDE 🚨" if is_fraud else "NORMALE ✅"

    # Coût métier simplifié pour une transaction individuelle
    amt = float(row.get("amt", 0.0))
    Ca = float(artifacts["Ca"])
    # Si fraude manquée → coût = montant ; si alerte → coût = Ca
    cost_est = Ca if is_fraud else amt if p_cal_val < threshold else 0.0
    n_alerts_est = 1 if is_fraud else 0

    anom = iso_score(X_enc)

    return PredictResponse(
        model_name=model_name,
        p_raw=p_raw_val,
        p_cal=p_cal_val,
        iso_anomaly_score=anom,
        threshold=threshold,
        decision=decision,
        is_fraud=is_fraud,
        n_alerts_estimated=n_alerts_est,
        cost_estimated=cost_est,
        beta=beta,
    )


@app.post("/explain", response_model=ExplainResponse, summary="Explication SHAP d'une transaction")
async def explain(tx: TransactionInput) -> ExplainResponse:
    """
    Retourne la prédiction complète + base_value SHAP, vecteur SHAP complet (24 valeurs),
    et les 3 à 5 codes motifs (raisons principales de l'alerte).
    """
    if not artifacts:
        raise HTTPException(status_code=503, detail="Artefacts non chargés.")

    model_name: str = tx.model_name or artifacts["retained"]
    if model_name in ("string", "", None):
        model_name = artifacts["retained"]
    if model_name not in artifacts["models"]:
        raise HTTPException(
            status_code=422,
            detail=f"Modèle '{model_name}' inconnu. Disponibles : {list(artifacts['models'].keys())}",
        )

    threshold: float = tx.threshold if tx.threshold is not None else float(artifacts["tau_star"])
    row = tx.model_dump(exclude={"threshold", "model_name"})

    X_enc = encode_transaction(row)
    p_cal, beta = predict_proba_calibrated(X_enc, model_name)
    p_raw_val = float(artifacts["models"][model_name]["pipe"].predict_proba(X_enc)[0, 1])
    p_cal_val = float(p_cal[0])

    is_fraud = bool(p_cal_val >= threshold)
    decision = "ALERTE FRAUDE 🚨" if is_fraud else "NORMALE ✅"
    anom = iso_score(X_enc)

    # SHAP
    try:
        shap_1d, base_value = compute_shap_values(X_enc, model_name)
    except Exception as e:
        # Fallback si TreeExplainer échoue (ex. modèle non compatible)
        feature_names_list = artifacts["feature_names"]
        shap_1d = np.zeros(len(feature_names_list))
        base_value = p_cal_val
        warnings.warn(f"SHAP computation failed: {e}")

    codes = reason_codes(shap_1d, row, top_n=5)

    return ExplainResponse(
        model_name=model_name,
        p_raw=p_raw_val,
        p_cal=p_cal_val,
        iso_anomaly_score=anom,
        threshold=threshold,
        decision=decision,
        is_fraud=is_fraud,
        base_value=base_value,
        shap_values=shap_1d.tolist(),
        feature_names=artifacts["feature_names"],
        reason_codes=codes,
    )


# ---------------------------------------------------------------------------
# Point d'entrée direct
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
