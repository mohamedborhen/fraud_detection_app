"""
app.py — Modules 1 à 5 : Application Streamlit Interactive de Détection de Fraude Explicable
==============================================================================================
4 Onglets :
  Onglet 1 : Données & Analyse Exploratoire (Modules 1 & 2)
  Onglet 2 : Comparaison Supervisé vs Non Supervisé (Modules 3 & 4 partie 1)
  Onglet 3 : Seuil d'Alerte & Coût Métier en Direct (Module 4 partie 2)
  Onglet 4 : Explicabilité (XAI SHAP) & Test API (Modules 5 & 6)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import shap
import streamlit as st

# Compatibilité désérialisation scikit-learn (pour artefacts créés sous sklearn < 1.7)
try:
    import sklearn.compose._column_transformer
    if not hasattr(sklearn.compose._column_transformer, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
except Exception:
    pass

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Configuration Streamlit
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🔍 Détection de Fraude Explicable",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
ARTIFACTS_PATH = ROOT / "artifacts" / "fraud_artifacts.joblib"
MASTER_CSV_PATH = ROOT / "artifacts" / "master_table.csv"
DATA_DIR = ROOT / "data"
API_URL = "http://127.0.0.1:8000"

# ---------------------------------------------------------------------------
# Chargement des artefacts (mis en cache)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Chargement des artefacts…")
def load_artifacts() -> Dict[str, Any]:
    if not ARTIFACTS_PATH.exists():
        st.error(f"❌ Artefact introuvable : {ARTIFACTS_PATH}")
        st.stop()
    art = joblib.load(ARTIFACTS_PATH)
    return art


@st.cache_data(show_spinner="Chargement du tableau comparatif…")
def load_master_table() -> pd.DataFrame:
    if MASTER_CSV_PATH.exists():
        return pd.read_csv(MASTER_CSV_PATH)
    art = load_artifacts()
    return art["master_table"]


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule et ajoute les features temporelles et comportementales nécessaires au preprocessor."""
    if df.empty:
        return df

    df = df.copy()

    # Dates et heure
    if "trans_date_trans_time" in df.columns:
        dt = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
        hour = dt.dt.hour.fillna(12)
        if "hour" not in df.columns:
            df["hour"] = hour
        if "hour_sin" not in df.columns:
            df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        if "hour_cos" not in df.columns:
            df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        if "day_of_week" not in df.columns:
            df["day_of_week"] = dt.dt.day_of_week.fillna(2).astype(float)
    else:
        if "hour" in df.columns and "hour_sin" not in df.columns:
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
        if "day_of_week" not in df.columns:
            df["day_of_week"] = 2.0

    # Distance au marchand (Haversine en km)
    if "dist_to_merchant" not in df.columns and all(c in df.columns for c in ["lat", "long", "merch_lat", "merch_long"]):
        lat1, lon1, lat2, lon2 = map(np.radians, [df["lat"], df["long"], df["merch_lat"], df["merch_long"]])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
        df["dist_to_merchant"] = 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    # Variables comportementales
    if "time_since_last_tx" not in df.columns:
        if "cc_num" in df.columns and "unix_time" in df.columns:
            df = df.sort_values(["cc_num", "unix_time"])
            df["time_since_last_tx"] = df.groupby("cc_num")["unix_time"].diff().fillna(86400.0)
        else:
            df["time_since_last_tx"] = 86400.0

    if "speed_kmh" not in df.columns:
        if "dist_to_merchant" in df.columns and "time_since_last_tx" in df.columns:
            hours = (df["time_since_last_tx"] / 3600.0).clip(lower=1e-3)
            df["speed_kmh"] = (df["dist_to_merchant"] / hours).clip(0, 1000)
        else:
            df["speed_kmh"] = 0.0

    if "tx_count_1h" not in df.columns:
        df["tx_count_1h"] = 1.0
    if "tx_count_24h" not in df.columns:
        df["tx_count_24h"] = 1.0
    if "amt_ratio_30d" not in df.columns:
        if "amt" in df.columns and "cc_num" in df.columns:
            user_mean = df.groupby("cc_num")["amt"].transform("mean")
            df["amt_ratio_30d"] = (df["amt"] / (user_mean + 1e-3)).clip(0.1, 100.0)
        else:
            df["amt_ratio_30d"] = 1.0

    return df


@st.cache_data(show_spinner="Chargement des données de test…", ttl=600)
def load_test_data(max_rows: int = 50_000) -> pd.DataFrame:
    """Charge test_transactions.csv (prioritaire) ou fraudTest.csv depuis data/."""
    candidates = [
        ROOT / "test_transactions.csv",
        DATA_DIR / "fraudTest.csv",
        DATA_DIR / "fraudTrain.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, nrows=max_rows)
            # Nettoyage basique
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])
            if "trans_date_trans_time" in df.columns:
                df["trans_date_trans_time"] = pd.to_datetime(
                    df["trans_date_trans_time"], errors="coerce"
                )
            # Enrichissement automatique si features manquantes
            df = enrich_features(df)
            return df
    st.warning("⚠️ Aucun fichier de données trouvé dans data/. Certains graphiques seront vides.")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bayes_correct(p: np.ndarray, beta: Optional[float]) -> np.ndarray:
    if beta is None or beta == 1.0:
        return p
    return (beta * p) / (beta * p + (1.0 - p))


def f2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    if p + r == 0:
        return 0.0
    return (1 + 4) * p * r / (4 * p + r)


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    p_cal: np.ndarray,
    threshold: float,
    amt: np.ndarray,
    Ca: float,
    k: int,
) -> Dict[str, Any]:
    """Calcule toutes les métriques métier pour un seuil τ donné."""
    y_pred = (p_cal >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    n_alerts = tp + fp
    prec = tp / n_alerts if n_alerts > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f2 = f2_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0

    # Coût métier : montants des FN + Ca * N_alertes
    fn_mask = (y_pred == 0) & (y_true == 1)
    cost = float(np.sum(amt[fn_mask])) + n_alerts * Ca

    # Précision@k et Rappel@k
    top_k_idx = np.argsort(p_cal)[::-1][:k]
    top_k_true = y_true[top_k_idx]
    p_at_k = float(np.mean(top_k_true)) if k > 0 else 0.0
    n_fraud = int(y_true.sum())
    r_at_k = float(np.sum(top_k_true) / n_fraud) if n_fraud > 0 else 0.0

    return {
        "threshold": threshold,
        "n_alerts": n_alerts,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec,
        "recall": rec,
        "f2": f2,
        "mcc": mcc,
        "cost": cost,
        "p_at_k": p_at_k,
        "r_at_k": r_at_k,
    }


def get_calibrated_probs(art: Dict[str, Any], df: pd.DataFrame, model_name: str) -> Optional[np.ndarray]:
    """Calcule les probabilités calibrées sur tout le DataFrame de test."""
    try:
        num_feats: List[str] = art["num_feats"]
        cat_feats: List[str] = art["cat_feats"]
        preprocessor = art["preprocessor"]
        pipe = art["models"][model_name]["pipe"]
        beta: float = art["models"][model_name]["beta"]

        df = enrich_features(df)
        all_feats = num_feats + cat_feats

        X = pd.DataFrame(index=df.index)
        for f in num_feats:
            if f in df.columns:
                X[f] = pd.to_numeric(df[f], errors="coerce").fillna(0.0)
            else:
                X[f] = 0.0
        for f in cat_feats:
            if f in df.columns:
                X[f] = df[f].astype(str).fillna("misc_net")
            else:
                X[f] = "misc_net"

        X_enc = preprocessor.transform(X[all_feats])
        p_raw = pipe.predict_proba(X_enc)[:, 1]
        return bayes_correct(p_raw, beta)
    except Exception as e:
        st.warning(f"Erreur lors du calcul des probabilités : {e}")
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(art: Dict[str, Any]) -> None:
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/ENSI_Logo.png/200px-ENSI_Logo.png",
        width=120,
    )
    st.sidebar.markdown("## 🛡️ Détection de Fraude")
    st.sidebar.markdown("**Projet Challenge 2**  \nENSI — II3-IA & II3-IF (2026-2027)")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Modèle retenu :** `{art['retained']}`")
    st.sidebar.markdown(f"**Seuil optimal τ*** : `{art['tau_star']:.4f}`")
    st.sidebar.markdown(f"**Coût analyse Ca :** `{art['Ca']:.2f} $`")
    st.sidebar.markdown(f"**Budget quotidien k :** `{art['k_daily']}`")
    st.sidebar.markdown("---")

    # Vérification API
    try:
        resp = requests.get(f"{API_URL}/health", timeout=2)
        if resp.status_code == 200:
            st.sidebar.success("✅ API FastAPI connectée")
        else:
            st.sidebar.warning("⚠️ API répond avec une erreur")
    except Exception:
        st.sidebar.error("❌ API FastAPI non disponible\n\n`uvicorn api:app --reload --port 8000`")

    st.sidebar.markdown("---")
    st.sidebar.caption("Modèles disponibles :")
    for mname in art["models"].keys():
        icon = "🏆" if mname == art["retained"] else "•"
        st.sidebar.caption(f"{icon} {mname}")


# ---------------------------------------------------------------------------
# ONGLET 1 : Données & Analyse Exploratoire (Modules 1 & 2)
# ---------------------------------------------------------------------------

def tab_eda(art: Dict[str, Any], df: pd.DataFrame) -> None:
    st.header("📊 Données & Analyse Exploratoire")

    if df.empty:
        st.error("Aucune donnée disponible. Vérifiez le dossier `data/`.")
        return

    # ── KPIs ────────────────────────────────────────────────────────────────
    st.subheader("📌 Indicateurs Clés (KPIs)")

    n_total = len(df)
    n_fraud = int(df["is_fraud"].sum()) if "is_fraud" in df.columns else 0
    fraud_rate = n_fraud / n_total if n_total > 0 else 0.0
    amt_fraud = df.loc[df["is_fraud"] == 1, "amt"].sum() if "is_fraud" in df.columns and "amt" in df.columns else 0
    amt_total = df["amt"].sum() if "amt" in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{n_total:,}")
    col2.metric("Transactions Frauduleuses", f"{n_fraud:,}")
    col3.metric("Taux de Fraude", f"{fraud_rate:.3%}")
    col4.metric("Montant Total Fraudé", f"${amt_fraud:,.2f}")

    st.markdown("---")

    # ── Séparation chronologique 60/20/20 ──────────────────────────────────
    st.subheader("🕐 Séparation Chronologique Train / Val / Test (60% / 20% / 20%)")
    if "trans_date_trans_time" in df.columns:
        df_sorted = df.sort_values("trans_date_trans_time").reset_index(drop=True)
        n = len(df_sorted)
        cut1, cut2 = int(n * 0.6), int(n * 0.8)

        split_info = pd.DataFrame({
            "Ensemble": ["Train (60%)", "Validation (20%)", "Test (20%)"],
            "De": [
                str(df_sorted["trans_date_trans_time"].iloc[0])[:10],
                str(df_sorted["trans_date_trans_time"].iloc[cut1])[:10],
                str(df_sorted["trans_date_trans_time"].iloc[cut2])[:10],
            ],
            "À": [
                str(df_sorted["trans_date_trans_time"].iloc[cut1 - 1])[:10],
                str(df_sorted["trans_date_trans_time"].iloc[cut2 - 1])[:10],
                str(df_sorted["trans_date_trans_time"].iloc[-1])[:10],
            ],
            "Transactions": [cut1, cut2 - cut1, n - cut2],
            "Fraudes": [
                int(df_sorted["is_fraud"].iloc[:cut1].sum()) if "is_fraud" in df.columns else "N/A",
                int(df_sorted["is_fraud"].iloc[cut1:cut2].sum()) if "is_fraud" in df.columns else "N/A",
                int(df_sorted["is_fraud"].iloc[cut2:].sum()) if "is_fraud" in df.columns else "N/A",
            ],
        })
        st.dataframe(split_info, use_container_width=True, hide_index=True)
    else:
        st.info("Colonne `trans_date_trans_time` non disponible pour la séparation chronologique.")

    st.markdown("---")

    # ── Distribution des montants ───────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("💰 Distribution des Montants par Classe")
        if "amt" in df.columns and "is_fraud" in df.columns:
            fig = px.box(
                df, x="is_fraud", y="amt",
                color="is_fraud",
                color_discrete_map={0: "#3498db", 1: "#e74c3c"},
                labels={"is_fraud": "Fraude (1=Oui)", "amt": "Montant ($)"},
                title="Distribution des montants : Normale vs Frauduleuse",
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("📈 Évolution Temporelle du Taux de Fraude")
        if "trans_date_trans_time" in df.columns and "is_fraud" in df.columns:
            df_tmp = df.copy()
            df_tmp["date"] = df_tmp["trans_date_trans_time"].dt.date
            daily = df_tmp.groupby("date")["is_fraud"].agg(["sum", "count"]).reset_index()
            daily.columns = ["date", "fraudes", "total"]
            daily["taux"] = daily["fraudes"] / daily["total"] * 100
            fig2 = px.line(
                daily, x="date", y="taux",
                title="Taux de fraude quotidien (%)",
                labels={"date": "Date", "taux": "Taux (%)"},
                color_discrete_sequence=["#e74c3c"],
            )
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── Taux de fraude par heure ────────────────────────────────────────────
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("🕰️ Taux de Fraude par Heure de la Journée")
        if "trans_date_trans_time" in df.columns and "is_fraud" in df.columns:
            df_tmp2 = df.copy()
            df_tmp2["hour"] = df_tmp2["trans_date_trans_time"].dt.hour
            hourly = df_tmp2.groupby("hour")["is_fraud"].agg(["sum", "count"]).reset_index()
            hourly.columns = ["hour", "fraudes", "total"]
            hourly["taux"] = hourly["fraudes"] / hourly["total"] * 100
            fig3 = px.bar(
                hourly, x="hour", y="taux",
                title="Taux de fraude par heure (%)",
                labels={"hour": "Heure", "taux": "Taux (%)"},
                color="taux",
                color_continuous_scale="Reds",
            )
            fig3.update_layout(height=350, coloraxis_showscale=False)
            st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.subheader("🏪 Taux de Fraude par Catégorie Marchande")
        if "category" in df.columns and "is_fraud" in df.columns:
            cat_fraud = df.groupby("category")["is_fraud"].agg(["sum", "count"]).reset_index()
            cat_fraud.columns = ["category", "fraudes", "total"]
            cat_fraud["taux"] = cat_fraud["fraudes"] / cat_fraud["total"] * 100
            cat_fraud = cat_fraud.sort_values("taux", ascending=True)
            fig4 = px.bar(
                cat_fraud, y="category", x="taux",
                orientation="h",
                title="Taux de fraude par catégorie (%)",
                labels={"category": "Catégorie", "taux": "Taux (%)"},
                color="taux",
                color_continuous_scale="Oranges",
            )
            fig4.update_layout(height=400, coloraxis_showscale=False)
            st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # ── Variables comportementales ──────────────────────────────────────────
    st.subheader("🧠 Variables Comportementales (sans fuite de données)")
    behavioral_cols = [
        "tx_count_1h", "tx_count_24h", "amt_ratio_30d",
        "time_since_last_tx", "dist_to_merchant", "speed_kmh",
    ]
    available_behavioral = [c for c in behavioral_cols if c in df.columns]

    if available_behavioral and "is_fraud" in df.columns:
        desc_data = []
        for col in available_behavioral:
            for fraud_val, label in [(0, "Normale"), (1, "Fraude")]:
                subset = df.loc[df["is_fraud"] == fraud_val, col].dropna()
                desc_data.append({
                    "Variable": col,
                    "Classe": label,
                    "Moyenne": f"{subset.mean():.3f}",
                    "Médiane": f"{subset.median():.3f}",
                    "Std": f"{subset.std():.3f}",
                    "Min": f"{subset.min():.3f}",
                    "Max": f"{subset.max():.3f}",
                })
        st.dataframe(pd.DataFrame(desc_data), use_container_width=True, hide_index=True)

        # Violinplots
        n_cols_beh = min(3, len(available_behavioral))
        for i in range(0, len(available_behavioral), n_cols_beh):
            cols_beh = st.columns(n_cols_beh)
            for j, col in enumerate(available_behavioral[i : i + n_cols_beh]):
                with cols_beh[j]:
                    fig_v = px.violin(
                        df.sample(min(5000, len(df))),
                        y=col, x="is_fraud", color="is_fraud",
                        color_discrete_map={0: "#3498db", 1: "#e74c3c"},
                        box=True,
                        title=col,
                        labels={"is_fraud": "Fraude"},
                    )
                    fig_v.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig_v, use_container_width=True)
    else:
        st.info(
            "Les variables comportementales (`tx_count_1h`, `dist_to_merchant`, etc.) "
            "sont calculées par le notebook. Elles ne sont pas présentes dans le fichier brut."
        )

    st.markdown("---")

    # ── Dérive temporelle (Temporal Drift) ─────────────────────────────────
    st.subheader("📉 Dérive Temporelle (Temporal Drift)")
    if "trans_date_trans_time" in df.columns and "is_fraud" in df.columns and "amt" in df.columns:
        df_drift = df.copy()
        df_drift["month"] = df_drift["trans_date_trans_time"].dt.to_period("M").astype(str)
        monthly = df_drift.groupby("month").agg(
            n_tx=("is_fraud", "count"),
            n_fraud=("is_fraud", "sum"),
            avg_amt=("amt", "mean"),
        ).reset_index()
        monthly["fraud_rate"] = monthly["n_fraud"] / monthly["n_tx"] * 100

        fig_drift = go.Figure()
        fig_drift.add_trace(go.Bar(
            x=monthly["month"], y=monthly["n_tx"],
            name="Total transactions", marker_color="#3498db", opacity=0.6, yaxis="y",
        ))
        fig_drift.add_trace(go.Scatter(
            x=monthly["month"], y=monthly["fraud_rate"],
            name="Taux de fraude (%)", mode="lines+markers",
            marker_color="#e74c3c", yaxis="y2",
        ))
        fig_drift.update_layout(
            title="Dérive temporelle mensuelle : volume et taux de fraude",
            yaxis=dict(title="Nb transactions"),
            yaxis2=dict(title="Taux de fraude (%)", overlaying="y", side="right"),
            legend=dict(orientation="h"),
            height=400,
        )
        st.plotly_chart(fig_drift, use_container_width=True)

    # ── Import CSV personnalisé ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📁 Importer un Fichier CSV Personnalisé")
    uploaded = st.file_uploader(
        "Glisser-déposer un fichier CSV de transactions :", type=["csv"]
    )
    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            st.success(f"✅ Fichier chargé : {df_up.shape[0]} lignes, {df_up.shape[1]} colonnes")
            st.dataframe(df_up.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Erreur de lecture : {e}")


# ---------------------------------------------------------------------------
# ONGLET 2 : Comparaison Supervisé vs Non Supervisé (Modules 3 & 4 partie 1)
# ---------------------------------------------------------------------------

def tab_benchmark(art: Dict[str, Any]) -> None:
    st.header("🏆 Comparaison Supervisé vs Non Supervisé")

    master: pd.DataFrame = load_master_table()

    # ── Tableau comparatif ──────────────────────────────────────────────────
    st.subheader("📋 Tableau Comparatif Complet (master_table)")

    # Formatage du tableau
    display_cols = [c for c in master.columns if c in [
        "Modèle", "Famille", "PR-AUC test", "ROC-AUC test",
        "P@100 test", "R@100 test", "τ*", "F2 test", "MCC test",
        "Alertes test", "Coût test ($)", "Sans modèle ($)", "Économie ($)", "Économie (%)",
    ]]
    if not display_cols:
        display_cols = master.columns.tolist()

    st.dataframe(
        master[display_cols] if display_cols else master,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # ── Graphique comparatif PR-AUC ─────────────────────────────────────────
    st.subheader("📊 Comparaison PR-AUC & ROC-AUC")

    col1, col2 = st.columns(2)
    with col1:
        if "PR-AUC test" in master.columns and "Modèle" in master.columns:
            mt_sorted = master.dropna(subset=["PR-AUC test"]).sort_values("PR-AUC test", ascending=True)
            colors = [
                "#2ecc71" if "Supervisé" in str(r) else "#e67e22"
                for r in mt_sorted.get("Famille", pd.Series([""] * len(mt_sorted)))
            ]
            fig_pr = px.bar(
                mt_sorted,
                y="Modèle", x="PR-AUC test",
                orientation="h",
                color="Famille" if "Famille" in mt_sorted.columns else None,
                color_discrete_map={"Supervisé": "#2ecc71", "Non supervisé": "#e67e22"},
                title="PR-AUC (test) — Supervisé vs Non Supervisé",
                labels={"PR-AUC test": "PR-AUC", "Modèle": ""},
            )
            fig_pr.update_layout(height=350)
            st.plotly_chart(fig_pr, use_container_width=True)

    with col2:
        if "ROC-AUC test" in master.columns and "Modèle" in master.columns:
            mt_sorted2 = master.dropna(subset=["ROC-AUC test"]).sort_values("ROC-AUC test", ascending=True)
            fig_roc = px.bar(
                mt_sorted2,
                y="Modèle", x="ROC-AUC test",
                orientation="h",
                color="Famille" if "Famille" in mt_sorted2.columns else None,
                color_discrete_map={"Supervisé": "#2ecc71", "Non supervisé": "#e67e22"},
                title="ROC-AUC (test) — Supervisé vs Non Supervisé",
                labels={"ROC-AUC test": "ROC-AUC", "Modèle": ""},
            )
            fig_roc.update_layout(height=350)
            st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")

    # ── Coût métier comparatif ──────────────────────────────────────────────
    st.subheader("💸 Coût Métier & Économies Réalisées")
    if "Coût test ($)" in master.columns and "Modèle" in master.columns:
        mt_cost = master.dropna(subset=["Coût test ($)"])
        fig_cost = go.Figure()
        if "Sans modèle ($)" in mt_cost.columns:
            sans_modele = mt_cost["Sans modèle ($)"].iloc[0]
            fig_cost.add_hline(
                y=sans_modele, line_dash="dash", line_color="red",
                annotation_text=f"Sans modèle : ${sans_modele:,.0f}",
            )
        for _, row in mt_cost.iterrows():
            color = "#27ae60" if float(row.get("Économie ($)", 0) or 0) >= 0 else "#e74c3c"
            fig_cost.add_trace(go.Bar(
                x=[row["Modèle"]], y=[row["Coût test ($)"]],
                name=row["Modèle"],
                marker_color=color,
            ))
        fig_cost.update_layout(
            title="Coût métier par modèle ($) — plus c'est bas, mieux c'est",
            yaxis_title="Coût total ($)",
            showlegend=False,
            height=400,
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    if "Économie (%)" in master.columns and "Modèle" in master.columns:
        mt_eco = master.dropna(subset=["Économie (%)"])
        fig_eco = px.bar(
            mt_eco.sort_values("Économie (%)"),
            y="Modèle", x="Économie (%)",
            orientation="h",
            color="Économie (%)",
            color_continuous_scale="RdYlGn",
            title="Économie réalisée vs. sans modèle (%)",
        )
        fig_eco.update_layout(height=350)
        st.plotly_chart(fig_eco, use_container_width=True)

    st.markdown("---")

    # ── Autres métriques ────────────────────────────────────────────────────
    st.subheader("📐 F2 Score, MCC et Précision@k")
    col3, col4 = st.columns(2)

    with col3:
        if "F2 test" in master.columns and "Modèle" in master.columns:
            mt_f2 = master.dropna(subset=["F2 test"]).sort_values("F2 test", ascending=True)
            fig_f2 = px.bar(
                mt_f2, y="Modèle", x="F2 test",
                orientation="h",
                color="Famille" if "Famille" in mt_f2.columns else None,
                color_discrete_map={"Supervisé": "#3498db", "Non supervisé": "#f39c12"},
                title="Score F2 (test)",
            )
            fig_f2.update_layout(height=300)
            st.plotly_chart(fig_f2, use_container_width=True)

    with col4:
        if "MCC test" in master.columns and "Modèle" in master.columns:
            mt_mcc = master.dropna(subset=["MCC test"]).sort_values("MCC test", ascending=True)
            fig_mcc = px.bar(
                mt_mcc, y="Modèle", x="MCC test",
                orientation="h",
                color="Famille" if "Famille" in mt_mcc.columns else None,
                color_discrete_map={"Supervisé": "#9b59b6", "Non supervisé": "#1abc9c"},
                title="Matthews Correlation Coefficient (test)",
            )
            fig_mcc.update_layout(height=300)
            st.plotly_chart(fig_mcc, use_container_width=True)


# ---------------------------------------------------------------------------
# ONGLET 3 : Seuil d'Alerte & Coût Métier en Direct (Module 4 partie 2)
# ---------------------------------------------------------------------------

def tab_threshold(art: Dict[str, Any], df: pd.DataFrame) -> None:
    st.header("🎛️ Simulateur de Seuil d'Alerte & Coût Métier")

    tau_star = float(art["tau_star"])
    Ca_default = float(art["Ca"])
    k_daily_default = int(art["k_daily"])
    retained = art["retained"]

    if df.empty or "is_fraud" not in df.columns:
        st.error("Données de test non disponibles.")
        return

    # ── Sélection du modèle ─────────────────────────────────────────────────
    model_choices = list(art["models"].keys())
    selected_model = st.selectbox(
        "Modèle supervisé à simuler :", model_choices,
        index=model_choices.index(retained) if retained in model_choices else 0,
    )

    # ── Calcul des probabilités calibrées ──────────────────────────────────
    with st.spinner("Calcul des probabilités calibrées…"):
        p_cal = get_calibrated_probs(art, df, selected_model)

    if p_cal is None:
        st.warning("Impossible de calculer les probabilités. Vérifiez que les colonnes numériques sont présentes.")
        return

    y_true = df["is_fraud"].values.astype(int)
    amt = df["amt"].values.astype(float) if "amt" in df.columns else np.zeros(len(y_true))

    # ── Curseurs interactifs ────────────────────────────────────────────────
    st.subheader("⚙️ Paramètres de Simulation")
    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        tau = st.slider(
            "Seuil d'alerte τ", min_value=0.01, max_value=0.99,
            value=float(round(tau_star, 4)), step=0.001,
            format="%.3f",
            help="Seuil de classification : p_cal ≥ τ → ALERTE",
        )
    with col_s2:
        k = st.slider(
            "Budget quotidien d'alertes k", min_value=1, max_value=max(1000, k_daily_default * 5),
            value=k_daily_default,
            help="Nombre maximum d'alertes quotidiennes à analyser",
        )
    with col_s3:
        Ca = st.number_input(
            "Coût d'analyse Ca ($)", min_value=0.0, max_value=10000.0,
            value=float(Ca_default), step=1.0,
            help="Coût unitaire d'analyse d'une alerte par un analyste",
        )

    # ── Métriques en direct ─────────────────────────────────────────────────
    metrics = compute_metrics_at_threshold(y_true, p_cal, tau, amt, Ca, k)

    st.markdown("---")
    st.subheader("📊 Métriques en Direct")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Alertes déclenchées", f"{metrics['n_alerts']:,}")
    m2.metric("Précision", f"{metrics['precision']:.3f}")
    m3.metric("Rappel", f"{metrics['recall']:.3f}")
    m4.metric("Score F₂", f"{metrics['f2']:.3f}")
    m5.metric("MCC", f"{metrics['mcc']:.3f}")
    m6.metric("Coût Métier Total", f"${metrics['cost']:,.0f}")

    m7, m8, m9, m10 = st.columns(4)
    m7.metric("TP", f"{metrics['tp']}")
    m8.metric("FP", f"{metrics['fp']}")
    m9.metric("FN", f"{metrics['fn']}")
    m10.metric("TN", f"{metrics['tn']}")

    prec_at_k = metrics["p_at_k"]
    rec_at_k = metrics["r_at_k"]
    st.info(f"**Précision@{k}** : {prec_at_k:.3f}  |  **Rappel@{k}** : {rec_at_k:.3f}")

    st.markdown("---")

    # ── Matrice de confusion ────────────────────────────────────────────────
    col_cm, col_pr = st.columns(2)

    with col_cm:
        st.subheader("🔢 Matrice de Confusion")
        cm = np.array([
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ])
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Prédit Normal", "Prédit Fraude"],
            y=["Réel Normal", "Réelle Fraude"],
            text=[[str(v) for v in row] for row in cm],
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        ))
        fig_cm.update_layout(title=f"Matrice de confusion (τ={tau:.3f})", height=300)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_pr:
        st.subheader("📈 Courbes Précision-Rappel")
        prec_curve, rec_curve, thresh_curve = precision_recall_curve(y_true, p_cal)
        fig_prc = go.Figure()
        fig_prc.add_trace(go.Scatter(
            x=rec_curve, y=prec_curve,
            mode="lines", name=selected_model,
            line=dict(color="#2ecc71", width=2),
        ))
        # Point au seuil actuel
        fig_prc.add_trace(go.Scatter(
            x=[metrics["recall"]], y=[metrics["precision"]],
            mode="markers", name=f"τ={tau:.3f}",
            marker=dict(color="red", size=12, symbol="star"),
        ))
        fig_prc.update_layout(
            title="Courbe Précision-Rappel",
            xaxis_title="Rappel", yaxis_title="Précision",
            height=300,
        )
        st.plotly_chart(fig_prc, use_container_width=True)

    st.markdown("---")

    # ── Courbe de coût en fonction du seuil ────────────────────────────────
    st.subheader("💰 Courbe de Coût Métier en Fonction du Seuil τ")

    thresholds_range = np.linspace(0.01, 0.99, 100)
    costs = []
    n_alerts_list = []
    prec_list = []
    rec_list = []
    f2_list = []

    for t in thresholds_range:
        m = compute_metrics_at_threshold(y_true, p_cal, t, amt, Ca, k)
        costs.append(m["cost"])
        n_alerts_list.append(m["n_alerts"])
        prec_list.append(m["precision"])
        rec_list.append(m["recall"])
        f2_list.append(m["f2"])

    fig_cost_curve = go.Figure()
    fig_cost_curve.add_trace(go.Scatter(
        x=thresholds_range, y=costs,
        mode="lines", name="Coût métier ($)",
        line=dict(color="#e74c3c", width=2),
    ))
    fig_cost_curve.add_vline(
        x=tau, line_dash="dash", line_color="#3498db",
        annotation_text=f"τ={tau:.3f}",
    )
    fig_cost_curve.add_vline(
        x=tau_star, line_dash="dot", line_color="#27ae60",
        annotation_text=f"τ*={tau_star:.3f}",
    )
    fig_cost_curve.update_layout(
        title="Coût métier total (fraudes manquées + coût d'analyse) vs Seuil τ",
        xaxis_title="Seuil τ", yaxis_title="Coût ($)",
        height=400,
    )
    st.plotly_chart(fig_cost_curve, use_container_width=True)

    # ── Métriques vs seuil ─────────────────────────────────────────────────
    fig_metrics_curve = go.Figure()
    fig_metrics_curve.add_trace(go.Scatter(
        x=thresholds_range, y=prec_list, mode="lines", name="Précision",
        line=dict(color="#2ecc71"),
    ))
    fig_metrics_curve.add_trace(go.Scatter(
        x=thresholds_range, y=rec_list, mode="lines", name="Rappel",
        line=dict(color="#3498db"),
    ))
    fig_metrics_curve.add_trace(go.Scatter(
        x=thresholds_range, y=f2_list, mode="lines", name="F₂",
        line=dict(color="#9b59b6", width=2),
    ))
    fig_metrics_curve.add_vline(
        x=tau, line_dash="dash", line_color="gray", annotation_text=f"τ={tau:.3f}",
    )
    fig_metrics_curve.add_vline(
        x=tau_star, line_dash="dot", line_color="#27ae60", annotation_text=f"τ*={tau_star:.3f}",
    )
    fig_metrics_curve.update_layout(
        title="Précision, Rappel et F₂ en fonction du seuil τ",
        xaxis_title="Seuil τ", yaxis_title="Score",
        height=400,
    )
    st.plotly_chart(fig_metrics_curve, use_container_width=True)

    st.markdown("---")

    # ── Économies vs sans modèle ────────────────────────────────────────────
    st.subheader("💡 Économies vs. Sans Modèle")
    cost_no_model = float(np.sum(amt[y_true == 1]))
    current_cost = metrics["cost"]
    saving = cost_no_model - current_cost
    saving_pct = (saving / cost_no_model * 100) if cost_no_model > 0 else 0

    col_save1, col_save2, col_save3 = st.columns(3)
    col_save1.metric("Coût sans modèle", f"${cost_no_model:,.0f}")
    col_save2.metric("Coût avec modèle (τ actuel)", f"${current_cost:,.0f}")
    col_save3.metric("Économie", f"${saving:,.0f}", delta=f"{saving_pct:.1f}%")


# ---------------------------------------------------------------------------
# ONGLET 4 : Explicabilité XAI & Test API (Modules 5 & 6)
# ---------------------------------------------------------------------------

def tab_xai(art: Dict[str, Any], df: pd.DataFrame) -> None:
    st.header("🔬 Explicabilité (XAI SHAP) & Test API en Direct")

    retained = art["retained"]
    num_feats: List[str] = art["num_feats"]
    cat_feats: List[str] = art["cat_feats"]
    feature_names: List[str] = art["feature_names"]
    tau_star = float(art["tau_star"])

    # ── Sélection du modèle ─────────────────────────────────────────────────
    model_choices = list(art["models"].keys())
    selected_model = st.selectbox(
        "Modèle pour les explications SHAP :", model_choices,
        index=model_choices.index(retained) if retained in model_choices else 0,
        key="xai_model",
    )

    pipe = art["models"][selected_model]["pipe"]
    beta = art["models"][selected_model]["beta"]
    preprocessor = art["preprocessor"]

    st.markdown("---")

    # ── Explication Globale ─────────────────────────────────────────────────
    st.subheader("🌍 Importance Globale des Variables (SHAP)")

    @st.cache_data(show_spinner="Calcul des valeurs SHAP globales…")
    def compute_global_shap(_pipe, _preprocessor, _df, _num_feats, _cat_feats, _feature_names, _model_name):
        """Calcule les valeurs SHAP sur un échantillon du jeu de test."""
        all_feats = _num_feats + _cat_feats
        missing = [f for f in all_feats if f not in _df.columns]
        if missing or _df.empty:
            return None, None

        sample_df = _df.sample(min(500, len(_df)), random_state=42)
        X = sample_df[all_feats].copy()
        for f in _num_feats:
            X[f] = pd.to_numeric(X[f], errors="coerce").fillna(0.0)
        for f in _cat_feats:
            X[f] = X[f].astype(str)

        try:
            X_enc = _preprocessor.transform(X)
            clf = _pipe.steps[-1][1] if hasattr(_pipe, "steps") else _pipe
            try:
                explainer = shap.TreeExplainer(clf)
                shap_vals = explainer(X_enc)
            except Exception:
                explainer = shap.Explainer(clf, X_enc[:50])
                shap_vals = explainer(X_enc)

            sv = shap_vals.values
            if sv.ndim == 3:
                sv = sv[:, :, 1]
            return sv, X_enc
        except Exception as e:
            st.warning(f"SHAP global non disponible : {e}")
            return None, None

    all_feats = num_feats + cat_feats
    shap_global, X_enc_global = compute_global_shap(
        pipe, preprocessor, df, num_feats, cat_feats, feature_names, selected_model
    )

    if shap_global is not None:
        tab_global_1, tab_global_2 = st.tabs(["🐝 Beeswarm (Dot) Plot", "📊 Importance Globale (Bar)"])

        with tab_global_1:
            st.markdown("**SHAP Beeswarm Plot** — chaque point = une transaction, couleur = valeur de la variable")
            fig_bee, ax_bee = plt.subplots(figsize=(10, 6))
            # Adapter les noms de features pour l'affichage
            n_features_shap = shap_global.shape[1]
            display_names = feature_names[:n_features_shap] if len(feature_names) >= n_features_shap else feature_names

            shap.summary_plot(
                shap_global, X_enc_global,
                feature_names=display_names,
                plot_type="dot",
                show=False,
                max_display=15,
            )
            plt.tight_layout()
            st.pyplot(fig_bee, use_container_width=True)
            plt.close()

        with tab_global_2:
            st.markdown("**SHAP Mean |SHAP| Bar Plot** — importance moyenne absolue par variable")
            mean_abs_shap = np.abs(shap_global).mean(axis=0)
            n_show = min(15, len(mean_abs_shap))
            top_idx = np.argsort(mean_abs_shap)[::-1][:n_show]
            top_names = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in top_idx]
            top_vals = mean_abs_shap[top_idx]

            fig_bar = px.bar(
                x=top_vals[::-1], y=top_names[::-1],
                orientation="h",
                labels={"x": "Importance SHAP Moyenne (|SHAP|)", "y": "Variable"},
                title=f"Top {n_show} Variables — Importance Globale ({selected_model})",
                color=top_vals[::-1],
                color_continuous_scale="Blues",
            )
            fig_bar.update_layout(height=450, coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info(
            "L'explication SHAP globale n'est pas disponible (données de test manquantes "
            "ou variables comportementales absentes du fichier brut)."
        )

    st.markdown("---")

    # ── Inspecteur d'Alertes Local ─────────────────────────────────────────
    st.subheader("🔎 Inspecteur d'Alerte Local — Explication d'une Transaction")

    input_mode = st.radio(
        "Mode d'entrée :", ["Saisie manuelle", "Sélection depuis les données de test"],
        horizontal=True,
    )

    row_input: Dict[str, Any] = {}

    if input_mode == "Saisie manuelle":
        st.markdown("#### Saisie des Variables de la Transaction")

        # Valeurs par défaut raisonnables
        defaults_num = {
            "amt": 250.0, "city_pop": 50000.0, "lat": 36.8, "long": 10.2,
            "merch_lat": 36.9, "merch_long": 10.1, "tx_count_1h": 1.0,
            "tx_count_24h": 3.0, "amt_ratio_30d": 1.5, "time_since_last_tx": 7200.0,
            "dist_to_merchant": 5.0, "speed_kmh": 2.5, "hour": 14.0,
            "hour_sin": -0.5, "hour_cos": -0.866, "day_of_week": 2.0,
        }

        cols_form = st.columns(3)
        for i, feat in enumerate(num_feats):
            with cols_form[i % 3]:
                row_input[feat] = st.number_input(
                    label=feat,
                    value=float(defaults_num.get(feat, 0.0)),
                    format="%.4f",
                    key=f"manual_{feat}",
                )

        for feat in cat_feats:
            unique_cats = []
            if "category" in df.columns and not df.empty:
                unique_cats = sorted(df["category"].dropna().unique().tolist())
            if not unique_cats:
                unique_cats = [
                    "gas_transport", "grocery_pos", "home", "kids_pets",
                    "misc_net", "misc_pos", "personal_care", "shopping_net",
                    "shopping_pos", "entertainment", "food_dining", "health_fitness",
                    "travel",
                ]
            row_input[feat] = st.selectbox(f"Catégorie ({feat})", unique_cats, key=f"manual_{feat}")

    else:  # Sélection depuis les données
        if df.empty:
            st.warning("Aucune donnée disponible pour la sélection.")
            return

        fraud_only = st.checkbox("Afficher uniquement les transactions frauduleuses", value=True)
        df_sel = df[df["is_fraud"] == 1] if fraud_only and "is_fraud" in df.columns else df
        df_sel = df_sel.reset_index(drop=True)

        n_sel = min(200, len(df_sel))
        tx_idx = st.slider(
            "Index de la transaction :", 0, max(0, n_sel - 1), 0, key="tx_select"
        )

        if len(df_sel) > 0:
            selected_row = df_sel.iloc[tx_idx]

            # Affichage de la ligne sélectionnée
            with st.expander("📄 Détails de la transaction sélectionnée", expanded=False):
                st.json(selected_row.to_dict())

            for feat in num_feats:
                row_input[feat] = float(selected_row.get(feat, 0.0))
            for feat in cat_feats:
                row_input[feat] = str(selected_row.get(feat, "misc_net"))

            if "is_fraud" in selected_row:
                truth = "🚨 Fraude" if selected_row["is_fraud"] == 1 else "✅ Normale"
                st.info(f"**Étiquette réelle :** {truth}")
            if "amt" in selected_row:
                st.info(f"**Montant :** ${float(selected_row['amt']):.2f}")

    # ── Paramètres de scoring ──────────────────────────────────────────────
    st.markdown("---")
    col_thresh, col_model2 = st.columns(2)
    with col_thresh:
        custom_threshold = st.slider(
            "Seuil τ pour cette transaction :", 0.01, 0.99,
            value=float(round(tau_star, 4)), step=0.001,
            key="local_tau",
        )
    with col_model2:
        local_model = st.selectbox(
            "Modèle :", model_choices,
            index=model_choices.index(selected_model),
            key="local_model_sel",
        )

    # ── Bouton d'envoi à l'API ─────────────────────────────────────────────
    st.markdown("---")
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        predict_clicked = st.button("🚀 Envoyer à l'API /predict", use_container_width=True, type="primary")
    with col_btn2:
        explain_clicked = st.button("🔬 Envoyer à l'API /explain", use_container_width=True, type="secondary")

    # ── Scoring via /predict ────────────────────────────────────────────────
    if predict_clicked:
        payload = {**row_input, "threshold": custom_threshold, "model_name": local_model}
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
            if resp.status_code == 200:
                result = resp.json()
                st.markdown("### 📋 Résultat du Scoring")

                decision = result.get("decision", "")
                if "FRAUDE" in decision or "ALERTE" in decision:
                    st.error(f"## {decision}")
                else:
                    st.success(f"## {decision}")

                cols_res = st.columns(4)
                cols_res[0].metric("Probabilité Brute (p_raw)", f"{result['p_raw']:.4f}")
                cols_res[1].metric("Probabilité Calibrée (p_cal)", f"{result['p_cal']:.4f}")
                cols_res[2].metric("Score Anomalie (Isolation Forest)", f"{result['iso_anomaly_score']:.4f}")
                cols_res[3].metric("Seuil τ appliqué", f"{result['threshold']:.4f}")

                with st.expander("📦 Réponse JSON complète"):
                    st.json(result)
            else:
                st.error(f"Erreur API {resp.status_code} : {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error("❌ API FastAPI non disponible. Lancez : `uvicorn api:app --reload --port 8000`")
        except Exception as e:
            st.error(f"Erreur : {e}")

    # ── Explication via /explain ────────────────────────────────────────────
    if explain_clicked:
        payload = {**row_input, "threshold": custom_threshold, "model_name": local_model}
        try:
            resp = requests.post(f"{API_URL}/explain", json=payload, timeout=30)
            if resp.status_code == 200:
                result = resp.json()
                st.markdown("### 🧠 Explication SHAP de la Transaction")

                decision = result.get("decision", "")
                if "FRAUDE" in decision or "ALERTE" in decision:
                    st.error(f"## {decision}")
                else:
                    st.success(f"## {decision}")

                cols_res = st.columns(4)
                cols_res[0].metric("p_raw", f"{result['p_raw']:.4f}")
                cols_res[1].metric("p_cal", f"{result['p_cal']:.4f}")
                cols_res[2].metric("Score Anomalie IF", f"{result['iso_anomaly_score']:.4f}")
                cols_res[3].metric("Seuil τ", f"{result['threshold']:.4f}")

                # ── Codes Motifs ────────────────────────────────────────────
                st.markdown("---")
                st.markdown("#### 🗝️ Codes Motifs — Raisons de l'Alerte")
                reason_codes = result.get("reason_codes", [])
                if reason_codes:
                    codes_df = pd.DataFrame(reason_codes)
                    codes_df.columns = ["Variable", "Valeur Observée", "Contribution SHAP"]
                    codes_df["Contribution SHAP"] = codes_df["Contribution SHAP"].apply(lambda x: f"{x:.4f}")
                    codes_df["Sens"] = codes_df["Contribution SHAP"].apply(
                        lambda x: "🔴 Augmente risque" if float(x) > 0 else "🟢 Réduit risque"
                    )

                    for _, code_row in codes_df.iterrows():
                        shap_val = float(code_row["Contribution SHAP"])
                        is_risk = shap_val > 0
                        bg_color = "rgba(231, 76, 60, 0.12)" if is_risk else "rgba(46, 204, 113, 0.12)"
                        border_color = "#e74c3c" if is_risk else "#2ecc71"
                        tag_color = "#ff6b6b" if is_risk else "#2ecc71"
                        tag_bg = "rgba(231, 76, 60, 0.25)" if is_risk else "rgba(46, 204, 113, 0.25)"
                        tag_label = "Augmente risque" if is_risk else "Réduit risque"
                        icon = "⬆️" if is_risk else "⬇️"

                        st.markdown(
                            f"""
                            <div style="background:{bg_color};border-left:5px solid {border_color};padding:10px 14px;border-radius:6px;margin:8px 0;display:flex;align-items:center;justify-content:space-between;color:#E0E0E0;">
                                <div>
                                    <span style="font-weight:700;font-size:15px;color:#FFFFFF;">{icon} {code_row['Variable']}</span>
                                    <span style="margin-left:15px;color:#BDC3C7;font-size:13px;">Valeur :</span>
                                    <span style="background:#1E272E;color:#F1C40F;padding:3px 8px;border-radius:4px;font-family:monospace;font-size:13px;font-weight:600;margin-left:6px;border:1px solid #34495E;">{code_row['Valeur Observée']}</span>
                                </div>
                                <div>
                                    <span style="color:#BDC3C7;font-size:13px;margin-right:6px;">Impact SHAP :</span>
                                    <span style="color:{tag_color};font-weight:bold;font-size:15px;font-family:monospace;">{shap_val:+.4f}</span>
                                    <span style="background:{tag_bg};color:{tag_color};border:1px solid {tag_color};padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600;margin-left:8px;">{tag_label}</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                # ── SHAP Waterfall ──────────────────────────────────────────
                st.markdown("---")
                st.markdown("#### 💧 SHAP Waterfall Plot")

                shap_values_list = result.get("shap_values", [])
                base_value = result.get("base_value", 0.0)
                feat_names_api = result.get("feature_names", feature_names)

                if shap_values_list:
                    sv_arr = np.array(shap_values_list)
                    n_display = min(20, len(sv_arr))
                    top_idx_wf = np.argsort(np.abs(sv_arr))[::-1][:n_display]

                    # Waterfall manuel avec Plotly
                    wf_names = [
                        feat_names_api[i] if i < len(feat_names_api) else f"f{i}"
                        for i in top_idx_wf
                    ]
                    wf_vals = sv_arr[top_idx_wf]

                    # Tri croissant pour waterfall lisible
                    sort_order = np.argsort(wf_vals)
                    wf_names_sorted = [wf_names[i] for i in sort_order]
                    wf_vals_sorted = wf_vals[sort_order]

                    colors_wf = ["#e74c3c" if v > 0 else "#3498db" for v in wf_vals_sorted]

                    fig_wf = go.Figure(go.Bar(
                        y=wf_names_sorted,
                        x=wf_vals_sorted,
                        orientation="h",
                        marker_color=colors_wf,
                        text=[f"{v:+.4f}" for v in wf_vals_sorted],
                        textposition="outside",
                    ))
                    fig_wf.add_vline(x=0, line_color="black", line_width=1)
                    fig_wf.update_layout(
                        title=f"SHAP Waterfall — {local_model} | Base = {base_value:.4f}",
                        xaxis_title="Contribution SHAP",
                        height=max(400, n_display * 25),
                        xaxis=dict(zeroline=True),
                    )
                    st.plotly_chart(fig_wf, use_container_width=True)

                    # Waterfall SHAP natif (si disponible)
                    try:
                        st.markdown("#### 📉 SHAP Waterfall (natif shap)")
                        sv_full = np.array(result["shap_values"])
                        expl_obj = shap.Explanation(
                            values=sv_full,
                            base_values=base_value,
                            feature_names=feat_names_api,
                        )
                        fig_wf_native, ax_wf = plt.subplots(figsize=(10, 6))
                        shap.waterfall_plot(expl_obj, max_display=15, show=False)
                        plt.tight_layout()
                        st.pyplot(plt.gcf(), use_container_width=True)
                        plt.close("all")
                    except Exception:
                        pass  # Le waterfall plotly ci-dessus est suffisant

                with st.expander("📦 Réponse JSON complète /explain"):
                    st.json({k: v for k, v in result.items() if k != "shap_values"})

            else:
                st.error(f"Erreur API {resp.status_code} : {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error("❌ API FastAPI non disponible. Lancez : `uvicorn api:app --reload --port 8000`")
        except Exception as e:
            st.error(f"Erreur : {e}")

    # ── Explication locale directe (sans API) ──────────────────────────────
    st.markdown("---")
    with st.expander("🔧 Explication SHAP Locale Directe (sans API FastAPI)", expanded=False):
        st.info("Cette section calcule les explications SHAP directement dans Streamlit, sans appeler l'API.")
        if st.button("▶️ Calculer SHAP local", key="local_shap_btn"):
            try:
                all_feats_local = num_feats + cat_feats
                X_input_df = pd.DataFrame([row_input])[all_feats_local]
                for f in num_feats:
                    X_input_df[f] = pd.to_numeric(X_input_df[f], errors="coerce").fillna(0.0)
                for f in cat_feats:
                    X_input_df[f] = X_input_df[f].astype(str)

                X_enc_local = preprocessor.transform(X_input_df)

                pipe_local = art["models"][local_model]["pipe"]
                beta_local = art["models"][local_model]["beta"]
                from sklearn.pipeline import Pipeline
                clf_local = pipe_local.steps[-1][1] if hasattr(pipe_local, "steps") else pipe_local

                p_raw_local = float(pipe_local.predict_proba(X_enc_local)[0, 1])
                p_cal_local = float(bayes_correct(np.array([p_raw_local]), beta_local)[0])

                explainer_local = shap.TreeExplainer(clf_local)
                shap_local = explainer_local(X_enc_local)
                sv_local = shap_local.values
                bv_local = shap_local.base_values

                if sv_local.ndim == 3:
                    sv_local = sv_local[:, :, 1]
                    bv_local = bv_local[:, 1] if np.ndim(bv_local) > 1 else bv_local

                sv_local_1d = sv_local[0]
                bv_local_val = float(bv_local[0]) if hasattr(bv_local, "__len__") else float(bv_local)

                decision_local = "ALERTE FRAUDE 🚨" if p_cal_local >= custom_threshold else "NORMALE ✅"
                if "FRAUDE" in decision_local:
                    st.error(f"**{decision_local}** — p_cal = {p_cal_local:.4f}")
                else:
                    st.success(f"**{decision_local}** — p_cal = {p_cal_local:.4f}")

                # Beeswarm / bar local
                top5_idx = np.argsort(np.abs(sv_local_1d))[::-1][:10]
                top5_names = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in top5_idx]
                top5_vals = sv_local_1d[top5_idx]

                fig_local_bar = go.Figure(go.Bar(
                    y=top5_names[::-1],
                    x=top5_vals[::-1],
                    orientation="h",
                    marker_color=["#e74c3c" if v > 0 else "#3498db" for v in top5_vals[::-1]],
                    text=[f"{v:+.4f}" for v in top5_vals[::-1]],
                    textposition="outside",
                ))
                fig_local_bar.add_vline(x=0, line_color="black")
                fig_local_bar.update_layout(
                    title=f"Top 10 Contributions SHAP — {local_model}",
                    xaxis_title="Contribution SHAP",
                    height=400,
                )
                st.plotly_chart(fig_local_bar, use_container_width=True)

            except Exception as e:
                st.error(f"Erreur calcul SHAP local : {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Chargement des artefacts
    art = load_artifacts()
    df = load_test_data()

    # Sidebar
    render_sidebar(art)

    # Titre principal
    st.title("🛡️ Application de Détection de Fraude Explicable")
    st.markdown(
        "> **Projet Challenge 2** — ENSI II3-IA & II3-IF (2026-2027) | "
        "Mme Rym Besrour | *Credit Card Transactions Fraud Detection (Sparkov)*"
    )

    # ── 4 Onglets ──────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Données & EDA",
        "🏆 Supervisé vs Non Supervisé",
        "🎛️ Seuil & Coût Métier",
        "🔬 XAI SHAP & Test API",
    ])

    with tab1:
        tab_eda(art, df)

    with tab2:
        tab_benchmark(art)

    with tab3:
        tab_threshold(art, df)

    with tab4:
        tab_xai(art, df)


if __name__ == "__main__":
    main()
