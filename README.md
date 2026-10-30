# 🛡️ Explainable Fraud Detection Platform (XAI)

An end-to-end machine learning and explainable AI (XAI) web application for credit card fraud detection. The platform combines a high-performance **FastAPI backend** for real-time inference and SHAP attribution with a feature-rich **Streamlit interactive dashboard** covering exploratory data analysis, benchmark evaluations, financial threshold optimization, and local alert explanations.

---

## 🌟 Key Features

* **Real-time Calibrated Scoring (`api.py`)**: Real-time transaction scoring using supervised tree ensembles with Bayesian probability calibration (`bayes_correct`) to correct for sampling bias, alongside unsupervised anomaly scoring via Isolation Forest.
* **Explainable AI (SHAP & Reason Codes)**: Global Beeswarm / feature importance plots, individual **SHAP Waterfall** visualizations, and automatic generation of top 3 to 5 human-readable **Reason Codes** explaining why an alert was flagged.
* **Cost-Sensitive Decision Optimization**: Interactive business cost simulation balancing missed fraud losses ($\text{FN} \times \text{amt}$) against human review overhead ($N_{\text{alerts}} \times C_a$), dynamically finding the optimal decision threshold $\tau^*$.
* **Robust Temporal Validation**: 60% / 20% / 20% chronological split preserving causality with leak-free behavioral feature engineering (`tx_count_1h`, `tx_count_24h`, `amt_ratio_30d`, `time_since_last_tx`, `dist_to_merchant`, `speed_kmh`).

---

## 📂 Repository Structure

```text
fraud_detection_app/
├── artifacts/
│   ├── fraud_artifacts.joblib            # Pre-trained bundle: preprocessor, models, isolation forest, tau_star, Ca
│   └── master_table.csv                  # Supervised vs Unsupervised benchmark results
├── fraud_detection_notebook.ipynb        # End-to-end research & experimentation notebook
├── api.py                                # Production FastAPI inference service (Module 6)
├── app.py                                # Streamlit 4-tab interactive web dashboard (Modules 1–5)
├── requirements.txt                      # Pinned Python package dependencies
├── .gitignore                            # Excluded cache, virtual environments, and data files
└── README.md                             # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.10 – 3.13 installed
* Git

### 2. Clone the Repository
```bash
git clone https://github.com/mohamedborhen/fraud_detection_app.git
cd fraud_detection_app
```

### 3. Create and Activate Virtual Environment
On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Download the Dataset
The models and exploratory modules utilize the *Credit Card Transactions Fraud Detection* dataset by Kartik Shenoy:
1. Download the dataset from Kaggle: **[Credit Card Transactions Fraud Detection (Sparkov)](https://www.kaggle.com/datasets/kartik2112/fraud-detection)**.
2. Create a `data/` directory at the project root:
   ```powershell
   mkdir data
   ```
3. Extract `fraudTrain.csv` and `fraudTest.csv` directly into the `data/` folder:
   ```text
   fraud_detection_app/
   └── data/
       ├── fraudTrain.csv
       └── fraudTest.csv
   ```
*(Note: Data files are automatically excluded by `.gitignore` to prevent committing large CSV files).*

---

## 🖥️ Running the Application

To take full advantage of live scoring and local XAI explanations, run the backend API and frontend dashboard concurrently in **two separate terminals**:

### Terminal 1 — Start the FastAPI Service
```powershell
# Ensure venv is activated
uvicorn api:app --reload --port 8000
```
* **API Base URL:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Interactive Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **ReDoc Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### Terminal 2 — Start the Streamlit Dashboard
```powershell
# Ensure venv is activated
streamlit run app.py --server.port 8501
```
* **Interactive Dashboard:** [http://localhost:8501](http://localhost:8501)

---

## 🧭 Dashboard Modules (`app.py`)

The Streamlit interface is structured into 4 interactive tabs:

| Tab | Modules | Overview & Capabilities |
|:---|:---|:---|
| **📊 Data & EDA** | Modules 1 & 2 | Real-time transaction KPIs, class imbalance overview, 60/20/20 chronological train/val/test split summary, transaction amount distributions (normal vs. fraudulent), temporal drift analysis, fraud rates by hour and merchant category, and custom CSV drag-and-drop inspection. |
| **🏆 Supervised vs. Unsupervised** | Module 3 & Module 4 (Part 1) | Interactive `master_table` comparing supervised models (*XGBoost*, *Random Forest*, *Logistic Regression*) against unsupervised anomaly detection (*Isolation Forest*, *LOF*, *One-Class SVM*, *Autoencoder*, *DBSCAN*). Metrics include PR-AUC, ROC-AUC, P@k, R@k, $F_2$, MCC, and business cost savings ($ and %). |
| **🎛️ Threshold & Cost Simulator** | Module 4 (Part 2) | Live interactive sliders for decision threshold $\tau \in [0.01, 0.99]$, daily alert budget $k$, and review cost $C_a$. Live update of triggered alerts, Precision, Recall, $F_2$, MCC, confusion matrix heatmap, overlaid PR curve with working point, and total business cost curve ($\sum_{\text{FN}} \text{amt} + N_{\text{alerts}} \times C_a$). |
| **🔬 XAI SHAP & Live API Testing** | Modules 5 & 6 | Global **SHAP Beeswarm** and Mean $\| \text{SHAP} \|$ feature importance plots. Local alert inspector with manual input or test set selection, direct testing of `/predict` and `/explain` endpoints, individual **SHAP Waterfall** plots, and high-contrast **Reason Codes** highlighting variables increasing or decreasing risk. |

---

## 🔌 API Endpoints (`api.py`)

| Method | Endpoint | Description | Sample Input / Output |
|:---:|:---:|:---|:---|
| **GET** | `/health` | Service health check and artifact verification | Returns `status: "ok"`, active champion model, optimal $\tau^*$, review cost $C_a$, and alert budget $k_{\text{daily}}$. |
| **GET** | `/metrics` | Master model benchmark comparison | Returns complete `master_table` records in JSON format and available model keys. |
| **POST** | `/predict` | Real-time transaction scoring | Input: transaction features (`amt`, `category`, `hour`, etc.) + optional `threshold` & `model_name`. Output: raw & calibrated fraud probabilities, Isolation Forest score, and final binary decision (`ALERTE FRAUDE 🚨` / `NORMALE ✅`). |
| **POST** | `/explain` | Full explainability (XAI) attribution | Input: transaction features. Output: prediction details, SHAP base value, complete 24-feature SHAP vector, and top 3–5 **Reason Codes** with signed contributions. |

### Testing via Command Line (PowerShell)
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"amt": 950.0, "category": "shopping_net", "hour": 2.0, "tx_count_1h": 5.0, "model_name": "XGBoost"}'
```

---

## 🛠️ Tech Stack

* **Backend & API:** FastAPI, Uvicorn, Pydantic
* **Frontend UI:** Streamlit, Plotly Express & Graph Objects, Matplotlib
* **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-Learn, Joblib
* **Explainability (XAI):** SHAP (*TreeExplainer*, *LinearExplainer*, *Explainer*), LIME
* **Data Processing:** Pandas, NumPy, SciPy

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
