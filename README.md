Explainable Healthcare Provider Risk Intelligence for Investigation Prioritization
🎯 Problem-Solution Highlights

Problem: Healthcare insurers and payers receive millions of claims, but investigation teams can audit only a small number of providers in each cycle. Fraud labels belong to providers, while the raw data are claim-level records. The fraud class is also rare (only 9.35% of labeled providers), so plain accuracy is misleading. Manual review is slow, inconsistent, and cannot scale to 558,000+ claims.

Solution:

Aggregated inpatient, outpatient, and beneficiary claims into a provider-level dataset (5,410 labeled providers) with a leakage-safe pipeline.
Audited the data, engineered financial, utilization, beneficiary-reuse, diagnosis-entropy, and physician-concentration features, and locked 62 predictors.
Benchmarked Logistic Regression, Random Forest, XGBoost, and LightGBM (cost-weighted) with 5-fold stratified cross-validation.
Calibrated the champion model with Platt scaling so scores behave like real probabilities.
Converted calibrated scores into a capacity-based investigation queue with three review tiers.
Explained every ranking with TreeSHAP, served through a FastAPI backend and a four-tab Streamlit dashboard.

Impact:

Reviewing only the top 50 of 1,353 unseen providers (3.69%) covers 41.18% of billed reimbursement, about 11× better than random selection.
Every score comes with a feature-level explanation, so investigators can see why a provider was ranked high.
Calibrated probabilities (ECE 0.0375 → 0.0059) make the risk score trustworthy for planning audit capacity.

⚠️ Risk scores support investigation prioritization. They are not legal determinations or proof of fraud.

📌 Project Overview

This project builds an end-to-end, explainable provider-risk system on the Kaggle Healthcare Provider Fraud Detection Analysis dataset. Instead of predicting fraud claim by claim, it scores providers and ranks them so that limited investigation capacity is spent where model-estimated risk is highest.

Raw claimsIP + OP + Beneficiary
Data audit &leakage checks
Provider-levelaggregation
Feature engineering& pruning: 62 predictors
Model benchmark5-fold CV
LightGBM champion+ Platt calibration
Investigation queue& tiers
TreeSHAPexplanations
FastAPI + Streamlit
🗂️ Dataset

Source: Healthcare Provider Fraud Detection Analysis – Kaggle (file used: HEALTHCARE PROVIDER FRAUD DETECTION.zip)

Item	Training	Public test
Providers	5,410	1,353
Label-positive (PotentialFraud = Yes)	506 (9.35%)	unlabeled
Inpatient claims	40,474	–
Outpatient claims	517,737	–
Total claims	558,211	135,392
Billed reimbursement	–	$132.87M
Four table types: provider labels, beneficiary details, inpatient claims, outpatient claims.
3,318 of 5,410 training providers are outpatient-only, so inpatient fields are structurally missing and handled with missingness indicators.
PotentialFraud is an administrative label, not independently verified fraud.

🔧 Key Steps
1. Data Audit & Leakage Control
Verified schema agreement between train and test tables, duplicates, unique claim IDs, and provider–beneficiary–claim linkage.
Checked chronology (no negative durations, no post-death claims).
Excluded identifiers and dates from the model; no label-derived features.
2. Provider-Level Aggregation
Inpatient and outpatient claims were aggregated separately and then merged, so each provider contributes exactly one row (no join multiplication).

3. Feature Engineering

Feature families built per provider:

Financial: total / mean / max / standard-deviation reimbursement, deductible patterns.
Utilization: claim counts, inpatient length of stay, claims per beneficiary.
Beneficiary reuse: repeat-beneficiary rate, unique beneficiaries, reimbursement per beneficiary.
Coding behavior: diagnosis diversity, normalized entropy, concentration (HHI).
Physician concentration across attending / operating / other physicians.

4. Feature Reliability & Pruning

Statistical screening (t-test, Mann–Whitney U, Spearman correlation) with multiple-testing correction.
Pruned 14 columns (exact duplicates, deterministic complements, zero variance, low denominator support, correlation ≥ 0.90) → 62 final predictors.
Preprocessing: log1p on 22 heavy-tailed predictors, median imputation + missingness indicators → 90 model inputs, always fitted inside each training fold.

5. Model Development
Models: Dummy baseline, Logistic Regression (L2), Random Forest, XGBoost, LightGBM.
Validation: 5-fold stratified cross-validation, same shuffled folds (seed 42), out-of-fold predictions.
Metrics: Average Precision (AP) as the primary metric, plus ROC-AUC, Precision@50 and Precision@100 (accuracy is misleading at 9.35% prevalence).
Imbalance: cost-weighting (weight = N-/N+ = 4,904 / 506 ≈ 9.69) was compared with SMOTE-NC.

7. Probability Calibration
Platt scaling fitted on out-of-fold log-odds (nested, leakage-free); isotonic regression compared as an alternative.

9. Investigation Prioritization
No arbitrary probability threshold: providers are sorted by calibrated risk and the top-K are reviewed, where K = investigation capacity.
Fixed rank tiers: Tier 1 – Immediate Review, Tier 2 – Priority Review, Tier 3 – Monitor.

11. Explainability & Deployment
TreeSHAP gives exact, additive, feature-level explanations in log-odds.
Artifacts are serialized and served via FastAPI; analysts use a Streamlit dashboard.

🚀 Results & Insights
Model Benchmark (5-fold stratified CV, out-of-fold)
Model	AP	ROC-AUC	Precision@50	Precision@100
LightGBM + cost weighting ✅	0.7769	0.9572	0.8640	0.7140
XGBoost + cost weighting	0.7709	0.9574	0.8480	0.7100
Logistic Regression (L2)	0.7142	0.9501	0.7960	0.6480
Random Forest (weighted)	0.7088	0.9476	0.7880	0.6580
Dummy (prior only)	0.0935	0.5000	0.1040	0.1080
Cost-weighting beat SMOTE-NC: LightGBM AP 0.7719 → 0.7543 (−0.0176) and XGBoost AP 0.7731 → 0.7543 (−0.0188) with synthetic oversampling, so weighting was retained.
Locked champion: LightGBM, 150 trees, learning rate 0.05, max depth 5, 25 leaves, min child samples 20, 0.80 row & feature subsampling, positive-class weight 9.6917, seed 42.
Calibration (Platt scaling)
Metric	Raw LightGBM	After Platt
ECE (Expected Calibration Error)	0.0375	0.0059
Brier score	0.0516	0.0387
Fitted Platt parameters: slope A* = 0.8366, intercept B* = −1.2705.
Ranking quality is unchanged (AP 0.7769, ROC-AUC 0.9572).
Isotonic regression reached ECE 0.0053 but lowered AP to 0.7599, so Platt scaling was chosen.
Investigation Capacity vs Billed Reimbursement Covered (1,353 unseen providers)
Review capacity (top-K)	Reimbursement covered by model queue	Random selection
Top 25	31.99%	1.85%
Top 50	41.18% (≈ $54.72M)	3.69%
Top 100	52.75%	7.39%
Top 200	70.22%	14.78%
Top 500	91.34%	36.96%

Exposure means billed reimbursement placed under review. It is not fraud found or money recovered, because public test labels are not available.

Tiers: Tier 1 = ranks 1–50 (50 providers) • Tier 2 = ranks 51–250 (200 providers) • Tier 3 = ranks 251–1,353 (1,103 providers).

Top Global Risk Drivers (mean |SHAP|, log-odds)
Rank	Feature	Mean |SHAP|
1	Total_Reimbursement	1.1483
2	Max_IP_LOS_Days	0.5622
3	Std_Claims_Per_Beneficiary	0.1932
4	Reimbursement_Per_Beneficiary	0.1370
5	Primary_Diagnosis_Normalized_Entropy	0.1272
6	Repeat_Beneficiary_Rate	0.1176
Example Case – Provider PRV53105
Calibrated risk 97.90%, Rank #1 of 1,353, Tier 1 – Immediate Review.
$2,895,670 billed across 2,979 claims (216 inpatient, 2,763 outpatient) for 1,169 beneficiaries.
Higher-risk SHAP contributions: Total_Reimbursement (+3.2638), Max_IP_LOS_Days (+1.4118), Max_Claims_Per_Beneficiary (+1.0392), Primary_Diagnosis_NonMissing (+0.3584), IP_Claim_Count (+0.2873).
Lower-risk contributions are small (for example Overall_Mean_Diag_Per_Claim, −0.0494).
🖥️ Interactive Dashboard (FastAPI + Streamlit)

The model, preprocessor, Platt calibrator and provider database are saved as deployment artifacts (champion_lightgbm.joblib, production_preprocessor.joblib, platt_calibrator.joblib, provider_database.pkl) and served by a FastAPI service (port 8000) to a four-tab Streamlit dashboard (port 8501).

Tab	What it does
1. Investigation Queue	Choose capacity (25 / 50 / 100 / 200 / 500) and view the ranked providers with calibrated risk, tier and recommended action.
2. Provider Deep Dive	Enter a provider ID to see risk summary, financial & utilization profile, and higher-/lower-risk SHAP contributions.
3. Provider Risk Lookup	Look up a provider and get the full structured JSON report.
4. Model Governance & Validation	Model, calibration parameters (slope 0.8366, intercept −1.2705), feature counts (62 → 90), imbalance ratio 9.69 and policy notes.

REST endpoints

Method	Endpoint	Purpose
GET	/health	Service health check
GET	/model-info	Model, calibration and pipeline metadata
GET	/providers/{provider_id}	Risk, rank, tier, financial profile and SHAP drivers (404 for unknown IDs)
GET	/investigation-queue/top	Top-K ranked providers
▶️ How to Run
Open the notebook in Google Colab (Python 3, GPU not required).
Upload HEALTHCARE PROVIDER FRAUD DETECTION.zip to Google Drive (or the Colab session) and set the path in the first data-loading cell.
Run the notebook top to bottom. It will:
extract and audit the data and build the provider-level features,
benchmark models, calibrate the champion and compute SHAP values,
save deployment artifacts to /content/deployment_artifacts,
create and start the FastAPI app (/content/api_service/app.py) on port 8000,
create and start the Streamlit dashboard (/content/dashboard_app/dashboard.py) on port 8501.
Open the Streamlit URL printed in the final cells to explore the four tabs.

In Colab the dashboard is exposed through a temporary public tunnel, which is suitable for demonstration only.

⚖️ Limitations & Responsible Use
PotentialFraud is an administrative label, not a verified or legal finding of fraud.
Public test providers are unlabeled, so no test-set accuracy can be reported; exposure figures are not fraud found.
The model leans on billing size (Total_Reimbursement dominates SHAP), so small fraudulent providers may be ranked lower.
Results come from a single data window and seed, without confidence intervals.
SHAP explains model behavior; it does not establish causality.
The dashboard is a prototype (temporary tunnel, no authentication or audit logging), not a production deployment.
Use scores only to order manual reviews; validate on independently reviewed cases before real-world use.
📊 Tools & Technologies
Python: Pandas, NumPy, SciPy, Statsmodels, Scikit-learn, LightGBM, XGBoost, imbalanced-learn (SMOTE-NC), SHAP, Matplotlib, Seaborn
Serving: FastAPI, Uvicorn
Dashboard: Streamlit
Environment: Google Colab, Jupyter Notebook, Google Drive
📈 Future Work
Graph neural networks to capture collusive provider–physician–beneficiary billing rings.
Drift monitoring (PSI) and size-stratified models to protect small providers.
Investigator feedback loops and uncertainty quantification (conformal prediction).
Clinical NLP on discharge notes and diagnosis text.
Secure production deployment (authentication, audit logging, containerization).
📖 References
Healthcare Provider Fraud Detection Analysis – Kaggle
Scikit-learn Documentation
LightGBM Documentation
XGBoost Documentation
SHAP Documentation
FastAPI Documentation
Streamlit Documentation
📬 Contact

Soumyadeep Basak – MSc Data Science & Artificial Intelligence, RKMVERI GitHub
