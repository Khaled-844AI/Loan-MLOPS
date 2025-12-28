import pandas as pd
import numpy as np
import xgboost as xgb
import wandb
import joblib
import os
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder

# ==========================================
# CONFIGURATION
# ==========================================
PROJECT_NAME = "lending_club_2015"
ENTITY = "mouadkhaled2004-esi"
DATA_PATH_2018 = "data_2018/data_2018.csv"
MODEL_ARTIFACT = "xgb_loan_classifier_2015:latest"

# 🟢 TRUE RESULTS FROM YOUR RUN 'dsgxqlt1' (From Screenshot)
# We use these if the API fails to connect.
REAL_2015_STATS = {
    "test_accuracy": 0.91938,
    "test_auc": 0.96648,
    "test_f1": 0.81177,
    "final_threshold": 0.7
}

# ==========================================
# 1. ATTEMPT TO FETCH METRICS (With Fallback)
# ==========================================
print(f"🔌 Connecting to Run dsgxqlt1 to fetch metrics...")

try:
    # We initialize API cleanly
    api = wandb.Api()
    run_path = f"{ENTITY}/{PROJECT_NAME}/dsgxqlt1"
    baseline_run = api.run(run_path)
    
    # Try to get values
    acc_2015 = baseline_run.summary.get("test_accuracy")
    auc_2015 = baseline_run.summary.get("test_auc")
    f1_2015  = baseline_run.summary.get("test_f1")
    threshold_2015 = baseline_run.summary.get("final_threshold")
    
    if acc_2015 is None: raise ValueError("Metrics missing")
    
    print("✅ API Success! Fetched metrics directly from W&B.")

except Exception as e:
    print(f"⚠️ API Fetch Failed ({e}).")
    print("✅ Using REAL PROVEN metrics from Run 'dsgxqlt1' (Recovered from logs/screenshot).")
    
    acc_2015 = REAL_2015_STATS["test_accuracy"]
    auc_2015 = REAL_2015_STATS["test_auc"]
    f1_2015  = REAL_2015_STATS["test_f1"]
    threshold_2015 = REAL_2015_STATS["final_threshold"]

print(f"   > 2015 Accuracy:  {acc_2015}")
print(f"   > 2015 Threshold: {threshold_2015}")

# ==========================================
# 2. START NEW RUN & PROCESS 2018 DATA
# ==========================================
run = wandb.init(project=PROJECT_NAME, job_type="comparison_final")

print(f"\nLoading 2018 data from {DATA_PATH_2018}...")
try:
    df = pd.read_csv(DATA_PATH_2018, low_memory=False)
    # Take random 10% sample
    df_sample = df.sample(frac=0.1, random_state=42)
    print(f"Original size: {len(df)}. Sampled 10% size: {len(df_sample)}")
except FileNotFoundError:
    print("Error: File not found.")
    run.finish()
    exit()

# Upload Artifact
df_sample.to_csv("data_2018_sample.csv", index=False)
artifact_2018 = wandb.Artifact("lending_club_2018_10pct_sample", type="dataset")
artifact_2018.add_file("data_2018_sample.csv")
run.log_artifact(artifact_2018)

# Preprocessing
print("Preprocessing...")
df_sample = df_sample[df_sample['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
df_sample['target'] = df_sample['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)

leakage_cols = [
    'loan_status', 'funded_amnt', 'funded_amnt_inv', 'total_pymnt', 
    'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 
    'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 
    'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 
    'last_credit_pull_d', 'debt_settlement_flag', 'hardship_flag',
    'pymnt_plan', 'out_prncp', 'out_prncp_inv'
]
irrelevant_cols = ['id', 'member_id', 'url', 'desc', 'emp_title', 'title', 'zip_code']
df_sample = df_sample.drop(columns=[c for c in leakage_cols + irrelevant_cols if c in df_sample.columns], errors='ignore')

if df_sample['int_rate'].dtype == 'O': df_sample['int_rate'] = df_sample['int_rate'].str.strip('%').astype(float)
if df_sample['revol_util'].dtype == 'O': df_sample['revol_util'] = df_sample['revol_util'].str.strip('%').astype(float)
if df_sample['term'].dtype == 'O': df_sample['term'] = df_sample['term'].apply(lambda x: 36 if '36' in str(x) else 60)
    
def clean_emp_length(x):
    if pd.isna(x): return np.nan
    if '<' in x: return 0
    if '+' in x: return 10
    return int(x.split()[0])
if 'emp_length' in df_sample.columns: df_sample['emp_length'] = df_sample['emp_length'].apply(clean_emp_length)

# Encoding
cat_cols = df_sample.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    df_sample[col] = le.fit_transform(df_sample[col].astype(str))

# ==========================================
# 3. DOWNLOAD MODEL & PREDICT
# ==========================================
print("\nDownloading model...")
try:
    artifact_path = f"{ENTITY}/{PROJECT_NAME}/{MODEL_ARTIFACT}"
    artifact = run.use_artifact(artifact_path)
    model_dir = artifact.download()
    model = joblib.load(os.path.join(model_dir, "model_2015.pkl"))
except Exception as e:
    print(f"Error downloading model: {e}")
    run.finish()
    exit()

# Align features
model_features = model.get_booster().feature_names
X_2018 = df_sample.drop('target', axis=1)
y_2018 = df_sample['target']

for feature in model_features:
    if feature not in X_2018.columns: X_2018[feature] = 0
X_2018 = X_2018[model_features]

# Predict
print(f"Predicting on 2018 data using 2015 Threshold ({threshold_2015})...")
y_proba = model.predict_proba(X_2018)[:, 1]

# Apply the 2015 Threshold
y_pred = (y_proba >= threshold_2015).astype(int)

# Calculate 2018 Metrics
acc_2018 = accuracy_score(y_2018, y_pred)
auc_2018 = roc_auc_score(y_2018, y_proba)
f1_2018  = f1_score(y_2018, y_pred)

print("\n" + "="*60)
print(f"COMPARISON (Threshold: {threshold_2015})")
print(f"{'METRIC':<20} | {'2015 (REAL)':<15} | {'2018 (NEW)':<15}")
print("="*60)
print(f"{'Accuracy':<20} | {acc_2015:.5f}          | {acc_2018:.5f}")
print(f"{'ROC AUC':<20} | {auc_2015:.5f}          | {auc_2018:.5f}")
print(f"{'F1 Score':<20} | {f1_2015:.5f}          | {f1_2018:.5f}")
print("="*60 + "\n")

# ==========================================
# 4. VISUALIZATION
# ==========================================
wandb.log({
    "2018_accuracy": acc_2018,
    "2018_auc": auc_2018,
    "2018_f1": f1_2018,
    "accuracy_drop": acc_2015 - acc_2018
})

# Table
table = wandb.Table(columns=["Metric", "2015 (Real)", "2018 (Sample)", "Diff"])
table.add_data("Accuracy", acc_2015, acc_2018, acc_2018 - acc_2015)
table.add_data("ROC AUC", auc_2015, auc_2018, auc_2018 - auc_2015)
table.add_data("F1 Score", f1_2015, f1_2018, f1_2018 - f1_2015)
wandb.log({"Metrics_Comparison_Table": table})

# Bar Chart
plot_data = [
    ["Accuracy (2015)", acc_2015],
    ["Accuracy (2018)", acc_2018],
    ["ROC AUC (2015)", auc_2015],
    ["ROC AUC (2018)", auc_2018],
    ["F1 Score (2015)", f1_2015],
    ["F1 Score (2018)", f1_2018]
]
plot_table = wandb.Table(data=plot_data, columns=["Metric", "Score"])

wandb.log({
    "Performance_Comparison_Chart": wandb.plot.bar(
        plot_table, "Metric", "Score", title="Comparison: 2015 (Real) vs 2018"
    )
})

run.finish()
print("Success! Charts generated using real 2015 data.")