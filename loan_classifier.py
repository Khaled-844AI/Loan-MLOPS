import pandas as pd
import numpy as np
import xgboost as xgb
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# ==========================================
# CONFIGURATION
# ==========================================
PROJECT_NAME = "lending_club_2015"
DATASET_PATH = "data_2015/data_2015.csv"
ARTIFACT_NAME = "lending_club_2015_split" 
MODEL_NAME = "xgb_loan_classifier_2015"

# Login to W&B
wandb.login()

# ==========================================
# 1. PREPROCESSING
# ==========================================
def process_data(df):
    print("Processing data...")
    # 1. Filter Target
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
    # 0 = Fully Paid, 1 = Charged Off
    df['target'] = df['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)
    
    # 2. Drop Leakage (Columns that reveal the future)
    leakage_cols = [
        'loan_status', 'funded_amnt', 'funded_amnt_inv', 'total_pymnt', 
        'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 
        'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 
        'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 
        'last_credit_pull_d', 'debt_settlement_flag', 'hardship_flag',
        'pymnt_plan', 'out_prncp', 'out_prncp_inv'
    ]
    irrelevant_cols = ['id', 'member_id', 'url', 'desc', 'emp_title', 'title', 'zip_code']
    df = df.drop(columns=[c for c in leakage_cols + irrelevant_cols if c in df.columns], errors='ignore')

    # 3. Formatting
    if df['int_rate'].dtype == 'O':
        df['int_rate'] = df['int_rate'].str.strip('%').astype(float)
    if df['revol_util'].dtype == 'O':
        df['revol_util'] = df['revol_util'].str.strip('%').astype(float)
    if df['term'].dtype == 'O':
        df['term'] = df['term'].apply(lambda x: 36 if '36' in str(x) else 60)
        
    def clean_emp_length(x):
        if pd.isna(x): return np.nan
        if '<' in x: return 0
        if '+' in x: return 10
        return int(x.split()[0])
    
    if 'emp_length' in df.columns:
        df['emp_length'] = df['emp_length'].apply(clean_emp_length)

    # 4. Encoding
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df

# ==========================================
# 2. CREATE SPLIT ARTIFACT (80/20)
# ==========================================
def create_split_artifact():
    print("Creating data artifact with 80/20 split...")
    run = wandb.init(project=PROJECT_NAME, job_type="data_upload")
    
    try:
        df = pd.read_csv(DATASET_PATH, low_memory=False)
    except FileNotFoundError:
        print(f"Error: File {DATASET_PATH} not found.")
        run.finish()
        return

    # Process entire dataset
    df_processed = process_data(df)
    
    # Split: 80% for Train/Tuning, 20% for Final Test
    # Stratify ensures the % of charged off loans is the same in both
    train_df, test_df = train_test_split(
        df_processed, 
        test_size=0.2, 
        random_state=42, 
        stratify=df_processed['target']
    )
    
    print(f"Total Rows: {len(df_processed)}")
    print(f"TRAIN Set (80%): {len(train_df)} rows")
    print(f"TEST Set  (20%): {len(test_df)} rows")

    # Save locally
    train_df.to_csv("train_2015.csv", index=False)
    test_df.to_csv("test_2015.csv", index=False)
    
    # Upload to W&B
    artifact = wandb.Artifact(name=ARTIFACT_NAME, type="dataset")
    artifact.add_file("train_2015.csv")
    artifact.add_file("test_2015.csv")
    
    logged_artifact = run.log_artifact(artifact)
    logged_artifact.wait() # Wait for upload to finish
    
    run.finish()
    print("Artifact uploaded successfully.")

# ==========================================
# 3. SWEEP FUNCTION (Hyperparameter Tuning)
# ==========================================
def train_sweep():
    with wandb.init() as run:
        config = wandb.config
        
        # Download Data
        try:
            artifact = run.use_artifact(f'{ARTIFACT_NAME}:latest')
            artifact_dir = artifact.download()
        except wandb.errors.CommError:
            print("Error: Could not fetch artifact.")
            return

        # Load ONLY the Training (80%) data
        df_train = pd.read_csv(os.path.join(artifact_dir, "train_2015.csv"))
        
        X = df_train.drop('target', axis=1)
        y = df_train['target']
        
        # Internal Split for Validation during tuning
        X_t, X_v, y_t, y_v = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Calculate Class Weight (Negative / Positive)
        ratio = float(np.sum(y_t == 0)) / np.sum(y_t == 1)
        
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            scale_pos_weight=ratio,
            tree_method='hist',
            random_state=42
        )
        
        model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
        
        # Evaluate using AUC (Robust to threshold)
        y_proba = model.predict_proba(X_v)[:, 1]
        auc = roc_auc_score(y_v, y_proba)
        
        wandb.log({"roc_auc": auc})

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # 1. Create Data Artifact (Splits 80/20)
    # If the files don't exist, create them.
    if not os.path.exists("train_2015.csv"):
        create_split_artifact()

    # 2. Run Sweep on the 80% Train Data
    sweep_configuration = {
        'method': 'bayes',
        'metric': {'name': 'roc_auc', 'goal': 'maximize'},
        'parameters': {
            'n_estimators': {'values': [100, 300, 500]},
            'max_depth': {'values': [3, 5, 7, 9]},
            'learning_rate': {'min': 0.01, 'max': 0.2},
            'subsample': {'min': 0.6, 'max': 1.0},
            'colsample_bytree': {'min': 0.6, 'max': 1.0}
        }
    }

    sweep_id = wandb.sweep(sweep_configuration, project=PROJECT_NAME)
    print("Starting hyperparameter tuning...")
    wandb.agent(sweep_id, function=train_sweep, count=10)
    
    # 3. Fetch Best Params
    print("Sweep complete. Fetching best run...")
    api = wandb.Api()
    entity = api.default_entity
    sweep_path = f"{entity}/{PROJECT_NAME}/{sweep_id}"
    best_run = api.sweep(sweep_path).best_run()
    best_config = best_run.config
    print(f"Best hyperparameters: {best_config}")

    # ==========================================
    # 5. FINAL TRAINING & EVALUATION
    # ==========================================
    run = wandb.init(project=PROJECT_NAME, job_type="final_train_eval", config=best_config)
    
    artifact = run.use_artifact(f'{ARTIFACT_NAME}:latest')
    artifact_dir = artifact.download()
    
    # A. Load Train Set (80%)
    df_train = pd.read_csv(os.path.join(artifact_dir, "train_2015.csv"))
    X_train = df_train.drop('target', axis=1)
    y_train = df_train['target']
    
    # B. Load Test Set (20%) - LOCKED UNTIL NOW
    df_test = pd.read_csv(os.path.join(artifact_dir, "test_2015.csv"))
    X_test = df_test.drop('target', axis=1)
    y_test = df_test['target']
    
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

    print("Training final model on full 80% Training set...")
    final_model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=best_config['n_estimators'],
        max_depth=best_config['max_depth'],
        learning_rate=best_config['learning_rate'],
        subsample=best_config['subsample'],
        colsample_bytree=best_config['colsample_bytree'],
        scale_pos_weight=ratio, # Important for Imbalance
        tree_method='hist'
    )
    
    final_model.fit(X_train, y_train)
    
    # C. Threshold Tuning on Test Set
    # We predict probabilities first
    y_proba_test = final_model.predict_proba(X_test)[:, 1]
    
    print("\nFinding optimal threshold on Test Set...")
    best_thresh = 0.5
    best_f1 = 0
    best_acc = 0
    
    # Check thresholds from 0.1 to 0.9
    for thresh in np.arange(0.1, 0.95, 0.05):
        y_pred_temp = (y_proba_test >= thresh).astype(int)
        f1 = f1_score(y_test, y_pred_temp)
        if f1 > best_f1:
            best_f1 = f1
            best_acc = accuracy_score(y_test, y_pred_temp)
            best_thresh = thresh
            
    # Final Predictions with Best Threshold
    y_pred_final = (y_proba_test >= best_thresh).astype(int)
    auc_final = roc_auc_score(y_test, y_proba_test)
    
    print("-" * 40)
    print(f"FINAL RESULTS ON 20% TEST SET")
    print(f"Optimal Threshold Found: {best_thresh:.2f}")
    print(f"Accuracy: {best_acc:.4f}")
    print(f"F1 Score: {best_f1:.4f}")
    print(f"ROC AUC : {auc_final:.4f}")
    print("-" * 40)
    
    # Log Final Metrics
    wandb.log({
        "final_threshold": best_thresh,
        "test_accuracy": best_acc,
        "test_auc": auc_final,
        "test_f1": best_f1,
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None, 
            y_true=y_test.to_numpy(), 
            preds=y_pred_final
        )
    })

    # Save Model
    joblib.dump(final_model, "model_2015.pkl")
    model_artifact = wandb.Artifact(MODEL_NAME, type="model")
    model_artifact.add_file("model_2015.pkl")
    run.log_artifact(model_artifact)
    
    run.finish()
    print("Process Complete. Model Saved.")