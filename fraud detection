# fraud_logistic.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

sns.set(style="whitegrid")

# -------------------------------
# STEP 1: Load CSV
# -------------------------------
try:
    data = pd.read_csv("data.csv", low_memory=False)
except FileNotFoundError:
    raise SystemExit("❌ File 'data.csv' not found. Please place your dataset in the same directory.")

data.columns = [c.strip() for c in data.columns]

print("✅ Dataset loaded successfully!")
print("Shape:", data.shape)
print("Columns:", data.columns.tolist(), "\n")

# -------------------------------
# STEP 2: Clean and preprocess data
# -------------------------------
def to_numeric_col(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.replace(" ", ""), errors="coerce")

num_cols_possible = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
num_cols_present = [c for c in num_cols_possible if c in data.columns]

for col in num_cols_present:
    data[col] = to_numeric_col(data[col])

for idcol in ["nameOrig", "nameDest"]:
    if idcol in data.columns:
        data.drop(columns=[idcol], inplace=True)

if "isFraud" not in data.columns:
    raise ValueError("Target column 'isFraud' not found!")

data["isFraud"] = pd.to_numeric(data["isFraud"], errors="coerce").fillna(0).astype(int)
num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
data[num_cols] = data[num_cols].fillna(0)

if "type" in data.columns:
    data["type"] = data["type"].astype(str).str.strip()

print("✅ Data cleaned. Rows:", len(data))
print("Class Distribution:\n", data["isFraud"].value_counts(), "\n")

# -------------------------------
# STEP 3: Split features & target
# -------------------------------
X = data.drop(columns=["isFraud"])
y = data["isFraud"]

categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
numeric_cols = [c for c in X.columns if c not in categorical_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}\n")

# -------------------------------
# STEP 4: Build preprocessing + model pipeline
# -------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_cols)
    ]
)

pipeline = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"))
])

# -------------------------------
# STEP 5: Train model
# -------------------------------
print("🧠 Training Logistic Regression model...")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("✅ Training complete!\n")

# -------------------------------
# STEP 6: Evaluate performance
# -------------------------------
print("📊 Model Performance Metrics:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("ROC AUC:", round(roc_auc_score(y_test, y_proba), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.4f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()
print("ROC curve saved as 'roc_curve.png'.")

# -------------------------------
# STEP 7: Save the model
# -------------------------------
joblib.dump(pipeline, "logistic_model.pkl")
print("💾 Model saved to 'logistic_model.pkl'.")
