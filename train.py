
import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score
)

# 1. Create output directories
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# 2. Load dataset
data_path = "data/ai4i2020.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError(
        "Dataset not found. Place ai4i2020.csv inside the data folder."
    )

df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)
print(df.head())

# 3. Select input features
features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

target = "Machine failure"

# Check required columns
missing = [col for col in features + [target] if col not in df.columns]

if missing:
    raise ValueError(f"Missing dataset columns: {missing}")

X = df[features].copy()
y = pd.to_numeric(df[target], errors="raise").astype(int)

if not set(y.unique()).issubset({0, 1}):
    raise ValueError("Machine failure must contain only 0 and 1.")

# 4. Separate categorical and numerical columns
categorical_features = ["Type"]

numerical_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# 5. Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 6. Build ML pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# 7. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 8. Train the model
print("\nTraining model...")
model.fit(X_train, y_train)

# 9. Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 10. Evaluate the model
report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["No Failure", "Failure"],
    zero_division=0
)

print("\nClassification Report:")
print(report)

metrics = {
    "precision_recall": report,
    "roc_auc": float(roc_auc_score(y_test, y_prob)),
    "average_precision": float(
        average_precision_score(y_test, y_prob)
    )
}

print("ROC-AUC:", round(metrics["roc_auc"], 4))
print("Average Precision:", round(metrics["average_precision"], 4))

# 11. Save trained model
joblib.dump(model, "models/model.joblib")

with open("models/metrics.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

# 12. Save classification report
with open(
    "reports/classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

# 13. Save confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Failure", "Failure"]
)

display.plot()
plt.title("Predictive Maintenance - Confusion Matrix")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=150)
plt.close()

print("\nTraining completed successfully!")
print("Saved: models/model.joblib")
print("Saved: models/metrics.json")
print("Saved: reports/classification_report.txt")
print("Saved: reports/confusion_matrix.png")
