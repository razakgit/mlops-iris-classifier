import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import mlflow
import mlflow.sklearn
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load Data
df = pd.read_csv("data/iris.csv")
X = df.drop(columns=["target"])
y = df["target"]

# Preprocess Data
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Convert back to DataFrame with correct column names
X = pd.DataFrame(X_scaled, columns=X.columns)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set MLflow Experiment
mlflow.set_experiment("IrisClassification")

# Define Models
models = {
    "logistic_regression": LogisticRegression(max_iter=200),
    "random_forest": RandomForestClassifier(n_estimators=100)
}

# Track best model
best_model = None
best_model_name = ""
best_score = 0.0
best_run_id = ""

for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Evaluation metrics
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro')

        # Logging
        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score_macro", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"{name} | Accuracy: {acc:.4f} | F1-macro: {f1:.4f}")
        print(classification_report(y_test, preds, digits=4))

        # Best model selection based on F1
        if f1 > best_score:
            best_score = f1
            best_model = model
            best_model_name = name
            best_run_id = run.info.run_id

# Save best model locally
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")
print(f" Best model: {best_model_name} | F1-macro: {best_score:.4f}")

# Register best model in MLflow Model Registry
model_uri = f"runs:/{best_run_id}/model"
mlflow.register_model(model_uri=model_uri, name="BestIrisClassifier")
