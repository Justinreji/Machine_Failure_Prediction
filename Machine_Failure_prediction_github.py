import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score
)

# Load dataset
df = pd.read_csv("machinefailure.csv")

# -----------------------------
# Exploratory Data Analysis
# -----------------------------

# Bar plot for class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=df["fail"])
plt.title("Class Distribution of Machine Failures")
plt.xlabel("Failure (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.grid(True, axis="y")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# -----------------------------
# Preprocessing
# -----------------------------
X = df.drop(columns="fail").to_numpy()
y = df["fail"].to_numpy()

# Normalize features
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -----------------------------
# Model 1: SVC (RBF Kernel)
# -----------------------------
print("\n--- Training SVC (RBF) ---")
param_grid_svc = {
    "C": np.linspace(0.5, 1.5, 5),
    "gamma": np.linspace(0.01, 1.0, 4)
}
grid_svc = GridSearchCV(SVC(kernel="rbf"), param_grid=param_grid_svc, cv=5, scoring="accuracy")
grid_svc.fit(X_train, y_train)

svc_best = grid_svc.best_estimator_
y_pred_svc = svc_best.predict(X_test)

print("Best SVC Params:", grid_svc.best_params_)
print("Train Accuracy:", svc_best.score(X_train, y_train))
print("Test Accuracy:", svc_best.score(X_test, y_test))
print("\nClassification Report (SVC):")
print(classification_report(y_test, y_pred_svc))

# Confusion matrix - SVC
cm_svc = confusion_matrix(y_test, y_pred_svc)
ConfusionMatrixDisplay(cm_svc).plot()
plt.title("Confusion Matrix: SVC (RBF)")
plt.show()

# -----------------------------
# Model 2: Random Forest
# -----------------------------
print("\n--- Training Random Forest ---")
param_grid_rf = {"max_depth": np.arange(2, 10)}
grid_rf = GridSearchCV(RandomForestClassifier(), param_grid=param_grid_rf, cv=5)
grid_rf.fit(X_train, y_train)

rf_best = grid_rf.best_estimator_
y_pred_rf = rf_best.predict(X_test)

print("Best RF Depth:", grid_rf.best_params_)
print("Train Accuracy:", rf_best.score(X_train, y_train))
print("Test Accuracy:", rf_best.score(X_test, y_test))
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

# Confusion matrix - RF
cm_rf = confusion_matrix(y_test, y_pred_rf)
ConfusionMatrixDisplay(cm_rf).plot()
plt.title("Confusion Matrix: Random Forest")
plt.show()

# -----------------------------
# Model Comparison Summary
# -----------------------------
model_scores = {
    "SVC (RBF)": {
        "Accuracy": accuracy_score(y_test, y_pred_svc),
        "Precision": precision_score(y_test, y_pred_svc),
        "Recall": recall_score(y_test, y_pred_svc),
        "F1 Score": f1_score(y_test, y_pred_svc)
    },
    "Random Forest": {
        "Accuracy": accuracy_score(y_test, y_pred_rf),
        "Precision": precision_score(y_test, y_pred_rf),
        "Recall": recall_score(y_test, y_pred_rf),
        "F1 Score": f1_score(y_test, y_pred_rf)
    }
}

comparison_df = pd.DataFrame(model_scores).T
print("\n--- Model Comparison ---")
print(comparison_df)