# Experiment: Confusion Matrix and ROC Curve with AUC

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# be aware, install certificates, use command:
# open "/Applications/Python 3.13/Install Certificates.command"

# csv of the wine quality dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

df = pd.read_csv(url, sep=';')
# print(df.head())  ### uncomment and check the data is loaded properly

print("=" * 60)
print("PART 1: Multi-class Classification - Confusion Matrix")
print("=" * 60)

# Multi-class classification (original quality values)
X = df.drop(columns=['quality'])
y = df['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model for multi-class classification
# increase the number of iterations to avoid 
# "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT" error
model_multi = LogisticRegression(max_iter=10000)
model_multi.fit(X_train, y_train)

# predict the test set
y_pred = model_multi.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

# OUTPUT:
# Confusion matrix:
# [[ 0  0  1  0  0  0]
#  [ 0  0  8  2  0  0]
#  [ 0  0 98 31  1  0]
#  [ 0  0 47 79  6  0]
#  [ 0  0  3 33  6  0]
#  [ 0  0  0  1  4  0]]

print("\n" + "=" * 60)
print("PART 2: Binary Classification - ROC Curve and AUC")
print("=" * 60)

# Binary classification: good wine (quality >= 6) vs bad wine (quality < 6)
y_binary = (df['quality'] >= 6).astype(int)

# Split the data into training and testing sets
X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(
    X, y_binary, test_size=0.2, random_state=42
)

# Train the model with scaling for binary classification
model_binary = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)
model_binary.fit(X_train_binary, y_train_binary)

# Get probability predictions for the positive class
y_pred_proba = model_binary.predict_proba(X_test_binary)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_proba)

# Calculate AUC
roc_auc = auc(fpr, tpr)

# Print AUC score
print(f"AUC (Area Under Curve): {roc_auc:.4f}")

# OUTPUT:
# AUC (Area Under Curve): 0.8190

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Wine Quality Classification')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

# Save figure to file
plt.savefig("../results/experiment.png", dpi=300)
print(f"\nROC curve saved to: ../results/experiment.png")

# Optionally show the plot on screen
# plt.show()

# Alternative: using roc_auc_score directly
auc_score = roc_auc_score(y_test_binary, y_pred_proba)
print(f"AUC using roc_auc_score: {auc_score:.4f}")

# OUTPUT:
# AUC using roc_auc_score: 0.8190
