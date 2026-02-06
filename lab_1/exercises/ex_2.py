# Exercise 1.3 (2) Show the ROC curve and AUC score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# be aware, install certificates, use command:
# open "/Applications/Python 3.13/Install Certificates.command"

# csv of the wine quality dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

df = pd.read_csv(url, sep=';')

X = df.drop(columns=['quality'])
# Convert to binary classification: good wine (quality >= 6) vs bad wine (quality < 6)
y = (df['quality'] >= 6).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model with scaling
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)
model.fit(X_train, y_train)

# Get probability predictions for the positive class
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

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
plt.show()

# RESULTS COULD BE CHECKED IN results/ex_2.png, they are saved by 
# plt.savefig("../results/ex_2.png", dpi=300)


# Alternative: using roc_auc_score directly
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC using roc_auc_score: {auc_score:.4f}")

# OUTPUT:
# AUC using roc_auc_score: 0.8190
