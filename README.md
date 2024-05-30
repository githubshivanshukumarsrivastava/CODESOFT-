# CODESOFT-

#CREDIT CARD FRAUD DETECTION
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Check for missing values
print(data.isnull().sum())

# Data preprocessing
X = data.drop(['Time', 'Class'], axis=1)
y = data['Class']

# Normalize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance using SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train a logistic regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Train a random forest classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Make predictions
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

# Evaluate the models
def evaluate_model(y_test, y_pred):
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return precision, recall, f1, cm

lr_precision, lr_recall, lr_f1, lr_cm = evaluate_model(y_test, lr_pred)
rf_precision, rf_recall, rf_f1, rf_cm = evaluate_model(y_test, rf_pred)

print("Logistic Regression:")
print(f"Precision: {lr_precision}")
print(f"Recall: {lr_recall}")
print(f"F1-Score: {lr_f1}")
print(f"Confusion Matrix:\n{lr_cm}")

print("\nRandom Forest Classifier:")
print(f"Precision: {rf_precision}")
print(f"Recall: {rf_recall}")
print(f"F1-Score: {rf_f1}")
print(f"Confusion Matrix:\n{rf_cm}")
