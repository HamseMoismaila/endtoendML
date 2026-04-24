"""
EC3355 – Machine Learning Techniques and Computing Environment (P5, PLO3)
Full end-to-end ML pipeline: DOSM Malaysia Labour Force Data
Usage: python src/ml_pipeline.py
"""

import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, ConfusionMatrixDisplay)

# ── Step A: Libraries already imported above ─────────────────────────────────
print("=" * 65)
print("EC3355 – DOSM Malaysia Labour Force ML Pipeline")
print("=" * 65)

# ── Step B: Load Data ─────────────────────────────────────────────────────────
print("\n[Step B] Loading DOSM dataset...")
df = pd.read_parquet('data/lfs_qtr_state.parquet')
df['date'] = pd.to_datetime(df['date'])
print(f"  Dataset shape : {df.shape}")
print(f"  Columns       : {df.columns.tolist()}")
print(df.head(3).to_string())

# ── Step C: Explore ───────────────────────────────────────────────────────────
print("\n[Step C] Data Exploration")
print(df.info())
print("\nMissing values:\n", df.isna().sum())
print("\nSummary statistics:\n", df.select_dtypes(include='number').describe().round(2))
print("\nStates:", df['state'].unique().tolist())
print("\nTop 5 states by avg unemployment rate:")
print(df.groupby('state')['u_rate'].mean().sort_values(ascending=False).head())

# ── Step D: Target Variable ───────────────────────────────────────────────────
print("\n[Step D] Creating target variable...")

def risk_level(u):
    if u < 3.0:   return 'Low'
    elif u < 4.5: return 'Medium'
    else:          return 'High'

df['risk'] = df['u_rate'].apply(risk_level)
print("Risk distribution:\n", df['risk'].value_counts().to_string())

# ── Step E: Feature Engineering ───────────────────────────────────────────────
print("\n[Step E] Feature engineering...")
df = df.sort_values(['state', 'date']).copy()
df['u_rate_lag1'] = df.groupby('state')['u_rate'].shift(1)
df['p_rate_lag1'] = df.groupby('state')['p_rate'].shift(1)
df['lf_lag1']     = df.groupby('state')['lf'].shift(1)
df['quarter']     = df['date'].dt.quarter
df['year']        = df['date'].dt.year

model_df = df.dropna().copy()
model_df = pd.get_dummies(model_df, columns=['state', 'quarter'], drop_first=True)
X = model_df.drop(columns=['risk', 'date', 'u_rate'])
y = model_df['risk']
print(f"  Feature matrix : {X.shape}")
print(f"  Class counts   :\n{y.value_counts().to_string()}")

# ── Step F: Split ─────────────────────────────────────────────────────────────
print("\n[Step F] Splitting data (80/20, stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"  Train : {X_train.shape}   Test : {X_test.shape}")

# ── Section 5A: SVM ───────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("[Model A] Support Vector Machine (SVM)")
svm_model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf', C=1.0, gamma='scale'))
])
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_acc  = accuracy_score(y_test, svm_pred)
print(f"  Accuracy : {svm_acc:.4f}  ({svm_acc*100:.2f}%)")
print(classification_report(y_test, svm_pred))

# ── Section 5B: Decision Tree ─────────────────────────────────────────────────
print("─" * 65)
print("[Model B] Decision Tree Classifier")
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)
tree_acc  = accuracy_score(y_test, tree_pred)
print(f"  Accuracy : {tree_acc:.4f}  ({tree_acc*100:.2f}%)")
print(classification_report(y_test, tree_pred))

# ── Section 5C: Random Forest ─────────────────────────────────────────────────
print("─" * 65)
print("[Model C] Random Forest Classifier")
rf_model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)
print(f"  Accuracy : {rf_acc:.4f}  ({rf_acc*100:.2f}%)")
print(classification_report(y_test, rf_pred))

# ── Section 5D: Deep Learning ─────────────────────────────────────────────────
print("─" * 65)
print("[Model D] Deep Learning (Neural Network)")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

le          = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)
scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)
y_train_cat = to_categorical(y_train_enc)
y_test_cat  = to_categorical(y_test_enc)

nn = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_sc.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])
nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
nn.fit(X_train_sc, y_train_cat, epochs=40, batch_size=16, validation_split=0.2, verbose=0)
_, dl_acc = nn.evaluate(X_test_sc, y_test_cat, verbose=0)
print(f"  Accuracy : {dl_acc:.4f}  ({dl_acc*100:.2f}%)")

# ── Section 6: Evaluation ─────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("MODEL COMPARISON RESULTS")
print("=" * 65)
results = pd.DataFrame({
    'Model':    ['SVM', 'Decision Tree', 'Random Forest', 'Deep Learning'],
    'Accuracy': [svm_acc, tree_acc, rf_acc, dl_acc]
}).sort_values('Accuracy', ascending=False).reset_index(drop=True)
results['Accuracy %'] = (results['Accuracy'] * 100).round(2)
print(results.to_string(index=False))

# Confusion matrix – Random Forest
fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(y_test, rf_pred, labels=['Low','Medium','High'])
ConfusionMatrixDisplay(cm, display_labels=['Low','Medium','High']).plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Random Forest – Confusion Matrix', fontsize=13, pad=12)
plt.tight_layout()
plt.savefig('outputs/confusion_matrix_rf.png', dpi=150, bbox_inches='tight')
print("\nConfusion matrix saved → outputs/confusion_matrix_rf.png")

# Feature importance
feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
fig2, ax2 = plt.subplots(figsize=(9, 5))
feat_imp.plot(kind='bar', ax=ax2, color='steelblue')
ax2.set_title('Top 15 Feature Importances – Random Forest', fontsize=13)
ax2.set_ylabel('Importance')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=150, bbox_inches='tight')
print("Feature importance chart saved → outputs/feature_importance.png")

# ── Conclusion ────────────────────────────────────────────────────────────────
best = results.iloc[0]
print("\n" + "=" * 65)
print("CONCLUSION")
print("=" * 65)
print(f"""
For the DOSM quarterly labour-force dataset, the Anaconda + Python
environment is suitable because it supports reproducible package
management, tabular-data processing, machine learning, and dashboard
deployment in one workflow.

Best model  : {best['Model']} ({best['Accuracy %']}% accuracy)
Recommended : Random Forest for production accuracy; Decision Tree
              for policy communication (interpretable rules).

Interactive Plotly charts and the Streamlit dashboard (dashboard.py)
make results accessible beyond the data-science notebook.
""")
