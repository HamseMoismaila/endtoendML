# %% [markdown]
# # EC3355 - Machine Learning Techniques and Computing Environment
# ## DOSM Malaysia Labour Force Analysis
# ### End-to-End Machine Learning Workflow
# 
# **Dataset:** Quarterly Principal Labour Force Statistics by State (DOSM Malaysia)  
# **Task:** Classify state-quarter unemployment into risk levels (Low / Medium / High)

# %% [markdown]
# ## Step A – Import Libraries

# %%
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

print("All libraries imported successfully.")

# %% [markdown]
# ## Step B – Load Official DOSM Data
# 
# *Note: The DOSM Parquet URL requires browser-based access. For this workflow,
# we use a locally cached version with the exact same schema as the official dataset.*

# %%
# Load the dataset (mirrors official DOSM schema)
df = pd.read_parquet('../data/lfs_qtr_state.parquet')
df['date'] = pd.to_datetime(df['date'])
print("Dataset loaded successfully.")
print(f"Shape: {df.shape}")
print(df.head())

# %% [markdown]
# ## Step C – Explore the Data

# %%
print("=== Data Info ===")
print(df.info())
print("\n=== Missing Values ===")
print(df.isna().sum())
print("\n=== Summary Statistics ===")
print(df.describe())

# %%
print("States in dataset:")
print(df['state'].unique())

# %%
# Which states have highest unemployment rates?
top_u = df.groupby('state')['u_rate'].mean().sort_values(ascending=False)
print("\nAverage Unemployment Rate by State:")
print(top_u.to_string())

# %% [markdown]
# ## Step D – Create Target Variable

# %%
def risk_level(u):
    if u < 3.0:
        return 'Low'
    elif u < 4.5:
        return 'Medium'
    else:
        return 'High'

df['risk'] = df['u_rate'].apply(risk_level)
print("Risk level distribution:")
print(df['risk'].value_counts())
print(df[['state', 'date', 'u_rate', 'risk']].head(10))

# %% [markdown]
# ## Step E – Feature Engineering

# %%
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
print(f"Feature matrix: {X.shape}")
print(f"Class distribution:\n{y.value_counts()}")

# %% [markdown]
# ## Step F – Split the Data

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

# %% [markdown]
# ## Section 5 – Build and Compare Models

# %% [markdown]
# ### A. Support Vector Machine (SVM)

# %%
svm_model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf', C=1.0, gamma='scale'))
])
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_acc  = accuracy_score(y_test, svm_pred)

print(f"SVM Accuracy: {svm_acc:.4f}")
print(classification_report(y_test, svm_pred))

# %% [markdown]
# ### B. Decision Tree

# %%
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)
tree_acc  = accuracy_score(y_test, tree_pred)

print(f"Decision Tree Accuracy: {tree_acc:.4f}")
print(classification_report(y_test, tree_pred))

# %% [markdown]
# ### C. Random Forest

# %%
rf_model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)

print(f"Random Forest Accuracy: {rf_acc:.4f}")
print(classification_report(y_test, rf_pred))

# %% [markdown]
# ### D. Deep Learning (Neural Network)

# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)
scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)
y_train_cat = to_categorical(y_train_enc)
y_test_cat  = to_categorical(y_test_enc)

nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_sc.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])
nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = nn_model.fit(X_train_sc, y_train_cat, epochs=40, batch_size=16,
                       validation_split=0.2, verbose=0)
_, dl_acc = nn_model.evaluate(X_test_sc, y_test_cat, verbose=0)

print(f"Deep Learning Accuracy: {dl_acc:.4f}")

# %% [markdown]
# ## Section 6 – Model Evaluation and Comparison

# %%
results = pd.DataFrame({
    'Model':    ['SVM', 'Decision Tree', 'Random Forest', 'Deep Learning'],
    'Accuracy': [svm_acc, tree_acc, rf_acc, dl_acc]
}).sort_values('Accuracy', ascending=False).reset_index(drop=True)

results['Accuracy %'] = (results['Accuracy'] * 100).round(2)
print(results.to_string(index=False))

# %%
# Confusion matrix for best model (Random Forest)
cm = confusion_matrix(y_test, rf_pred, labels=['Low','Medium','High'])
fig, ax = plt.subplots(figsize=(7,5))
disp = ConfusionMatrixDisplay(cm, display_labels=['Low','Medium','High'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Random Forest – Confusion Matrix')
plt.tight_layout()
plt.savefig('../outputs/confusion_matrix_rf.png', dpi=150)
plt.show()
print("Confusion matrix saved.")

# %%
# Feature importance from Random Forest
feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
fig2, ax2 = plt.subplots(figsize=(9,5))
feat_imp.plot(kind='bar', ax=ax2, color='steelblue')
ax2.set_title('Top 15 Feature Importances – Random Forest')
ax2.set_ylabel('Importance')
plt.tight_layout()
plt.savefig('../outputs/feature_importance.png', dpi=150)
plt.show()
print("Feature importance chart saved.")

# %% [markdown]
# ## Section 7 – Interactive Data Visualisation

# %%
viz_df = df[['date','state','u_rate','p_rate']].copy()

fig_line = px.line(
    viz_df, x='date', y='u_rate', color='state',
    title='DOSM Unemployment Rate by State (Quarterly)',
    labels={'u_rate': 'Unemployment Rate (%)', 'date': 'Quarter'}
)
fig_line.update_layout(hovermode='x unified')
fig_line.show()

# %%
fig_scatter = px.scatter(
    df, x='p_rate', y='u_rate', color='state',
    animation_frame=df['date'].dt.strftime('%Y-%m-%d'),
    title='Participation Rate vs Unemployment Rate (Animated by Quarter)',
    labels={'p_rate': 'Participation Rate (%)', 'u_rate': 'Unemployment Rate (%)'}
)
fig_scatter.show()

# %%
# Accuracy comparison chart
fig_bar = px.bar(
    results.sort_values('Accuracy'),
    x='Accuracy', y='Model', orientation='h',
    title='Model Accuracy Comparison',
    color='Accuracy', color_continuous_scale='Blues',
    text='Accuracy %'
)
fig_bar.update_traces(texttemplate='%{text}%', textposition='outside')
fig_bar.update_layout(xaxis_range=[0.7, 1.0])
fig_bar.show()

# %% [markdown]
# ## Section 8 – Dynamic Report Summary

# %%
best_model = results.iloc[0]['Model']
best_acc   = results.iloc[0]['Accuracy %']

print("=" * 60)
print("DYNAMIC REPORT SUMMARY")
print("=" * 60)
print(f"\nDataset     : DOSM Malaysia Labour Force (Quarterly by State)")
print(f"Records     : {len(df):,}  |  Features (after encoding): {X.shape[1]}")
print(f"\nModel Results:")
for _, row in results.iterrows():
    marker = " ★ BEST" if row['Model'] == best_model else ""
    print(f"  {row['Model']:<18} {row['Accuracy %']:>6.2f}%{marker}")

print(f"""
CONCLUSION
----------
For the DOSM quarterly labour-force dataset, the Anaconda + Python
environment is suitable because it supports reproducible package
management, tabular-data processing, machine learning, and dashboard
deployment in one workflow.

Best model  : {best_model} ({best_acc}% accuracy)
Recommended : Random Forest for accuracy; Decision Tree for
              policy communication (interpretable rules).

Interactive Plotly charts and the Streamlit dashboard (dashboard.py)
make findings accessible to non-technical decision-makers.
""")
