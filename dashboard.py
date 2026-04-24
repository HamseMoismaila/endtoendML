"""
EC3355 – DOSM Malaysia Labour Force Dashboard
Run with: streamlit run dashboard.py
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EC3355 – DOSM Labour Force ML Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 10px; padding: 16px 20px;
    }
    .stMetric label { color: #8b949e !important; font-size: 12px !important; }
    .stMetric [data-testid="metric-container"] { background: #161b22; border-radius: 8px; padding: 12px; }
    h1, h2, h3 { color: #e6edf3; }
    .section-title { color: #58a6ff; font-size: 1rem; font-weight: 600;
                     border-left: 3px solid #58a6ff; padding-left: 10px; margin: 16px 0 12px; }
</style>
""", unsafe_allow_html=True)

# ── Load & prepare data ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_parquet('data/lfs_qtr_state.parquet')
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data
def prepare_model_data(df):
    def risk_level(u):
        if u < 3.0:   return 'Low'
        elif u < 4.5: return 'Medium'
        else:          return 'High'

    df = df.copy()
    df['risk'] = df['u_rate'].apply(risk_level)
    df = df.sort_values(['state', 'date'])
    df['u_rate_lag1'] = df.groupby('state')['u_rate'].shift(1)
    df['p_rate_lag1'] = df.groupby('state')['p_rate'].shift(1)
    df['lf_lag1']     = df.groupby('state')['lf'].shift(1)
    df['quarter']     = df['date'].dt.quarter
    df['year']        = df['date'].dt.year
    model_df = df.dropna().copy()
    model_df = pd.get_dummies(model_df, columns=['state', 'quarter'], drop_first=True)
    X = model_df.drop(columns=['risk', 'date', 'u_rate'])
    y = model_df['risk']
    return X, y, df

@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    svm   = Pipeline([('sc', StandardScaler()), ('clf', SVC(kernel='rbf', C=1.0, gamma='scale'))])
    tree  = DecisionTreeClassifier(max_depth=5, random_state=42)
    rf    = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)

    svm.fit(X_train, y_train);  svm_acc  = accuracy_score(y_test, svm.predict(X_test))
    tree.fit(X_train, y_train); tree_acc = accuracy_score(y_test, tree.predict(X_test))
    rf.fit(X_train, y_train);   rf_acc   = accuracy_score(y_test, rf.predict(X_test))

    return {
        'SVM':           (svm,  svm_acc,  svm.predict(X_test)),
        'Decision Tree': (tree, tree_acc, tree.predict(X_test)),
        'Random Forest': (rf,   rf_acc,   rf.predict(X_test)),
    }, X_test, y_test, X_train, rf

df = load_data()
X, y, df_feat = prepare_model_data(df)
models, X_test, y_test, X_train, rf_model = train_models(X, y)

COLORS = {'Low': '#3fb950', 'Medium': '#f0883e', 'High': '#f85149'}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 EC3355 Dashboard")
    st.markdown("**DOSM Malaysia Labour Force**")
    st.markdown("---")

    page = st.radio("Navigate", ["🏠 Overview", "🔍 Explorer", "🤖 Models", "🗺️ State Analysis"])
    st.markdown("---")

    st.markdown("**Dataset Info**")
    st.markdown(f"- Records: **{len(df):,}**")
    st.markdown(f"- States: **{df['state'].nunique()}**")
    st.markdown(f"- Period: **{df['date'].dt.year.min()}–{df['date'].dt.year.max()}**")
    st.markdown("---")
    st.markdown("*Run: `streamlit run dashboard.py`*")

# ═════════════════════════════════════════════════════════
# PAGE 1 – OVERVIEW
# ═════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🇲🇾 DOSM Labour Force – ML Dashboard")
    st.caption("EC3355 · Machine Learning Techniques and Computing Environment · P5, PLO3")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Unemployment Rate", f"{df['u_rate'].mean():.2f}%")
    col2.metric("Avg Participation Rate", f"{df['p_rate'].mean():.2f}%")
    col3.metric("Best Model Accuracy",    f"{max(v[1] for v in models.values())*100:.2f}%")
    col4.metric("Total Records",          f"{len(df):,}")

    st.markdown("---")

    # Risk distribution pie
    def risk_label(u):
        if u < 3.0: return 'Low'
        elif u < 4.5: return 'Medium'
        return 'High'
    df['risk'] = df['u_rate'].apply(risk_label)
    risk_counts = df['risk'].value_counts().reset_index()
    risk_counts.columns = ['Risk', 'Count']

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown('<div class="section-title">Risk Level Distribution</div>', unsafe_allow_html=True)
        fig_pie = px.pie(risk_counts, names='Risk', values='Count',
                         color='Risk', color_discrete_map=COLORS,
                         hole=0.5)
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               font_color='#e6edf3', margin=dict(t=10,b=10), height=300,
                               legend=dict(font=dict(size=12)))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">National Unemployment Trend</div>', unsafe_allow_html=True)
        nat = df.groupby('date')['u_rate'].mean().reset_index()
        fig_nat = px.area(nat, x='date', y='u_rate',
                          color_discrete_sequence=['#58a6ff'])
        fig_nat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               font_color='#e6edf3', xaxis=dict(gridcolor='#21262d'),
                               yaxis=dict(gridcolor='#21262d', title='Avg U-Rate (%)'),
                               height=300, margin=dict(t=10,b=40,l=0,r=0))
        fig_nat.add_vline(x='2020-04-01', line_dash='dash', line_color='#f85149',
                          annotation_text='COVID-19', annotation_font_color='#f85149')
        st.plotly_chart(fig_nat, use_container_width=True)

    # Model comparison
    st.markdown('<div class="section-title">Model Accuracy Comparison</div>', unsafe_allow_html=True)
    model_df_res = pd.DataFrame({
        'Model':    list(models.keys()),
        'Accuracy': [v[1]*100 for v in models.values()]
    }).sort_values('Accuracy', ascending=True)
    fig_acc = px.bar(model_df_res, x='Accuracy', y='Model', orientation='h',
                     color='Accuracy', color_continuous_scale='Greens',
                     text=model_df_res['Accuracy'].apply(lambda x: f'{x:.2f}%'),
                     range_x=[70, 100])
    fig_acc.update_traces(textposition='outside')
    fig_acc.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font_color='#e6edf3', coloraxis_showscale=False,
                           xaxis=dict(gridcolor='#21262d'), yaxis=dict(gridcolor='#21262d'),
                           height=280, margin=dict(t=10,b=40,l=0,r=60))
    st.plotly_chart(fig_acc, use_container_width=True)

# ═════════════════════════════════════════════════════════
# PAGE 2 – EXPLORER
# ═════════════════════════════════════════════════════════
elif page == "🔍 Explorer":
    st.title("🔍 Data Explorer")

    col1, col2 = st.columns(2)
    with col1:
        state_sel = st.selectbox("Select State", sorted(df['state'].unique()))
    with col2:
        metric_sel = st.selectbox("Select Metric",
            ['u_rate', 'p_rate', 'lf', 'lf_employed', 'lf_unemployed'],
            format_func=lambda x: {
                'u_rate':'Unemployment Rate (%)', 'p_rate':'Participation Rate (%)',
                'lf':'Labour Force (000s)', 'lf_employed':'Employed (000s)',
                'lf_unemployed':'Unemployed (000s)'
            }[x])

    filtered = df[df['state'] == state_sel]

    # Line chart
    fig_s = px.line(filtered, x='date', y=metric_sel,
                    title=f"{metric_sel} – {state_sel}",
                    markers=True, color_discrete_sequence=['#58a6ff'])
    fig_s.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                         font_color='#e6edf3', xaxis=dict(gridcolor='#21262d'),
                         yaxis=dict(gridcolor='#21262d'), height=380)
    st.plotly_chart(fig_s, use_container_width=True)

    # State comparison heatmap
    st.markdown('<div class="section-title">Unemployment Rate Heatmap – All States</div>', unsafe_allow_html=True)
    pivot = df.pivot_table(index='state', columns=df['date'].dt.year, values='u_rate', aggfunc='mean')
    fig_heat = px.imshow(pivot, color_continuous_scale='RdYlGn_r',
                          labels=dict(color='U-Rate %'), aspect='auto')
    fig_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font_color='#e6edf3', height=500, margin=dict(t=10))
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown('<div class="section-title">Recent Data</div>', unsafe_allow_html=True)
    st.dataframe(filtered[['date','u_rate','p_rate','lf','lf_employed','lf_unemployed']].tail(12)
                 .sort_values('date', ascending=False).reset_index(drop=True),
                 use_container_width=True)

# ═════════════════════════════════════════════════════════
# PAGE 3 – MODELS
# ═════════════════════════════════════════════════════════
elif page == "🤖 Models":
    st.title("🤖 Model Evaluation")

    model_sel = st.selectbox("Select Model", list(models.keys()))
    mdl, acc, preds = models[model_sel]

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc*100:.2f}%")

    report = classification_report(y_test, preds, output_dict=True)
    col2.metric("Macro F1-Score", f"{report['macro avg']['f1-score']:.3f}")
    col3.metric("Weighted Precision", f"{report['weighted avg']['precision']:.3f}")

    # Confusion matrix
    st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
    labels = ['High', 'Low', 'Medium']
    cm = confusion_matrix(y_test, preds, labels=labels)
    fig_cm = px.imshow(cm, x=labels, y=labels, text_auto=True,
                       color_continuous_scale='Blues',
                       labels=dict(x='Predicted', y='Actual'))
    fig_cm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#e6edf3', height=380, margin=dict(t=10))
    st.plotly_chart(fig_cm, use_container_width=True)

    # Per-class metrics
    st.markdown('<div class="section-title">Per-Class Performance</div>', unsafe_allow_html=True)
    class_metrics = []
    for cls in ['High', 'Low', 'Medium']:
        r = report[cls]
        class_metrics.append({'Class': cls, 'Precision': round(r['precision'],3),
                               'Recall': round(r['recall'],3), 'F1-Score': round(r['f1-score'],3),
                               'Support': int(r['support'])})
    st.dataframe(pd.DataFrame(class_metrics), use_container_width=True, hide_index=True)

    # Feature importance (RF only)
    if model_sel == 'Random Forest':
        st.markdown('<div class="section-title">Top 15 Feature Importances</div>', unsafe_allow_html=True)
        feat_imp = pd.Series(mdl.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
        fig_fi = px.bar(feat_imp.reset_index(), x='importance', y='index', orientation='h',
                        color='importance', color_continuous_scale='Blues')
        fig_fi.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font_color='#e6edf3', coloraxis_showscale=False,
                              xaxis=dict(gridcolor='#21262d', title='Importance'),
                              yaxis=dict(gridcolor='#21262d', title=''),
                              height=420, margin=dict(t=10,l=0))
        st.plotly_chart(fig_fi, use_container_width=True)

# ═════════════════════════════════════════════════════════
# PAGE 4 – STATE ANALYSIS
# ═════════════════════════════════════════════════════════
elif page == "🗺️ State Analysis":
    st.title("🗺️ State-Level Analysis")

    # Average u_rate bar chart
    st.markdown('<div class="section-title">Average Unemployment Rate by State</div>', unsafe_allow_html=True)
    state_avg = df.groupby('state')[['u_rate','p_rate']].mean().reset_index().sort_values('u_rate', ascending=True)
    fig_sb = px.bar(state_avg, x='u_rate', y='state', orientation='h',
                    color='u_rate', color_continuous_scale='RdYlGn_r',
                    labels={'u_rate':'Avg U-Rate (%)', 'state':'State'},
                    text=state_avg['u_rate'].apply(lambda x: f'{x:.2f}%'))
    fig_sb.update_traces(textposition='outside')
    fig_sb.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#e6edf3', coloraxis_showscale=False,
                          xaxis=dict(gridcolor='#21262d'), height=500, margin=dict(t=10,r=60))
    st.plotly_chart(fig_sb, use_container_width=True)

    # Risk distribution per state
    st.markdown('<div class="section-title">Risk Level Distribution by State</div>', unsafe_allow_html=True)
    risk_state = df.groupby(['state','risk']).size().reset_index(name='count')
    fig_rs = px.bar(risk_state, x='state', y='count', color='risk',
                    color_discrete_map=COLORS, barmode='stack')
    fig_rs.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#e6edf3', xaxis=dict(gridcolor='#21262d', tickangle=-35),
                          yaxis=dict(gridcolor='#21262d', title='Quarters'),
                          height=400, margin=dict(t=10,b=100))
    st.plotly_chart(fig_rs, use_container_width=True)

    # Scatter p_rate vs u_rate
    st.markdown('<div class="section-title">Participation Rate vs Unemployment Rate</div>', unsafe_allow_html=True)
    fig_sc2 = px.scatter(df, x='p_rate', y='u_rate', color='state',
                          hover_data=['date','risk'],
                          labels={'p_rate':'Participation Rate (%)','u_rate':'Unemployment Rate (%)'})
    fig_sc2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font_color='#e6edf3', xaxis=dict(gridcolor='#21262d'),
                           yaxis=dict(gridcolor='#21262d'), height=420, margin=dict(t=10))
    st.plotly_chart(fig_sc2, use_container_width=True)
