DOSM Malaysia Labour Force ML Project

An end-to-end Machine Learning project analyzing and predicting unemployment risk levels in Malaysia using historical Labour Force Survey (LFS) data. This project was developed 

---

## 🚀 Project Overview

This repository contains a full data science workflow, including data processing, feature engineering, multi-model evaluation, and an interactive deployment dashboard. The primary goal is to classify quarterly state-level unemployment into "Low," "Medium," or "High" risk categories based on historical indicators.

### Key Features:
*   **End-to-End Pipeline**: From raw `.parquet` data to trained models and visual outputs.
*   **Multi-Model Benchmarking**: Comparison between SVM, Decision Trees, Random Forests, and Deep Learning (Keras/TensorFlow).
*   **Interactive Dashboard**: A modern Streamlit-based interface for exploring national and state-level trends.
*   **Reproducible Environment**: Fully managed via Conda/Anaconda.

---

## 📂 Project Structure

```text
dosm-ml-project/
├── dashboard.py           # Streamlit interactive dashboard
├── environment.yml        # Conda environment configuration
├── data/
│   └── lfs_qtr_state.parquet  # DOSM Malaysia Labour Force dataset
├── src/
│   └── ml_pipeline.py     # Main ML training and evaluation script
├── notebooks/
│   └── dosm_ml_report.py  # Report generation script
├── outputs/               # Visualizations and HTML reports
│   ├── confusion_matrix_rf.png
│   ├── feature_importance.png
│   └── dosm_ml_report.html
└── README.md              # Project documentation
```

---

## 📊 Dataset & Target

*   **Source**: [DOSM Malaysia / data.gov.my](https://data.gov.my)
*   **Variables**: Date, State, Labour Force (`lf`), Employed, Unemployed, Outside Labour Force, Unemployment Rate (`u_rate`), Participation Rate (`p_rate`).
*   **Target Variable (`risk`)**:
    *   🟢 **Low**: `u_rate` < 3.0%
    *   🟡 **Medium**: 3.0% ≤ `u_rate` < 4.5%
    *   🔴 **High**: `u_rate` ≥ 4.5%

---

## ⚙️ Setup & Installation

Ensure you have [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd dosm-ml-project
    ```

2.  **Create and activate the environment**:
    ```bash
    conda env create -f environment.yml
    conda activate dosm-ml
    ```

---

## 🛠️ Usage

### 1. Run the ML Pipeline
This script processes the data, trains four models, generates performance metrics, and saves visualizations to the `outputs/` folder.
```bash
python src/ml_pipeline.py
```

### 2. Launch the Interactive Dashboard
Open the Plotly-powered Streamlit dashboard to explore state-level data and model performance.
```bash
streamlit run dashboard.py
```

### 3. View the Static Report
The automated report can be found at `outputs/dosm_ml_report.html`. Open it in any web browser to view the analysis.

---

## 📈 Model Performance

| Model | Accuracy | Recommended Use Case |
| :--- | :--- | :--- |
| **Deep Learning** | **92.05%** | Highest predictive power |
| **Random Forest** | **89.77%** | Best balance of accuracy and feature analysis |
| **Decision Tree** | **86.36%** | High interpretability for policy communication |
| **SVM** | **86.36%** | Robust performance on high-dimensional data |

---

## 🛠️ Built With

*   **Python 3.11**
*   **Scikit-Learn**: Traditional ML models and pipelines
*   **TensorFlow/Keras**: Deep Learning implementation
*   **Pandas & NumPy**: Data manipulation
*   **Plotly & Matplotlib**: Visualizations
*   **Streamlit**: Dashboard deployment

