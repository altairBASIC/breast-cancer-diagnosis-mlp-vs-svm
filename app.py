"""
Streamlit app for Breast Cancer Diagnosis (MLP vs SVM)

- Loads preprocessed data from data/processed if available
- Otherwise falls back to raw CSV or sklearn's built-in dataset
- Trains/evaluates MLPClassifier and SVC
- Provides simple prediction UI

Run:
    streamlit run app.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
)

import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PROC_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"

st.set_page_config(page_title="Breast Cancer: MLP vs SVM", layout="wide")

# -------------
# Data loading
# -------------

@st.cache_data(show_spinner=False)
def load_processed() -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]]:
    """Load processed train/test arrays and feature names if available."""
    xtr = PROC_DIR / "X_train.npy"
    xte = PROC_DIR / "X_test.npy"
    ytr = PROC_DIR / "y_train.npy"
    yte = PROC_DIR / "y_test.npy"
    finfo = PROC_DIR / "feature_info.json"

    if all(p.exists() for p in [xtr, xte, ytr, yte]):
        X_train = np.load(xtr)
        X_test = np.load(xte)
        y_train = np.load(ytr)
        y_test = np.load(yte)
        feature_names: List[str] = []
        if finfo.exists():
            try:
                with finfo.open("r", encoding="utf-8") as f:
                    info = json.load(f)
                    feature_names = info.get("feature_names", [])
            except Exception:
                feature_names = []
        return X_train, X_test, y_train, y_test, feature_names
    return None


def _prepare_from_dataframe(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = df.copy()
    # Try typical Kaggle-style breast cancer CSV conventions
    target = None
    if "diagnosis" in df.columns:
        # Map M (malignant) = 1, B (benign) = 0
        if df["diagnosis"].dtype == object:
            mapping = {"M": 1, "B": 0, "benign": 0, "malignant": 1}
            df["diagnosis"] = df["diagnosis"].map(lambda v: mapping.get(str(v).strip(), v))
        target = "diagnosis"
    elif "target" in df.columns:
        target = "target"

    # Drop common non-feature columns if present
    drop_cols = [c for c in ["id", "Unnamed: 32"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if target is None:
        # Fallback: last column as target if it looks binary
        last = df.columns[-1]
        if df[last].nunique() <= 3:
            target = last
        else:
            # No clear target, assume sklearn dataset-like won't hit this path normally
            raise ValueError("No target column found. Expected 'diagnosis' or 'target'.")

    y = df[target].astype(int).to_numpy()
    X = df.drop(columns=[target]).to_numpy(dtype=float)
    feature_names = [c for c in df.columns if c != target]
    return X, y, feature_names


@st.cache_data(show_spinner=False)
def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load data with preference: processed arrays > CSV > sklearn dataset."""
    processed = load_processed()
    if processed is not None:
        X_train, X_test, y_train, y_test, feature_names = processed
        return X_train, X_test, y_train, y_test, feature_names

    # Try CSV if present
    csv_path = DATA_DIR / "breast_cancer.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        X, y, feature_names = _prepare_from_dataframe(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test, feature_names

    # Fallback to sklearn dataset
    from sklearn.datasets import load_breast_cancer

    bunch = load_breast_cancer()
    X, y = bunch.data, bunch.target
    feature_names = list(bunch.feature_names)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, feature_names


# ----------------
# Model utilities
# ----------------

def build_models() -> Dict[str, object]:
    models: Dict[str, object] = {
        "MLPClassifier": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", probability=True),
    }
    return models


def evaluate(model, X_train, y_train, X_test, y_test) -> Dict[str, float]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Note: for ROC AUC we need probabilities for positive class (index 1)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        # Some models may not expose predict_proba; try decision_function
        try:
            scores = model.decision_function(X_test)
            # Min-max scale decision scores to [0,1]
            s_min, s_max = scores.min(), scores.max()
            y_prob = (scores - s_min) / (s_max - s_min + 1e-9)
        except Exception:
            y_prob = None

    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
        except Exception:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")

    cm = confusion_matrix(y_test, y_pred)
    return metrics | {"confusion_matrix": cm}


# Cache training to avoid recomputation on UI tweaks
@st.cache_resource(show_spinner=True)
def train_cached(model_name: str, X_train, y_train, X_test, y_test):
    model = build_models()[model_name]
    results = evaluate(model, X_train, y_train, X_test, y_test)
    return model, results


# ---------
# Sidebar
# ---------

with st.sidebar:
    st.title("⚙️ Config")
    st.caption("Model and data options")

    model_name = st.selectbox("Modelo", list(build_models().keys()), index=0)
    st.divider()

    st.caption("Datos")
    data_source = "processed/CSV/Sklearn (auto)"
    if (PROC_DIR / "X_train.npy").exists():
        st.success("Usando datos procesados (data/processed)")
    elif (DATA_DIR / "breast_cancer.csv").exists():
        st.info("Usando CSV local (data/breast_cancer.csv)")
    else:
        st.warning("Usando dataset de sklearn (fallback)")


# ---------
# Main UI
# ---------

st.title("Breast Cancer Diagnosis — MLP vs SVM")
st.caption("Entrena, evalúa y compara modelos rápidamente. Usa tus datos procesados si están disponibles.")

X_train, X_test, y_train, y_test, feature_names = load_data()

# Train/Eval section
col_left, col_right = st.columns([1, 1])
with col_left:
    st.subheader("Entrenamiento y evaluación")
    run_train = st.button("Entrenar y evaluar", type="primary")

    if run_train:
        with st.spinner("Entrenando modelos..."):
            model, results = train_cached(model_name, X_train, y_train, X_test, y_test)
        st.success(f"Entrenamiento completado: {model_name}")

        # Metrics
        m = {k: v for k, v in results.items() if k != "confusion_matrix"}
        st.write({k: (float(v) if isinstance(v, (np.floating, np.float64)) else v) for k, v in m.items()})

        # Confusion matrix plot
        cm = results["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Matriz de confusión")
        st.pyplot(fig)

        # ROC curve (if probability available)
        try:
            RocCurveDisplay.from_estimator(model, X_test, y_test)
            st.pyplot(plt.gcf())
        except Exception:
            st.caption("ROC no disponible para este modelo/configuración.")

with col_right:
    st.subheader("Predicción interactiva")
    st.caption("Introduce valores para generar una predicción. Se usarán estadísticas simples si no hay entrenamiento previo en esta sesión.")

    # For input defaults, compute medians from training data
    X_df = pd.DataFrame(X_train, columns=feature_names if feature_names else None)
    defaults = X_df.median(numeric_only=True).to_list() if not X_df.empty else [0.0] * (len(feature_names) or 30)

    n_features = X_train.shape[1]
    user_vals: List[float] = []
    # Split form into two columns for readability
    cols = st.columns(2)
    for i in range(n_features):
        label = feature_names[i] if feature_names else f"feature_{i}"
        col = cols[i % 2]
        with col:
            val = st.number_input(label, value=float(defaults[i] if i < len(defaults) else 0.0), format="%.4f")
        user_vals.append(val)

    do_predict = st.button("Predecir")
    if do_predict:
        with st.spinner("Entrenando/recuperando modelo..."):
            model, _ = train_cached(model_name, X_train, y_train, X_test, y_test)
        x = np.array(user_vals, dtype=float).reshape(1, -1)
        try:
            prob = float(model.predict_proba(x)[0, 1])
        except Exception:
            # As backup use decision function and squash
            try:
                s = float(model.decision_function(x)[0])
                prob = 1.0 / (1.0 + np.exp(-s))
            except Exception:
                prob = float("nan")
        pred = int(model.predict(x)[0])
        st.metric("Predicción (1=maligno, 0=benigno)", value=str(pred))
        st.progress(0.0 if np.isnan(prob) else prob, text=f"Prob. maligno: {prob:.3f}" if not np.isnan(prob) else "Prob. no disponible")

# Reports section (if any)
st.divider()
st.subheader("Reportes de entrenamiento (si existen)")
rep_cols = st.columns(2)
rep_map = {
    "MLP": MODELS_DIR / "mlp_training_report.json",
    "SVM": MODELS_DIR / "svm_training_report.json",
}
for (label, path), col in zip(rep_map.items(), rep_cols):
    with col:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    report = json.load(f)
                st.json(report)
            except Exception as e:
                st.error(f"No se pudo leer {path.name}: {e}")
        else:
            st.caption(f"{path.name} no encontrado.")

# Dataset preview tab
st.divider()
st.subheader("Vista rápida del dataset")
preview_tabs = st.tabs(["DataFrame", "Descripción"])
with preview_tabs[0]:
    # Try CSV first, then processed arrays as DataFrame, then sklearn
    csv_path = DATA_DIR / "breast_cancer.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            st.dataframe(df.head(50))
        except Exception as e:
            st.error(f"No se pudo leer CSV: {e}")
    else:
        # Show X_test snapshot
        df = pd.DataFrame(X_test, columns=feature_names if feature_names else None)
        st.dataframe(df.head(50))

with preview_tabs[1]:
    sum_path = DATA_DIR / "data_summary.txt"
    if sum_path.exists():
        try:
            st.code(sum_path.read_text(encoding="utf-8"), language="text")
        except Exception as e:
            st.error(f"No se pudo leer data_summary.txt: {e}")
    else:
        st.caption("data_summary.txt no encontrado; mostrando descripción de X_test")
        st.write(pd.DataFrame(X_test).describe())
