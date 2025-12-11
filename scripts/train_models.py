import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from datetime import datetime

def train_models():
    print("="*70)
    print("ENTRENAMIENTO DE MODELOS (MLP vs SVM)")
    print("="*70)

    # Rutas
    DATA_DIR = 'data/processed'
    MODEL_DIR = 'models'
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Cargar datos preprocesados (YA ESCALADOS)
    print("Cargando datos...")
    try:
        X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
        X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
        y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
        y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
        print(f"Datos cargados: Train {X_train.shape}, Test {X_test.shape}")
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        return

    # --- ENTRENAMIENTO SVM ---
    print("\n" + "-"*30)
    print("Entrenando SVM...")
    # SVM con probabilidad para curvas ROC
    svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    svm.fit(X_train, y_train)
    
    # Evaluación SVM
    svm_pred = svm.predict(X_test)
    svm_prob = svm.predict_proba(X_test)[:, 1]
    
    svm_metrics = {
        "accuracy": accuracy_score(y_test, svm_pred),
        "precision": precision_score(y_test, svm_pred),
        "recall": recall_score(y_test, svm_pred),
        "f1_score": f1_score(y_test, svm_pred),
        "auc": roc_auc_score(y_test, svm_prob)
    }
    print(f"SVM Metrics: {svm_metrics}")
    
    # Guardar SVM
    joblib.dump(svm, os.path.join(MODEL_DIR, 'svm_model.pkl'))
    
    # --- ENTRENAMIENTO MLP ---
    print("\n" + "-"*30)
    print("Entrenando MLP...")
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, activation='relu', solver='adam', random_state=42)
    mlp.fit(X_train, y_train)
    
    # Evaluación MLP
    mlp_pred = mlp.predict(X_test)
    mlp_prob = mlp.predict_proba(X_test)[:, 1]
    
    mlp_metrics = {
        "accuracy": accuracy_score(y_test, mlp_pred),
        "precision": precision_score(y_test, mlp_pred),
        "recall": recall_score(y_test, mlp_pred),
        "f1_score": f1_score(y_test, mlp_pred),
        "auc": roc_auc_score(y_test, mlp_prob)
    }
    print(f"MLP Metrics: {mlp_metrics}")
    
    # Guardar MLP
    joblib.dump(mlp, os.path.join(MODEL_DIR, 'mlp_model.pkl'))

    print("\n" + "="*70)
    print("Modelos guardados exitosamente en 'models/'")

if __name__ == "__main__":
    train_models()
