# supervised_classification.py

import streamlit as st
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import pandas as pd
import time

def train_and_evaluate(model, X, y, hyperparameters={}, scaler_type="RobustScaler", test_size=0.2, random_state=42, stratify_option=True):
    model_name = model.__class__.__name__
    stratify_param = y if stratify_option else None

    # Dividir el conjunto de datos
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
    except ValueError as e:
        st.error(f"Error: {e}")
        return None, None

    # Asignar hiperparámetros
    for param, value in hyperparameters.items():
        setattr(model, param, value)

    scaler = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "RobustScaler": RobustScaler()}.get(scaler_type, RobustScaler())
    pipeline = Pipeline(steps=[('scaler', scaler), ('classifier', model)])

    start_time = time.time()
    pipeline.fit(X_train, y_train)
    execution_time = time.time() - start_time

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')

    st.write(f"### Model: {model_name}")
    st.write(f"Execution Time: {execution_time:.2f} seconds")
    st.write(f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
    st.write(f"Train Precision: {train_precision:.4f}, Test Precision: {test_precision:.4f}")
    st.write(f"Train Recall: {train_recall:.4f}, Test Recall: {test_recall:.4f}")

    result = pd.DataFrame({
        'Model': [model_name],
        'Train Accuracy': [train_accuracy],
        'Test Accuracy': [test_accuracy],
        'Train Precision': [train_precision],
        'Test Precision': [test_precision],
        'Train Recall': [train_recall],
        'Test Recall': [test_recall],
        'Execution Time': [execution_time]
    })

    return pipeline, result

def run_classification_models(X, y):
    st.write("## Step 1: Configure Classification Model and Hyperparameters")

    # Seleccionar características (X)
    feature_columns = st.multiselect("Select Feature Columns (X):", options=X.columns.tolist(), default=X.columns.tolist())
    X = X[feature_columns]

    scaler_type = st.selectbox("Select Scaler:", ["StandardScaler", "MinMaxScaler", "RobustScaler"])
    stratify_option = st.checkbox("Stratify Split", value=True)

    # Configuración de hiperparámetros para XGBClassifier
    model = XGBClassifier()
    st.write("### XGBClassifier Hyperparameters")
    learning_rate = st.number_input("Learning Rate", min_value=0.01, max_value=1.0, value=0.1)
    n_estimators = st.slider("Number of Estimators", min_value=50, max_value=300, value=100)
    max_depth = st.slider("Max Depth", min_value=1, max_value=10, value=3)
    hyperparameters = {'learning_rate': learning_rate, 'n_estimators': n_estimators, 'max_depth': max_depth}

    if st.button("Train and Evaluate Model"):
        pipeline, results = train_and_evaluate(
            model, X, y, hyperparameters=hyperparameters, scaler_type=scaler_type, stratify_option=stratify_option
        )
        if results is not None:
            st.write("### Evaluation Results")
            st.dataframe(results)