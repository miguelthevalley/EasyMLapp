import streamlit as st
import pandas as pd
import time
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

def train_and_evaluate(model, X, y, hyperparameters={}, scaler_type="RobustScaler", test_size=0.2, random_state=42, stratify_option=True, cv=0):
    model_name = model.__class__.__name__
    stratify_param = y if stratify_option else None

    # Asignar hiperparámetros
    for param, value in hyperparameters.items():
        setattr(model, param, value)

    # Configurar el escalador
    scaler = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "RobustScaler": RobustScaler()}.get(scaler_type, None)
    pipeline = Pipeline(steps=[('scaler', scaler), ('classifier', model)])

    # Dividir el conjunto de datos en train-test
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
    except ValueError as e:
        st.error(f"Error: {e}")
        return None, None

    if cv > 0:
        # Validación cruzada en el conjunto de entrenamiento
        start_time = time.time()
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
        execution_time = time.time() - start_time

        # Entrenar el modelo en el conjunto de entrenamiento completo y evaluar en test
        pipeline.fit(X_train, y_train)
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')

        # Resultados de CV y de test final
        result = pd.DataFrame({
            'Model': [model_name],
            'Scaler': [scaler_type],
            'CV': [cv],
            'Hyperparameters': [", ".join(f"{k}={v}" for k, v in hyperparameters.items())],
            'CV Accuracy Mean': [cv_scores.mean()],
            'CV Accuracy Std': [cv_scores.std()],
            'Test Accuracy': [test_accuracy],
            'Test Precision': [test_precision],
            'Test Recall': [test_recall],
            'Execution Time': [execution_time]
        })

    else:
        # Entrenar y evaluar sin validación cruzada
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

        result = pd.DataFrame({
            'Model': [model_name],
            'Scaler': [scaler_type],
            'CV': '0',
            'Hyperparameters': [", ".join(f"{k}={v}" for k, v in hyperparameters.items())],
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

    # Seleccionar modelo de clasificación
    model_type = st.selectbox("Select Classification Model:", ["XGBClassifier", "RandomForestClassifier", "LogisticRegression"])

    # Seleccionar escalador
    scaler_type = st.selectbox("Select Scaler:", ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"])
    stratify_option = st.checkbox("Stratify Split", value=True)

    # Seleccionar cross-validation
    cv = st.slider("Select Cross-Validation (CV) folds:", min_value=0, max_value=10, step=2)

    # Configuración de hiperparámetros basados en el modelo seleccionado
    if model_type == "XGBClassifier":
        model = XGBClassifier()
        st.write("### XGBClassifier Hyperparameters")
        learning_rate = st.number_input("Learning Rate", min_value=0.01, max_value=1.0, value=0.1)
        n_estimators = st.slider("Number of Estimators", min_value=50, max_value=300, value=100)
        max_depth = st.slider("Max Depth", min_value=1, max_value=10, value=3)
        hyperparameters = {'learning_rate': learning_rate, 'n_estimators': n_estimators, 'max_depth': max_depth}

    elif model_type == "RandomForestClassifier":
        model = RandomForestClassifier()
        st.write("### RandomForestClassifier Hyperparameters")
        n_estimators = st.slider("Number of Estimators", min_value=50, max_value=300, value=100)
        max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=10)
        hyperparameters = {'n_estimators': n_estimators, 'max_depth': max_depth}

    elif model_type == "LogisticRegression":
        model = LogisticRegression()
        st.write("### LogisticRegression Hyperparameters")
        C = st.number_input("Inverse of Regularization Strength (C)", min_value=0.01, max_value=10.0, value=1.0)
        max_iter = st.slider("Maximum Iterations", min_value=50, max_value=500, value=100)
        hyperparameters = {'C': C, 'max_iter': max_iter}

    # Inicializar historial de simulaciones y diccionario de modelos
    if "simulation_history" not in st.session_state:
        st.session_state.simulation_history = pd.DataFrame()
    if "trained_models" not in st.session_state:
        st.session_state.trained_models = {}  # Inicializar trained_models si no existe

    # Entrenar y evaluar el modelo seleccionado con los hiperparámetros configurados
    if st.button("Train and Evaluate Model"):
        pipeline, results = train_and_evaluate(
            model, X, y, hyperparameters=hyperparameters, scaler_type=scaler_type, stratify_option=stratify_option, cv=cv
        )
        if results is not None:
            st.write("### Evaluation Results")
            st.dataframe(results)

            # Guardar el modelo entrenado en `st.session_state.trained_models`
            st.session_state.trained_models[model_type] = {
                "model": pipeline,          # El pipeline del modelo entrenado
                "features": list(X.columns) # Las características utilizadas para entrenar
}

            # Guardar el resultado en el historial de simulaciones
            st.session_state.simulation_history = pd.concat([st.session_state.simulation_history, results], ignore_index=True)

    # Mostrar historial de simulaciones
    if not st.session_state.simulation_history.empty:
        st.write("## Simulation History")
        st.dataframe(st.session_state.simulation_history)