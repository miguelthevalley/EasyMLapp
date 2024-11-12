from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import time
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def train_and_evaluate(model, X, y, hyperparameters={}, scaler_type="RobustScaler", test_size=0.2, random_state=42, stratify_option=False):
    model_name = model.__class__.__name__

    # Dividir el conjunto de datos sin estratificación (para regresión)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Asignar hiperparámetros al modelo
    for param, value in hyperparameters.items():
        setattr(model, param, value)

    # Selección del escalador
    scaler = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "RobustScaler": RobustScaler()}.get(scaler_type, None)

    # Crear pipeline sin escalador si `scaler_type` es `None`
    if scaler:
        pipeline = Pipeline(steps=[('scaler', scaler), ('regressor', model)])
    else:
        pipeline = Pipeline(steps=[('regressor', model)])

    # Entrenar el modelo
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    execution_time = time.time() - start_time

    # Predicciones y evaluación
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Crear DataFrame de resultados
    result = pd.DataFrame({
        'Model': [model_name],
        'Train MSE': [train_mse],
        'Test MSE': [test_mse],
        'Train MAE': [train_mae],
        'Test MAE': [test_mae],
        'Train R2': [train_r2],
        'Test R2': [test_r2],
        'Execution Time': [execution_time]
    })

    return pipeline, result

import streamlit as st
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def run_regression_models(X, y):
    st.write("## Step 1: Configure Regression Model and Hyperparameters")

    # Seleccionar modelo de regresión
    model_type = st.selectbox("Select Regression Model:", ["LinearRegression", "RandomForestRegressor", "XGBRegressor"])

    # Seleccionar escalador
    scaler_type = st.selectbox("Select Scaler:", ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"])

    # Configuración de hiperparámetros basados en el modelo seleccionado
    if model_type == "LinearRegression":
        model = LinearRegression()
        st.write("### Linear Regression Hyperparameters")
        fit_intercept = st.checkbox("Fit Intercept", value=True)
        hyperparameters = {'fit_intercept': fit_intercept}


    elif model_type == "RandomForestRegressor":
        model = RandomForestRegressor()
        st.write("### Random Forest Regressor Hyperparameters")
        n_estimators = st.slider("Number of Estimators", min_value=50, max_value=300, value=100)
        max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=10)
        hyperparameters = {'n_estimators': n_estimators, 'max_depth': max_depth}

    elif model_type == "XGBRegressor":
        model = XGBRegressor()
        st.write("### XGBoost Regressor Hyperparameters")
        learning_rate = st.number_input("Learning Rate", min_value=0.01, max_value=1.0, value=0.1)
        n_estimators = st.slider("Number of Estimators", min_value=50, max_value=300, value=100)
        max_depth = st.slider("Max Depth", min_value=1, max_value=10, value=3)
        hyperparameters = {'learning_rate': learning_rate, 'n_estimators': n_estimators, 'max_depth': max_depth}

    # Inicializar historial de simulaciones
    if "simulation_history" not in st.session_state:
        st.session_state.simulation_history = pd.DataFrame()

    # Entrenar y evaluar el modelo seleccionado con los hiperparámetros configurados
    if st.button("Train and Evaluate Model"):
        pipeline, results = train_and_evaluate(
            model, X, y, hyperparameters=hyperparameters, scaler_type=scaler_type
        )
        if results is not None:
            st.write("### Evaluation Results")
            st.dataframe(results)

            # Guardar el resultado en el historial de simulaciones
            st.session_state.simulation_history = pd.concat([st.session_state.simulation_history, results], ignore_index=True)

    # Mostrar historial de simulaciones
    if not st.session_state.simulation_history.empty:
        st.write("## Simulation History")
        st.dataframe(st.session_state.simulation_history)