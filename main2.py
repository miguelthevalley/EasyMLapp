import streamlit as st
from supervised_classification import run_classification_models
from supervised_regression import run_regression_models
from unsupervised_clustering import run_clustering_models

def run_ml_pipeline(df):
    # Verificar que el DataFrame esté disponible
    if df is None:
        st.error("No processed data available. Please complete data preprocessing first.")
        return

    st.write("## Machine Learning Model Selection")
    st.dataframe(df)  # Mostrar el DataFrame actual

    # Selección del tipo de problema (supervisado o no supervisado)
    problem_type = st.selectbox("Select Problem Type:", ["Supervised Learning", "Unsupervised Learning"])

    target_column = None  # Default value for target column

    if problem_type == "Supervised Learning":
        # Selección de la variable objetivo (Y)
        target_column = st.selectbox("Select target (Y) variable:", options=df.columns)

        # Si se selecciona una columna de destino
        if target_column:
            # Separar las características (X) y la variable objetivo (y)
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Elegir entre clasificación o regresión
            supervised_type = st.selectbox("Is it a regression or classification problem?", ["Regression", "Classification"])

            # Ejecutar modelos de regresión
            if supervised_type == "Regression":
                run_regression_models(X, y)

            # Ejecutar modelos de clasificación
            elif supervised_type == "Classification":
                run_classification_models(X, y)  # Pasa tanto X como y a la función
        else:
            st.warning("Please select a target variable to proceed.")

    # Flujo para modelos no supervisados
    elif problem_type == "Unsupervised Learning":
        run_clustering_models(df)
    else:
        st.warning("Please select a valid problem type.")