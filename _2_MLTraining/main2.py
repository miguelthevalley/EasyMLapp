# main2.py

import streamlit as st
from _2_MLTraining.supervised_classification import run_classification_models
from _2_MLTraining.supervised_regression import run_regression_models
from _2_MLTraining.unsupervised_clustering import run_unsupervised_models

def run_ml_pipeline(df):
    # Verificar que el DataFrame esté disponible
    if df is None:
        st.error("No processed data available. Please complete data preprocessing first.")
        return

    st.write("## Machine Learning Model Selection")
    st.dataframe(df)

    # Inicializamos el diccionario para almacenar modelos entrenados
    trained_models = {}

    # Selección del tipo de problema con una clave única
    problem_type = st.selectbox("Select Problem Type:", ["Supervised Learning", "Unsupervised Learning"], key="ml_problem_type_select")

    if problem_type == "Supervised Learning":
        # Seleccionar la variable objetivo con una clave única
        target_column = st.selectbox("Select target (Y) variable:", options=df.columns, key="ml_target_column_select")

        if target_column:
            # Selección de variables de características con una clave única
            feature_columns = st.multiselect("Select feature (X) variables:", options=[col for col in df.columns if col != target_column], key="ml_feature_columns_multiselect")

            # Verificar si se han seleccionado las variables de características
            if feature_columns:
                X = df[feature_columns]
                y = df[target_column]

                # Selección del tipo de aprendizaje supervisado con una clave única
                supervised_type = st.selectbox("Is it a regression or classification problem?", ["Regression", "Classification"], key="ml_supervised_type_select")

                if supervised_type == "Regression":
                    # st.session_state.trained_models = run_regression_models(X, y)  # Guardar modelos de regresión entrenados
                    trained_models = run_regression_models(X, y)
                elif supervised_type == "Classification":
                    # st.session_state.trained_models = run_classification_models(X, y)  # Guardar modelos de clasificación entrenados
                    trained_models = run_classification_models(X, y)
            else:
                st.warning("Please select at least one feature variable for supervised learning.")
        else:
            st.warning("Please select a target variable for supervised learning.")

    elif problem_type == "Unsupervised Learning":
        # Seleccionar variables de características para clustering con una clave única
        feature_columns = st.multiselect("Select features (X) variables for clustering:", options=df.columns, key="ml_clustering_feature_columns_multiselect")

        if feature_columns:
            X = df[feature_columns]
            st.session_state.trained_models = run_unsupervised_models(X)  # Guardar modelos de clustering entrenados
        else:
            st.warning("Please select at least one feature for clustering.")
    
        return trained_models