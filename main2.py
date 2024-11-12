# main2.py

import streamlit as st
from supervised_classification import run_classification_models
from supervised_regression import run_regression_models
from unsupervised_clustering import run_unsupervised_models

def run_ml_pipeline(df):
    # Verificar que el DataFrame esté disponible
    if df is None:
        st.error("No processed data available. Please complete data preprocessing first.")
        return

    st.write("## Machine Learning Model Selection")
    st.dataframe(df)

    problem_type = st.selectbox("Select Problem Type:", ["Supervised Learning", "Unsupervised Learning"])

    if problem_type == "Supervised Learning":
        target_column = st.selectbox("Select target (Y) variable:", options=df.columns)


        # Si se selecciona una columna de destino
        
        # Si se selecciona una columna de destino
        if target_column:
            # Selección de variables de características
            feature_columns = st.multiselect("Select feature (X) variables:", options=[col for col in df.columns if col != target_column])

            # Si el usuario ha seleccionado al menos una variable de características
            if feature_columns:
                X = df[feature_columns]
                y = df[target_column]

                supervised_type = st.selectbox("Is it a regression or classification problem?", ["Regression", "Classification"])

                if supervised_type == "Regression":
                    run_regression_models(X, y)

                elif supervised_type == "Classification":
                    run_classification_models(X, y)

    elif problem_type == "Unsupervised Learning":
        feature_columns = st.multiselect("Select features (X) variables for clustering:", options=df.columns)

        if feature_columns:
            X = df[feature_columns]
            run_unsupervised_models(X)
        else:
            st.warning("Please select at least one feature for clustering.")