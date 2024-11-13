import streamlit as st
from _2_MLTraining.supervised_classification import run_classification_models
from _2_MLTraining.supervised_regression import run_regression_models
from _2_MLTraining.unsupervised_clustering import run_unsupervised_models

def run_ml_pipeline(df):
    if df is None:
        st.error("No processed data available. Please complete data preprocessing first.")
        return None

    st.write("## Machine Learning Model Selection")
    st.dataframe(df)

    # Inicializamos el diccionario para almacenar modelos entrenados
    trained_models = {}

    # Selección del tipo de problema
    problem_type = st.selectbox("Select Problem Type:", ["Supervised Learning", "Unsupervised Learning"], key="ml_problem_type_select")

    if problem_type == "Supervised Learning":
        target_column = st.selectbox("Select target (Y) variable:", options=df.columns, key="ml_target_column_select")

        if target_column:
            feature_columns = st.multiselect("Select feature (X) variables:", options=[col for col in df.columns if col != target_column], key="ml_feature_columns_multiselect")

            if feature_columns:
                X = df[feature_columns]
                y = df[target_column]

                supervised_type = st.selectbox("Is it a regression or classification problem?", ["Regression", "Classification"], key="ml_supervised_type_select")

                if supervised_type == "Regression":
                    trained_models = run_regression_models(X, y)
                elif supervised_type == "Classification":
                    trained_models = run_classification_models(X, y)
            else:
                st.warning("Please select at least one feature variable.")
        else:
            st.warning("Please select a target variable.")

    elif problem_type == "Unsupervised Learning":
        feature_columns = st.multiselect("Select features (X) variables for clustering:", options=df.columns, key="ml_clustering_feature_columns_multiselect")

        if feature_columns:
            X = df[feature_columns]
            trained_models = run_unsupervised_models(X)
        else:
            st.warning("Please select at least one feature for clustering.")

    return trained_models