import streamlit as st
import pandas as pd
from eda import initial_eda, final_eda
from imputation import impute_missing_values
from transformations import select_target_and_apply_transformations

def load_data():
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            st.session_state.df = df  # Guardar el DataFrame en `st.session_state`
            st.success("File uploaded successfully!")
            st.write("### Data Preview")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error loading file: {e}")

def run_preprocessing_pipeline(df):
    # Verificar si el DataFrame existe
    if df is None:
        load_data()  # Permitir cargar el archivo si aún no se ha cargado
        return None

    # Paso 1: EDA inicial
    st.write("### Step 1: Initial Exploratory Data Analysis (EDA)")
    initial_eda(df)
    st.write("### Data Preview")
    st.dataframe(df)

    # Paso 2: Imputación de valores nulos
    st.write("### Step 2: Impute Missing Values")
    df_imputed = impute_missing_values(df)
    st.write("### Data after Imputation")
    st.dataframe(df_imputed)

    # Paso 3: Selección de variable objetivo y aplicación de transformaciones
    st.write("### Step 3: Select Target & Apply Transformations")
    df_transformed = select_target_and_apply_transformations(df_imputed)
    st.write("### Data after Transformations")
    st.dataframe(df_transformed)

    # Paso 4: EDA final después de las transformaciones
    st.write("### Step 4: Final EDA After Transformations")
    final_eda(df_transformed)

    # Guardar el DataFrame transformado en `st.session_state` y devolverlo
    st.session_state.df_transformed = df_transformed
    return st.session_state.df_transformed