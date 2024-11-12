import streamlit as st
import pandas as pd

def impute_missing_values(df):
    st.write("### Handle Missing Values")

    # Separar columnas numéricas y categóricas
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns
    
    # Detectar columnas con valores nulos
    missing_numeric_cols = [col for col in numeric_columns if df[col].isnull().any()]
    missing_categorical_cols = [col for col in categorical_columns if df[col].isnull().any()]
    
    if len(missing_numeric_cols) == 0 and len(missing_categorical_cols) == 0:
        st.info("No missing values detected.")
        return df
    
    # Diccionarios para almacenar las estrategias de imputación seleccionadas
    impute_strategies = {}
    constant_values = {}

    # Imputación para columnas numéricas
    if missing_numeric_cols:
        st.write("#### Numeric Columns with Missing Values")
        for col in missing_numeric_cols:
            st.write(f"Column `{col}` has {df[col].isnull().sum()} missing values.")
            strategy = st.selectbox(
                f"Choose strategy for `{col}`", 
                options=["Mean", "Median", "Drop Rows"],
                key=f"impute_numeric_{col}"
            )
            impute_strategies[col] = strategy
    
    # Imputación para columnas categóricas
    if missing_categorical_cols:
        st.write("#### Categorical Columns with Missing Values")
        for col in missing_categorical_cols:
            st.write(f"Column `{col}` has {df[col].isnull().sum()} missing values.")
            strategy = st.selectbox(
                f"Choose strategy for `{col}`", 
                options=["Mode", "Constant", "Drop Rows"],
                key=f"impute_categorical_{col}"
            )
            impute_strategies[col] = strategy
            if strategy == "Constant":
                constant_values[col] = st.text_input(f"Specify constant value for `{col}`:", key=f"constant_value_{col}")

    # Botón para aplicar todas las estrategias de imputación seleccionadas
    if st.button("Apply Imputation Strategies"):
        for col, strategy in impute_strategies.items():
            if strategy == "Mean" and col in missing_numeric_cols:
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == "Median" and col in missing_numeric_cols:
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == "Mode" and col in missing_categorical_cols:
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif strategy == "Constant" and col in constant_values:
                df[col].fillna(constant_values[col], inplace=True)
            elif strategy == "Drop Rows":
                df.dropna(subset=[col], inplace=True)
        
        st.success("Imputation and/or row removal applied successfully.")
        return df

    return None