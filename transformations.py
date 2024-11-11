import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def select_target_and_apply_transformations(df):
    # Selección de la columna objetivo (opcional)
    st.write("### Select Target Column")
    target_column = st.selectbox("Select the target (y) variable (optional):", [None] + list(df.columns))
    st.session_state.target_column = target_column

    # Aplicar transformaciones solo si df no está vacío
    if df is not None:
        st.write("## Step 4: Apply Transformations")

        # Opción para eliminar columnas
        st.write("### Drop Columns")
        drop_cols = st.multiselect("Select columns to drop:", [col for col in df.columns if col != target_column])
        if drop_cols and st.button("Apply Drop Columns"):
            df.drop(columns=drop_cols, inplace=True)
            st.success(f"Columns {', '.join(drop_cols)} dropped successfully.")
            st.write("Updated DataFrame:")
            st.dataframe(df)

        # Log Transformation para columnas numéricas
        numeric_columns = [col for col in df.select_dtypes(include=['float64', 'int64']) if col != target_column]
        st.write("### Log Transformation")
        log_cols = st.multiselect("Select columns for log transformation (or select 'All'):", ["All"] + numeric_columns)
        if "All" in log_cols:
            log_cols = numeric_columns
        if log_cols and st.button("Apply Log Transformation"):
            for col in log_cols:
                df[col] = np.log1p(df[col])  # log(1 + x)
            st.success("Log transformation applied.")
            st.write("Updated DataFrame:")
            st.dataframe(df)

        # Label Encoding para columnas categóricas
        categorical_columns = [col for col in df.select_dtypes(include=['object', 'category']) if col != target_column]
        st.write("### Label Encoding")
        label_cols = st.multiselect("Select columns for Label Encoding (or select 'All'):", ["All"] + categorical_columns)
        if "All" in label_cols:
            label_cols = categorical_columns
        if label_cols and st.button("Apply Label Encoding"):
            for col in label_cols:
                df[col] = LabelEncoder().fit_transform(df[col])
            st.success("Label Encoding applied.")
            st.write("Updated DataFrame:")
            st.dataframe(df)
        
        # One-Hot Encoding para columnas categóricas
        st.write("### One-Hot Encoding")
        ohe_cols = st.multiselect("Select columns for One-Hot Encoding (or select 'All'):", ["All"] + categorical_columns)
        if "All" in ohe_cols:
            ohe_cols = categorical_columns
        if ohe_cols and st.button("Apply One-Hot Encoding"):
            df = pd.get_dummies(df, columns=ohe_cols)
            st.success("One-Hot Encoding applied.")
            st.write("Updated DataFrame:")
            st.dataframe(df)

        # Target Encoding (Media de la columna objetivo `y` para cada categoría)
        if target_column:
            st.write("### Target Encoding")
            target_encode_cols = st.multiselect("Select columns for Target Encoding:", categorical_columns)
            if target_encode_cols and st.button("Apply Target Encoding"):
                for col in target_encode_cols:
                    # Calcula la media de la variable objetivo para cada categoría de la columna seleccionada
                    encoding_map = df.groupby(col)[target_column].mean()
                    df[col] = df[col].map(encoding_map)
                st.success("Target Encoding applied.")
                st.write("Updated DataFrame:")
                st.dataframe(df)
    
    return df