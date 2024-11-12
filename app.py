import streamlit as st
import pandas as pd
from main1 import run_preprocessing_pipeline
from main2 import run_ml_pipeline


st.set_page_config(page_title="EasyMLApp", layout="wide")  # Debe estar en la primera línea

# Título de la aplicación
st.title("EasyMLApp")

# Inicializar `st.session_state` para el DataFrame y el paso seleccionado
if "df" not in st.session_state:
    st.session_state.df = None  # DataFrame original cargado
if "df_transformed" not in st.session_state:
    st.session_state.df_transformed = None  # DataFrame modificado tras cada paso
if "selected_step" not in st.session_state:
    st.session_state.selected_step = "Upload Data"  # Paso inicial para subir el archivo

# Menú de navegación que muestra el paso actual en `st.session_state.selected_step`
steps = ["Upload Data", "Data Preprocessing", "Machine Learning Pipeline"]
selected_step = st.sidebar.selectbox("Select a step:", steps, index=steps.index(st.session_state.selected_step))

# Actualizar el estado del paso seleccionado en `st.session_state`
st.session_state.selected_step = selected_step

# Función para actualizar el DataFrame transformado y mantenerlo en `st.session_state`
def update_processed_data(df):
    if df is not None:
        st.session_state.df_transformed = df

# Paso 1: Subir Datos
if st.session_state.selected_step == "Upload Data":
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

# Paso 2: Preprocesamiento de datos
if st.session_state.selected_step == "Data Preprocessing":
    st.write("### Step 1: Data Preprocessing")
    # Ejecutar el pipeline de preprocesamiento y actualizar el DataFrame transformado
    if st.session_state.df is not None:
        st.session_state.df_transformed = run_preprocessing_pipeline(st.session_state.df)  # Pasa el DataFrame
        update_processed_data(st.session_state.df_transformed)  # Guardar el DataFrame transformado
    else:
        st.warning("Please upload data first.")

# Paso 3: Pipeline de Machine Learning
if st.session_state.selected_step == "Machine Learning Pipeline":
    st.write("### Step 2: Machine Learning Pipeline")
    if st.session_state.df_transformed is not None:
        run_ml_pipeline(st.session_state.df_transformed)  # Usa el último DataFrame procesado
    else:
        st.warning("Please complete the Data Preprocessing step first.")