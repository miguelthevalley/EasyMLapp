import streamlit as st
import pandas as pd

def select_and_predict():
    # Asegúrate de que hay modelos entrenados
    if "trained_models" in st.session_state and st.session_state.trained_models:
        # Selección del modelo a guardar
        model_name = st.selectbox("Select a model to save:", list(st.session_state.trained_models.keys()))
        st.session_state.saved_model = st.session_state.trained_models[model_name]
        st.success(f"Model '{model_name}' saved successfully for predictions!")

        # Entrada de nuevas variables
        st.write("### Enter new data for prediction")
        new_data_method = st.radio("Choose how to input new data:", ["Manual Input", "Upload CSV"])

        # Entrada de datos manual
        if new_data_method == "Manual Input":
            input_data = {}
            for feature in st.session_state.df_transformed.columns:
                input_value = st.number_input(f"Enter value for {feature}", format="%.4f")
                input_data[feature] = input_value
            new_data = pd.DataFrame([input_data])

        # Subir datos desde CSV
        elif new_data_method == "Upload CSV":
            new_data_file = st.file_uploader("Upload a CSV file with new data for prediction", type=["csv"])
            if new_data_file is not None:
                try:
                    new_data = pd.read_csv(new_data_file)
                    st.write("### New Data Preview")
                    st.dataframe(new_data)
                except Exception as e:
                    st.error(f"Error loading new data file: {e}")
                    new_data = None

        # Realizar predicciones
        if st.button("Predict") and st.session_state.saved_model is not None:
            if new_data is not None:
                predictions = st.session_state.saved_model.predict(new_data)
                st.write("### Predictions")
                st.write(predictions)
            else:
                st.warning("Please input new data for prediction.")
    else:
        st.warning("Please complete the Machine Learning Pipeline step to train models.")