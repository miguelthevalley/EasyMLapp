import streamlit as st
import pandas as pd

def select_and_predict():
    # Verificar que `trained_models` esté en `st.session_state` y no esté vacío
    if "trained_models" in st.session_state and st.session_state.trained_models:
        # Seleccionar el modelo para realizar predicciones
        model_key = st.selectbox("Select a model for prediction:", list(st.session_state.trained_models.keys()))
        
        # Obtener modelo y su información asociada
        selected_model_info = st.session_state.trained_models[model_key]
        selected_model = selected_model_info["model"]
        selected_features = selected_model_info["features"]
        evaluation_metrics = selected_model_info.get("evaluation", None)  # Métricas pueden no existir

        # Mostrar detalles del modelo seleccionado
        st.write(f"### Details for Model: {model_key}")
        if evaluation_metrics:
            st.write("#### Evaluation Metrics:")
            st.table(pd.DataFrame([evaluation_metrics]))
        else:
            st.info(f"The selected model '{model_key}' has no evaluation metrics (e.g., clustering).")

        st.success(f"Model '{model_key}' is selected and ready for predictions!")

        # Entrada de nuevas variables para predicción
        st.write("### Enter new data for prediction")
        new_data_method = st.radio("Choose how to input new data:", ["Manual Input", "Upload CSV"])

        # Entrada de datos manual
        if new_data_method == "Manual Input":
            input_data = {}
            for feature in selected_features:
                input_value = st.number_input(f"Enter value for {feature}", format="%.4f")
                input_data[feature] = input_value
            new_data = pd.DataFrame([input_data])  # Crear un DataFrame de una fila con los datos ingresados manualmente

        # Subir datos desde un archivo CSV
        elif new_data_method == "Upload CSV":
            new_data_file = st.file_uploader("Upload a CSV file with new data for prediction", type=["csv"])
            if new_data_file is not None:
                try:
                    new_data = pd.read_csv(new_data_file)

                    # Validar que las columnas coincidan con las características del modelo
                    if set(selected_features).issubset(new_data.columns):
                        new_data = new_data[selected_features]
                        st.write("### New Data Preview")
                        st.dataframe(new_data)
                    else:
                        st.error(f"The uploaded CSV does not contain all required features: {selected_features}")
                        new_data = None
                except Exception as e:
                    st.error(f"Error loading new data file: {e}")
                    new_data = None

        # Realizar predicciones si el modelo está guardado y hay datos disponibles
        if st.button("Predict") and selected_model is not None:
            if new_data is not None:
                try:
                    # Para modelos de clustering, mostramos etiquetas en lugar de métricas predictivas
                    if hasattr(selected_model, "predict"):
                        predictions = selected_model.predict(new_data)
                        st.write("### Predictions")
                        st.write(predictions)
                    else:
                        st.warning(f"The selected model '{model_key}' does not support predictions.")
                except Exception as e:
                    st.error(f"Error making predictions: {e}")
            else:
                st.warning("Please input new data for prediction.")
    else:
        st.warning("Please complete the Machine Learning Pipeline step to train models.")