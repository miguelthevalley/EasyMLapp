import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def run_regression_models(df):
    target_column = st.selectbox("Select the target (y) variable for regression:", options=df.columns)
    feature_columns = st.multiselect("Select feature (X) variables:", options=[col for col in df.columns if col != target_column])

    if target_column and feature_columns:
        X = df[feature_columns]
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Selección del modelo de regresión
        model_choice = st.selectbox("Choose your regression model:", ["Linear Regression", "Random Forest Regressor"])
        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Random Forest Regressor":
            model = RandomForestRegressor()

        # Entrenar el modelo
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Evaluación del modelo
        mse = mean_squared_error(y_test, y_pred)
        st.write("### Regression Model Evaluation")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")