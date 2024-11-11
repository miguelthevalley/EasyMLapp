# Dentro de supervised_classification.py
import streamlit as st
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

def train_and_evaluate(model, X, y, hyperparameters={}, scaler_type="RobustScaler", test_size=0.2, random_state=42, stratify_option=True):
    # Obtener el nombre del modelo
    model_name = model.__class__.__name__

    # Separar en conjunto de entrenamiento y prueba
    stratify_param = y if stratify_option else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_param)

    # Asignar hiperparámetros manuales al modelo
    for param, value in hyperparameters.items():
        setattr(model, param, value)

    # Seleccionar el tipo de escalador según la elección del usuario
    scaler = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "RobustScaler": RobustScaler()}.get(scaler_type, RobustScaler())
    pipeline = Pipeline(steps=[('scaler', scaler), ('classifier', model)])

    start_time = time.time()
    pipeline.fit(X_train, y_train)
    execution_time = time.time() - start_time

    # Realizar predicciones en el conjunto de entrenamiento y prueba
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Calcular métricas de evaluación
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')

    # Mostrar resultados
    st.write(f"### Model: {model_name}")
    st.write(f"Execution Time: {execution_time:.2f} seconds")
    st.write(f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
    st.write(f"Train Precision: {train_precision:.4f}, Test Precision: {test_precision:.4f}")
    st.write(f"Train Recall: {train_recall:.4f}, Test Recall: {test_recall:.4f}")

    # Crear un DataFrame temporal con los resultados
    result = pd.DataFrame({
        'Model': [model_name],
        'Train Accuracy': [train_accuracy],
        'Test Accuracy': [test_accuracy],
        'Train Precision': [train_precision],
        'Test Precision': [test_precision],
        'Train Recall': [train_recall],
        'Test Recall': [test_recall],
        'Execution Time': [execution_time]
    })

    return pipeline, result

def run_classification_models(X, y):
    # Aquí estamos recibiendo X y y
    st.write("## Configure Model Hyperparameters and Scaler")

    # Selección del tipo de escalador
    scaler_type = st.selectbox("Select Scaler:", ["StandardScaler", "MinMaxScaler", "RobustScaler"])

    # Selección de hiperparámetros para XGBClassifier
    st.write("### Hyperparameter Settings for XGBClassifier")
    learning_rate = st.number_input("Learning Rate", min_value=0.01, max_value=1.0, value=0.1)
    n_estimators = st.slider("Number of Estimators", min_value=50, max_value=300, value=100)
    max_depth = st.slider("Max Depth", min_value=1, max_value=10, value=3)

    # Configurar el modelo y los hiperparámetros
    model = XGBClassifier()
    hyperparameters = {
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        'max_depth': max_depth
    }

    # Ejecutar el entrenamiento y la evaluación cuando se haga clic en el botón
    if st.button("Train and Evaluate Model"):
        pipeline, results = train_and_evaluate(model, X, y, hyperparameters=hyperparameters, scaler_type=scaler_type)
        st.write("### Evaluation Results")
        st.dataframe(results)