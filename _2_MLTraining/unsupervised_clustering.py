import streamlit as st
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def calculate_elbow_method(X, scaler):
    if scaler:
        X = scaler.fit_transform(X)
    
    # Calcular la suma de distancias al cuadrado para diferentes valores de K
    sse = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", max_iter=300, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
    
    # Graficar el método del codo
    plt.figure(figsize=(8, 6))
    plt.plot(K_range, sse, marker='o')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Sum of Squared Distances")
    plt.title("Elbow Method for Optimal K")
    st.pyplot(plt)

def train_and_evaluate_unsupervised(model, X, scaler_type="StandardScaler"):
    # Seleccionar escalador
    scaler = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "RobustScaler": RobustScaler()}.get(scaler_type, None)

    # Aplicar escalador si está seleccionado
    if scaler:
        X = scaler.fit_transform(X)

    # Entrenar el modelo
    model.fit(X)

    # Predecir etiquetas de clusters si el modelo lo soporta (KMeans)
    if hasattr(model, 'predict'):
        labels = model.predict(X)
    else:
        labels = model.fit_predict(X)

    return labels, X, scaler

def run_unsupervised_models(X):
    st.write("## Step 1: Configure Unsupervised Model and Hyperparameters")

    # Inicializar almacenamiento para modelos si no existe
    if "trained_models" not in st.session_state:
        st.session_state.trained_models = {}

    # Seleccionar modelo no supervisado
    model_type = st.selectbox("Select Unsupervised Model:", ["KMeans", "DBSCAN"])

    # Seleccionar escalador
    scaler_type = st.selectbox("Select Scaler:", ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"])

    # Seleccionar visualización
    visualization_type = st.selectbox("Select Visualization Type:", ["2D", "3D"])

    # Configuración de hiperparámetros en función del modelo seleccionado
    if model_type == "KMeans":
        model = KMeans()
        st.write("### K-Means Hyperparameters")
        calculate_elbow = st.checkbox("Calculate Elbow Method for Optimal K")
        if calculate_elbow:
            st.write("### Elbow Method")
            scaler = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "RobustScaler": RobustScaler()}.get(scaler_type, None)
            calculate_elbow_method(X, scaler)

        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=100, value=3)
        init = st.selectbox("Initialization Method", ["k-means++", "random"])
        max_iter = st.slider("Max Iterations", min_value=100, max_value=1000, value=300)
        hyperparameters = {'n_clusters': n_clusters, 'init': init, 'max_iter': max_iter}
        model.set_params(**hyperparameters)

    elif model_type == "DBSCAN":
        model = DBSCAN()
        st.write("### DBSCAN Hyperparameters")
        eps = st.slider("Epsilon", min_value=0.1, max_value=10.0, value=0.5)
        min_samples = st.slider("Minimum Samples", min_value=1, max_value=20, value=5)
        hyperparameters = {'eps': eps, 'min_samples': min_samples}
        model.set_params(**hyperparameters)

    # Entrenar y evaluar el modelo no supervisado
    if st.button("Train and Visualize Model"):
        labels, X_transformed, scaler = train_and_evaluate_unsupervised(model, X, scaler_type=scaler_type)

        # Guardar modelo entrenado
        st.session_state.trained_models[model_type] = {
            "model": model,
            "scaler": scaler,
            "features": list(X.columns) if isinstance(X, pd.DataFrame) else None
        }

        # Reducción de dimensionalidad para visualización (PCA a 2 o 3 componentes)
        n_components = 3 if visualization_type == "3D" else 2
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X_transformed)

        # Visualización de los clusters
        st.write("### Cluster Visualization")
        if visualization_type == "2D":
            plt.figure(figsize=(8, 6))
            plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap="viridis", s=50, alpha=0.7)
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.title(f"Cluster Visualization for {model_type} (2D)")
            plt.colorbar(label="Cluster Label")
            st.pyplot(plt)

        elif visualization_type == "3D":
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=labels, cmap="viridis", s=50, alpha=0.7)
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_zlabel("Principal Component 3")
            ax.set_title(f"Cluster Visualization for {model_type} (3D)")
            st.pyplot(fig)

        # Mostrar etiquetas de clusters
        st.write("### Cluster Labels")
        st.write(labels)