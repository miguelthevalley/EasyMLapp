import streamlit as st
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

def calculate_elbow_method(X, scaler):
    """Calcula y muestra el método del codo para seleccionar el número óptimo de clusters."""
    if scaler:
        X = scaler.fit_transform(X)

    sse = []
    K_range = range(1, 100)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", max_iter=300, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(4, 4))
    plt.plot(K_range, sse, marker='o')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Sum of Squared Distances")
    plt.title("Elbow Method for Optimal K")
    st.pyplot(plt)

def train_and_evaluate_unsupervised(model, X, scaler_type="StandardScaler"):
    """Entrena un modelo no supervisado y devuelve etiquetas, datos transformados y el escalador."""
    scaler = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "RobustScaler": RobustScaler()}.get(scaler_type, None)
    X_transformed = scaler.fit_transform(X) if scaler else X

    model.fit(X_transformed)
    labels = model.fit_predict(X_transformed) if not hasattr(model, 'predict') else model.predict(X_transformed)

    return labels, X_transformed, scaler

def run_unsupervised_models(X):
    st.write("## Step 1: Configure Unsupervised Model and Hyperparameters")

    if "trained_models" not in st.session_state:
        st.session_state.trained_models = {}

    model_type = st.selectbox("Select Unsupervised Model:", ["KMeans", "DBSCAN"])
    scaler_type = st.selectbox("Select Scaler:", ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"])
    visualization_type = st.selectbox("Select Visualization Type:", ["2D", "3D"])

    if model_type == "KMeans":
        model = KMeans()
        st.write("### K-Means Hyperparameters")
        if st.checkbox("Calculate Elbow Method for Optimal K"):
            calculate_elbow_method(X, scaler={"StandardScaler": StandardScaler(), 
                                              "MinMaxScaler": MinMaxScaler(), 
                                              "RobustScaler": RobustScaler()}.get(scaler_type, None))
        model.set_params(
            n_clusters=st.slider("Number of Clusters", min_value=2, max_value=100, value=3),
            init=st.selectbox("Initialization Method", ["k-means++", "random"]),
            max_iter=st.slider("Max Iterations", min_value=100, max_value=1000, value=300)
        )
    elif model_type == "DBSCAN":
        model = DBSCAN()
        st.write("### DBSCAN Hyperparameters")
        model.set_params(
            eps=st.slider("Epsilon", min_value=0.1, max_value=10.0, value=0.5),
            min_samples=st.slider("Minimum Samples", min_value=1, max_value=20, value=5)
        )

    if st.button("Train and Visualize Model"):
        labels, X_transformed, scaler = train_and_evaluate_unsupervised(model, X, scaler_type)
        model_key = f"{model_type}_{len([key for key in st.session_state.trained_models.keys() if model_type in key])}"
        st.session_state.trained_models[model_key] = {
            "model": model,
            "scaler": scaler,
            "features": list(X.columns) if isinstance(X, pd.DataFrame) else None
        }

        n_components = 3 if visualization_type == "3D" else 2
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X_transformed)

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

        st.write("### Cluster Labels")
        st.write(labels)