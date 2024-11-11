import streamlit as st
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

def run_clustering_models(df):
    feature_columns = st.multiselect("Select features for clustering:", options=df.columns)

    if feature_columns:
        X = df[feature_columns]

        # Selección del modelo de clustering
        clustering_model = st.selectbox("Choose your clustering model:", ["KMeans", "DBSCAN"])

        if clustering_model == "KMeans":
            n_clusters = st.slider("Select number of clusters (k):", min_value=2, max_value=10, value=3)
            model = KMeans(n_clusters=n_clusters, random_state=42)
            model.fit(X)
            df["Cluster"] = model.labels_

            # Evaluación con Silhouette Score
            silhouette = silhouette_score(X, model.labels_)
            st.write("### KMeans Clustering Results")
            st.write(f"Silhouette Score: {silhouette:.2f}")
            st.dataframe(df[["Cluster"] + feature_columns])

        elif clustering_model == "DBSCAN":
            eps = st.slider("Select eps parameter:", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
            min_samples = st.slider("Select min_samples parameter:", min_value=1, max_value=20, value=5)
            model = DBSCAN(eps=eps, min_samples=min_samples)
            model.fit(X)
            df["Cluster"] = model.labels_

            # Evaluación con Silhouette Score
            silhouette = silhouette_score(X, model.labels_)
            st.write("### DBSCAN Clustering Results")
            st.write(f"Silhouette Score: {silhouette:.2f}")
            st.dataframe(df[["Cluster"] + feature_columns])