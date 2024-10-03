import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os
from _0config import config, CLUSTERS_PATH, MODELS_DIRECTORY
from _2utility import debug_print

def create_clusters(preprocessed_data, clustering_config):
    st.write("Creating clusters...")
    debug_print("Entering create_clusters function.")
    
    clustered_data = {}
    cluster_models = {}
    
    for column, config in clustering_config.items():
        method = config['method']
        params = config['params']
        
        if method == 'None':
            clustered_data[column] = {'label_0': preprocessed_data.index}
        elif method == 'DBSCAN':
            clustered_data[column], cluster_models[column] = perform_dbscan_clustering(preprocessed_data[column], params)
        elif method == 'KMeans':
            clustered_data[column], cluster_models[column] = perform_kmeans_clustering(preprocessed_data[column], params)
    
    # Save cluster models
    save_clustering_models(cluster_models, MODELS_DIRECTORY)
    
    return clustered_data

def perform_dbscan_clustering(data, parameters):
    st.write(f"Performing DBSCAN clustering")
    
    # Reshape data for DBSCAN
    X = data.values.reshape(-1, 1)
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    dbscan = DBSCAN(**parameters)
    cluster_labels = dbscan.fit_predict(X_scaled)
    
    unique_labels = np.unique(cluster_labels)
    st.write(f"DBSCAN clustering done. Number of clusters: {len(unique_labels)}")
    
    clusters = {f'label_{label}': data.index[cluster_labels == label].tolist() for label in unique_labels}
    return clusters, dbscan

def perform_kmeans_clustering(data, parameters):
    st.write(f"Performing KMeans clustering")
    
    # Reshape data for KMeans
    X = data.values.reshape(-1, 1)
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(**parameters)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    unique_labels = np.unique(cluster_labels)
    st.write(f"KMeans clustering done. Number of clusters: {len(unique_labels)}")
    
    clusters = {f'label_{label}': data.index[cluster_labels == label].tolist() for label in unique_labels}
    return clusters, kmeans

def save_clustering_models(cluster_models, save_path):
    """Save clustering models to the specified directory."""
    for column, model in cluster_models.items():
        joblib.dump(model, os.path.join(save_path, f'cluster_model_{column}.joblib'))
    st.success("Clustering models saved successfully.")

def load_clustering_models(models_directory):
    """Load clustering models from the specified directory."""
    cluster_models = {}
    for filename in os.listdir(models_directory):
        if filename.startswith('cluster_model_') and filename.endswith('.joblib'):
            model_path = os.path.join(models_directory, filename)
            column = filename.replace('cluster_model_', '').replace('.joblib', '')
            cluster_models[column] = joblib.load(model_path)
    return cluster_models

def predict_cluster(data_point, cluster_models, clustering_config):
    clusters = {}
    for column, config in clustering_config.items():
        method = config['method']
        if method == 'None':
            clusters[column] = 'label_0'
        elif column in cluster_models:
            model = cluster_models[column]
            if isinstance(model, DBSCAN):
                cluster = model.fit_predict(data_point[column].values.reshape(1, -1))
                clusters[column] = f'label_{cluster[0]}'
            elif isinstance(model, KMeans):
                cluster = model.predict(data_point[column].values.reshape(1, -1))
                clusters[column] = f'label_{cluster[0]}'
        else:
            clusters[column] = 'unknown'
    return clusters

def display_cluster_info(clustered_data):
    st.subheader("Cluster Information")
    for column, cluster_labels in clustered_data.items():
        st.write(f"Column: {column}")
        for label, indices in cluster_labels.items():
            st.write(f"  Label {label}: {len(indices)} data points")
        st.write("---")

def plot_cluster_distributions(preprocessed_data, clustered_data):
    st.subheader("Cluster Distributions")
    for column, cluster_labels in clustered_data.items():
        st.write(f"Column: {column}")
        cluster_data = preprocessed_data.loc[np.concatenate(list(cluster_labels.values()))]
        fig = px.histogram(cluster_data, x=column, color=cluster_data.index.map(lambda x: next(label for label, indices in cluster_labels.items() if x in indices)),
                           title=f"Distribution of {column} in clusters")
        st.plotly_chart(fig)
        st.write("---")
