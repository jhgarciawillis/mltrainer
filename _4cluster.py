import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os
from _0config import config, CLUSTERS_PATH, MODELS_DIRECTORY
from _2misc_utils import debug_print

def create_clusters(preprocessed_data, clustering_config, clustering_2d_config):
    st.write("Creating clusters...")
    debug_print("Entering create_clusters function.")
    
    clustered_data = {}
    cluster_models = {}
    
    # 1D Clustering
    for column, config in clustering_config.items():
        method = config['method']
        params = config['params']
        
        if method == 'None':
            clustered_data[column] = {'label_0': preprocessed_data.index}
        elif method == 'DBSCAN':
            clustered_data[column], cluster_models[column] = perform_dbscan_clustering(preprocessed_data[column], params)
        elif method == 'KMeans':
            clustered_data[column], cluster_models[column] = perform_kmeans_clustering(preprocessed_data[column], params)
    
    # 2D Clustering
    for column_pair, config in clustering_2d_config.items():
        if set(column_pair).issubset(set(preprocessed_data.columns)):
            method = config['method']
            params = config['params']
            
            if method == 'None':
                clustered_data[column_pair] = {'label_0': preprocessed_data.index}
            elif method == 'DBSCAN':
                clustered_data[column_pair], cluster_models[column_pair] = perform_2d_clustering(preprocessed_data[list(column_pair)], method, params)
            elif method == 'KMeans':
                clustered_data[column_pair], cluster_models[column_pair] = perform_2d_clustering(preprocessed_data[list(column_pair)], method, params)
        else:
            st.warning(f"Skipping 2D clustering for {column_pair} as one or more columns are not present in the data.")
    
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

def perform_2d_clustering(data, method, parameters):
    st.write(f"Performing 2D {method} clustering")
    
    # Ensure only specified columns are used
    data_subset = data[list(data.columns)]
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_subset)
    
    if method == 'DBSCAN':
        model = DBSCAN(**parameters)
    elif method == 'KMeans':
        model = KMeans(**parameters)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    
    cluster_labels = model.fit_predict(X_scaled)
    
    unique_labels = np.unique(cluster_labels)
    st.write(f"2D {method} clustering done. Number of clusters: {len(unique_labels)}")
    
    clusters = {f'label_{label}': data_subset.index[cluster_labels == label].tolist() for label in unique_labels}
    return clusters, model

def generate_2d_cluster_filename(column_pair, method):
    col1, col2 = column_pair
    return f"2D_{col1[:3]}_{col2[:3]}_{method}.joblib"

def save_clustering_models(cluster_models, save_path):
    """Save clustering models to the specified directory."""
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
    for column, model in cluster_models.items():
        if isinstance(column, tuple):  # 2D clustering
            filename = generate_2d_cluster_filename(column, model.__class__.__name__)
        else:  # 1D clustering
            filename = f"cluster_model_{column}.joblib"
        model_path = os.path.join(save_path, filename)
        joblib.dump(model, model_path)
    st.success("Clustering models saved successfully.")

def load_clustering_models(models_directory):
    """Load clustering models from the specified directory."""
    cluster_models = {}
    for filename in os.listdir(models_directory):
        if filename.startswith('cluster_model_') or filename.startswith('2D_'):
            model_path = os.path.join(models_directory, filename)
            if filename.startswith('2D_'):
                # Extract column names from 2D cluster filename
                parts = filename.split('_')
                col1, col2 = parts[1], parts[2]
                column = (col1, col2)
            else:
                column = filename.replace('cluster_model_', '').replace('.joblib', '')
            cluster_models[column] = joblib.load(model_path)
    return cluster_models

def predict_cluster(data_point, cluster_models, clustering_config, clustering_2d_config):
    clusters = {}
    
    # 1D Clustering prediction
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
    
    # 2D Clustering prediction
    for column_pair, config in clustering_2d_config.items():
        method = config['method']
        if method == 'None':
            clusters[column_pair] = 'label_0'
        elif column_pair in cluster_models:
            model = cluster_models[column_pair]
            data = data_point[list(column_pair)].values.reshape(1, -1)
            if isinstance(model, DBSCAN):
                cluster = model.fit_predict(data)
                clusters[column_pair] = f'label_{cluster[0]}'
            elif isinstance(model, KMeans):
                cluster = model.predict(data)
                clusters[column_pair] = f'label_{cluster[0]}'
        else:
            clusters[column_pair] = 'unknown'
    
    return clusters
