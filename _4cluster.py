import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from _0config import config, CLUSTERS_PATH, MODELS_DIRECTORY
from _2utility import debug_print, plot_feature_importance

def create_clusters(preprocessed_data, clustering_config):
    st.write("Creating clusters...")
    debug_print("Entering create_clusters function.")
    try:
        clusters_file_path = CLUSTERS_PATH

        # Remove the existing Clusters file if found
        if os.path.exists(clusters_file_path):
            os.remove(clusters_file_path)
            debug_print("Existing Clusters file found and deleted.")

        clustered_data = {}
        models_to_save = {}

        debug_print(f"Selected clustering configurations: {clustering_config}")

        progress_bar = st.progress(0)
        
        cluster_method = clustering_config['method']
        cluster_parameters = clustering_config['params']
        cluster_columns = clustering_config['columns']

        if cluster_method == 'DBSCAN':
            cluster_labels, dbscan_model = perform_dbscan_clustering(preprocessed_data, cluster_columns, **cluster_parameters)
            models_to_save['dbscan'] = dbscan_model
        elif cluster_method == 'KMeans':
            cluster_labels, kmeans_model = perform_kmeans_clustering(preprocessed_data, cluster_columns, **cluster_parameters)
            models_to_save['kmeans'] = kmeans_model
        else:
            st.warning(f"Unsupported clustering method: {cluster_method}")
            return None

        clustered_data['cluster'] = {label: preprocessed_data.index[cluster_labels == label].tolist() for label in np.unique(cluster_labels)}
        progress_bar.progress(1.0)

        # Save the clustering models
        save_clustering_models(models_to_save)

        st.success("Clustering completed successfully.")
        st.write(f"Number of clusters created: {len(np.unique(cluster_labels))}")
        for label, indices in clustered_data['cluster'].items():
            st.write(f"Cluster {label} has {len(indices)} data points.")

        return clustered_data

    except Exception as e:
        st.error(f"Error in create_clusters: {str(e)}")
        raise

def perform_dbscan_clustering(preprocessed_data, columns, **parameters):
    st.write(f"Performing DBSCAN clustering for columns: {columns}")
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(preprocessed_data[columns])
    
    dbscan_model = DBSCAN(**parameters)
    cluster_labels = dbscan_model.fit_predict(scaled_data)
    st.write(f"DBSCAN clustering done. Number of clusters: {len(np.unique(cluster_labels))}")
    st.write("DBSCAN cluster label counts:")
    st.write(pd.Series(cluster_labels).value_counts())
    return cluster_labels, dbscan_model

def perform_kmeans_clustering(preprocessed_data, columns, **parameters):
    st.write(f"Performing KMeans clustering for columns: {columns}")
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(preprocessed_data[columns])
    
    kmeans_model = KMeans(random_state=42, **parameters)
    cluster_labels = kmeans_model.fit_predict(scaled_data)
    st.write(f"KMeans clustering done. Number of clusters: {len(np.unique(cluster_labels))}")
    st.write("KMeans cluster label counts:")
    st.write(pd.Series(cluster_labels).value_counts())
    return cluster_labels, kmeans_model

def save_clustering_models(models_to_save):
    if models_to_save:
        for model_name, model in models_to_save.items():
            joblib.dump(model, os.path.join(MODELS_DIRECTORY, f'{model_name}_model.joblib'))
        st.success("Clustering models saved.")
    else:
        st.warning("No clustering models to save.")

@st.cache(allow_output_mutation=True)
def load_clusters(clusters_file_path):
    st.write("Loading clusters from file.")
    try:
        clustered_data = joblib.load(clusters_file_path)
        st.success(f"Clusters loaded successfully. Number of clusters: {len(clustered_data)}")
        return clustered_data
    except Exception as e:
        st.error(f"Error loading clusters: {str(e)}")
        return None

def determine_cluster(data_point, cluster_models):
    for model_name, model in cluster_models.items():
        if model_name == 'dbscan':
            cluster = model.fit_predict(data_point.values.reshape(1, -1))
            if cluster[0] != -1:  # Not an outlier
                return 'cluster', cluster[0]
        elif model_name == 'kmeans':
            cluster = model.predict(data_point.values.reshape(1, -1))
            return 'cluster', cluster[0]
    return 'default', 0  # If no cluster is determined, use a default cluster

@st.cache(allow_output_mutation=True)
def load_clustering_models(models_directory):
    cluster_models = {}
    for filename in os.listdir(models_directory):
        if filename.endswith('_model.joblib'):
            model_path = os.path.join(models_directory, filename)
            model_name = filename.replace('_model.joblib', '')
            cluster_models[model_name] = joblib.load(model_path)
    return cluster_models

def predict_cluster(data_point, cluster_models):
    for model_name, model in cluster_models.items():
        if model_name == 'dbscan':
            cluster = model.fit_predict(data_point.values.reshape(1, -1))
            if cluster[0] != -1:  # Not an outlier
                return 'cluster', cluster[0]
        elif model_name == 'kmeans':
            cluster = model.predict(data_point.values.reshape(1, -1))
            return 'cluster', cluster[0]
    return 'default', 0  # If no cluster is determined, use a default cluster

def display_cluster_info(clustered_data):
    st.subheader("Cluster Information")
    for cluster_name, cluster_labels in clustered_data.items():
        st.write(f"Cluster: {cluster_name}")
        for label, indices in cluster_labels.items():
            st.write(f"  Label {label}: {len(indices)} data points")
        st.write("---")

def plot_cluster_distributions(preprocessed_data, clustered_data):
    st.subheader("Cluster Distributions")
    for cluster_name, cluster_labels in clustered_data.items():
        st.write(f"Cluster: {cluster_name}")
        cluster_data = preprocessed_data.loc[np.concatenate(list(cluster_labels.values()))]
        for column in preprocessed_data.columns:
            fig = px.histogram(cluster_data, x=column, color=cluster_data.index.map(lambda x: next(label for label, indices in cluster_labels.items() if x in indices)),
                               title=f"Distribution of {column} in {cluster_name}")
            st.plotly_chart(fig)
        st.write("---")