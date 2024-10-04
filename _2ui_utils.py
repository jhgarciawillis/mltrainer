import pandas as pd
import streamlit as st
from _0config import (STREAMLIT_THEME, AVAILABLE_CLUSTERING_METHODS, DBSCAN_PARAMETERS, KMEANS_PARAMETERS, 
                      MODEL_CLASSES, config, STREAMLIT_APP_NAME, CHART_WIDTH)
from _2misc_utils import truncate_sheet_name, validate_file_upload

def set_streamlit_theme():
    """Set the Streamlit theme based on the configuration."""
    st.set_page_config(
        page_title=STREAMLIT_APP_NAME,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Apply custom theme
    st.markdown(f"""
        <style>
            .reportview-container .main .block-container{{
                max-width: {CHART_WIDTH}px;
                padding-top: 5rem;
                padding-right: 1rem;
                padding-left: 1rem;
                padding-bottom: 5rem;
            }}
            .reportview-container .main {{
                color: {STREAMLIT_THEME['textColor']};
                background-color: {STREAMLIT_THEME['backgroundColor']};
            }}
            .sidebar .sidebar-content {{
                background-color: {STREAMLIT_THEME['secondaryBackgroundColor']};
            }}
            .Widget>label {{
                color: {STREAMLIT_THEME['textColor']};
            }}
            .stButton>button {{
                color: {STREAMLIT_THEME['backgroundColor']};
                background-color: {STREAMLIT_THEME['primaryColor']};
                border-radius: 0.3rem;
            }}
            .stSelectbox, .stMultiSelect {{
                width: 50%;
            }}
        </style>
        """, unsafe_allow_html=True)

def display_metrics(metrics):
    """Display metrics in a formatted way."""
    st.subheader("Model Performance Metrics")
    col1, col2 = st.columns(2)
    for i, (metric, value) in enumerate(metrics.items()):
        if i % 2 == 0:
            col1.metric(metric, f"{value:.4f}")
        else:
            col2.metric(metric, f"{value:.4f}")

def display_column_selection(columns):
    st.subheader("Column Selection")
    
    column_types = {}
    outlier_removal = {}
    
    col1, col2 = st.columns([2, 1])
    
    for i, col in enumerate(columns):
        with col1:
            column_types[col] = st.selectbox(f"Type for {col}", 
                ['numerical', 'categorical', 'unused', 'target'],
                key=f'col_type_{i}',
                index=['numerical', 'categorical', 'unused', 'target'].index(
                    'numerical' if col in config.numerical_columns 
                    else 'categorical' if col in config.categorical_columns
                    else 'target' if col == config.target_column
                    else 'unused'
                ))
        
        with col2:
            if column_types[col] == 'numerical':
                outlier_removal[col] = st.checkbox(f"Remove outliers for {col}", key=f'outlier_{i}')
    
    return column_types, outlier_removal

def display_clustering_options():
    """Display options for clustering configuration."""
    st.subheader("Clustering Configuration")
    
    # 1D Clustering
    st.write("1D Clustering Options:")
    for col in config.numerical_columns:
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            method = st.selectbox(f"Method for {col}", AVAILABLE_CLUSTERING_METHODS, key=f"cluster_method_{col}")
        
        if method != 'None':
            if method == 'DBSCAN':
                with col2:
                    eps = st.slider(f"DBSCAN eps for {col}", 0.1, 1.0, DBSCAN_PARAMETERS['eps'], key=f"dbscan_eps_{col}")
                with col3:
                    min_samples = st.slider(f"DBSCAN min_samples for {col}", 2, 10, DBSCAN_PARAMETERS['min_samples'], key=f"dbscan_min_samples_{col}")
                params = {'eps': eps, 'min_samples': min_samples}
            elif method == 'KMeans':
                with col2:
                    n_clusters = st.slider(f"KMeans n_clusters for {col}", 2, 10, KMEANS_PARAMETERS['n_clusters'], key=f"kmeans_n_clusters_{col}")
                params = {'n_clusters': n_clusters}
            config.clustering_config[col] = {'method': method, 'params': params}
        else:
            config.clustering_config[col] = {'method': 'None', 'params': {}}
    
    # 2D Clustering
    st.write("2D Clustering Options:")
    col_pairs = select_2d_clustering_columns()
    for i, pair in enumerate(col_pairs):
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            method = st.selectbox(f"Method for {pair}", AVAILABLE_CLUSTERING_METHODS, key=f"cluster_method_{i}")
        
        if method != 'None':
            if method == 'DBSCAN':
                with col2:
                    eps = st.slider(f"DBSCAN eps for {pair}", 0.1, 1.0, DBSCAN_PARAMETERS['eps'], key=f"dbscan_eps_{i}")
                with col3:
                    min_samples = st.slider(f"DBSCAN min_samples for {pair}", 2, 10, DBSCAN_PARAMETERS['min_samples'], key=f"dbscan_min_samples_{i}")
                params = {'eps': eps, 'min_samples': min_samples}
            elif method == 'KMeans':
                with col2:
                    n_clusters = st.slider(f"KMeans n_clusters for {pair}", 2, 10, KMEANS_PARAMETERS['n_clusters'], key=f"kmeans_n_clusters_{i}")
                params = {'n_clusters': n_clusters}
            config.set_2d_clustering([pair], method, params)
        else:
            config.set_2d_clustering([pair], 'None', {})

def select_2d_clustering_columns():
    """Allow users to select pairs of columns for 2D clustering."""
    st.write("Select column pairs for 2D clustering:")
    col_pairs = []
    valid_columns = [col for col in config.numerical_columns if col not in config.unused_columns]
    num_pairs = st.number_input("Number of column pairs for 2D clustering", min_value=0, max_value=len(valid_columns)//2, value=0, key='num_2d_pairs')
    
    for i in range(num_pairs):
        col1, col2 = st.columns(2)
        with col1:
            first_col = st.selectbox(f"First column for pair {i+1}", valid_columns, key=f"2d_cluster_col1_{i}")
        with col2:
            remaining_columns = [col for col in valid_columns if col != first_col]
            second_col = st.selectbox(f"Second column for pair {i+1}", remaining_columns, key=f"2d_cluster_col2_{i}")
        
        if first_col != second_col:
            col_pairs.append((first_col, second_col))
        else:
            st.warning(f"Pair {i+1}: Please select different columns.")
    
    config.set_2d_clustering_columns([col for pair in col_pairs for col in pair])
    return col_pairs

def get_prediction_inputs():
    """Get user inputs for Prediction mode."""
    st.header("Prediction Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_saved_models = st.radio("Use saved models?", ["Yes", "No"], key='use_saved_models_radio')
        
        if use_saved_models == "No":
            uploaded_models = st.file_uploader("Upload trained models", type="joblib", accept_multiple_files=True, key='upload_models')
            uploaded_preprocessor = st.file_uploader("Upload preprocessor", type="joblib", key='upload_preprocessor')
            config.update(uploaded_models=uploaded_models, uploaded_preprocessor=uploaded_preprocessor)
    
    with col2:
        new_data_file = st.file_uploader("Choose a CSV file with new data for prediction", type="csv", key='new_data_upload')
        if new_data_file is not None:
            config.update(new_data_file=new_data_file)
    
    return config
