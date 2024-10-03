import streamlit as st
from _0config import (STREAMLIT_THEME, AVAILABLE_CLUSTERING_METHODS, DBSCAN_PARAMETERS, KMEANS_PARAMETERS, 
                      MODEL_CLASSES, config)
from _2misc_utils import truncate_sheet_name

def set_streamlit_theme():
    """Set the Streamlit theme based on the configuration."""
    st.set_page_config(
        page_title=config.get('STREAMLIT_APP_NAME', "ML Algo Trainer"),
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

def get_user_inputs(mode):
    """Get user inputs for both Training and Prediction modes."""
    if mode == "Training":
        return get_training_inputs()
    else:
        return get_prediction_inputs()

def get_training_inputs():
    """Get user inputs for Training mode."""
    st.header("Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
        if uploaded_file is not None:
            config.update(file_path=uploaded_file)
            
            if uploaded_file.name.endswith('.xlsx'):
                xls = pd.ExcelFile(uploaded_file)
                sheet_name = st.selectbox("Select sheet", xls.sheet_names)
                config.update(sheet_name=sheet_name)
        
        train_size = st.slider("Select percentage of data for training", 0.1, 0.9, 0.8)
        config.update(train_size=train_size)
    
    with col2:
        use_clustering = st.checkbox("Use clustering", value=False)
        config.update(use_clustering=use_clustering)
        
        if use_clustering:
            display_clustering_options()
        
        models_to_use = st.multiselect("Select models to use", list(MODEL_CLASSES.keys()))
        config.update(models_to_use=models_to_use)
        
        tuning_method = st.selectbox("Select tuning method", ["None", "GridSearchCV", "RandomizedSearchCV"])
        config.update(tuning_method=tuning_method)
    
    return config

def display_clustering_options():
    """Display options for clustering configuration."""
    st.subheader("Clustering Configuration")
    
    # 1D Clustering
    st.write("1D Clustering Options:")
    for col in config.numerical_columns:
        method = st.selectbox(f"Select clustering method for {col}", AVAILABLE_CLUSTERING_METHODS, key=f"cluster_method_{col}")
        if method != 'None':
            if method == 'DBSCAN':
                eps = st.slider(f"DBSCAN eps for {col}", 0.1, 1.0, DBSCAN_PARAMETERS['eps'], key=f"dbscan_eps_{col}")
                min_samples = st.slider(f"DBSCAN min_samples for {col}", 2, 10, DBSCAN_PARAMETERS['min_samples'], key=f"dbscan_min_samples_{col}")
                params = {'eps': eps, 'min_samples': min_samples}
            elif method == 'KMeans':
                n_clusters = st.slider(f"KMeans n_clusters for {col}", 2, 10, KMEANS_PARAMETERS['n_clusters'], key=f"kmeans_n_clusters_{col}")
                params = {'n_clusters': n_clusters}
            config.clustering_config[col] = {'method': method, 'params': params}
        else:
            config.clustering_config[col] = {'method': 'None', 'params': {}}
    
    # 2D Clustering
    st.write("2D Clustering Options:")
    col_pairs = select_2d_clustering_columns()
    for i, pair in enumerate(col_pairs):
        method = st.selectbox(f"Select clustering method for {pair}", AVAILABLE_CLUSTERING_METHODS, key=f"cluster_method_{i}")
        if method != 'None':
            if method == 'DBSCAN':
                eps = st.slider(f"DBSCAN eps for {pair}", 0.1, 1.0, DBSCAN_PARAMETERS['eps'], key=f"dbscan_eps_{i}")
                min_samples = st.slider(f"DBSCAN min_samples for {pair}", 2, 10, DBSCAN_PARAMETERS['min_samples'], key=f"dbscan_min_samples_{i}")
                params = {'eps': eps, 'min_samples': min_samples}
            elif method == 'KMeans':
                n_clusters = st.slider(f"KMeans n_clusters for {pair}", 2, 10, KMEANS_PARAMETERS['n_clusters'], key=f"kmeans_n_clusters_{i}")
                params = {'n_clusters': n_clusters}
            config.set_2d_clustering([pair], method, params)

def select_2d_clustering_columns():
    """Allow users to select pairs of columns for 2D clustering."""
    st.write("Select column pairs for 2D clustering:")
    col_pairs = []
    num_pairs = st.number_input("Number of column pairs for 2D clustering", min_value=0, max_value=len(config.numerical_columns)//2, value=0)
    
    for i in range(num_pairs):
        col1 = st.selectbox(f"Select first column for pair {i+1}", config.numerical_columns, key=f"2d_cluster_col1_{i}")
        col2 = st.selectbox(f"Select second column for pair {i+1}", config.numerical_columns, key=f"2d_cluster_col2_{i}")
        if col1 != col2:
            col_pairs.append((col1, col2))
        else:
            st.warning(f"Pair {i+1}: Please select different columns.")
    
    return col_pairs

def get_prediction_inputs():
    """Get user inputs for Prediction mode."""
    st.header("Prediction Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_saved_models = st.radio("Use saved models?", ["Yes", "No"])
        
        if use_saved_models == "No":
            uploaded_models = st.file_uploader("Upload trained models", type="joblib", accept_multiple_files=True)
            uploaded_preprocessor = st.file_uploader("Upload preprocessor", type="joblib")
            config.update(uploaded_models=uploaded_models, uploaded_preprocessor=uploaded_preprocessor)
    
    with col2:
        new_data_file = st.file_uploader("Choose a CSV file with new data for prediction", type="csv")
        if new_data_file is not None:
            config.update(new_data_file=new_data_file)
    
    return config