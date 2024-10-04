import pandas as pd
import streamlit as st
from _0config import (STREAMLIT_THEME, AVAILABLE_CLUSTERING_METHODS, DBSCAN_PARAMETERS, KMEANS_PARAMETERS, 
                      MODEL_CLASSES, config, STREAMLIT_APP_NAME, CHART_WIDTH, TOOLTIPS, INFO_TEXTS)
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
        </style>
        """, unsafe_allow_html=True)

def create_tooltip(text):
    """Create a tooltip with the given text."""
    st.markdown(f"""
    <style>
    .tooltip {{
      position: relative;
      display: inline-block;
      border-bottom: 1px dotted black;
    }}
    .tooltip .tooltiptext {{
      visibility: hidden;
      width: 120px;
      background-color: black;
      color: #fff;
      text-align: center;
      border-radius: 6px;
      padding: 5px 0;
      position: absolute;
      z-index: 1;
      bottom: 125%;
      left: 50%;
      margin-left: -60px;
      opacity: 0;
      transition: opacity 0.3s;
    }}
    .tooltip:hover .tooltiptext {{
      visibility: visible;
      opacity: 1;
    }}
    </style>
    <div class="tooltip">ℹ️
      <span class="tooltiptext">{text}</span>
    </div>
    """, unsafe_allow_html=True)

def create_info_button(key):
    """Create an info button that displays detailed information when clicked."""
    if st.button(f"ℹ️ More Info: {key.replace('_', ' ').title()}"):
        st.info(INFO_TEXTS[key])

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
        create_tooltip(TOOLTIPS["file_upload"])
        if uploaded_file is not None:
            config.update(file_path=uploaded_file)
            
            if uploaded_file.name.endswith('.xlsx'):
                try:
                    xls = pd.ExcelFile(uploaded_file)
                    sheet_name = st.selectbox("Select sheet", xls.sheet_names)
                    create_tooltip(TOOLTIPS["sheet_selection"])
                    config.update(sheet_name=sheet_name)
                except Exception as e:
                    st.error(f"Error reading Excel file: {str(e)}")
                    return None
        
        train_size = st.slider("Select percentage of data for training", 0.1, 0.9, 0.8)
        create_tooltip(TOOLTIPS["train_test_split"])
        config.update(train_size=train_size)
    
    with col2:
        use_clustering = st.checkbox("Use clustering", value=False)
        create_tooltip(TOOLTIPS["use_clustering"])
        config.update(use_clustering=use_clustering)
        
        if use_clustering:
            display_clustering_options()
        
        models_to_use = st.multiselect("Select models to use", list(MODEL_CLASSES.keys()))
        create_tooltip(TOOLTIPS["models_to_use"])
        config.update(models_to_use=models_to_use)
        
        tuning_method = st.selectbox("Select tuning method", ["None", "GridSearchCV", "RandomizedSearchCV"])
        create_tooltip(TOOLTIPS["tuning_method"])
        config.update(tuning_method=tuning_method)
    
    display_outlier_removal_options()
    
    return config

def display_clustering_options():
    """Display options for clustering configuration."""
    st.subheader("Clustering Configuration")
    create_info_button("clustering_configuration")
    
    # 1D Clustering
    st.write("1D Clustering Options:")
    for col in config.numerical_columns:
        method = st.selectbox(f"Select clustering method for {col}", AVAILABLE_CLUSTERING_METHODS, key=f"cluster_method_{col}")
        create_tooltip(TOOLTIPS["clustering_method"])
        if method != 'None':
            if method == 'DBSCAN':
                eps = st.slider(f"DBSCAN eps for {col}", 0.1, 1.0, DBSCAN_PARAMETERS['eps'], key=f"dbscan_eps_{col}")
                create_tooltip(TOOLTIPS["dbscan_eps"])
                min_samples = st.slider(f"DBSCAN min_samples for {col}", 2, 10, DBSCAN_PARAMETERS['min_samples'], key=f"dbscan_min_samples_{col}")
                create_tooltip(TOOLTIPS["dbscan_min_samples"])
                params = {'eps': eps, 'min_samples': min_samples}
            elif method == 'KMeans':
                n_clusters = st.slider(f"KMeans n_clusters for {col}", 2, 10, KMEANS_PARAMETERS['n_clusters'], key=f"kmeans_n_clusters_{col}")
                create_tooltip(TOOLTIPS["kmeans_n_clusters"])
                params = {'n_clusters': n_clusters}
            config.clustering_config[col] = {'method': method, 'params': params}
        else:
            config.clustering_config[col] = {'method': 'None', 'params': {}}
    
    # 2D Clustering
    st.write("2D Clustering Options:")
    col_pairs = select_2d_clustering_columns()
    for pair in col_pairs:
        method = st.selectbox(f"Select clustering method for {pair}", AVAILABLE_CLUSTERING_METHODS, key=f"cluster_method_{pair}")
        create_tooltip(TOOLTIPS["clustering_method"])
        if method != 'None':
            if method == 'DBSCAN':
                eps = st.slider(f"DBSCAN eps for {pair}", 0.1, 1.0, DBSCAN_PARAMETERS['eps'], key=f"dbscan_eps_{pair}")
                create_tooltip(TOOLTIPS["dbscan_eps"])
                min_samples = st.slider(f"DBSCAN min_samples for {pair}", 2, 10, DBSCAN_PARAMETERS['min_samples'], key=f"dbscan_min_samples_{pair}")
                create_tooltip(TOOLTIPS["dbscan_min_samples"])
                params = {'eps': eps, 'min_samples': min_samples}
            elif method == 'KMeans':
                n_clusters = st.slider(f"KMeans n_clusters for {pair}", 2, 10, KMEANS_PARAMETERS['n_clusters'], key=f"kmeans_n_clusters_{pair}")
                create_tooltip(TOOLTIPS["kmeans_n_clusters"])
                params = {'n_clusters': n_clusters}
            config.set_2d_clustering([pair], method, params)
        else:
            config.set_2d_clustering([pair], 'None', {})

def select_2d_clustering_columns():
    """Allow users to select pairs of columns for 2D clustering."""
    st.write("Select column pairs for 2D clustering:")
    create_tooltip(TOOLTIPS["2d_clustering"])
    col_pairs = []
    valid_columns = [col for col in config.numerical_columns if col not in config.unused_columns]
    num_pairs = st.number_input("Number of column pairs for 2D clustering", min_value=0, max_value=len(valid_columns)//2, value=0)
    
    for i in range(num_pairs):
        col1 = st.selectbox(f"Select first column for pair {i+1}", valid_columns, key=f"2d_cluster_col1_{i}")
        remaining_columns = [col for col in valid_columns if col != col1]
        col2 = st.selectbox(f"Select second column for pair {i+1}", remaining_columns, key=f"2d_cluster_col2_{i}")
        if col1 != col2:
            col_pairs.append((col1, col2))
        else:
            st.warning(f"Pair {i+1}: Please select different columns.")
    
    config.set_2d_clustering_columns([col for pair in col_pairs for col in pair])
    return col_pairs

def display_outlier_removal_options():
    """Display options for outlier removal."""
    st.subheader("Outlier Removal Options")
    create_info_button("outlier_removal")
    outlier_columns = []
    for col in config.numerical_columns:
        if st.checkbox(f"Remove outliers for {col}", key=f"outlier_{col}"):
            outlier_columns.append(col)
    config.update_outlier_removal_columns(outlier_columns)

def get_prediction_inputs():
    """Get user inputs for Prediction mode."""
    st.header("Prediction Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_saved_models = st.radio("Use saved models?", ["Yes", "No"])
        create_tooltip(TOOLTIPS["use_saved_models"])
        
        if use_saved_models == "No":
            uploaded_models = st.file_uploader("Upload trained models", type="joblib", accept_multiple_files=True)
            create_tooltip(TOOLTIPS["upload_models"])
            uploaded_preprocessor = st.file_uploader("Upload preprocessor", type="joblib")
            create_tooltip(TOOLTIPS["upload_preprocessor"])
            config.update(uploaded_models=uploaded_models, uploaded_preprocessor=uploaded_preprocessor)
    
    with col2:
        new_data_file = st.file_uploader("Choose a CSV file with new data for prediction", type="csv")
        create_tooltip(TOOLTIPS["new_data_file"])
        if new_data_file is not None:
            config.update(new_data_file=new_data_file)
    
    return config
