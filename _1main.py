import streamlit as st
import pandas as pd
import os
import joblib

from _0config import config, STREAMLIT_APP_NAME, STREAMLIT_APP_ICON, TOOLTIPS, INFO_TEXTS
from _2data_utils import load_data, display_data_info, handle_missing_values, auto_detect_column_types, display_column_selection, save_unused_data
from _2ui_utils import display_metrics, get_user_inputs, get_training_inputs, display_clustering_options, select_2d_clustering_columns, get_prediction_inputs, create_tooltip, create_info_button
from _2misc_utils import debug_print, validate_file_upload

from _3preprocessing import load_and_preprocess_data, split_and_preprocess_data, create_global_preprocessor, save_global_preprocessor, load_global_preprocessor
from _4cluster import create_clusters, load_clustering_models, predict_cluster
from _5feature import (apply_feature_generation, generate_polynomial_features, generate_interaction_terms, generate_statistical_features,
                       combine_feature_engineered_data, generate_features_for_prediction)
from _6training import (train_and_validate_models, create_ensemble_model, train_models_on_flattened_data, load_trained_models, predict_with_model,
                        save_trained_models)
from _7metrics import calculate_metrics, display_metrics, plot_residuals, plot_actual_vs_predicted
from _8prediction import PredictionProcessor, load_saved_models, predict_for_new_data

# Set page config as the first Streamlit command
st.set_page_config(
    page_title=STREAMLIT_APP_NAME,
    page_icon=STREAMLIT_APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    st.title(STREAMLIT_APP_NAME)

    # Mode selection
    mode = st.sidebar.radio("Select Mode", ["Training", "Prediction"])

    if mode == "Training":
        run_training_mode()
    else:
        run_prediction_mode()

def run_training_mode():
    st.header("Training Mode")

    # 1. Data Input and Initial Configuration
    st.subheader("1. Data Input and Initial Configuration")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    create_tooltip(TOOLTIPS["file_upload"])
    
    if uploaded_file is not None:
        if validate_file_upload(uploaded_file):
            config.update(file_path=uploaded_file)
            
            if uploaded_file.name.endswith('.xlsx'):
                sheet_name = display_sheet_selection(uploaded_file)
                config.update(sheet_name=sheet_name)
            
            data = load_data(config.file_path, config.sheet_name)
            if data is not None:
                display_data_info(data)
                
                # 2. Data Preprocessing
                st.subheader("2. Data Preprocessing")
                create_info_button("data_preprocessing")
                data = preprocess_data(data)
                
                # 3. Feature Engineering
                st.subheader("3. Feature Engineering")
                create_info_button("feature_engineering")
                config = configure_feature_engineering(config)
                
                # 4. Clustering Configuration
                st.subheader("4. Clustering Configuration")
                create_info_button("clustering_configuration")
                config = configure_clustering(config)
                
                # 5. Model Selection and Training
                st.subheader("5. Model Selection and Training")
                create_info_button("model_selection_training")
                config = configure_model_training(config)
                
                # 6. Advanced Options
                st.subheader("6. Advanced Options")
                create_info_button("advanced_options")
                config = configure_advanced_options(config)
                
                # 7. Execution and Results
                if st.button("Start Training"):
                    with st.spinner("Training in progress..."):
                        execute_training(data, config)
                
                # 8. Model Saving
                save_models(config)

def run_prediction_mode():
    st.header("Prediction Mode")
    
    # Load saved models and preprocessors
    create_info_button("load_saved_models")
    models, preprocessors = load_saved_models_and_preprocessors()
    
    # Upload new data for prediction
    create_info_button("upload_prediction_data")
    new_data = upload_prediction_data()
    
    if new_data is not None and models:
        # Make predictions
        create_info_button("make_predictions")
        predictions = predict_for_new_data(models, new_data, preprocessors)
        
        # Display predictions
        display_predictions(predictions)

def display_sheet_selection(uploaded_file):
    create_tooltip(TOOLTIPS["sheet_selection"])
    xls = pd.ExcelFile(uploaded_file)
    return st.selectbox("Select sheet", xls.sheet_names)

def preprocess_data(data):
    # Automatic column type detection
    create_tooltip(TOOLTIPS["auto_detect_column_types"])
    initial_types = auto_detect_column_types(data)
    
    # Manual column selection
    create_tooltip(TOOLTIPS["manual_column_selection"])
    selected_columns = display_column_selection(data.columns, initial_types)
    
    if selected_columns:
        config.set_column_types(**selected_columns)
        
        # Handle missing values
        create_tooltip(TOOLTIPS["handle_missing_values"])
        data = handle_missing_values(data)
        
        # Outlier removal
        create_tooltip(TOOLTIPS["outlier_removal"])
        config = display_outlier_removal_options(config)
    
    return data

def configure_feature_engineering(config):
    config.use_polynomial_features = st.checkbox("Use polynomial features", value=config.use_polynomial_features)
    create_tooltip(TOOLTIPS["polynomial_features"])
    
    config.use_interaction_terms = st.checkbox("Use interaction terms", value=config.use_interaction_terms)
    create_tooltip(TOOLTIPS["interaction_terms"])
    
    config.use_statistical_features = st.checkbox("Use statistical features", value=config.use_statistical_features)
    create_tooltip(TOOLTIPS["statistical_features"])
    
    return config

def configure_clustering(config):
    config.use_clustering = st.checkbox("Use clustering", value=config.use_clustering)
    create_tooltip(TOOLTIPS["use_clustering"])
    
    if config.use_clustering:
        config = display_clustering_options(config)
    
    return config

def configure_model_training(config):
    config.models_to_use = st.multiselect("Select models to use", list(config.MODEL_CLASSES.keys()), default=config.models_to_use)
    create_tooltip(TOOLTIPS["models_to_use"])
    
    config.tuning_method = st.selectbox("Select tuning method", ["None", "GridSearchCV", "RandomizedSearchCV"], index=["None", "GridSearchCV", "RandomizedSearchCV"].index(config.tuning_method))
    create_tooltip(TOOLTIPS["tuning_method"])
    
    config.train_size = st.slider("Train/Test split ratio", 0.1, 0.9, config.train_size)
    create_tooltip(TOOLTIPS["train_test_split"])
    
    return config

def configure_advanced_options(config):
    config.RANDOM_STATE = st.number_input("Random state", value=config.RANDOM_STATE)
    create_tooltip(TOOLTIPS["random_state"])
    
    config.MODEL_CV_SPLITS = st.number_input("Cross-validation folds", min_value=2, max_value=10, value=config.MODEL_CV_SPLITS)
    create_tooltip(TOOLTIPS["cv_folds"])
    
    return config

def execute_training(data, config):
    preprocessed_data = load_and_preprocess_data(data, config)
    
    if config.use_clustering:
        clustered_data = create_clusters(preprocessed_data, config.clustering_config, config.clustering_2d_config)
    else:
        clustered_data = {'no_cluster': {'label_0': preprocessed_data.index}}
    
    data_splits = split_and_preprocess_data(preprocessed_data, clustered_data, config.target_column, config.train_size)
    
    feature_generation_functions = []
    if config.use_polynomial_features:
        feature_generation_functions.append(generate_polynomial_features)
    if config.use_interaction_terms:
        feature_generation_functions.append(generate_interaction_terms)
    if config.use_statistical_features:
        feature_generation_functions.append(generate_statistical_features)
    
    clustered_X_train_combined, clustered_X_test_combined = apply_feature_generation(
        data_splits, feature_generation_functions
    )
    
    all_models, ensemble_cv_results, all_evaluation_metrics = train_and_validate_models(
        data_splits, clustered_X_train_combined, clustered_X_test_combined, 
        config.models_to_use, config.tuning_method
    )
    
    save_trained_models(all_models, config.MODELS_DIRECTORY)
    save_unused_data(data[config.unused_columns], os.path.join(config.MODELS_DIRECTORY, "unused_data.csv"))
    
    joblib.dump(config.clustering_config, os.path.join(config.MODELS_DIRECTORY, "clustering_config.joblib"))
    joblib.dump(config.clustering_2d_config, os.path.join(config.MODELS_DIRECTORY, "clustering_2d_config.joblib"))
    
    st.success("Training completed successfully!")
    
    display_training_results(all_evaluation_metrics)

def save_models(config):
    if st.button("Save trained models"):
        save_trained_models(config.all_models, config.MODELS_DIRECTORY)
        st.success("Models saved successfully")

def load_saved_models_and_preprocessors():
    all_models = load_saved_models(config.MODELS_DIRECTORY)
    global_preprocessor = load_global_preprocessor(config.MODELS_DIRECTORY)
    clustering_config = joblib.load(os.path.join(config.MODELS_DIRECTORY, "clustering_config.joblib"))
    clustering_2d_config = joblib.load(os.path.join(config.MODELS_DIRECTORY, "clustering_2d_config.joblib"))
    cluster_models = load_clustering_models(config.MODELS_DIRECTORY)
    return all_models, (global_preprocessor, clustering_config, clustering_2d_config, cluster_models)

def upload_prediction_data():
    uploaded_file = st.file_uploader("Upload new data for prediction", type=["csv"])
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

def display_predictions(predictions):
    st.write("Predictions:")
    st.dataframe(predictions)
    
    if config.target_column in predictions.columns:
        metrics = calculate_metrics(predictions[config.target_column], predictions['Prediction'])
        st.subheader("Prediction Metrics")
        display_metrics(metrics)

def display_training_results(all_evaluation_metrics):
    st.subheader("Evaluation Metrics")
    for cluster_name, metrics in all_evaluation_metrics.items():
        st.write(f"Cluster: {cluster_name}")
        for model_name, model_metrics in metrics.items():
            st.write(f"Model: {model_name}")
            display_metrics(model_metrics)

if __name__ == "__main__":
    main()
