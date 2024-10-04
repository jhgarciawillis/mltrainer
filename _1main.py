import streamlit as st
import pandas as pd
import os
import joblib
import traceback

from _0config import config, STREAMLIT_APP_NAME, STREAMLIT_APP_ICON
from _2data_utils import load_data, display_data_info, handle_missing_values, auto_detect_column_types, display_column_selection, save_unused_data
from _2ui_utils import display_metrics, get_user_inputs, get_training_inputs, display_clustering_options, select_2d_clustering_columns, get_prediction_inputs
from _2misc_utils import debug_print, validate_file_upload

from _3preprocessing import load_and_preprocess_data, split_and_preprocess_data, create_global_preprocessor, save_global_preprocessor, load_global_preprocessor
from _4cluster import create_clusters, load_clustering_models, predict_cluster
from _5feature import (apply_feature_generation, generate_polynomial_features, generate_interaction_terms, generate_statistical_features,
                       combine_feature_engineered_data, generate_features_for_prediction)
from _6training import (train_and_validate_models, create_ensemble_model, train_models_on_flattened_data, load_trained_models, predict_with_model,
                        save_trained_models)
from _7metrics import calculate_metrics, display_metrics, plot_residuals, plot_actual_vs_predicted
from _8prediction import PredictionProcessor, load_saved_models, predict_for_new_data

def main():
    debug_print("Starting main function")
    st.set_page_config(
        page_title=STREAMLIT_APP_NAME,
        page_icon=STREAMLIT_APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    debug_print("Page config set")

    st.title(STREAMLIT_APP_NAME)
    debug_print("Title set")

    # Mode selection
    mode = st.radio("Select Mode", ["Training", "Prediction"])
    debug_print(f"Mode selected: {mode}")

    # Get user inputs based on the selected mode
    debug_print("Getting user inputs")
    user_config = get_user_inputs(mode)
    debug_print(f"User config received: {user_config}")

    if mode == "Training":
        debug_print("Entering Training mode")
        run_training_mode(user_config)
    else:
        debug_print("Entering Prediction mode")
        run_prediction_mode(user_config)
    debug_print("Main function completed")

def run_training_mode(user_config):
    debug_print("Starting run_training_mode")
    st.header("Training Mode")

    if user_config.file_path is not None:
        debug_print(f"File path provided: {user_config.file_path}")
        try:
            # Load data
            debug_print("Loading data")
            data = load_data(user_config.file_path, user_config.sheet_name)
            st.write("Data loaded successfully.")
            debug_print("Data loaded successfully")
            display_data_info(data)
            debug_print("Data info displayed")

            # Auto-detect column types
            debug_print("Auto-detecting column types")
            initial_types = auto_detect_column_types(data)
            debug_print(f"Initial types detected: {initial_types}")

            # Manual column selection
            debug_print("Displaying column selection")
            selected_columns = display_column_selection(data.columns, initial_types)
            if selected_columns is None:
                debug_print("Column selection returned None")
                return

            debug_print(f"Selected columns: {selected_columns}")
            config.set_column_types(
                numerical=selected_columns['numerical'],
                categorical=selected_columns['categorical'],
                unused=selected_columns['unused'],
                target=selected_columns['target']
            )
            debug_print("Column types set in config")

            if st.button("Start Training"):
                debug_print("Start Training button clicked")
                with st.spinner("Training in progress..."):
                    # Preprocess data
                    debug_print("Preprocessing data")
                    preprocessed_data = load_and_preprocess_data(data, config)
                    debug_print("Data preprocessed")
                    
                    # Create clusters if clustering is enabled
                    if user_config.use_clustering:
                        debug_print("Creating clusters")
                        clustered_data = create_clusters(preprocessed_data, user_config.clustering_config, user_config.clustering_2d_config)
                        debug_print("Clusters created")
                    else:
                        debug_print("Clustering not used")
                        clustered_data = {'no_cluster': {'label_0': preprocessed_data.index}}
                    
                    # Split and preprocess data
                    debug_print("Splitting and preprocessing data")
                    data_splits = split_and_preprocess_data(preprocessed_data, clustered_data, config.target_column, user_config.train_size)
                    debug_print("Data split and preprocessed")
                    
                    # Generate features
                    debug_print("Generating features")
                    feature_generation_functions = []
                    if user_config.use_polynomial_features:
                        feature_generation_functions.append(generate_polynomial_features)
                    if user_config.use_interaction_terms:
                        feature_generation_functions.append(generate_interaction_terms)
                    if user_config.use_statistical_features:
                        feature_generation_functions.append(generate_statistical_features)
                    
                    debug_print("Applying feature generation")
                    clustered_X_train_combined, clustered_X_test_combined = apply_feature_generation(
                        data_splits, feature_generation_functions
                    )
                    debug_print("Feature generation completed")
                    
                    # Train models
                    debug_print("Training and validating models")
                    all_models, ensemble_cv_results, all_evaluation_metrics = train_and_validate_models(
                        data_splits, clustered_X_train_combined, clustered_X_test_combined, 
                        user_config.models_to_use, user_config.tuning_method
                    )
                    debug_print("Models trained and validated")
                    
                    # Save models and unused data
                    debug_print("Saving trained models")
                    save_trained_models(all_models, config.MODELS_DIRECTORY)
                    debug_print("Saving unused data")
                    save_unused_data(data[config.unused_columns], os.path.join(config.MODELS_DIRECTORY, "unused_data.csv"))
                    
                    # Save clustering configuration
                    debug_print("Saving clustering configuration")
                    joblib.dump(user_config.clustering_config, os.path.join(config.MODELS_DIRECTORY, "clustering_config.joblib"))
                    joblib.dump(config.clustering_2d_config, os.path.join(config.MODELS_DIRECTORY, "clustering_2d_config.joblib"))
                    
                    st.success("Training completed successfully!")
                    debug_print("Training completed successfully")
                    
                    # Display evaluation metrics
                    st.subheader("Evaluation Metrics")
                    debug_print("Displaying evaluation metrics")
                    for cluster_name, metrics in all_evaluation_metrics.items():
                        st.write(f"Cluster: {cluster_name}")
                        for model_name, model_metrics in metrics.items():
                            st.write(f"Model: {model_name}")
                            st.table(pd.DataFrame(model_metrics, index=[0]))
                    debug_print("Evaluation metrics displayed")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Check the Streamlit log for more details.")
            debug_print(f"Error in run_training_mode: {str(e)}")
            debug_print(f"Traceback: {traceback.format_exc()}")
    else:
        debug_print("No file path provided")
    debug_print("run_training_mode completed")

def run_prediction_mode(user_config):
    debug_print("Starting run_prediction_mode")
    st.header("Prediction Mode")

    if user_config.use_saved_models == "Yes":
        debug_print("Using saved models")
        # Load saved models and preprocessors
        with st.spinner("Loading saved models and preprocessors..."):
            debug_print("Loading saved models")
            all_models = load_saved_models(config.MODELS_DIRECTORY)
            debug_print("Loading global preprocessor")
            global_preprocessor = load_global_preprocessor(config.MODELS_DIRECTORY)
            debug_print("Loading clustering config")
            clustering_config = joblib.load(os.path.join(config.MODELS_DIRECTORY, "clustering_config.joblib"))
            clustering_2d_config = joblib.load(os.path.join(config.MODELS_DIRECTORY, "clustering_2d_config.joblib"))
            debug_print("Loading clustering models")
            cluster_models = load_clustering_models(config.MODELS_DIRECTORY)
            st.success("Models and preprocessors loaded successfully.")
            debug_print("Models and preprocessors loaded successfully")
    else:
        debug_print("Using uploaded models")
        # Use uploaded models and preprocessors
        if user_config.uploaded_models and user_config.uploaded_preprocessor:
            debug_print("Loading uploaded models")
            all_models = {model.name: joblib.load(model) for model in user_config.uploaded_models}
            debug_print("Loading uploaded preprocessor")
            global_preprocessor = joblib.load(user_config.uploaded_preprocessor)
            clustering_config = {}
            clustering_2d_config = {}
            cluster_models = None
            st.success("Uploaded models and preprocessor loaded successfully.")
            debug_print("Uploaded models and preprocessor loaded successfully")
        else:
            st.warning("Please upload all required files.")
            debug_print("Required files not uploaded")
            return

    # Make predictions using the new data file
    if user_config.new_data_file is not None:
        debug_print("New data file provided")
        new_data = pd.read_csv(user_config.new_data_file)
        st.write("New data shape:", new_data.shape)
        debug_print(f"New data shape: {new_data.shape}")

        # Make predictions
        with st.spinner("Generating predictions..."):
            debug_print("Generating predictions")
            predictions = predict_for_new_data(all_models, new_data, cluster_models, clustering_config, clustering_2d_config)
            st.write("Predictions:")
            st.dataframe(predictions)
            debug_print("Predictions displayed")

            # Calculate and display metrics if target column is available
            if config.target_column in new_data.columns:
                debug_print("Calculating prediction metrics")
                metrics = calculate_metrics(new_data[config.target_column], predictions['Prediction'])
                st.subheader("Prediction Metrics")
                st.table(pd.DataFrame(metrics, index=[0]))
                debug_print("Prediction metrics displayed")

        st.success("Predictions generated successfully!")
        debug_print("Predictions generated successfully")
    else:
        debug_print("No new data file provided")
    debug_print("run_prediction_mode completed")

if __name__ == "__main__":
    debug_print("Script started")
    main()
    debug_print("Script completed")
