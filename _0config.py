import streamlit as st
import os
import joblib
import pandas as pd
import traceback

from _0config import (config, MODELS_DIRECTORY, PREDICTIONS_PATH, AVAILABLE_CLUSTERING_METHODS,
                      MODEL_CLASSES, LOG_FILE, STREAMLIT_APP_NAME, STREAMLIT_APP_ICON)
from _2utility import (setup_logging, setup_directory, truncate_sheet_name, 
                       check_and_remove_duplicate_columns, check_and_reset_indices, 
                       display_dataframe, get_user_inputs, validate_file_upload, 
                       load_data, display_data_info, handle_missing_values,
                       display_column_selection, save_unused_data)
from _3preprocessing import (load_and_preprocess_data, split_and_preprocess_data, 
                             create_global_preprocessor, save_global_preprocessor, load_global_preprocessor)
from _4cluster import create_clusters, load_clustering_models, predict_cluster
from _5feature import (apply_feature_generation, generate_polynomial_features, 
                       generate_interaction_terms, generate_statistical_features, 
                       combine_feature_engineered_data, generate_features_for_prediction)
from _6training import (train_and_validate_models, create_ensemble_model, 
                        train_models_on_flattened_data, load_trained_models, predict_with_model,
                        save_trained_models)
from _7metrics import calculate_metrics, display_metrics, plot_residuals, plot_actual_vs_predicted
from _8prediction import PredictionProcessor, load_saved_models, predict_for_new_data

# Initialize logger
logger = setup_logging(LOG_FILE)

def main():
    st.set_page_config(page_title=STREAMLIT_APP_NAME, page_icon=STREAMLIT_APP_ICON, layout="wide")
    st.title(STREAMLIT_APP_NAME)

    # Mode selection
    mode = st.radio("Select Mode", ["Training", "Prediction"])

    # Get user inputs based on the selected mode
    user_config = get_user_inputs(mode)

    if mode == "Training":
        run_training_mode(user_config)
    else:
        run_prediction_mode(user_config)

def run_training_mode(user_config):
    st.header("Training Mode")

    if user_config.file_path is not None:
        try:
            # Load data
            data = load_data(user_config.file_path, user_config.sheet_name)
            st.write("Data loaded successfully.")
            display_data_info(data)

            # Manual column selection
            selected_columns = display_column_selection(data.columns)
            config.update(
                numerical_columns=selected_columns['numerical'],
                categorical_columns=selected_columns['categorical'],
                target_column=selected_columns['target'],
                unused_columns=selected_columns['unused']
            )

            if st.button("Start Training"):
                with st.spinner("Training in progress..."):
                    # Preprocess data
                    preprocessed_data = load_and_preprocess_data(data, config)
                    
                    # Create clusters
                    clustering_config = {
                        'method': user_config.clustering_method,
                        'params': user_config.clustering_parameters,
                        'columns': config.get('numerical_columns')
                    }
                    clustered_data = create_clusters(preprocessed_data, clustering_config)
                    
                    # Split and preprocess data
                    data_splits = split_and_preprocess_data(preprocessed_data, clustered_data, config.get('target_column'), user_config.train_size)
                    
                    # Generate features
                    feature_generation_functions = []
                    if user_config.use_polynomial_features:
                        feature_generation_functions.append(generate_polynomial_features)
                    if user_config.use_interaction_terms:
                        feature_generation_functions.append(generate_interaction_terms)
                    if user_config.use_statistical_features:
                        feature_generation_functions.append(generate_statistical_features)
                    
                    clustered_X_train_combined, clustered_X_test_combined = apply_feature_generation(
                        data_splits, feature_generation_functions
                    )
                    
                    # Train models
                    all_models, ensemble_cv_results, all_evaluation_metrics = train_and_validate_models(
                        data_splits, clustered_X_train_combined, clustered_X_test_combined, 
                        user_config.models_to_use, user_config.tuning_method
                    )
                    
                    # Save models and unused data
                    save_trained_models(all_models, MODELS_DIRECTORY)
                    save_unused_data(data_splits['unused_data'], os.path.join(MODELS_DIRECTORY, "unused_data.csv"))
                    
                    st.success("Training completed successfully!")
                    
                    # Display evaluation metrics
                    st.subheader("Evaluation Metrics")
                    for cluster_name, metrics in all_evaluation_metrics.items():
                        st.write(f"Cluster: {cluster_name}")
                        for model_name, model_metrics in metrics.items():
                            st.write(f"Model: {model_name}")
                            st.table(pd.DataFrame(model_metrics, index=[0]))

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"An error occurred: {str(e)}")
            st.error("Check the log file for more details.")

def run_prediction_mode(user_config):
    st.header("Prediction Mode")

    if user_config.use_saved_models == "Yes":
        # Load saved models and preprocessors
        with st.spinner("Loading saved models and preprocessors..."):
            all_models = load_saved_models(MODELS_DIRECTORY)
            global_preprocessor = load_global_preprocessor(MODELS_DIRECTORY)
            cluster_models = load_clustering_models(MODELS_DIRECTORY)
            st.success("Models and preprocessors loaded successfully.")
    else:
        # Use uploaded models and preprocessors
        if user_config.uploaded_models and user_config.uploaded_preprocessor:
            all_models = {model.name: joblib.load(model) for model in user_config.uploaded_models}
            global_preprocessor = joblib.load(user_config.uploaded_preprocessor)
            st.success("Uploaded models and preprocessor loaded successfully.")
        else:
            st.warning("Please upload all required files.")
            return

    # Make predictions using the new data file
    if user_config.new_data_file is not None:
        new_data = pd.read_csv(user_config.new_data_file)
        st.write("New data shape:", new_data.shape)

        # Make predictions
        with st.spinner("Generating predictions..."):
            predictions = predict_for_new_data(all_models, new_data, cluster_models)
            st.write("Predictions:")
            st.dataframe(predictions)

            # Calculate and display metrics if target column is available
            if config.get('target_column') in new_data.columns:
                metrics = calculate_metrics(new_data[config.get('target_column')], predictions['Prediction'])
                st.subheader("Prediction Metrics")
                st.table(pd.DataFrame(metrics, index=[0]))

        st.success("Predictions generated successfully!")

if __name__ == "__main__":
    main()
