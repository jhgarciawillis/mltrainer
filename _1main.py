import streamlit as st
import os
import joblib
import pandas as pd
import traceback

from _0config import (config, MODELS_DIRECTORY, PREDICTIONS_PATH, AVAILABLE_CLUSTERING_METHODS,
                      MODEL_CLASSES, LOG_FILE, identify_column_types)
from _2utility import (setup_logging, setup_directory, truncate_sheet_name, 
                       check_and_remove_duplicate_columns, check_and_reset_indices, 
                       display_dataframe, get_user_inputs, validate_file_upload, 
                       load_data, display_data_info, handle_missing_values)
from _3preprocessing import (load_and_preprocess_data, split_and_preprocess_data, flatten_data,
                             create_global_preprocessor, save_global_preprocessor, load_global_preprocessor)
from _4cluster import create_clusters, load_clustering_models, predict_cluster
from _5feature import (apply_feature_generation, generate_polynomial_features, 
                       generate_interaction_terms, generate_statistical_features, 
                       combine_feature_engineered_data, generate_features_for_prediction)
from _6training import (train_and_validate_models, create_ensemble_model, 
                        train_models_on_flattened_data, load_trained_models, predict_with_model)
from _7metrics import calculate_metrics
from _8prediction import PredictionProcessor

# Initialize logger
logger = setup_logging(LOG_FILE)

def main():
    st.set_page_config(page_title="Real Estate Price Prediction App", page_icon="🏠", layout="wide")
    st.title("Real Estate Price Prediction App")

    # Sidebar configuration
    st.sidebar.header("Configuration")
    config = get_user_inputs()

    # File upload and validation
    if config.file_path and validate_file_upload(config.file_path):
        try:
            data = load_and_preprocess_data(config.file_path, config.sheet_name)
            config.set_column_types(data)
            display_data_info(data)
            
            # Handle missing values
            data = handle_missing_values(data)

            # Get user inputs for model configuration
            models_to_use = st.sidebar.multiselect("Select models to use", list(MODEL_CLASSES.keys()))
            tuning_method = st.sidebar.selectbox("Select tuning method", ["None", "GridSearchCV", "RandomizedSearchCV"])
            
            clustering_method = st.sidebar.selectbox("Select clustering method", AVAILABLE_CLUSTERING_METHODS)
            if clustering_method == 'DBSCAN':
                eps = st.sidebar.slider("DBSCAN eps", 0.1, 1.0, 0.5)
                min_samples = st.sidebar.slider("DBSCAN min_samples", 2, 10, 5)
                clustering_params = {'eps': eps, 'min_samples': min_samples}
            elif clustering_method == 'KMeans':
                n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 5)
                clustering_params = {'n_clusters': n_clusters}
            
            execution_mode = st.sidebar.radio("Execution mode", ["full", "prediction"])

            # Setup directory
            setup_directory(MODELS_DIRECTORY)

            # Create and save global preprocessor
            with st.spinner("Creating global preprocessor..."):
                global_preprocessor = create_global_preprocessor(data)
                if global_preprocessor is not None:
                    save_global_preprocessor(global_preprocessor, MODELS_DIRECTORY)
                    st.success("Global preprocessor created and saved.")
                else:
                    st.warning("No global preprocessor created.")

            if execution_mode == 'full':
                # Full execution mode
                clustering_config = {
                    'method': clustering_method,
                    'params': clustering_params,
                    'columns': config.numerical_columns
                }

                with st.spinner("Creating clusters..."):
                    clustered_data = create_clusters(data, clustering_config)
                    logger.info("Clustering complete.")

                # Split and preprocess data for each cluster
                with st.spinner("Splitting and preprocessing data..."):
                    data_splits = split_and_preprocess_data(data, clustered_data, config.target_column)
                    logger.info("Data splitting and preprocessing complete.")

                # Generate features
                with st.spinner("Generating features..."):
                    feature_generation_functions = [
                        generate_polynomial_features,
                        generate_interaction_terms,
                        generate_statistical_features
                    ]
                    clustered_X_train_combined, clustered_X_test_combined = apply_feature_generation(
                        data_splits, feature_generation_functions
                    )
                    clustered_X_train_combined, clustered_X_test_combined = combine_feature_engineered_data(
                        data_splits, clustered_X_train_combined, clustered_X_test_combined
                    )
                    logger.info("Feature generation complete.")

                # Train and validate models
                if st.button("Train and Validate Models"):
                    with st.spinner("Training and validating models..."):
                        all_models, ensemble_cv_results, all_evaluation_metrics = train_and_validate_models(
                            data_splits, clustered_X_train_combined, clustered_X_test_combined, 
                            models_to_use, tuning_method
                        )
                        st.success("Model training and validation complete.")

                    # Display evaluation metrics
                    st.subheader("Evaluation Metrics")
                    for cluster_name, metrics in all_evaluation_metrics.items():
                        st.write(f"Cluster: {cluster_name}")
                        for model_name, model_metrics in metrics.items():
                            st.write(f"Model: {model_name}")
                            st.table(pd.DataFrame(model_metrics, index=[0]))

                    # Flatten data and train models on flattened data
                    flattened_X_train, flattened_y_train = flatten_data(clustered_X_train_combined, data_splits)
                    
                    with st.spinner("Training models on flattened data..."):
                        flattened_models, flattened_cv_results = train_models_on_flattened_data(
                            flattened_X_train, flattened_y_train, models_to_use, tuning_method, global_preprocessor
                        )
                        st.success("Model training on flattened data complete.")

                    # Create ensemble model
                    with st.spinner("Creating ensemble model..."):
                        ensemble, ensemble_cv_scores = create_ensemble_model(all_models, flattened_X_train, flattened_y_train, global_preprocessor)
                        st.success("Ensemble model created and evaluated.")

                # Generate predictions
                if st.button("Generate Predictions"):
                    with st.spinner("Generating predictions..."):
                        prediction_processor = PredictionProcessor(
                            truncate_sheet_name_func=truncate_sheet_name,
                            calculate_metrics_func=calculate_metrics,
                            debug_print_func=logger.debug
                        )
                        prediction_processor.create_predictions_file(
                            PREDICTIONS_PATH,
                            all_models,
                            flattened_models,
                            ensemble,
                            clustered_X_train_combined,
                            clustered_X_test_combined,
                            flattened_X_train,
                            flattened_X_train,  # Use flattened_X_train as flattened_X_test for consistency
                            {
                                **ensemble_cv_results,
                                'flattened': flattened_cv_results,
                                'ensemble': ensemble_cv_scores
                            }
                        )
                        st.success(f"Predictions generated and saved to {PREDICTIONS_PATH}")

            elif execution_mode == 'prediction':
                st.subheader("Prediction Mode")
                
                # Load saved models and preprocessors
                with st.spinner("Loading models and preprocessors..."):
                    all_models = load_trained_models(MODELS_DIRECTORY)
                    global_preprocessor = load_global_preprocessor(MODELS_DIRECTORY)
                    cluster_models = load_clustering_models(MODELS_DIRECTORY)
                    st.success("Models and preprocessors loaded.")

                # Load or recreate clustered data
                with st.spinner("Creating clusters..."):
                    clustered_data = create_clusters(data, clustering_config)
                    logger.info("Clustering complete.")

                # Generate features for prediction
                feature_generation_functions = [
                    generate_polynomial_features,
                    generate_interaction_terms,
                    generate_statistical_features
                ]

                # Create a file uploader for new data
                new_data_file = st.file_uploader("Choose a CSV file with new data for prediction", type="csv")
                if new_data_file is not None:
                    new_data = pd.read_csv(new_data_file)
                    st.write("New data shape:", new_data.shape)

                    predictions = []
                    for index, row in new_data.iterrows():
                        cluster_name, cluster_label = predict_cluster(row, cluster_models)
                        preprocessed_row = global_preprocessor.transform(row.to_frame().T)
                        features = generate_features_for_prediction(preprocessed_row, feature_generation_functions)
                        
                        if cluster_name in all_models:
                            model = all_models[f"{cluster_name}_{cluster_label}"]
                            prediction = predict_with_model(model, features)
                            predictions.append({"Index": index, "Cluster": cluster_name, "Label": cluster_label, "Prediction": prediction[0]})
                        else:
                            predictions.append({"Index": index, "Cluster": cluster_name, "Label": cluster_label, "Prediction": "No model available"})

                    predictions_df = pd.DataFrame(predictions)
                    st.write("Predictions:")
                    st.dataframe(predictions_df)

            logger.info("Process completed successfully.")
            st.success("Process completed successfully.")

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"An error occurred: {str(e)}")
            st.error("Check the log file for more details.")

if __name__ == "__main__":
    main()
