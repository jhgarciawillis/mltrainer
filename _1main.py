import streamlit as st
import os
import joblib
import pandas as pd
import traceback

from _0config import (config, MODELS_DIRECTORY, PREDICTIONS_PATH, AVAILABLE_CLUSTERING_METHODS,
                      MODEL_CLASSES, LOG_FILE, GITHUB_USERNAME, GITHUB_REPOSITORY, STREAMLIT_APP_NAME,
                      STREAMLIT_APP_URL)
from _2utility import (setup_logging, setup_directory, truncate_sheet_name, 
                       check_and_remove_duplicate_columns, check_and_reset_indices, 
                       display_dataframe, get_user_inputs, validate_file_upload, 
                       load_data, display_data_info, handle_missing_values,
                       display_column_selection, save_unused_data)
from _3preprocessing import (load_and_preprocess_data, split_and_preprocess_data, flatten_data,
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
    st.set_page_config(page_title=STREAMLIT_APP_NAME, page_icon="🧠", layout="wide")
    st.title(STREAMLIT_APP_NAME)

    st.sidebar.write(f"Application URL: [{STREAMLIT_APP_NAME}]({STREAMLIT_APP_URL})")
    st.sidebar.write(f"GitHub Repository: [{ GITHUB_USERNAME }/{ GITHUB_REPOSITORY }](https://github.com/{ GITHUB_USERNAME }/{ GITHUB_REPOSITORY })")

    # Mode selection
    mode = st.sidebar.radio("Select Mode", ["Training", "Prediction"])

    if mode == "Training":
        run_training_mode()
    else:
        run_prediction_mode()

def run_training_mode():
    st.header("Training Mode")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            # Load data
            data = load_data(uploaded_file)
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

            # Get user inputs for model configuration
            models_to_use = st.multiselect("Select models to use", list(MODEL_CLASSES.keys()))
            tuning_method = st.selectbox("Select tuning method", ["None", "GridSearchCV", "RandomizedSearchCV"])
            
            # Clustering configuration
            clustering_method = st.selectbox("Select clustering method", AVAILABLE_CLUSTERING_METHODS)
            if clustering_method == 'DBSCAN':
                eps = st.slider("DBSCAN eps", 0.1, 1.0, 0.5)
                min_samples = st.slider("DBSCAN min_samples", 2, 10, 5)
                clustering_params = {'eps': eps, 'min_samples': min_samples}
            elif clustering_method == 'KMeans':
                n_clusters = st.slider("Number of clusters", 2, 10, 5)
                clustering_params = {'n_clusters': n_clusters}
            
            # Train/test split
            train_size = st.slider("Select percentage of data for training", 0.1, 0.9, 0.8)
            
            if st.button("Start Training"):
                with st.spinner("Training in progress..."):
                    # Preprocess data
                    preprocessed_data = load_and_preprocess_data(data, config)
                    
                    # Create clusters
                    clustering_config = {
                        'method': clustering_method,
                        'params': clustering_params,
                        'columns': config.get('numerical_columns')
                    }
                    clustered_data = create_clusters(preprocessed_data, clustering_config)
                    
                    # Split and preprocess data
                    data_splits = split_and_preprocess_data(preprocessed_data, clustered_data, config.get('target_column'), train_size)
                    
                    # Generate features
                    feature_generation_functions = [
                        generate_polynomial_features,
                        generate_interaction_terms,
                        generate_statistical_features
                    ]
                    clustered_X_train_combined, clustered_X_test_combined = apply_feature_generation(
                        data_splits, feature_generation_functions
                    )
                    
                    # Train models
                    all_models, ensemble_cv_results, all_evaluation_metrics = train_and_validate_models(
                        data_splits, clustered_X_train_combined, clustered_X_test_combined, 
                        models_to_use, tuning_method
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

def run_prediction_mode():
    st.header("Prediction Mode")

    # Option to use saved models or upload new ones
    use_saved_models = st.radio("Use saved models?", ["Yes", "No"])

    if use_saved_models == "Yes":
        # Load saved models and preprocessors
        with st.spinner("Loading saved models and preprocessors..."):
            all_models = load_saved_models(MODELS_DIRECTORY)
            global_preprocessor = load_global_preprocessor(MODELS_DIRECTORY)
            cluster_models = load_clustering_models(MODELS_DIRECTORY)
            st.success("Models and preprocessors loaded successfully.")
    else:
        # Allow user to upload models and preprocessors
        st.write("Please upload your trained models and preprocessors.")
        uploaded_models = st.file_uploader("Upload trained models", type="joblib", accept_multiple_files=True)
        uploaded_preprocessor = st.file_uploader("Upload preprocessor", type="joblib")
        if uploaded_models and uploaded_preprocessor:
            all_models = {model.name: joblib.load(model) for model in uploaded_models}
            global_preprocessor = joblib.load(uploaded_preprocessor)
            st.success("Uploaded models and preprocessor loaded successfully.")
        else:
            st.warning("Please upload all required files.")
            return

    # Upload new data for prediction
    new_data_file = st.file_uploader("Choose a CSV file with new data for prediction", type="csv")
    if new_data_file is not None:
        new_data = pd.read_csv(new_data_file)
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
