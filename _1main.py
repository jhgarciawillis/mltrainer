import streamlit as st
import pandas as pd
import os
import joblib

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

            # Auto-detect column types
            initial_types = auto_detect_column_types(data)

            # Manual column selection
            selected_columns = display_column_selection(data.columns, initial_types)
            if selected_columns is None:
                return

            config.set_column_types(
                numerical=selected_columns['numerical'],
                categorical=selected_columns['categorical'],
                unused=selected_columns['unused'],
                target=selected_columns['target']
            )

            if st.button("Start Training"):
                with st.spinner("Training in progress..."):
                    # Preprocess data
                    preprocessed_data = load_and_preprocess_data(data, config)
                    
                    # Create clusters if clustering is enabled
                    if user_config.use_clustering:
                        clustered_data = create_clusters(preprocessed_data, user_config.clustering_config, user_config.clustering_2d_config)
                    else:
                        clustered_data = {'no_cluster': {'label_0': preprocessed_data.index}}
                    
                    # Split and preprocess data
                    data_splits = split_and_preprocess_data(preprocessed_data, clustered_data, config.target_column, user_config.train_size)
                    
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
                    save_trained_models(all_models, config.MODELS_DIRECTORY)
                    save_unused_data(data[config.unused_columns], os.path.join(config.MODELS_DIRECTORY, "unused_data.csv"))
                    
                    # Save clustering configuration
                    joblib.dump(user_config.clustering_config, os.path.join(config.MODELS_DIRECTORY, "clustering_config.joblib"))
                    joblib.dump(config.clustering_2d_config, os.path.join(config.MODELS_DIRECTORY, "clustering_2d_config.joblib"))
                    
                    st.success("Training completed successfully!")
                    
                    # Display evaluation metrics
                    st.subheader("Evaluation Metrics")
                    for cluster_name, metrics in all_evaluation_metrics.items():
                        st.write(f"Cluster: {cluster_name}")
                        for model_name, model_metrics in metrics.items():
                            st.write(f"Model: {model_name}")
                            st.table(pd.DataFrame(model_metrics, index=[0]))

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Check the Streamlit log for more details.")

def run_prediction_mode(user_config):
    st.header("Prediction Mode")

    if user_config.use_saved_models == "Yes":
        # Load saved models and preprocessors
        with st.spinner("Loading saved models and preprocessors..."):
            all_models = load_saved_models(config.MODELS_DIRECTORY)
            global_preprocessor = load_global_preprocessor(config.MODELS_DIRECTORY)
            clustering_config = joblib.load(os.path.join(config.MODELS_DIRECTORY, "clustering_config.joblib"))
            clustering_2d_config = joblib.load(os.path.join(config.MODELS_DIRECTORY, "clustering_2d_config.joblib"))
            cluster_models = load_clustering_models(config.MODELS_DIRECTORY)
            st.success("Models and preprocessors loaded successfully.")
    else:
        # Use uploaded models and preprocessors
        if user_config.uploaded_models and user_config.uploaded_preprocessor:
            all_models = {model.name: joblib.load(model) for model in user_config.uploaded_models}
            global_preprocessor = joblib.load(user_config.uploaded_preprocessor)
            clustering_config = {}
            clustering_2d_config = {}
            cluster_models = None
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
            predictions = predict_for_new_data(all_models, new_data, cluster_models, clustering_config, clustering_2d_config)
            st.write("Predictions:")
            st.dataframe(predictions)

            # Calculate and display metrics if target column is available
            if config.target_column in new_data.columns:
                metrics = calculate_metrics(new_data[config.target_column], predictions['Prediction'])
                st.subheader("Prediction Metrics")
                st.table(pd.DataFrame(metrics, index=[0]))

        st.success("Predictions generated successfully!")

if __name__ == "__main__":
    main()
    debug_print("Script completed")
