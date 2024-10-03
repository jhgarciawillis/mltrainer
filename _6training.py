import os
import joblib
import pandas as pd
import numpy as np
import scipy.sparse
import streamlit as st

from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import VotingRegressor
from sklearn.pipeline import Pipeline
from _0config import (
    config, MODELS_DIRECTORY, MODEL_CLASSES, HYPERPARAMETER_GRIDS, RANDOM_STATE,
    ENSEMBLE_CV_SPLITS, ENSEMBLE_CV_SHUFFLE, MODEL_CV_SPLITS, RANDOMIZED_SEARCH_ITERATIONS
)
from _2utility import debug_print, plot_prediction_vs_actual
from _7metrics import calculate_metrics

def get_model_instance(model_name):
    """Create an instance of a machine learning model based on its name."""
    if model_name in MODEL_CLASSES:
        return MODEL_CLASSES[model_name]()
    else:
        raise ValueError(f"Model {model_name} is not recognized.")

def create_pipeline(model, preprocessor):
    """Create a pipeline that includes the preprocessor and the model."""
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

def tune_hyperparameters(model_name, model, x_train, y_train, preprocessor, tuning_strategy):
    """Tune the hyperparameters of the model."""
    st.write(f"Tuning hyperparameters for {model_name} using {tuning_strategy}")
    pipeline = create_pipeline(model, preprocessor)
    param_grid = {'model__' + key: value for key, value in HYPERPARAMETER_GRIDS[model_name].items()}
    
    progress_bar = st.progress(0)
    if tuning_strategy == 'GridSearchCV':
        grid_search = GridSearchCV(pipeline, param_grid, cv=MODEL_CV_SPLITS, n_jobs=-1, verbose=1)
        grid_search.fit(x_train, y_train)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        best_estimator = grid_search.best_estimator_
    elif tuning_strategy == 'RandomizedSearchCV':
        random_search = RandomizedSearchCV(pipeline, param_grid, n_iter=RANDOMIZED_SEARCH_ITERATIONS, cv=MODEL_CV_SPLITS, n_jobs=-1, verbose=1)
        random_search.fit(x_train, y_train)
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        best_estimator = random_search.best_estimator_
    else:
        pipeline.fit(x_train, y_train)
        best_params = {}
        best_score = cross_val_score(pipeline, x_train, y_train, cv=MODEL_CV_SPLITS).mean()
        best_estimator = pipeline

    progress_bar.progress(1.0)
    st.write(f"Best parameters: {best_params}")
    st.write(f"Best cross-validation score: {best_score:.4f}")
    return best_estimator

def save_model(model, filename, save_path=MODELS_DIRECTORY):
    """Save the trained model."""
    model_filename = os.path.join(save_path, f"{filename}.joblib")
    joblib.dump(model, model_filename)
    st.write(f"Model saved at: {model_filename}")

def train_and_validate_models(data_splits, clustered_X_train_combined, clustered_X_test_combined, models_to_use, tuning_method):
    st.subheader("Training and Validating Models")

    all_models = {}
    ensemble_cv_results = {}
    all_evaluation_metrics = {}

    progress_bar = st.progress(0)
    for i, (cluster_name, split_data) in enumerate(data_splits.items()):
        st.write(f"\nProcessing data for cluster: {cluster_name}")

        X_train = clustered_X_train_combined[cluster_name]
        X_test = clustered_X_test_combined[cluster_name]
        y_train = split_data['y_train']
        y_test = split_data['y_test']
        preprocessor = split_data['preprocessor']

        st.write(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        st.write(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

        models, cv_results, evaluation_metrics = train_models(
            X_train, y_train, cluster_name, models_to_use, MODELS_DIRECTORY, tuning_method, preprocessor
        )

        all_models[cluster_name] = models
        ensemble_cv_results[cluster_name] = cv_results
        all_evaluation_metrics[cluster_name] = evaluation_metrics

        progress_bar.progress((i + 1) / len(data_splits))

    st.success("Training and validation completed for all selected clusters and models.")

    return all_models, ensemble_cv_results, all_evaluation_metrics

def train_models_on_flattened_data(flattened_x_train, flattened_y_train, models_to_use, tuning_method, global_preprocessor):
    st.subheader("Training Models on Flattened Data")

    flattened_models = {}
    flattened_cv_results = {}

    if not flattened_x_train.empty:
        # Use the global_preprocessor to transform the data
        flattened_x_train_transformed = global_preprocessor.transform(flattened_x_train)
        
        # Get feature names from the preprocessor
        feature_names = (global_preprocessor.named_transformers_['num'].get_feature_names_out().tolist() +
                         global_preprocessor.named_transformers_['cat'].get_feature_names_out().tolist())

        # Convert the transformed data to a DataFrame
        flattened_x_train_transformed_df = pd.DataFrame(flattened_x_train_transformed, columns=feature_names, index=flattened_x_train.index)

        # Train models on the flattened data
        flattened_models, flattened_cv_results, _ = train_models(
            flattened_x_train_transformed_df, flattened_y_train, "flattened", models_to_use, MODELS_DIRECTORY, tuning_method, None
        )

        st.success("Model training on flattened data completed.")
    else:
        st.warning("No data available for training models on flattened data.")

    return flattened_models, flattened_cv_results

def create_ensemble_model(all_models, x_train, y_train, preprocessor, save_path=MODELS_DIRECTORY):
    st.subheader("Creating Ensemble Model")

    if not all_models:
        st.warning("No models available for ensembling.")
        return None, []

    # Print the models being included in the ensemble
    st.write("Models included in the ensemble:")
    for model_name in all_models:
        st.write(f" - {model_name}")

    ensemble_models = [(f"{model_name}", model) for model_name, model in all_models.items()]
    ensemble = VotingRegressor(estimators=ensemble_models)

    # Create a pipeline with preprocessor and ensemble
    ensemble_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('ensemble', ensemble)
    ])

    st.write("Initiated cross-validation for the ensemble model...")
    cv_scores = cross_val_score(ensemble_pipeline, x_train, y_train, cv=KFold(n_splits=ENSEMBLE_CV_SPLITS, shuffle=ENSEMBLE_CV_SHUFFLE, random_state=RANDOM_STATE))

    # Print detailed cross-validation scores
    st.write("Detailed cross-validation scores:")
    for fold_index, score in enumerate(cv_scores, start=1):
        st.write(f" - Fold {fold_index}: Score = {score:.4f}")

    st.write(f"Average cross-validation score for ensemble: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} (std dev)")

    # Fit the ensemble on the entire training data
    ensemble_pipeline.fit(x_train, y_train)

    # Saving the ensemble model
    ensemble_filename = os.path.join(save_path, 'ensemble_model.joblib')
    joblib.dump(ensemble_pipeline, ensemble_filename)
    st.success(f"Ensemble model saved successfully at: {ensemble_filename}")

    # Optionally, provide insights into the ensemble's composition
    st.write("Ensemble model composition:")
    for model_name, _ in ensemble_models:
        st.write(f" - {model_name}")

    return ensemble_pipeline, cv_scores.tolist()

def train_models(x_train, y_train, cluster_name, models_to_use, save_path, tuning_method, preprocessor):
    """Train and evaluate multiple models for a specific cluster."""
    st.write(f"Training models for cluster: {cluster_name}")

    models = {}
    cv_results = {}
    evaluation_metrics = {}

    for model_name in models_to_use:
        st.write(f"Training {model_name} for cluster {cluster_name}")
        model = get_model_instance(model_name)

        if tuning_method == 'GridSearchCV' or tuning_method == 'RandomizedSearchCV':
            best_model = tune_hyperparameters(model_name, model, x_train, y_train, preprocessor, tuning_method)
            cv_scores = cross_val_score(best_model, x_train, y_train, cv=MODEL_CV_SPLITS)
        else:
            st.write(f"Training {model_name} without hyperparameter tuning on {cluster_name}...")
            if preprocessor is not None:
                best_model = create_pipeline(model, preprocessor)
            else:
                best_model = model
            best_model.fit(x_train, y_train)
            cv_scores = cross_val_score(best_model, x_train, y_train, cv=MODEL_CV_SPLITS)

        save_model(best_model, f"{cluster_name}_{model_name}")

        models[model_name] = best_model
        cv_results[model_name] = cv_scores.tolist()
        y_pred = best_model.predict(x_train)
        evaluation_metrics[model_name] = calculate_metrics(y_train, y_pred)

        st.write(f"Model {model_name} for cluster {cluster_name}:")
        st.write(f"  CV scores: {cv_scores}")
        st.write(f"  Evaluation metrics: {evaluation_metrics[model_name]}")

        # Plot actual vs predicted values
        plot_prediction_vs_actual(y_train, y_pred, title=f"{model_name} - Actual vs Predicted ({cluster_name})")

    return models, cv_results, evaluation_metrics

def predict_with_model(model, x_data, preprocessor=None):
    """Make predictions using a trained model."""
    if preprocessor is not None:
        x_data = preprocessor.transform(x_data)
    return model.predict(x_data)

@st.cache(allow_output_mutation=True)
def load_trained_models(models_directory):
    """Load all trained models from the specified directory."""
    trained_models = {}
    for filename in os.listdir(models_directory):
        if filename.endswith('.joblib'):
            model_path = os.path.join(models_directory, filename)
            model_name = filename.replace('.joblib', '')
            trained_models[model_name] = joblib.load(model_path)
    return trained_models

def display_model_performance(all_evaluation_metrics):
    st.subheader("Model Performance")
    for cluster_name, cluster_metrics in all_evaluation_metrics.items():
        st.write(f"Cluster: {cluster_name}")
        for model_name, metrics in cluster_metrics.items():
            st.write(f"Model: {model_name}")
            for metric_name, metric_value in metrics.items():
                st.write(f"  {metric_name}: {metric_value:.4f}")
        st.write("---")

def save_trained_models(all_models, save_path):
    """Save all trained models to the specified directory."""
    for cluster_name, models in all_models.items():
        for model_name, model in models.items():
            filename = f"{cluster_name}_{model_name}"
            save_model(model, filename, save_path)
    st.success("All trained models saved successfully.")
