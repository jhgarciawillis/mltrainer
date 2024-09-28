import os
import joblib
import pandas as pd
import numpy as np
import traceback
import streamlit as st

from scipy.stats import zscore
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from _0config import config, MODELS_DIRECTORY, OUTLIER_THRESHOLD, TEST_SIZE, RANDOM_STATE
from _2utility import debug_print, check_and_remove_duplicate_columns, check_and_reset_indices, identify_column_types, display_dataframe

def load_and_preprocess_data(file_path, sheet_name=None):
    debug_print("Starting data load and preprocessing from path: " + str(file_path))
    preprocessed_data = load_data(file_path, sheet_name)
    preprocessed_data = remove_outliers(preprocessed_data, config.numerical_columns, OUTLIER_THRESHOLD)
    preprocessed_data = convert_to_numeric(preprocessed_data)
    return preprocessed_data

@st.cache(allow_output_mutation=True)
def load_data(file_path, sheet_name=None):
    if isinstance(file_path, str):
        if file_path.endswith('.xlsx'):
            data = pd.read_excel(file_path, sheet_name=sheet_name)
        elif file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Please use .xlsx or .csv")
    else:  # Assume it's a file-like object (e.g., from st.file_uploader)
        if file_path.name.endswith('.xlsx'):
            data = pd.read_excel(file_path, sheet_name=sheet_name)
        elif file_path.name.endswith('.csv'):
            data = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Please use .xlsx or .csv")

    data = check_and_remove_duplicate_columns(data)
    data = check_and_reset_indices(data)
    debug_print(f"Initial data loaded with shape: {data.shape}")
    return data

def remove_outliers(data, columns, threshold):
    debug_print(f"Removing outliers based on Z-scores in columns: {columns}")
    initial_shape = data.shape
    z_scores = np.abs(zscore(data[columns].astype(float), nan_policy='omit'))
    data = data[(z_scores < threshold).all(axis=1)].copy()
    debug_print(f"Outliers removed. Data shape changed from {initial_shape} to {data.shape}")
    return data

def convert_to_numeric(df):
    debug_print(f"Converting numerical columns to numeric type")
    for col in config.numerical_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    debug_print("Columns converted to numeric types successfully.")
    return df

def create_preprocessor_pipeline(numerical_cols, categorical_cols):
    debug_print(f"Creating preprocessor pipeline with numerical_cols: {numerical_cols} and categorical_cols: {categorical_cols}")
    transformers = []
    
    if numerical_cols:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_transformer, numerical_cols))
    
    if categorical_cols:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
        transformers.append(('cat', categorical_transformer, categorical_cols))

    if not transformers:
        debug_print("No transformers created, returning None")
        return None

    debug_print(f"Created transformers: {transformers}")
    return ColumnTransformer(transformers=transformers, remainder='passthrough')
  
def split_and_preprocess_data(preprocessed_data, clustered_data, target_column):
    debug_print("Splitting and preprocessing data...")
    data_splits = {}
    flattened_clustered_data = flatten_clustered_data(clustered_data)

    progress_bar = st.progress(0)
    for i, (cluster_name, label, indices) in enumerate(flattened_clustered_data):
        debug_print(f"\n[DEBUG] Processing cluster: {cluster_name}, label: {label}")

        if not indices:
            debug_print(f"Skipping label {label} of cluster {cluster_name} as its indices are empty")
            continue

        data_subset = preprocessed_data.loc[indices].reset_index(drop=True)
        X, y = split_features_and_target(data_subset, target_column)

        debug_print(f"Data subset for cluster {cluster_name}, label {label} created with shape: {data_subset.shape}")
        debug_print(f"X shape: {X.shape}, y shape: {y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        debug_print(f"Split data for cluster {cluster_name}, label {label}: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

        numerical_cols, categorical_cols = identify_column_types(X_train)
        preprocessor = create_preprocessor_pipeline(numerical_cols, categorical_cols)

        try:
            if preprocessor is None:
                X_train_prepared, X_test_prepared = X_train, X_test
                feature_names = X_train.columns.tolist()
                debug_print("No preprocessing applied")
            else:
                debug_print("Fitting preprocessor...")
                preprocessor.fit(X_train)
                debug_print("Transforming X_train...")
                X_train_prepared = preprocessor.transform(X_train)
                debug_print("Transforming X_test...")
                X_test_prepared = preprocessor.transform(X_test)

                feature_names = get_feature_names(preprocessor)
                debug_print(f"Feature names after preprocessing: {feature_names}")
                
                # Convert to DataFrame and ensure correct shape
                X_train_prepared = pd.DataFrame(X_train_prepared, columns=feature_names, index=X_train.index)
                X_test_prepared = pd.DataFrame(X_test_prepared, columns=feature_names, index=X_test.index)

            debug_print(f"Columns after preprocessing for cluster {cluster_name}, label {label}: {X_train_prepared.columns.tolist()}")
            debug_print(f"Number of columns after preprocessing for cluster {cluster_name}, label {label}: {len(X_train_prepared.columns)}")

            save_preprocessor(preprocessor, cluster_name, label)
        except Exception as e:
            debug_print(f"Error occurred during preprocessor fitting or transformation: {str(e)}")
            debug_print(f"Error type: {type(e).__name__}")
            debug_print(f"Error traceback: {traceback.format_exc()}")
            debug_print(f"Skipping preprocessing for cluster {cluster_name}, label {label} due to the error.")
            X_train_prepared, X_test_prepared = X_train, X_test
            feature_names = X_train.columns.tolist()  # Use the original column names
            preprocessor = None

        data_splits[f"{cluster_name}_{label}"] = {
            'X_train': X_train_prepared,
            'X_test': X_test_prepared,
            'y_train': y_train,
            'y_test': y_test,
            'preprocessor': preprocessor,
            'feature_names': feature_names  # Add the feature names to the data split
        }
        debug_print(f"Data split and preprocessing completed for cluster {cluster_name}, label {label}.")
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(flattened_clustered_data))

    debug_print("\n[DEBUG] Finished processing all clusters and labels")
    return data_splits

def split_features_and_target(data_subset, target_column):
    X = data_subset.drop(columns=[target_column])
    y = data_subset[target_column]
    return X, y

def get_feature_names(preprocessor):
    if preprocessor is None:
        return []
    
    feature_names = []
    
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(columns)
        elif name == 'cat':
            if hasattr(transformer, 'get_feature_names_out'):
                if isinstance(transformer, Pipeline):
                    transformer_feature_names = transformer.steps[-1][1].get_feature_names_out(columns)
                else:
                    transformer_feature_names = transformer.get_feature_names_out(columns)
                feature_names.extend(transformer_feature_names)
            else:
                feature_names.extend(columns)
    
    return feature_names

def save_preprocessor(preprocessor, cluster_name, label):
    if preprocessor is not None:
        preprocessor_filename = os.path.join(MODELS_DIRECTORY, f'preprocessor_{cluster_name}_{label}.joblib')
        joblib.dump(preprocessor, preprocessor_filename)
        
        # Save individual components
        numeric_transformer = preprocessor.named_transformers_.get('num')
        if numeric_transformer:
            imputer = numeric_transformer.named_steps['imputer']
            scaler = numeric_transformer.named_steps['scaler']
            
            joblib.dump(imputer, os.path.join(MODELS_DIRECTORY, f'imputer_{cluster_name}_{label}.joblib'))
            joblib.dump(scaler, os.path.join(MODELS_DIRECTORY, f'scaler_{cluster_name}_{label}.joblib'))
        
        categorical_transformer = preprocessor.named_transformers_.get('cat')
        if categorical_transformer:
            onehot_encoder = categorical_transformer.named_steps['onehot']
            joblib.dump(onehot_encoder, os.path.join(MODELS_DIRECTORY, f'onehot_encoder_{cluster_name}_{label}.joblib'))
        
        debug_print(f"Preprocessing components saved for cluster {cluster_name}, label {label}")
    else:
        debug_print(f"No preprocessor saved for cluster {cluster_name}, label {label} as it encountered an error.")

def flatten_clustered_data(clustered_data):
    return [(cluster_name, label, indices) 
            for cluster_name, cluster_labels in clustered_data.items() 
            for label, indices in cluster_labels.items()]

def flatten_data(clustered_X_train, data_splits):
    if clustered_X_train:
        flattened_X_train = pd.concat(list(clustered_X_train.values()))
        flattened_y_train = pd.concat([split_data['y_train'] for split_data in data_splits.values() if 'y_train' in split_data])
    else:
        flattened_X_train = pd.DataFrame()
        flattened_y_train = pd.Series()
    return flattened_X_train, flattened_y_train

def create_global_preprocessor(data):
    debug_print("Creating global preprocessor...")
    numerical_cols, categorical_cols = identify_column_types(data)
    all_cols = numerical_cols + categorical_cols
    
    debug_print(f"Numerical columns: {numerical_cols}")
    debug_print(f"Categorical columns: {categorical_cols}")
    
    transformers = []
    
    if numerical_cols:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_transformer, numerical_cols))
    
    if categorical_cols:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        transformers.append(('cat', categorical_transformer, categorical_cols))
    
    if not transformers:
        debug_print("No transformers created, returning None")
        return None
    
    debug_print(f"Created transformers: {transformers}")
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    
    # Fit the preprocessor
    preprocessor.fit(data[all_cols])
    
    return preprocessor
  
def save_global_preprocessor(preprocessor, save_path):
    if preprocessor is not None:
        joblib.dump(preprocessor, os.path.join(save_path, 'global_preprocessor.joblib'))
        debug_print("Global preprocessor saved successfully.")
    else:
        debug_print("No global preprocessor to save (preprocessor is None).")

def load_global_preprocessor(save_path):
    preprocessor = joblib.load(os.path.join(save_path, 'global_preprocessor.joblib'))
    debug_print("Global preprocessor loaded successfully.")
    return preprocessor

def display_preprocessed_data(preprocessed_data):
    st.subheader("Preprocessed Data")
    display_dataframe(preprocessed_data)

def display_data_splits(data_splits):
    st.subheader("Data Splits")
    for cluster_name, split_data in data_splits.items():
        st.write(f"Cluster: {cluster_name}")
        st.write(f"X_train shape: {split_data['X_train'].shape}")
        st.write(f"X_test shape: {split_data['X_test'].shape}")
        st.write(f"y_train shape: {split_data['y_train'].shape}")
        st.write(f"y_test shape: {split_data['y_test'].shape}")
        st.write("---")