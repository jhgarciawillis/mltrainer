import pandas as pd
import numpy as np
import streamlit as st

from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from _0config import config, MAX_INTERACTION_DEGREE, STATISTICAL_AGG_FUNCTIONS, TOP_K_FEATURES, POLYNOMIAL_DEGREE
from _2misc_utils import debug_print, plot_feature_importance, flatten_clustered_data

def generate_polynomial_features(X, degree=POLYNOMIAL_DEGREE):
    numerical_cols = config.numerical_columns
    X_numeric = X[numerical_cols]
    if X_numeric.empty:
        st.warning("No numerical columns found. Skipping polynomial feature generation.")
        return pd.DataFrame(index=X.index)

    st.write(f"Generating polynomial features with degree {degree}")
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_numeric)
    feature_names = poly.get_feature_names_out(X_numeric.columns)
    new_features = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
    st.write(f"Generated {new_features.shape[1]} polynomial features")
    return new_features

def generate_interaction_terms(X, max_degree=MAX_INTERACTION_DEGREE):
    numerical_cols = config.numerical_columns
    categorical_cols = config.categorical_columns
    X_numeric = X[numerical_cols]
    X_categorical = X[categorical_cols]
    
    if X_numeric.empty and X_categorical.empty:
        st.warning("No numerical or categorical columns found. Skipping interaction feature generation.")
        return pd.DataFrame(index=X.index)

    st.write(f"Generating interaction terms with max degree {max_degree}")
    interaction_columns = []
    
    # Numeric-Numeric interactions
    for i in range(2, max_degree + 1):
        for combo in combinations(X_numeric.columns, i):
            col_name = '_X_'.join(combo)
            interaction_columns.append(pd.Series(X_numeric[list(combo)].product(axis=1), name=col_name))
    
    # Numeric-Categorical interactions
    for num_col in X_numeric.columns:
        for cat_col in X_categorical.columns:
            col_name = f"{num_col}_X_{cat_col}"
            interaction_columns.append(pd.Series(X_numeric[num_col] * pd.factorize(X[cat_col])[0], name=col_name))

    if interaction_columns:
        new_features = pd.concat(interaction_columns, axis=1)
        st.write(f"Generated {new_features.shape[1]} interaction features")
    else:
        new_features = pd.DataFrame(index=X.index)
        st.warning("No interaction features generated")

    return new_features

def generate_statistical_features(X):
    numerical_cols = config.numerical_columns
    X_numeric = X[numerical_cols]

    if X_numeric.empty:
        st.warning("No numerical columns found. Skipping statistical feature generation.")
        return pd.DataFrame(index=X.index)

    st.write(f"Generating statistical features using functions: {STATISTICAL_AGG_FUNCTIONS}")
    new_features = pd.DataFrame(index=X.index)
    for func in STATISTICAL_AGG_FUNCTIONS:
        new_features[f"{func}_all"] = X_numeric.agg(func, axis=1)

    st.write(f"Generated {new_features.shape[1]} statistical features")
    return new_features

def apply_feature_generation(data_splits, feature_generation_functions):
    st.subheader("Applying Feature Generation")
    clustered_X_train_combined = {}
    clustered_X_test_combined = {}

    flattened_data_splits = flatten_clustered_data(data_splits)

    progress_bar = st.progress(0)
    for i, (cluster_key, split_data) in enumerate(flattened_data_splits.items()):
        X_train, y_train = split_data['X_train'], split_data['y_train']
        X_test = split_data['X_test']

        st.write(f"Applying feature generation for cluster: {cluster_key}")
        st.write(f"Initial X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

        new_features_train = pd.DataFrame(index=X_train.index)
        new_features_test = pd.DataFrame(index=X_test.index)

        for feature_gen_func in feature_generation_functions:
            try:
                new_features_train_func = feature_gen_func(X_train)
                new_features_test_func = feature_gen_func(X_test)
                
                # Ensure unique column names
                new_features_train_func = new_features_train_func.add_prefix(f"{feature_gen_func.__name__}_")
                new_features_test_func = new_features_test_func.add_prefix(f"{feature_gen_func.__name__}_")
                
                new_features_train = pd.concat([new_features_train, new_features_train_func], axis=1)
                new_features_test = pd.concat([new_features_test, new_features_test_func], axis=1)
            except Exception as e:
                st.error(f"Error in {feature_gen_func.__name__} for cluster {cluster_key}: {str(e)}")
                continue

        X_train_combined = pd.concat([X_train, new_features_train], axis=1)
        X_test_combined = pd.concat([X_test, new_features_test], axis=1)

        # Remove duplicate columns
        X_train_combined = X_train_combined.loc[:, ~X_train_combined.columns.duplicated()]
        X_test_combined = X_test_combined.loc[:, ~X_test_combined.columns.duplicated()]

        # Apply feature selection
        X_train_selected = select_top_features(X_train_combined, y_train)
        X_test_selected = X_test_combined[X_train_selected.columns]

        clustered_X_train_combined[cluster_key] = X_train_selected
        clustered_X_test_combined[cluster_key] = X_test_selected

        st.write(f"Final X_train shape for cluster {cluster_key}: {X_train_selected.shape}")
        st.write(f"Final X_test shape for cluster {cluster_key}: {X_test_selected.shape}")
        
        progress_bar.progress((i + 1) / len(flattened_data_splits))

    st.success("Feature generation completed for all clusters.")
    return clustered_X_train_combined, clustered_X_test_combined
  
def select_top_features(X, y, k=TOP_K_FEATURES):
    X_numeric = X.select_dtypes(include=['number'])
    if X_numeric.shape[1] == 0:
        st.warning("No numeric features found for selection. Returning original features.")
        return X
    
    st.write(f"Selecting top {k} features")
    selector = SelectKBest(score_func=f_regression, k=min(k, X_numeric.shape[1]))
    X_selected = selector.fit_transform(X_numeric, y)
    selected_feature_mask = selector.get_support()
    selected_numeric_features = X_numeric.columns[selected_feature_mask]
    
    # Include all non-numeric columns
    non_numeric_features = X.select_dtypes(exclude=['number']).columns
    selected_features = list(selected_numeric_features) + list(non_numeric_features)
    
    st.write(f"Selected {len(selected_features)} features")
    
    # Plot feature importance
    feature_importance = pd.DataFrame({
        'feature': selected_numeric_features,
        'importance': selector.scores_[selected_feature_mask]
    }).sort_values('importance', ascending=False)
    
    plot_feature_importance(feature_importance)
    
    return X[selected_features]

def combine_feature_engineered_data(data_splits, clustered_X_train_combined, clustered_X_test_combined):
    st.subheader("Combining Feature Engineered Data")
    flattened_data_splits = flatten_clustered_data(data_splits)
    for cluster_key in flattened_data_splits.keys():
        X_train = clustered_X_train_combined[cluster_key]
        X_test = clustered_X_test_combined[cluster_key]

        # Remove duplicate columns
        X_train = X_train.loc[:, ~X_train.columns.duplicated()]
        X_test = X_test.loc[:, ~X_test.columns.duplicated()]

        # Get the union of columns from both train and test
        all_columns = X_train.columns.union(X_test.columns)

        # Reindex both dataframes with the union of columns
        X_train = X_train.reindex(columns=all_columns, fill_value=0)
        X_test = X_test.reindex(columns=all_columns, fill_value=0)

        # Sort the columns alphabetically
        X_train = X_train.reindex(sorted(X_train.columns), axis=1)
        X_test = X_test.reindex(sorted(X_test.columns), axis=1)

        clustered_X_train_combined[cluster_key] = X_train
        clustered_X_test_combined[cluster_key] = X_test

        st.write(f"Combined data for cluster {cluster_key}:")
        st.write(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    st.success("Feature engineered data combined for all clusters.")
    return clustered_X_train_combined, clustered_X_test_combined

def generate_features_for_prediction(X, feature_generation_functions):
    st.subheader("Generating Features for Prediction")
    new_features = pd.DataFrame(index=X.index)

    for feature_gen_func in feature_generation_functions:
        try:
            new_features_func = feature_gen_func(X)
            new_features_func = new_features_func.add_prefix(f"{feature_gen_func.__name__}_")
            new_features = pd.concat([new_features, new_features_func], axis=1)
        except Exception as e:
            st.error(f"Error in {feature_gen_func.__name__} for prediction: {str(e)}")
            continue

    X_combined = pd.concat([X, new_features], axis=1)
    X_combined = X_combined.loc[:, ~X_combined.columns.duplicated()]

    st.write(f"Generated features for prediction. Final shape: {X_combined.shape}")
    return X_combined

def display_feature_info(clustered_X_train_combined, clustered_X_test_combined):
    st.subheader("Feature Information")
    flattened_data_splits = flatten_clustered_data(clustered_X_train_combined)
    for cluster_key in flattened_data_splits.keys():
        st.write(f"Cluster: {cluster_key}")
        st.write(f"Number of features: {clustered_X_train_combined[cluster_key].shape[1]}")
        st.write("Top 10 features:")
        st.write(clustered_X_train_combined[cluster_key].columns[:10].tolist())
        st.write("---")

def plot_feature_correlations(X, y):
    st.subheader("Feature Correlations with Target")
    correlations = X.apply(lambda x: x.corr(y) if x.dtype in ['int64', 'float64'] else 0)
    correlations = correlations.sort_values(ascending=False)
    
    fig = px.bar(x=correlations.index, y=correlations.values, 
                 labels={'x': 'Features', 'y': 'Correlation with Target'},
                 title='Feature Correlations with Target Variable')
    st.plotly_chart(fig)
