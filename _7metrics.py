import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    explained_variance_score, max_error, median_absolute_error
)
from sklearn.inspection import PartialDependenceDisplay
import plotly.graph_objects as go
from _0config import config
from _2utility import debug_print, plot_feature_importance

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    metrics = {
        'Mean Absolute Error': mean_absolute_error(y_true, y_pred),
        'Mean Squared Error': mean_squared_error(y_true, y_pred),
        'Root Mean Squared Error': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R-squared Value': r2_score(y_true, y_pred),
        'Explained Variance Score': explained_variance_score(y_true, y_pred),
        'Max Error': max_error(y_true, y_pred),
        'Median Absolute Error': median_absolute_error(y_true, y_pred)
    }
    
    # Calculate Mean Absolute Percentage Error (MAPE) and Mean Bias Deviation (MBD)
    # Avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    mbd = np.mean((y_pred[mask] - y_true[mask]) / y_true[mask]) * 100
    
    metrics['Mean Absolute Percentage Error'] = mape
    metrics['Mean Bias Deviation'] = mbd
    
    return metrics

def display_metrics(metrics, title="Model Metrics"):
    """Display metrics in Streamlit."""
    st.subheader(title)
    for metric_name, metric_value in metrics.items():
        st.metric(label=metric_name, value=f"{metric_value:.4f}")

def plot_residuals(y_true, y_pred):
    """Plot regression residuals."""
    residuals = y_true - y_pred
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers'))
    fig.update_layout(
        title='Residual Plot',
        xaxis_title='Predicted Values',
        yaxis_title='Residuals'
    )
    st.plotly_chart(fig)

def plot_actual_vs_predicted(y_true, y_pred):
    """Plot actual vs predicted values."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers'))
    fig.add_trace(go.Scatter(x=[y_true.min(), y_true.max()], 
                             y=[y_true.min(), y_true.max()], 
                             mode='lines', name='Ideal'))
    fig.update_layout(
        title='Actual vs Predicted',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values'
    )
    st.plotly_chart(fig)

def plot_feature_importance(model, X):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plot_feature_importance(feature_importance)
    else:
        st.write("Feature importance not available for this model type.")

def plot_partial_dependence(model, X, features):
    """Plot partial dependence for specified features."""
    fig, ax = plt.subplots(figsize=(10, 6))
    display = PartialDependenceDisplay.from_estimator(model, X, features, ax=ax)
    st.pyplot(fig)

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and display various metrics and plots."""
    st.subheader("Model Evaluation")
    
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    display_metrics(metrics)
    
    plot_actual_vs_predicted(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    
    if X_test.shape[1] <= 20:  # Limit to 20 features for readability
        plot_feature_importance(model, X_test)
    
    # Plot partial dependence for top 3 important features
    if hasattr(model, 'feature_importances_'):
        top_features = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)['feature'][:3].tolist()
        
        plot_partial_dependence(model, X_test, top_features)

def calculate_feature_correlations(X, y):
    """Calculate correlations between features and target variable."""
    correlations = X.apply(lambda x: x.corr(y) if x.dtype in ['int64', 'float64'] else 0)
    return correlations.sort_values(ascending=False)

def display_feature_correlations(correlations):
    """Display feature correlations in Streamlit."""
    st.subheader("Feature Correlations with Target")
    fig = go.Figure(go.Bar(
        x=correlations.values,
        y=correlations.index,
        orientation='h'
    ))
    fig.update_layout(
        title='Feature Correlations with Target Variable',
        xaxis_title='Correlation',
        yaxis_title='Features'
    )
    st.plotly_chart(fig)

def evaluate_predictions(y_true, y_pred, cluster_name=None):
    """Evaluate predictions and display results."""
    metrics = calculate_metrics(y_true, y_pred)
    
    title = "Prediction Metrics"
    if cluster_name:
        title += f" for Cluster: {cluster_name}"
    
    display_metrics(metrics, title)
    plot_actual_vs_predicted(y_true, y_pred)
    plot_residuals(y_true, y_pred)

def compare_models(models_metrics):
    """Compare multiple models based on their metrics."""
    st.subheader("Model Comparison")
    
    comparison_df = pd.DataFrame(models_metrics).T
    st.dataframe(comparison_df)
    
    # Plot comparison for each metric
    for metric in comparison_df.columns:
        fig = go.Figure(go.Bar(
            x=comparison_df.index,
            y=comparison_df[metric],
            text=comparison_df[metric].round(4),
            textposition='auto'
        ))
        fig.update_layout(
            title=f'Comparison of {metric}',
            xaxis_title='Models',
            yaxis_title=metric
        )
        st.plotly_chart(fig)

def calculate_cluster_metrics(clustered_predictions, y_true):
    """Calculate metrics for each cluster."""
    cluster_metrics = {}
    for cluster, predictions in clustered_predictions.items():
        cluster_metrics[cluster] = calculate_metrics(y_true[predictions.index], predictions)
    return cluster_metrics

def display_cluster_metrics(cluster_metrics):
    """Display metrics for each cluster."""
    st.subheader("Cluster-wise Metrics")
    for cluster, metrics in cluster_metrics.items():
        st.write(f"Cluster: {cluster}")
        display_metrics(metrics)
        st.write("---")

# Add any additional metric functions as needed