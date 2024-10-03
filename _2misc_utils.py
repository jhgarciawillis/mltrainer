import os
import shutil
import logging
from logging.handlers import RotatingFileHandler
import streamlit as st
import pandas as pd
import plotly.express as px
from _0config import CHART_HEIGHT, CHART_WIDTH

def setup_directory(directory_path):
    """Ensures that the directory exists; if not, it creates it."""
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)

def setup_logging(log_file):
    """Sets up logging to both console and file with rotation."""
    logger = logging.getLogger('mltrainer')
    logger.setLevel(logging.INFO)
    
    file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=5)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def debug_print(logger, *args):
    """Logs debugging information and displays it in Streamlit."""
    message = ' '.join(map(str, args))
    logger.debug(message)
    st.text(message)

def truncate_sheet_name(sheet_name, max_length=31):
    """Truncates Excel sheet names to the maximum length allowed by Excel."""
    return sheet_name[:max_length]

def check_and_remove_duplicate_columns(df):
    """Check and remove duplicate columns from a DataFrame."""
    duplicate_columns = df.columns[df.columns.duplicated()]
    if len(duplicate_columns) > 0:
        st.warning(f"Duplicate columns found and removed: {', '.join(duplicate_columns)}")
        df = df.loc[:, ~df.columns.duplicated()]
    return df

def check_and_reset_indices(df):
    """Check and reset indices if they are not unique or continuous."""
    if not df.index.is_unique or not df.index.is_monotonic_increasing:
        st.info("Indices are not unique or continuous. Resetting index.")
        df = df.reset_index(drop=True)
    return df

def display_dataframe(df, title="DataFrame"):
    """Display a DataFrame in Streamlit with pagination."""
    st.subheader(title)
    page_size = 10
    total_pages = (len(df) - 1) // page_size + 1
    page_number = st.number_input(f"Page (1-{total_pages})", min_value=1, max_value=total_pages, value=1)
    start_idx = (page_number - 1) * page_size
    end_idx = min(start_idx + page_size, len(df))
    st.dataframe(df.iloc[start_idx:end_idx])
    st.write(f"Showing rows {start_idx+1} to {end_idx} of {len(df)}")

def plot_feature_importance(feature_importance, title="Feature Importance"):
    """Plot feature importance using Plotly."""
    fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                 title=title, height=CHART_HEIGHT, width=CHART_WIDTH)
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig)

def plot_prediction_vs_actual(y_true, y_pred, title="Prediction vs Actual"):
    """Plot prediction vs actual values using Plotly."""
    df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
    fig = px.scatter(df, x='Actual', y='Predicted', title=title,
                     height=CHART_HEIGHT, width=CHART_WIDTH)
    fig.add_shape(type="line", line=dict(dash='dash'),
                  x0=df['Actual'].min(), y0=df['Actual'].min(),
                  x1=df['Actual'].max(), y1=df['Actual'].max())
    st.plotly_chart(fig)

def validate_file_upload(uploaded_file):
    """Validate the uploaded file."""
    if uploaded_file is None:
        st.error("Please upload a file.")
        return False
    
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    if file_extension not in ['.xlsx', '.csv']:
        st.error("Invalid file format. Please upload an Excel (.xlsx) or CSV (.csv) file.")
        return False
    
    return True