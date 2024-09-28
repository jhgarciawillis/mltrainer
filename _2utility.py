import os
import shutil
import logging
import streamlit as st
import pandas as pd
import plotly.express as px
from _0config import STREAMLIT_THEME, MAX_ROWS_TO_DISPLAY, CHART_HEIGHT, CHART_WIDTH, config

def setup_directory(directory_path):
    """Ensures that the directory exists; if not, it creates it."""
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)

def setup_logging(log_file):
    """Sets up logging to both console and file."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def debug_print(*args):
    """Logs debugging information and displays it in Streamlit."""
    message = ' '.join(map(str, args))
    logging.debug(message)
    st.text(message)

def truncate_sheet_name(sheet_name, max_length=31):
    """Truncates Excel sheet names to the maximum length allowed by Excel."""
    return sheet_name[:max_length]

def check_and_remove_duplicate_columns(df):
    """Check and remove duplicate columns from a DataFrame."""
    duplicate_columns = df.columns[df.columns.duplicated()]
    if len(duplicate_columns) > 0:
        debug_print(f"Duplicate columns found: {', '.join(duplicate_columns)}")
        df = df.loc[:, ~df.columns.duplicated()]
        debug_print("Duplicate columns removed.")
    return df

def check_and_reset_indices(df):
    """Check and reset indices if they are not unique or continuous."""
    if not df.index.is_unique or not df.index.is_monotonic_increasing:
        debug_print("Indices are not unique or continuous. Resetting index.")
        df = df.reset_index(drop=True)
    return df

def identify_column_types(df):
    """Identify numerical and categorical columns in a DataFrame."""
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
    return numerical_cols, categorical_cols

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

def set_streamlit_theme():
    """Set the Streamlit theme based on the configuration."""
    st.set_page_config(
        page_title="Real Estate Price Prediction",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Apply custom theme
    st.markdown(f"""
        <style>
            .reportview-container .main .block-container{{
                max-width: {CHART_WIDTH}px;
                padding-top: 5rem;
                padding-right: 1rem;
                padding-left: 1rem;
                padding-bottom: 5rem;
            }}
            .reportview-container .main {{
                color: {STREAMLIT_THEME['textColor']};
                background-color: {STREAMLIT_THEME['backgroundColor']};
            }}
            .sidebar .sidebar-content {{
                background-color: {STREAMLIT_THEME['secondaryBackgroundColor']};
            }}
            .Widget>label {{
                color: {STREAMLIT_THEME['textColor']};
            }}
            .stButton>button {{
                color: {STREAMLIT_THEME['backgroundColor']};
                background-color: {STREAMLIT_THEME['primaryColor']};
                border-radius: 0.3rem;
            }}
        </style>
        """, unsafe_allow_html=True)

def display_metrics(metrics):
    """Display metrics in a formatted way."""
    st.subheader("Model Performance Metrics")
    col1, col2 = st.columns(2)
    for i, (metric, value) in enumerate(metrics.items()):
        if i % 2 == 0:
            col1.metric(metric, f"{value:.4f}")
        else:
            col2.metric(metric, f"{value:.4f}")

def get_user_inputs():
    """Get user inputs for file selection and target column."""
    st.sidebar.header("Data Configuration")
    
    uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type="xlsx")
    if uploaded_file is not None:
        config.update(file_path=uploaded_file)
        
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = st.sidebar.selectbox("Select sheet", xls.sheet_names)
        config.update(sheet_name=sheet_name)
        
        if config.file_path and config.sheet_name:
            df = pd.read_excel(config.file_path, sheet_name=config.sheet_name)
            target_column = st.sidebar.selectbox("Select target column", df.columns)
            config.update(target_column=target_column)
            
            config.set_column_types(df)
    
    return config

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

def load_data(file_path, sheet_name=None):
    """Load data from Excel or CSV file."""
    if file_path.name.endswith('.xlsx'):
        if sheet_name is None:
            st.error("Please select a sheet name for Excel files.")
            return None
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    elif file_path.name.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        st.error("Unsupported file format.")
        return None
    
    return df

def display_data_info(df):
    """Display information about the loaded data."""
    st.subheader("Data Information")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    st.write(f"Columns: {', '.join(df.columns)}")
    st.write(f"Numerical columns: {', '.join(config.numerical_columns)}")
    st.write(f"Categorical columns: {', '.join(config.categorical_columns)}")
    
    display_dataframe(df.head(), "Data Preview")

def handle_missing_values(df):
    """Handle missing values in the DataFrame."""
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        st.warning("Missing values detected in the following columns:")
        st.write(missing_values[missing_values > 0])
        
        strategy = st.selectbox("Choose a strategy to handle missing values:", 
                                ["Drop rows", "Fill with mean/mode", "Fill with median"])
        
        if strategy == "Drop rows":
            df = df.dropna()
            st.success("Rows with missing values have been dropped.")
        elif strategy == "Fill with mean/mode":
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
            st.success("Missing values have been filled with mean/mode.")
        elif strategy == "Fill with median":
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
            st.success("Missing values have been filled with median.")
    
    return df