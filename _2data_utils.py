import streamlit as st
import pandas as pd
from _0config import config

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

def auto_detect_column_types(data):
    """Automatically detect column types."""
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = data.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    target_column = data.columns[-1]
    
    # Remove target column from numeric or categorical lists
    if target_column in numeric_columns:
        numeric_columns.remove(target_column)
    elif target_column in categorical_columns:
        categorical_columns.remove(target_column)
    
    return {
        'numerical': numeric_columns,
        'categorical': categorical_columns,
        'target': target_column,
        'unused': []
    }

def display_column_selection(columns, initial_types):
    """Display interface for manual column selection."""
    st.subheader("Column Selection")
    
    column_types = {}
    for col in columns:
        if col == initial_types['target']:
            column_types[col] = st.selectbox(f"Select type for {col}", 
                                             ['target', 'numerical', 'categorical', 'unused'],
                                             index=0)
        else:
            column_types[col] = st.selectbox(f"Select type for {col}", 
                                             ['numerical', 'categorical', 'unused', 'target'],
                                             index=['numerical', 'categorical', 'unused', 'target'].index(
                                                 'numerical' if col in initial_types['numerical'] 
                                                 else 'categorical' if col in initial_types['categorical']
                                                 else 'unused'
                                             ))
    
    # Ensure we have a target column
    if 'target' not in column_types.values():
        st.error("Please select a target column")
        return None
    
    return {
        'numerical': [col for col, type in column_types.items() if type == 'numerical'],
        'categorical': [col for col, type in column_types.items() if type == 'categorical'],
        'target': next(col for col, type in column_types.items() if type == 'target'),
        'unused': [col for col, type in column_types.items() if type == 'unused']
    }

def save_unused_data(unused_data, file_path):
    """Save unused data as CSV."""
    unused_data.to_csv(file_path, index=False)
    st.success(f"Unused data saved to {file_path}")