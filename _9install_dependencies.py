import subprocess
import sys
import streamlit as st

def install(package):
    try:
        __import__(package)
        st.write(f"{package} is already installed.")
    except ImportError:
        st.write(f"{package} is not installed. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            st.success(f"{package} installed successfully.")
        except subprocess.CalledProcessError:
            st.error(f"Failed to install {package}. Please install it manually.")

def install_packages():
    # List of required packages
    required_packages = [
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'xgboost',
        'lightgbm',
        'catboost',
        'openpyxl',
        'xlsxwriter',
        'matplotlib',
        'seaborn',
        'plotly',
        'streamlit',
        'jupyter',
        'ipywidgets',
        'pytest',
        'pylint',
        'autopep8',
        'rope',
        'yapf',
        'mypy',
        'black',
        'flake8',
        'tqdm',
        'joblib',
        'itables'
    ]

    st.subheader("Installing Required Packages")
    progress_bar = st.progress(0)
    for i, package in enumerate(required_packages):
        install(package)
        progress_bar.progress((i + 1) / len(required_packages))
    st.success("All required packages installed successfully.")

def update_packages():
    st.subheader("Updating Packages")
    try:
        st.write("Updating pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        st.success("Pip updated successfully.")
        
        st.write("Updating packages from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "-r", "requirements.txt"])
        st.success("Packages updated successfully.")
    except subprocess.CalledProcessError as e:
        st.error(f"An error occurred while updating packages: {str(e)}")

def dependency_management():
    st.title("Dependency Management")
    
    action = st.radio("Select action:", ("Install Packages", "Update Packages"))
    
    if st.button("Execute"):
        if action == "Install Packages":
            install_packages()
        elif action == "Update Packages":
            update_packages()

if __name__ == "__main__":
    dependency_management()