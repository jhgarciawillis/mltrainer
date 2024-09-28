import os
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths and names
DATA_PATH = os.path.join(BASE_DIR, "Data.xlsx")
PREDICTIONS_PATH = os.path.join(BASE_DIR, "Predictions.xlsx")
LOG_FILE = os.path.join(BASE_DIR, "app.log")
MODELS_DIRECTORY = os.path.join(BASE_DIR, "Trained")
CLUSTERS_PATH = os.path.join(BASE_DIR, "Clusters.joblib")
GLOBAL_STATS_PATH = os.path.join(BASE_DIR, "global_stats.joblib")

# Sheet name configuration
SHEET_NAME_MAX_LENGTH = 31
TRUNCATE_SHEET_NAME_REPLACEMENT = "_cluster_db"

# Data processing parameters
OUTLIER_THRESHOLD = 3
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Clustering parameters
AVAILABLE_CLUSTERING_METHODS = ['DBSCAN', 'KMeans']
DBSCAN_PARAMETERS = {
    'eps': 0.5,
    'min_samples': 5
}
KMEANS_PARAMETERS = {
    'n_clusters': 5
}

# Feature engineering parameters
STATISTICAL_AGG_FUNCTIONS = ['mean', 'median', 'std']
TOP_K_FEATURES = 20
MAX_INTERACTION_DEGREE = 2
POLYNOMIAL_DEGREE = 2
FEATURE_SELECTION_SCORE_FUNC = 'f_regression'

# Model configurations
MODEL_CLASSES = {
    'rf': RandomForestRegressor,
    'xgb': XGBRegressor,
    'lgbm': LGBMRegressor,
    'ada': AdaBoostRegressor,
    'catboost': CatBoostRegressor,
    'knn': KNeighborsRegressor
}

# Hyperparameter grids
HYPERPARAMETER_GRIDS = {
    'rf': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    },
    'xgb': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 9]
    },
    'lgbm': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 62, 124]
    },
    'ada': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    },
    'catboost': {
        'iterations': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'depth': [4, 6, 8]
    },
    'knn': {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }
}

# Model training parameters
MODEL_CV_SPLITS = 5
RANDOMIZED_SEARCH_ITERATIONS = 10
ENSEMBLE_CV_SPLITS = 10
ENSEMBLE_CV_SHUFFLE = True

# Streamlit configurations
STREAMLIT_THEME = {
    'primaryColor': '#FF4B4B',
    'backgroundColor': '#FFFFFF',
    'secondaryBackgroundColor': '#F0F2F6',
    'textColor': '#262730',
    'font': 'sans serif'
}

# File upload configurations
ALLOWED_EXTENSIONS = ['csv', 'xlsx']
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB

# Visualization configurations
MAX_ROWS_TO_DISPLAY = 100
CHART_HEIGHT = 400
CHART_WIDTH = 600

# Dynamic column identification function
def identify_column_types(df):
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    return numerical_cols, categorical_cols

# Configuration class for dynamic settings
class Config:
    def __init__(self):
        self.file_path = None
        self.sheet_name = None
        self.target_column = None
        self.numerical_columns = None
        self.categorical_columns = None

    def update(self, file_path=None, sheet_name=None, target_column=None):
        if file_path:
            self.file_path = file_path
        if sheet_name:
            self.sheet_name = sheet_name
        if target_column:
            self.target_column = target_column

    def set_column_types(self, df):
        self.numerical_columns, self.categorical_columns = identify_column_types(df)

# Initialize global configuration
config = Config()