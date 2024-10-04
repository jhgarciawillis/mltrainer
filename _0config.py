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
MODELS_DIRECTORY = os.path.join(BASE_DIR, "Trained")
CLUSTERS_PATH = os.path.join(BASE_DIR, "Clusters.joblib")
GLOBAL_STATS_PATH = os.path.join(BASE_DIR, "global_stats.joblib")

# Sheet name configuration
SHEET_NAME_MAX_LENGTH = 31
TRUNCATE_SHEET_NAME_REPLACEMENT = "_cluster_db"

# Data processing parameters
OUTLIER_THRESHOLD = 3
RANDOM_STATE = 42

# Clustering parameters
AVAILABLE_CLUSTERING_METHODS = ['None', 'DBSCAN', 'KMeans']
DEFAULT_CLUSTERING_METHOD = 'None'
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

# Streamlit configurations
STREAMLIT_APP_NAME = 'ML Algo Trainer'
STREAMLIT_APP_ICON = '🧠'

# File upload configurations
ALLOWED_EXTENSIONS = ['csv', 'xlsx']
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB

# Visualization configurations
MAX_ROWS_TO_DISPLAY = 100
CHART_HEIGHT = 400
CHART_WIDTH = 600

class Config:
    def __init__(self):
        self.file_path = None
        self.sheet_name = None
        self.target_column = None
        self.numerical_columns = []
        self.categorical_columns = []
        self.unused_columns = []
        self.all_columns = []
        self.use_clustering = False
        self.clustering_config = {}  # Will store {column: {'method': method, 'params': params}}
        self.clustering_2d_config = {}  # Will store {(col1, col2): {'method': method, 'params': params}}
        self.clustering_2d_columns = []  # Will store columns selected for 2D clustering
        self.train_size = 0.8
        self.models_to_use = []
        self.tuning_method = 'None'
        self.use_polynomial_features = True
        self.use_interaction_terms = True
        self.use_statistical_features = True
        self.use_saved_models = 'Yes'
        self.uploaded_models = None
        self.uploaded_preprocessor = None
        self.new_data_file = None
        self.STREAMLIT_APP_NAME = STREAMLIT_APP_NAME

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Update all_columns to include all columns
        self.all_columns = list(set(self.numerical_columns + self.categorical_columns + [self.target_column] + self.unused_columns))

    def set_column_types(self, numerical, categorical, unused, target):
        self.numerical_columns = numerical
        self.categorical_columns = categorical
        self.unused_columns = unused
        self.target_column = target
        self.all_columns = list(set(numerical + categorical + unused + [target]))

    def set_2d_clustering(self, column_pairs, method, params):
        for pair in column_pairs:
            self.clustering_2d_config[pair] = {'method': method, 'params': params}

    def set_2d_clustering_columns(self, columns):
        self.clustering_2d_columns = columns

# Initialize global configuration
config = Config()
