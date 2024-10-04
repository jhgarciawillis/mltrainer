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

# Tooltips for UI elements
TOOLTIPS = {
    "file_upload": "Upload your dataset in CSV or Excel format.",
    "sheet_selection": "Select the sheet containing your data (for Excel files).",
    "train_test_split": "Set the proportion of data to use for training. The rest will be used for testing.",
    "use_clustering": "Enable clustering to group similar data points before training models.",
    "models_to_use": "Select one or more machine learning models to train on your data.",
    "tuning_method": "Choose a method for optimizing model hyperparameters.",
    "clustering_method": "Select a clustering algorithm to apply to this column.",
    "dbscan_eps": "Set the maximum distance between two samples for them to be considered as in the same neighborhood.",
    "dbscan_min_samples": "Set the number of samples in a neighborhood for a point to be considered as a core point.",
    "kmeans_n_clusters": "Set the number of clusters to form and centroids to generate.",
    "2d_clustering": "Select pairs of columns for two-dimensional clustering.",
    "use_saved_models": "Choose whether to use previously saved models or upload new ones.",
    "upload_models": "Upload trained model files (in joblib format).",
    "upload_preprocessor": "Upload the preprocessor used for data transformation.",
    "new_data_file": "Upload a CSV file containing new data for prediction.",
    "auto_detect_column_types": "Automatically detect the column types in the dataset.",
    "manual_column_selection": "Manually select the column types for your dataset.",
    "handle_missing_values": "Choose a strategy to handle missing values in your dataset.",
    "outlier_removal": "Identify and remove data points that significantly differ from other observations.",
    "polynomial_features": "Generate polynomial features to capture non-linear relationships.",
    "interaction_terms": "Create interaction features between numerical and categorical columns.",
    "statistical_features": "Calculate statistical features like mean, median, and standard deviation.",
    "random_state": "Set the random state for reproducibility of the machine learning models.",
    "cv_folds": "Specify the number of cross-validation folds to use during model training."
}

# Detailed information for UI sections
INFO_TEXTS = {
    "data_preprocessing": "Data preprocessing involves cleaning and transforming raw data into a format suitable for machine learning models. This includes handling missing values, encoding categorical variables, and scaling numerical features.",
    "feature_engineering": "Feature engineering is the process of using domain knowledge to create new features or transform existing ones. This can improve model performance by providing more relevant information to the algorithms.",
    "clustering_configuration": "Clustering is an unsupervised learning technique that groups similar data points together. It can be used to segment your data before applying regression models, potentially improving overall performance.",
    "model_selection_training": "In this section, you can choose which machine learning models to train on your data. You can also select a method for tuning the hyperparameters of these models to optimize their performance.",
    "advanced_options": "Advanced options allow you to fine-tune various aspects of the machine learning pipeline, such as the random state for reproducibility and the number of cross-validation folds.",
    "outlier_removal": "Outlier removal is the process of identifying and removing data points that significantly differ from other observations. This can improve model performance by reducing the impact of anomalous data.",
    "load_saved_models": "Loading saved models allows you to use previously trained models for making predictions on new data without having to retrain them.",
    "upload_prediction_data": "Upload new data on which you want to make predictions using your trained models.",
    "make_predictions": "Use your trained models to generate predictions for the new data you've uploaded.",
    "manual_column_selection": "Manually select the column types for your dataset. This allows you to specify which columns should be treated as numerical, categorical, target, or unused.",
    "handle_missing_values": "Choose a strategy to handle missing values in your dataset. This can include dropping rows with missing values, filling with the mean/mode, or filling with the median.",
    "polynomial_features": "Generating polynomial features can capture non-linear relationships in your data, potentially improving model performance.",
    "interaction_terms": "Creating interaction features between numerical and categorical columns can help the model learn more complex patterns in the data.",
    "statistical_features": "Calculating statistical features like mean, median, and standard deviation can provide additional information to the machine learning models.",
    "random_state": "Setting the random state ensures reproducibility of the machine learning models, so that the results can be replicated.",
    "cv_folds": "The number of cross-validation folds determines how the dataset is split during model training and validation, which can affect the model's generalization performance."
}

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
        self.outlier_removal_columns = []  # New attribute for outlier removal

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

    def update_outlier_removal_columns(self, columns):
        self.outlier_removal_columns = columns

# Initialize global configuration
config = Config()
