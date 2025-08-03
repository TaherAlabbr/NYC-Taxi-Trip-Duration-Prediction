# Standard library
import pandas as pd
import numpy as np

# Third-party libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint, uniform


# Local application imports
from trip_duration_utils_preprocess import preprocessing_pipeline

def prepare_data(args):
    """
    Load, preprocess, and split the training and testing datasets for taxi trip duration prediction.

    This function reads CSV files, applies the preprocessing pipeline (including outlier removal
    with IQR multiplier from args), and returns the feature matrices and target vectors for both
    training and testing datasets.

    Parameters:
    -----------
    args : argparse.Namespace
        Arguments namespace expected to contain:
        - train_dir (str): Path to the training dataset CSV file.
        - test_dir (str): Path to the testing dataset CSV file.
        - iqr_multiplier (float): Multiplier for IQR-based outlier removal in preprocessing.

    Returns:
    --------
    X_train : pd.DataFrame
        Feature matrix for the training dataset after preprocessing.
    X_test : pd.DataFrame
        Feature matrix for the testing dataset after preprocessing.
    y_train : pd.Series
        Target variable (log-transformed trip duration) for training.
    y_test : pd.Series
        Target variable (log-transformed trip duration) for testing.
    """
    train = pd.read_csv(args.train_dir)
    test = pd.read_csv(args.test_dir)


    df_train, iqr_boundries = preprocessing_pipeline(train, iqr_multiplier = args.iqr_multiplier, dataset_name='Training dataset')
    df_test, _ = preprocessing_pipeline(test, iqr_multiplier = args.iqr_multiplier, iqr_bound= iqr_boundries, dataset_name='Testing dataset')


    X_train, y_train = df_train.drop('log_trip_duration', axis=1), df_train['log_trip_duration']
    X_test, y_test = df_test.drop('log_trip_duration', axis=1), df_test['log_trip_duration']

    return X_train, X_test, y_train, y_test, iqr_boundries


def build_preprocessor(scale_strategy, numeric_features, categorical_features):
    """
    Build a ColumnTransformer for preprocessing numeric and categorical features.

    Parameters:
        scale_strategy (str): Scaling strategy to apply to numeric features.
                              Options: 'standard', 'minmax', or 'raw' (no scaling).
        numeric_features (list of str): List of column names corresponding to numeric features.
        categorical_features (list of str): List of column names corresponding to categorical features.

    Returns:
        sklearn.compose.ColumnTransformer: A transformer that applies scaling to numeric features
                                           and one-hot encoding to categorical features.
    
    Raises:
        ValueError: If an unsupported scale_strategy is provided.
    """
    if scale_strategy == 'standard':
        scaler = StandardScaler()
    elif scale_strategy == 'minmax':
        scaler = MinMaxScaler()
    elif scale_strategy == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unsupported scaling strategy: {scale_strategy}")

    preprocessor = ColumnTransformer([
        ('num', scaler, numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], remainder='drop')  # or 'passthrough' to retain unused columns

    return preprocessor

def identify_feature_types(X: pd.DataFrame, args):
    """
    Identify numeric and categorical feature names from the input DataFrame.

    Numeric features are automatically detected by selecting columns with numeric data types,
    excluding those specified as categorical in the arguments. Categorical features are
    provided explicitly via the args.

    Parameters:
    -----------
    X : pd.DataFrame
        Input DataFrame containing feature columns.

    args : argparse.Namespace
        Namespace containing:
        - categorical_features (list of str): List of manually defined categorical feature names.

    Returns:
    --------
    tuple:
        numeric_features (list of str): List of numeric feature names excluding those marked categorical.
        categorical_features (list of str): List of categorical feature names from args.
    """
    # Automatically detect numeric features, excluding those known to be categorical
    numeric_features = [
        col for col in X.select_dtypes(include='number').columns
        if col not in args.categorical_features
    ]

    return numeric_features, args.categorical_features


def setup_random_search(model, args, cv=5, refit='r2', n_jobs=-1, verbose=1, n_iter=30):
    """
    Sets up RandomizedSearchCV for regression models used in the NYC Taxi Project.

    Parameters:
    -----------
    model : Pipeline
        The model or pipeline to tune.

    args : Namespace
        Should include args.model (e.g., 'RR', 'XGB', 'NN') and args.use_stacking.

    cv : int, default=5
        Number of cross-validation folds.

    scoring : str, default='r2'
        Metric to optimize.

    n_jobs : int, default=-1
        Number of parallel jobs.

    verbose : int, default=1
        Controls verbosity.

    n_iter : int, default=30
        Number of parameter settings sampled.

    Returns:
    --------
    RandomizedSearchCV object.
    """

    if args.use_stacking:
        param_dist = {
            # Ridge
            'stacking__ridge__alpha': loguniform(1e-4, 1e2),
            'stacking__ridge__max_iter': randint(1000, 5000),

            'stacking__xgb__n_estimators': randint(50, 150),
            'stacking__xgb__max_depth': randint(3, 12),
                
            'stacking__final_estimator__hidden_layer_sizes': [(64,), (64, 32)],
            'stacking__final_estimator__activation': ['relu'],
            'stacking__final_estimator__alpha': [1e-4, 1e-3],  # Fixed small range
            'stacking__final_estimator__learning_rate_init': [1e-3],  # Default is 0.001
            'stacking__final_estimator__solver': ['adam'],  # Stick to 'adam' to save time
            'stacking__final_estimator__max_iter': [1000, 1500],  # Enough for convergence
        }



    elif args.model == 'RR':
        param_dist = {
            'model__alpha': loguniform(1e-3, 1e1),          # Samples from [0.01, 20.01)
            'model__max_iter': randint(1000, 5000)         # Random integers between 1000 and 4999
        }

    elif args.model == 'XGB':
        param_dist = {
            'model__n_estimators': randint(50, 150),       # Random ints from [50, 150)
            'model__max_depth': randint(3, 15)             # Random ints from [3, 15)
        }

    elif args.model == 'NN':
        param_dist = {
            'model__hidden_layer_sizes': [(64,), (128,), (64, 32)],
            'model__activation': ['relu'],
            'model__alpha': loguniform(1e-5, 1e-2),         # Log-uniform over small regularization values
            'model__learning_rate_init': loguniform(1e-4, 1e-1),  # More realistic continuous range
            'model__solver': ['adam'],
            'model__max_iter': [1000]
        }

    else:
        raise ValueError("Unsupported model type or args configuration")
    
    scoring = { 'r2': 'r2',
                'neg_mae': 'neg_mean_absolute_error',
                'neg_mse': 'neg_mean_squared_error'}

    return RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        refit=refit,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=42
    )
