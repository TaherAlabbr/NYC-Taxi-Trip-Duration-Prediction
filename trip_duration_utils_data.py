# Standard library
import pandas as pd

# Third-party libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer

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

    return X_train.values, X_test.values, y_train.values, y_test.values


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