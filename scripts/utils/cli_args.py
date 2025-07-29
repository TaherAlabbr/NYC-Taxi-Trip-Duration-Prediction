import argparse

def parse_hidden_layers(s):
    """
    Parse hidden layer sizes from a string.

    Accepts either:
    - A single integer: "100" → (100,)
    - A comma-separated list: "64,32" → (64, 32)

    Raises:
        argparse.ArgumentTypeError: If the input is not valid.
    """
    try:
        # Handle comma-separated input
        if ',' in s:
            return tuple(map(int, s.split(',')))
        # Handle single integer input
        return (int(s),)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Hidden layers must be an integer or a comma-separated list of integers, e.g., '64,32' or '100'"
        )

def get_args():
    parser = argparse.ArgumentParser('NYC Taxi Trip Duration Predictor')

    parser.add_argument('--train_dir',
                        type=str,
                        metavar='TRAIN_FILE_PATH',
                        default='data/split/train.csv',
                        help='Path to the training dataset CSV file (default: data/split/train.csv)')

    parser.add_argument('--test_dir',
                        type=str,
                        metavar='TEST_FILE_PATH',
                        default='data/split/val.csv',
                        help='Path to the testing dataset CSV file (default: data/split/val.csv)')

    parser.add_argument('--save-dir',
                        type=str,
                        metavar='SAVE_MODEL_DIR',
                        default='saved_models',
                        help='Directory path where the trained model will be saved (default: saved_models)')

    parser.add_argument('--load-dir',
                        type=str,
                        metavar='LOAD_MODEL_PATH',
                        default='saved_models/ridge_model.pkl',
                        help='File path of the saved model to load for testing or inference (default: saved_models/ridge_model.pkl)')

    # File name for the saved model
    parser.add_argument('--model-name',
                        type=str,
                        metavar='MODEL_FILENAME',
                        default='ridge_model.pkl',
                        help='Filename to save the trained model (e.g., ridge_model.pkl).')

    default_categorical_features = [
        'month', 'hour', 'passenger_count', 'weekday', 'minute',
        'weekend_times_month', 'hour_times_weekend',
        'month_times_weekend', 'vendor_passenger_interaction'
        ]

    parser.add_argument(
        '--categorical-features',
        nargs='*',  # zero or more args, separated by space
        default=default_categorical_features,
        help='List of categorical feature names (default provided)'
    )

    parser.add_argument(
        '--scaling',
        type=str,
        metavar='SCALING',
        choices=['robust', 'standard', 'minmax'],
        default='standard',
        help=(
            'Feature scaling strategy:\n'
            '  standard → StandardScaler (zero mean, unit variance)\n'
            '  minmax   → MinMaxScaler (scale features to [0, 1])\n'
            '  robust   → RobustScaler (uses median and IQR, better for outliers)'
        )
    )

    parser.add_argument(
        '--iqr-multiplier',
        type=float,
        metavar='IQR_MULTIPLIER',
        default=8,
        help='Multiplier for IQR when removing outliers (default: 8).'
    )

    parser.add_argument(
        '--model',
        type=str,
        metavar='MODEL',
        choices=['RR', 'XGB', 'NN'],   # Added 'NN' here
        default='RR',
        help=(
            'Select the model to use:\n'
            '  RR  → RidgeRegression\n'
            '  XGB → XGBRegressor\n'
            '  NN  → Neural Network\n'
        )
    )

    # Ridge Regression
    parser.add_argument('--ridge-alpha', type=float, default=1.0,
        help='Regularization strength (alpha) for Ridge Regression (default: 1.0)')
    parser.add_argument('--ridge-max-iter', type=int, default= None,
        help='Maximum number of iterations for Ridge Regression solver (default: None)')

    # XGBoost
    parser.add_argument('--xgb-n-estimators', type=int, default=100,
        help='Number of boosting rounds (trees) for XGBoost (default: 100)')
    parser.add_argument('--xgb-max-depth', type=int, default=6,
        help='Maximum tree depth for base learners in XGBoost (default: 6)')
    parser.add_argument('--xgb-learning-rate', type=float, default=0.1,
        help='Learning rate (eta) for XGBoost (default: 0.1)')
    
    # Neural Network (MLP)
    parser.add_argument('--nn-hidden-layers', type=parse_hidden_layers, default='64,32',
        help='Comma-separated hidden layer sizes for Neural Network (e.g., "64,32")')
    parser.add_argument('--nn-max-iter', type=int, default=300,
        help='Maximum iterations for training Neural Network (default: 300)')
    parser.add_argument('--nn-activation', type=str, default='relu', choices=['relu', 'tanh', 'logistic'],
        help='Activation function for Neural Network (default: relu)')
    parser.add_argument('--nn-lr', type=float, default=0.001,
    help='Initial learning rate for Neural Network (default: 0.001)')
   
    return parser.parse_args()
