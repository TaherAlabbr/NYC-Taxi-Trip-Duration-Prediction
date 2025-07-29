# Standard library
from pathlib import Path
import pickle

# Third-party libraries
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# Local application imports
from trip_duration_utils_eval import evaluate
from trip_duration_utils_data import *
from cli_args import get_args

def build_model(args, preprocessor):
    """
    Build a regression pipeline using the selected model and preprocessing pipeline.

    Parameters:
        args (Namespace): Parsed command-line arguments containing model configuration.
        preprocessor (ColumnTransformer): The preprocessing pipeline.

    Returns:
        pipeline (Pipeline): A scikit-learn Pipeline with preprocessing and model.
    """
    if args.model == 'RR':
        model = Ridge(
            alpha=args.ridge_alpha,
            max_iter=args.ridge_max_iter
        )
    
    elif args.model == 'XGB':
        model = XGBRegressor(
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            objective='reg:squarederror',
            random_state=42
        )

    elif args.model == 'NN':
        model = MLPRegressor(
            hidden_layer_sizes=args.nn_hidden_layers,
            max_iter=args.nn_max_iter,
            activation=args.nn_activation,
            learning_rate_init=args.nn_lr,
            random_state=42
        )

    else:
        raise ValueError(f"Unsupported model choice: {args.model}")

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', model)
    ])

    return pipeline


def save_model_bundle(pipeline, target_column, args):
    """
    Saves the trained pipeline and related metadata to a pickle file.

    Parameters:
    -----------
    pipeline : sklearn.pipeline.Pipeline
        The full pipeline containing preprocessing steps and the trained model.
    target_column : str
        The name of the target variable used during training.
    args : argparse.Namespace
        Command-line arguments, including the output file path.

    Returns:
    --------
    None
    """

    filename = Path(args.save_dir) / f"{args.model_name}.pkl"
    # Create directory if it doesn't exist
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    model_bundle = {
        'model_pipeline': pipeline,
        'target_column': target_column
    }

    with open(filename, 'wb') as file:
        pickle.dump(model_bundle, file)

    print(f"\nModel saved as '{filename}'")


if __name__=='__main__':
    args = get_args()
    X_train, X_test, y_train, y_test = prepare_data(args)

    numeric_features, categorical_features = identify_feature_types(X_train, args)
    preprocessor = build_preprocessor(args.scaling, numeric_features, categorical_features)

    model = build_model(args, preprocessor)
    model.fit(X_train, y_train)

    evaluate(model, X_train, y_train, 'Training')
    evaluate(model, X_test, y_test, 'Testing')

    save_model_bundle(model, 'log_trip_duration', args)
