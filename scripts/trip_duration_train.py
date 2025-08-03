# Standard library
from pathlib import Path
import pickle
import pandas as pd

# Third-party libraries
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint

# Local application imports
from trip_duration_utils_eval import evaluate
from trip_duration_utils_data import *
from cli_args import parse_args


def build_model(args, preprocessor):
    """
    Build a regression pipeline using the selected model and preprocessing pipeline.

    Parameters:
        args (Namespace): Parsed command-line arguments containing model configuration.
        preprocessor (ColumnTransformer): The preprocessing pipeline.

    Returns:
        pipeline (Pipeline): A scikit-learn Pipeline with preprocessing and model.
    """
    RANDOM = 42
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
            random_state=RANDOM
        )

    elif args.model == 'NN':
        model = MLPRegressor(
            hidden_layer_sizes=args.nn_hidden_layers,
            max_iter=args.nn_max_iter,
            activation=args.nn_activation,
            learning_rate_init=args.nn_lr,
            random_state= RANDOM
        )

    else:
        raise ValueError(f"Unsupported model choice: {args.model}")

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', model)
    ])

    return pipeline

def build_stacked_model(args, preprocessor):
    random_state = 42

    # Base learners
    base_estimators = [
        ('ridge', Ridge(
            alpha=args.ridge_alpha,
            max_iter=args.ridge_max_iter
        )),
        ('xgb', XGBRegressor(
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            verbosity=0,
            random_state=random_state
        ))
    ]

    # Meta-model
    meta_model = MLPRegressor(
        hidden_layer_sizes=args.nn_hidden_layers,
        max_iter=args.nn_max_iter,
        random_state=random_state
    )

    # Stacking ensemble
    stacking_model = StackingRegressor(
        estimators=base_estimators,
        final_estimator=meta_model,
        cv=5,
        passthrough=False,
        n_jobs=-1
    )

    return Pipeline([
        ('preprocessor', preprocessor),
        ('stacking', stacking_model)
    ])
    
def save_model_bundle(pipeline, iqr_bd,target_column, args):
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
        'target_column': target_column,
        'iqr_boundaries': iqr_bd,
    }

    with open(filename, 'wb') as file:
        pickle.dump(model_bundle, file)

    print(f"\nModel saved as '{filename}'")


if __name__ == '__main__':
    args = parse_args()

    X_train, X_test, y_train, y_test, iqr_bd = prepare_data(args)

    numeric_features, categorical_features = identify_feature_types(X_train, args)
    preprocessor = build_preprocessor(args.scaling, numeric_features, categorical_features)
    print('-' * 40)

    if args.use_stacking:
        print("🔁 Using stacked model with Ridge ,XGBoost and Neural Network")
        model = build_stacked_model(args, preprocessor)

    else:
        model = build_model(args, preprocessor)
        
    if args.random_search:
       random_search = setup_random_search(model ,args, cv=3, n_iter=10)
       random_search.fit(X_train, y_train)
       model = random_search.best_estimator_
       print('Best params of the model:', random_search.best_params_)
    else:
        model.fit(X_train,y_train)

        
    if args.random_search:
        cv_results = pd.DataFrame(random_search.cv_results_)
        print('Top 5 score:')
        print(cv_results[['mean_test_r2', 'mean_test_neg_mae', 'mean_test_neg_mse']].sort_values(by='mean_test_r2', ascending=False).head(5))


    evaluate(model, X_train, y_train, 'Training')
    evaluate(model, X_test, y_test, 'Testing')

    save_model_bundle(model, iqr_bd, 'log_trip_duration', args)
