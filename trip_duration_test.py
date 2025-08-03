# Standard library
import pickle

# Local application imports
from cli_args import parse_args
from trip_duration_utils_data import prepare_data
from trip_duration_utils_eval import evaluate

def load_model(model_path="model.pkl"):
    """
    Load a trained model or pipeline from a .pkl file.

    Parameters:
    -----------
    model_path : str
        Path to the saved pickle file (default is 'model.pkl').

    Returns:
    --------
    model : object
        The deserialized model or pipeline.
    """
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            print(f"Model loaded successfully from '{model_path}'")
            return model
    except Exception as e:
        print(f"Failed to load model from '{model_path}': {e}")
        return None
    

if __name__=='__main__':
    args = parse_args()
    X_train, X_test, y_train, y_test = prepare_data(args)

    model_bundle = load_model(args.load_dir)
    model = model_bundle['model_pipeline']

    evaluate(model, X_train, y_train, 'Training')
    evaluate(model, X_test, y_test, 'Testing')