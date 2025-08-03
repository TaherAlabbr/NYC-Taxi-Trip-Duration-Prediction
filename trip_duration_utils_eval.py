from sklearn.metrics import r2_score, root_mean_squared_error

def evaluate(model, X, y, data_name):
    """
    Evaluate a regression model by computing and printing R² and RMSE metrics.

    Parameters:
    -----------
    model : estimator object
        A trained regression model or pipeline implementing the `predict` method.

    X : array-like or pd.DataFrame
        Feature matrix used for prediction.

    y : array-like
        True target values corresponding to X.

    data_name : str
        Label identifying the dataset (e.g., 'Train', 'Test') used in the output display.
    """

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = root_mean_squared_error(y, y_pred)

    print(f"{data_name} Dataset → R² = {r2:.3f} | RMSE = {rmse:.3f}")