# NYC Taxi Trip Duration Prediction: From Feature Engineering to Model Deployment

This project predicts the duration of taxi trips in New York City using supervised machine learning, with a **strong emphasis on feature engineering, model comparison, and real-world deployment**. It is divided into two major phases: the first focuses on EDA and interpretable feature creation using Ridge Regression, and the second benchmarks more advanced models—including XGBoost, Neural Networks, and a stacked ensemble—followed by hyperparameter tuning and API deployment.

---

## Project Overview

This project is structured into two main parts:

### • Part I: Feature Engineering and Baseline Modeling

* Extensive **exploratory data analysis (EDA)** was conducted to identify patterns in spatial, temporal, and behavioral variables.
* More than **30 engineered features** were crafted, ranging from distance metrics and spatial interactions to temporal and behavioral indicators.
* A Ridge Regression model (α = 1) was used to **isolate the effect of engineered features** while minimizing the complexity of the model.
* Rigorous **IQR-based outlier removal** was tuned (k = 8) to balance between data cleaning and generalization.
* Log transformation was applied to the target variable (`trip_duration`) to reduce skewness and stabilize model training.

### • Part II: Model Comparison, Hyperparameter Tuning & API

* Benchmarked multiple models including:

  * Ridge Regression
  * XGBoost
  * Neural Network (PyTorch)
  * A **stacked ensemble** using Ridge & XGBoost as base learners and a Neural Network as the meta-learner
* Conducted **RandomizedSearchCV** for hyperparameter tuning (e.g., `max_depth=13`, `n_estimators=137` for XGBoost).
* **XGBoost** outperformed all other models and was selected for production due to its strong generalization and interpretability.
* A lightweight **API** was developed to serve real-time predictions using the best-performing model.
* A comprehensive **command-line interface (CLI)** is provided for flexible training, tuning, and deployment of all models in the project.
  
---

## Dataset

The dataset includes detailed records of NYC taxi trips with attributes such as timestamps, coordinates, and passenger count.

* **Target:** `trip_duration` (in seconds)
* **Features include:**

  * `pickup_latitude`, `pickup_longitude`
  * `dropoff_latitude`, `dropoff_longitude`
  * `passenger_count`, `vendor_id`, and more

---

##  Model Performance Overview

### Phase I – Ridge Regression (α = 1)

| Dataset        | R² Score | RMSE    |
| -------------- | -------- | ------- |
| Training Set   | 0.68638  | 0.43218 |
| Validation Set | 0.68688  | 0.43290 |

### Phase II – Final Model Comparison

| Model            | Val R² | Val RMSE | Comments                        |
| ---------------- | ------ | -------- | ------------------------------- |
| XGBoost          | 0.759  | 0.381    | **Selected for production**     |
| Stacked Ensemble | 0.754  | 0.385    | Good but more complex           |
| Neural Network   | 0.747  | 0.390    | Competitive, less interpretable |
| Ridge Regression | 0.689  | 0.433    | Strong linear baseline          |

### 🎯 Final XGBoost Test Results:

| Dataset  | R² Score | RMSE  |
| -------- | -------- | ----- |
| Test Set | 0.759    | 0.380 |

---

## 📄 Full Report

This report presents a complete machine learning pipeline for NYC taxi trip duration prediction. It includes:

* **Extensive Feature Engineering**: From spatial transformations (e.g., Haversine distance, lat/lon sums) to temporal patterns and interaction effects.
* **Robust Data Preprocessing**: Outlier detection using the IQR method (with an optimal multiplier of *k = 8*), log transformation of the target variable, and strict separation of training and validation sets to prevent data leakage.
* **Model Development and Comparison**: Evaluation of Ridge Regression, XGBoost, Neural Network, and a stacked ensemble—with XGBoost emerging as the top-performing model.
* **Hyperparameter Tuning**: Randomized search was used to identify the best configurations for each model, improving generalization and stability.
* **Final Results**: The best model (XGBoost) achieved an R² score of **0.759** and RMSE of **0.380** on the test set, demonstrating strong predictive performance.

For a detailed breakdown of the methodology, engineered features, data insights, model development, and performance metrics, refer to the full project report:

👉 [**Read Full Report (PDF)**](reports/Project_Report.pdf)

---


## 📂 Project Structure

```
NYC-Taxi-Trip-Duration-Prediction/
├── notebooks/
│   └── trip_duration.ipynb
├── reports/
│   └── project_report.pdf
├── scripts/
│   ├── trip_duration_train.py
│   ├── trip_duration_test.py
│   ├── trip_duration_utils_data.py
│   ├── trip_duration_utils_preprocess.py
│   ├── trip_duration_utils_eval.py
│   └── cli_args.py
├── api/
│   └── predict_api.py
├── saved_models/
│   └── final-xgb.pkl  # not included in repo (31 MB — can be reproduced via CLI)
├── requirements.txt
└── README.md
```

> ⚠️ **Note:** The final trained model (`final-xgb.pkl`, ~31 MB) is **not included in the repository** due to GitHub's file size limitations.  
> However, it is fully **reproducible** using the command-line interface:
>
> ```bash
> python scripts/trip_duration_train.py \
>   --model XGB \
>   --xgb-n-estimators 137 \
>   --xgb-max-depth 13 \
>   --model-name final_xgb
> ```


---
## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/NYC-Taxi-Trip-Duration-Prediction.git
cd NYC-Taxi-Trip-Duration-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Python 3.9 or later is required for full compatibility.
All models and preprocessing steps are version-locked for reproducibility.



---

## Command-Line Interface (CLI)

This project includes a fully configurable command-line interface using `argparse`, enabling flexible training, tuning, and deployment.

### Available Arguments

#### File Paths

| Argument       | Description                                            |
| -------------- | ------------------------------------------------------ |
| `--train_dir`  | Path to the training dataset CSV file                  |
| `--test_dir`   | Path to the testing dataset CSV file                   |
| `--save-dir`   | Directory to save the trained model                    |
| `--load-dir`   | Path to load a saved model for inference               |
| `--model-name` | Name of the saved model file (e.g., `ridge_model.pkl`) |

#### Model Selection

| Argument          | Description                                                               |
| ----------------- | ------------------------------------------------------------------------- |
| `--model`         | Select the model: `RR` (Ridge), `XGB` (XGBoost), or `NN` (Neural Network) |
| `--use-stacking`  | Enable stacked ensemble using Ridge, XGBoost, and Neural Network          |
| `--random-search` | Use `RandomizedSearchCV` for hyperparameter tuning                        |

#### Preprocessing

| Argument                 | Description                                               |
| ------------------------ | --------------------------------------------------------- |
| `--categorical-features` | List of categorical features to one-hot encode            |
| `--scaling`              | Scaling method: `standard`, `robust`, or `minmax`         |
| `--iqr-multiplier`       | IQR threshold multiplier for outlier removal (default: 8) |

#### Ridge Regression Parameters

| Argument           | Description                        |
| ------------------ | ---------------------------------- |
| `--ridge-alpha`    | Regularization strength (α)        |
| `--ridge-max-iter` | Maximum iterations (default: None) |

#### XGBoost Parameters

| Argument              | Description               |
| --------------------- | ------------------------- |
| `--xgb-n-estimators`  | Number of boosting rounds |
| `--xgb-max-depth`     | Maximum tree depth        |
| `--xgb-learning-rate` | Learning rate (eta)       |

#### Neural Network Parameters

| Argument             | Description                                        |
| -------------------- | -------------------------------------------------- |
| `--nn-hidden-layers` | Comma-separated hidden layer sizes (e.g., `64,32`) |
| `--nn-max-iter`      | Maximum training iterations                        |
| `--nn-activation`    | Activation function: `relu`, `tanh`, or `logistic` |
| `--nn-lr`            | Learning rate                                      |

### Examples

Train a Ridge Regression model with default preprocessing:

```bash
python scripts/train/trip_duration_train.py --model RR
```

Train a tuned XGBoost model using RobustScaler and stacking:

```bash
python scripts/train/trip_duration_train.py --model XGB --use-stacking --scaling robust --random-search
```
---

## Tools and Technologies

* Python, pandas, NumPy, datetime
* scikit-learn (preprocessing, Ridge, RandomizedSearchCV)
* XGBoost
* PyTorch (Neural Network + Ensemble)
* matplotlib, seaborn (EDA)
* haversine (geospatial distance)
* FastAPI (API deployment)

---

## API Deployment

A RESTful API is provided to serve model predictions using the final XGBoost model.

### Running the API Locally

```bash
python api/predict_api.py
```

Ensure your `pipeline` object (preprocessing + model) is correctly loaded inside the script.

---

### 📥 Request Schema

The endpoint expects a JSON payload with the following fields:

```json
{
  "pickup_datetime": "2025-01-15T08:45:00",
  "vendor_id": 2,
  "passenger_count": 1,
  "pickup_longitude": -73.985428,
  "pickup_latitude": 40.748817,
  "dropoff_longitude": -73.985130,
  "dropoff_latitude": 40.758896
}
```

* `pickup_datetime` (ISO 8601): e.g., `"2025-01-15T08:45:00"`
* `vendor_id`: 1 or 2
* `passenger_count`: between 1 and 6
* `pickup_latitude`, `pickup_longitude`, `dropoff_latitude`, `dropoff_longitude`: float values for location

---

###  Example Request (using `curl`)

```bash
curl -X POST http://127.0.0.1:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_datetime": "2025-01-15T08:45:00",
    "vendor_id": 2,
    "passenger_count": 1,
    "pickup_longitude": -73.985428,
    "pickup_latitude": 40.748817,
    "dropoff_longitude": -73.985130,
    "dropoff_latitude": 40.758896
}'
```

---

###  Response Format

```json
{
  "prediction": 9.37
}
```

* `prediction`: Estimated trip duration in **minutes**, rounded to two decimal places
* Internally, the model returns a log-transformed prediction which is exponentiated and converted to minutes before being returned.

---

### Error Handling

If the input is invalid (e.g., missing fields or out-of-range values), the API responds with:

```json
{
  "detail": "Error message explaining what went wrong"
}
```

---

## Contact

**Author:** Taher Alabbar
**Email:** [t.alabbar.ca@gmail.com](mailto:t.alabbar.ca@gmail.com)  
[**LinkedIn**](https://www.linkedin.com/in/taher-alabbar/)

Let’s connect if you're interested in ML for real-world problems, interpretable modeling, or data-driven feature design!
