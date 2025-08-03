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

---

## Dataset

The dataset includes detailed records of NYC taxi trips with attributes such as timestamps, coordinates, and passenger count.

* **Target:** `trip_duration` (in seconds)
* **Features include:**

  * `pickup_latitude`, `pickup_longitude`
  * `dropoff_latitude`, `dropoff_longitude`
  * `passenger_count`, `vendor_id`, and more

---

##  EDA & Feature Engineering Highlights

* **Temporal Patterns:**

  * Trip durations peak in summer and during afternoon rush hours.
  * Weekday effects observed: longer durations midweek, shorter on Sundays.

* **Spatial Insights:**

  * Aggregated `latitude_sum` and `longitude_sum` were more robust than raw coordinates.
  * Pickup/dropoff clustering reflects real-world zones (e.g., Midtown vs. JFK).

* **Outlier Removal via IQR (k = 8):**

  * Experimented with multiple thresholds and empirically selected the one yielding the highest R² without sacrificing valid data.

* **Log-Transforming `trip_duration`:**

  * Improved distribution shape and increased correlation with predictors.

* **Engineered Feature Examples:**

  * Haversine `trip_distance`, its square and log
  * `trip_distance × latitude_sum` / `longitude_sum`
  * Temporal flags: `is_night`, `is_weekend`, `hour × is_night`, `month × is_weekend`
  * Behavioral interactions: `vendor_id × passenger_count`, `trip_distance × weekday`

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

For in-depth insights including methodology, feature details, visualizations, and model comparison:

👉 [**Read Full Report (PDF)**](project_report.pdf)

---

## 📂 Project Structure

```
NYC-Taxi-Trip-Duration/
├── notebooks/
│   └── trip_duration.ipynb
├── reports/
│   └── project_report.pdf
├── scripts/
│   ├── train/
│   │   └── trip_duration_train.py
│   ├── test/
│   │   └── trip_duration_test.py
│   ├── utils/
│   │   ├── trip_duration_utils_data.py
│   │   ├── trip_duration_utils_preprocess.py
│   │   ├── trip_duration_utils_eval.py
│   │   └── cli_args.py
│   └── api/
│       └── predict_api.py
└── saved_models/
```

---

## Installation

```bash
pip install -r requirements.txt
```

* Python 3.9+ recommended
* Models and preprocessing steps are version-locked for reproducibility

---

## Tools and Technologies

* Python, pandas, NumPy, datetime
* scikit-learn (preprocessing, Ridge, RandomizedSearchCV)
* XGBoost
* PyTorch (Neural Network + Ensemble)
* matplotlib, seaborn (EDA)
* haversine (geospatial distance)
* Flask / FastAPI (API deployment)

---

## API Deployment

A RESTful API is provided to serve model predictions:

```bash
python scripts/api/predict_api.py
```

* Accepts pickup time, location, and trip features
* Returns predicted `trip_duration`
* See `api/predict_api.py` for usage and sample request format

---

## 📬 Contact

**Author:** Taher Alabbar
**Email:** [t.alabbar.ca@gmail.com](mailto:t.alabbar.ca@gmail.com)
[**LinkedIn**](https://www.linkedin.com/in/taher-alabbar/)

Let’s connect if you're interested in ML for real-world problems, interpretable modeling, or data-driven feature design!
