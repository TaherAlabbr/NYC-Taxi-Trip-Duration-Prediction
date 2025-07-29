# 🚖 NYC Taxi Trip Duration Prediction: A Feature Engineering-Centric Approach

This project predicts the duration of taxi trips in New York City using supervised machine learning, with a **strong emphasis on feature engineering**. A single, interpretable model—**Ridge Regression (α = 1)**—is used throughout to highlight the impact of each transformation and engineered feature.

---

## 📊 Dataset

The dataset includes detailed information about taxi rides in NYC, such as timestamps, pickup/dropoff coordinates, and passenger count.

* Target: `trip_duration` (in seconds)
* Key features: `pickup_datetime`, `dropoff_datetime`, `passenger_count`, `pickup/dropoff lat/lon`, `vendor_id`, and more

---

## 🧠 Key Highlights

* Applied **log transformation** to the skewed target variable (`trip_duration`) to stabilize variance and improve correlation.
* Performed detailed **EDA** to uncover trends across time, location, and passenger behavior.
* Designed and evaluated **30+ engineered features**, including:

  * Haversine-based `trip_distance` and interaction terms
  * `latitude_sum`, `longitude_sum` to simplify spatial encoding
  * Time-of-day, day-of-week, seasonal flags
* Conducted **IQR-based outlier removal** and tuned strictness parameter `k` using validation R² performance.
* Ensured **data leakage prevention** by computing thresholds from the training set only.
* Achieved R² score of **0.68688** on validation set using only Ridge Regression.

---

## 🧪 Model Performance Overview

| Dataset        | R² Score | RMSE    |
| -------------- | -------- | ------- |
| Training Set   | 0.68638  | 0.43218 |
| Validation Set | 0.68688  | 0.43290 |

---

## 📄 Full Report

The full report offers a comprehensive breakdown of:

* Motivation and objective of the task
* Exploratory data analysis with time and space-based insights
* Detailed reasoning behind every preprocessing and feature engineering step
* Description of IQR outlier tuning and its empirical effect on model performance
* Final evaluation and key findings

👉 [**Review Full Report (PDF)**](project_report.pdf)

---

## 🧰 Tools and Technologies

* Python 3.9+
* scikit-learn (for Ridge Regression, scaling, encoding, metrics)
* numpy, pandas (for data handling and preprocessing)
* matplotlib, seaborn (for EDA visualizations)
* datetime (for temporal feature engineering)
* haversine (custom function for spatial distance calculation)

---

## 📂 Project Structure

```
NYC-Taxi-Duration-Prediction/
│
├── nyc_taxi_analysis.ipynb           # Main notebook with EDA, FE, modeling
├── project_report.pdf                # Full detailed write-up
│
├── data_preprocessing.py            # Functions for IQR filtering, encoding
├── feature_engineering.py           # Spatial and temporal feature construction
├── model_training.py                # Ridge Regression training & evaluation
│
└── plots/                           # EDA and feature visualization outputs
```

---

## 📬 Contact

**Author:** Taher Alabbar
**Email:** [t.alabbar.ca@gmail.com](mailto:t.alabbar.ca@gmail.com)
[**LinkedIn**](https://www.linkedin.com/in/taher-alabbar/)

Feel free to connect, collaborate, or ask any questions!
