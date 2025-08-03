import numpy as np
import pandas as pd

def extract_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract datetime-related features from the 'pickup_datetime' column.

    This function converts the 'pickup_datetime' column to datetime format and extracts
    additional features such as hour, minute, day, month, and weekday.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a 'pickup_datetime' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with added datetime features.
    """
    df = df.copy()
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['minute'] = df['pickup_datetime'].dt.minute
    df['day'] = df['pickup_datetime'].dt.day
    df['month'] = df['pickup_datetime'].dt.month
    df['weekday'] = df['pickup_datetime'].dt.weekday
    return df


def remove_outliers_iqr(df: pd.DataFrame, columns: list, multiplier=8, iqr_boundaries=None)  -> pd.DataFrame:
    """
    Remove outliers from specified columns using the IQR (Interquartile Range) method.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list
        List of column names to apply IQR filtering to.
    multiplier : float, optional
        IQR multiplier to determine the outlier boundaries (default is 1.5).
    iqr_boundaries : list of tuple, optional
        Precomputed IQR boundaries [(lower, upper), ...] for each column. If not provided,
        the function will compute them from the data.

    Returns
    -------
    pd.DataFrame
        DataFrame with outliers removed.
    list
        List of IQR boundaries used for each column.
    """
    df = df.copy()
    if not iqr_boundaries:
        iqr_boundaries = []
        for col in columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr
            iqr_boundaries.append((lower_bound, upper_bound))
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    else:
        for iqr, col in zip(iqr_boundaries, columns):
            df = df[(df[col] >= iqr[0]) & (df[col] <= iqr[1])]
    return df, iqr_boundaries


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new features from spatial and temporal data.

    This includes:
    - Calculating haversine distance
    - Creating binary flags (e.g. night, weekend)
    - Interaction features (e.g. hour x is_night)
    - Seasonal flags

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with necessary columns including GPS coordinates and datetime parts.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional engineered features.
    """
    df = df.copy()

    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in kilometers
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    # Spatial features
    df['trip_distance'] = haversine_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    df['trip_distance_squared'] = df['trip_distance'] ** 2
    df['log_trip_distance'] = np.log1p(df['trip_distance'])

    # Temporal binary features
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] >= 20) | (df['hour'] <= 5)).astype(int)
    df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
    df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 21)).astype(int)

    # Seasonal flags
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)

    # Interaction terms
    df['night_and_weekend'] = df['is_night'] * df['is_weekend']
    df['weekend_times_month'] = df['is_weekend'] * df['month']
    df['hour_x_is_night'] = df['hour'] * df['is_night']
    df['hour_times_weekend'] = df['hour'] * df['is_weekend']
    df['month_times_weekend'] = df['month'] * df['is_weekend']
    df['latitude_sum'] = df['pickup_latitude'] + df['dropoff_latitude']
    df['longitude_sum'] = df['pickup_longitude'] + df['dropoff_longitude']
    df['distance_times_latitude_sum'] = df['trip_distance'] * df['latitude_sum']
    df['distance_times_longitude_sum'] = df['trip_distance'] * df['longitude_sum']
    df['vendor_passenger_interaction'] = df['vendor_id'] * df['passenger_count']
    df['hour_x_passenger_count'] = df['hour'] * df['passenger_count']
    df['trip_distance_x_passenger_count'] = df['trip_distance'] * df['passenger_count']
    df['trip_distance_x_weekday'] = df['trip_distance'] * df['weekday']

    return df


def preprocessing_pipeline(df: pd.DataFrame, iqr_multiplier:float, iqr_bound=None, dataset_name: str = "dataset") -> pd.DataFrame:
    """
    Complete preprocessing pipeline for the NYC Taxi Trip Duration dataset.

    Steps:
    - Remove duplicates
    - Extract datetime features
    - Remove outliers from spatial and duration columns
    - Generate spatial, temporal, and interaction features
    - Drop unused or irrelevant columns
    - Apply log transform to the 'trip_duration' column

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame containing trip data.
    iqr_bound : list of tuple, optional
        IQR boundaries to reuse for consistent outlier filtering between datasets.
    dataset_name : str
        Name of the dataset (used for logging).

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame ready for modeling.
    list
        IQR boundaries computed or reused for filtering.
    """
    print(f"Starting preprocessing for '{dataset_name}', shape: {df.shape}")
    
    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape}")
    
    df = extract_datetime(df)
    
    df, iqr_boundaries = remove_outliers_iqr(
        df,
        columns=['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'trip_duration'],
        iqr_boundaries=iqr_bound,
        multiplier= iqr_multiplier
    )
    print(f"After outlier removal: {df.shape}")
    
    df = engineer_features(df)

    columns_to_drop = [
        'id', 'pickup_datetime', 'pickup_latitude',
        'dropoff_latitude', 'dropoff_longitude', 'pickup_longitude',
        'store_and_fwd_flag'
    ]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    print(f"After dropping unused columns: {df.shape}")
    
    print("Applying log1p transform to 'trip_duration'...")
    df['log_trip_duration'] = np.log1p(df['trip_duration'])
    df.drop(columns=['trip_duration'], inplace=True)
    
    print(f"Final dataframe shape after preprocessing: {df.shape}")
    print("-" * 40)

    return df, iqr_boundaries

# 'store_and_fwd_flag' useless
# lat_sum gives better performance than indivisual lat values!
# long_sum gives better performance than indivisual lat values!
# tried the slope, which was terrible!
# log transformation with lat,long sum useless!
# 'trip_distance_x_lat_sum' increase 0.015 for the one with long_sum slightly better effect since long_sum has more correlation with the target?!
# trip_distance_x_month' and  'trip_distance_x_weekday' good effect
# lat_sum_x_long_sum', long_sum_squared', lat_sum_squared no effect!
# 'distance_per_day' very slight improvement
# 'vendor_id_x_passenge a very good chanfe of 0.01 did not excpect it, when tried to move the both of the features the model's performance degraded!
# otherwise the value store_and_fwd_flag is littrally useless and i think that b/c the nature of the feature it's binary and also dominated by one class = so no_use of it.
#'one_passenger' very slight effect of 0.0001
# tried is_rush_day and rush weekday degrading in the performance
# i probably have nothing to add more xD
# one passenger flag useless b/c its already dominating the passanger count
# lat sum and lon sum better than diffrence ?
