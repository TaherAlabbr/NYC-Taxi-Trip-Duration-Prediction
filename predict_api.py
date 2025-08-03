import sys
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
import pandas as pd

scripts_path = os.path.abspath('../scripts/')
sys.path.append(scripts_path)

# Import from scripts/
from trip_duration_utils_data import *
from trip_duration_test import load_model
from trip_duration_utils_preprocess import *

app = FastAPI()

model_bundle = load_model('../saved_models/final_xgb.pkl')
pipeline = model_bundle['model_pipeline']


class TripInput(BaseModel):
    pickup_datetime: datetime
    vendor_id: int = Field(ge=1, le=2)
    passenger_count: int = Field(ge=1, le=6)
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float


@app.post('/predict/')
def predict(trip_data: TripInput):
    try:
        data = pd.DataFrame([trip_data.model_dump()])

        data = extract_datetime(data)

        data = engineer_features(data)

        columns_to_drop = [
            'pickup_datetime', 'pickup_latitude',
            'dropoff_latitude', 'dropoff_longitude', 'pickup_longitude'
        ]

        data.drop(columns=columns_to_drop, inplace=True, errors='ignore')

        prediction = pipeline.predict(data)

        return {"prediction": round(float(np.expm1(prediction[0])/60), 2)}
    
    except Exception as e:
        raise HTTPException(status_code = 400, detail=str(e))