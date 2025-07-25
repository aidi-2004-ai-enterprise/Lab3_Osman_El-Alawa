from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
import joblib
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define enums for categorical variables
class Island(str, Enum):
    Torgersen = "Torgersen"
    Biscoe = "Biscoe"
    Dream = "Dream"

class Sex(str, Enum):
    Male = "male"
    Female = "female"

# Define Pydantic model for input validation
class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    # year: int  # Commented out to match training data
    sex: Sex
    island: Island

# Global variables for model and transformers
model: xgb.XGBClassifier = None
column_transformer: ColumnTransformer = None
label_encoder: LabelEncoder = None

@app.on_event("startup")
def load_model() -> None:
    """Load the model and initialize transformers on startup."""
    global model, column_transformer, label_encoder

    # Load the trained model
    model = xgb.XGBClassifier()
    model.load_model('app/data/model.json')
    logger.info("Model loaded successfully from 'app/data/model.json'")

    # Load the saved transformer and encoder
    column_transformer = joblib.load('app/data/column_transformer.joblib')
    label_encoder = joblib.load('app/data/label_encoder.joblib')
    logger.info("Column transformer and label encoder loaded from saved files")

@app.get("/")
def read_root():
    """Return a welcome message for the root endpoint."""
    return {"message": "Welcome to the Penguin Species Prediction API! Visit /docs for the Swagger UI."}

@app.post("/predict")
def predict(features: PenguinFeatures) -> dict:
    """Predict penguin species based on input features."""
    try:
        # Log incoming request
        logger.info(f"Received prediction request: {features.dict()}")

        # Convert input to DataFrame with explicit column order and lowercase categorical values
        input_data = pd.DataFrame([{
            'island': features.island.value.lower(),
            'bill_length_mm': features.bill_length_mm,
            'bill_depth_mm': features.bill_depth_mm,
            'flipper_length_mm': features.flipper_length_mm,
            'body_mass_g': features.body_mass_g,
            'sex': features.sex.value.lower()
        }], columns=['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex'])

        # Apply preprocessing and log shape for debugging
        input_encoded = column_transformer.transform(input_data)
        logger.info(f"Input encoded shape: {input_encoded.shape}")

        # Make prediction
        prediction = model.predict(input_encoded)[0]
        species: str = label_encoder.inverse_transform([prediction])[0]

        logger.info(f"Prediction successful: {species}")
        return {"species": species}
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")