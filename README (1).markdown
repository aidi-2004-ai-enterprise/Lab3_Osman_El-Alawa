# Penguin Species Prediction API

This project implements a machine learning API to predict penguin species using physical measurements and categorical features (island and sex). The API is built with FastAPI and uses an XGBoost model trained on the penguins dataset from Seaborn.

## Requirements

- Python 3.8 or higher
- Virtual environment with the following dependencies:
  - `seaborn`
  - `pandas`
  - `scikit-learn`
  - `xgboost`
  - `fastapi`
  - `uvicorn`
  - `joblib`

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/aidi-2004-ai-enterprise/Lab3_Osman_El-Alawa
   cd ci-lab-osman
   ```

2. Set up a virtual environment and install dependencies:

   ```
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # or .venv/bin/activate  # On macOS/Linux
   uv sync  # Installs dependencies from pyproject.toml
   ```

3. Train the model:

   ```
   python train.py
   ```

## Usage

### Run the API

Start the server with:

```
uvicorn app.main:app --reload
```

- Access the API at `http://127.0.0.1:8000`.
- View the Swagger UI at `http://127.0.0.1:8000/docs` for interactive documentation.

### API Endpoints

- **GET /**: Returns a welcome message.

  - Response: `{"message": "Welcome to the Penguin Species Prediction API! Visit /docs for the Swagger UI."}`

- **POST /predict**: Predicts penguin species based on input features.

  - **Request Body**:

    ```json
    {
      "bill_length_mm": 39.1,
      "bill_depth_mm": 18.7,
      "flipper_length_mm": 181.0,
      "body_mass_g": 3750.0,
      "sex": "male",
      "island": "Biscoe"
    }
    ```

  - **Response**: `{"species": "<predicted_species>"}` (e.g., `{"species": "Adelie"}`).

## Project Structure

- `train.py`: Trains the XGBoost model and saves it to `app/data/model.json`.
- `app/main.py`: Defines the FastAPI application with the `/predict` endpoint.
- `app/data/`: Stores the trained model (`model.json`), `column_transformer.joblib`, and `label_encoder.joblib`.
- `test_api.py`: Contains a basic test script for the API.
- `README.md`: This documentation file.
- `pyproject.toml`: Dependency and project configuration.

## Testing

Run the included test script to verify the API:

```
python test_api.py
```

This sends a sample prediction request and checks for a valid response.

## Logging

Logs are output to the console for debugging, including model loading, prediction requests, and errors.