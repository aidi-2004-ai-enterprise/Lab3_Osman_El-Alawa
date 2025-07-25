import seaborn as sns
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import os
import joblib

def load_and_preprocess_data() -> tuple:
    """Load and preprocess the penguins dataset."""
    # Load dataset
    df: pd.DataFrame = sns.load_dataset('penguins')
    print("Original columns:", df.columns.tolist())  # Debug original columns
    print("NaN counts per column:", df.isna().sum())  # Debug NaN counts

    # Drop rows with NaN only in specific columns if needed, but retain year
    df = df.dropna(subset=['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex'])
    print("Columns after targeted dropna:", df.columns.tolist())

    # Convert categorical columns to lowercase for consistency
    df['island'] = df['island'].str.lower()
    df['sex'] = df['sex'].str.lower()

    # Separate features and target
    X: pd.DataFrame = df.drop('species', axis=1)
    y: pd.Series = df['species']

    # Define categories matching the standardized data
    island_categories: list = ['torgersen', 'biscoe', 'dream']
    sex_categories: list = ['male', 'female']

    # Debugging: Print column info
    print("Columns in X:", X.columns.tolist())
    print("Unique values in island:", X['island'].unique())
    print("Unique values in sex:", X['sex'].unique())

    # Apply one-hot encoding
    column_transformer = ColumnTransformer(
        transformers=[
            ('island', OneHotEncoder(categories=[island_categories], sparse_output=False), ['island']),
            ('sex', OneHotEncoder(categories=[sex_categories], sparse_output=False), ['sex'])
        ],
        remainder='passthrough'
    )
    X_encoded = column_transformer.fit_transform(X)
    print("Number of features after encoding:", X_encoded.shape[1])

    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Save transformer and encoder for reuse
    os.makedirs('app/data', exist_ok=True)
    joblib.dump(column_transformer, 'app/data/column_transformer.joblib')
    joblib.dump(label_encoder, 'app/data/label_encoder.joblib')

    return X_encoded, y_encoded, column_transformer, label_encoder

def train_model(X_encoded, y_encoded) -> xgb.XGBClassifier:
    """Train an XGBoost model with the preprocessed data."""
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    # Initialize and train model
    model = xgb.XGBClassifier(max_depth=3, n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    test_f1 = f1_score(y_test, y_test_pred, average='macro')

    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Train F1-score: {train_f1:.4f}")
    print(f"Test F1-score: {test_f1:.4f}")

    return model

if __name__ == "__main__":
    # Load and preprocess data
    X_encoded, y_encoded, column_transformer, label_encoder = load_and_preprocess_data()

    # Train and save model
    model = train_model(X_encoded, y_encoded)
    model.save_model('app/data/model.json')
    print("Model saved to 'app/data/model.json'")

