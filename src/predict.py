import argparse
from pathlib import Path

import json
import joblib

import pandas as pd

ROOT_PATH = Path(__file__).parent.parent
MODEL_PATH = ROOT_PATH / "models" / "fraud_pipeline.joblib"
METADATA_PATH = ROOT_PATH / "models" / "metadata.json"

def predict(X, model, metadata):
    threshold = metadata["threshold"]

    X = X.copy()
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    X['Class'] = y_pred
    return X

def load_model():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return model, metadata

def main():
    parser = argparse.ArgumentParser(description="Score transactions with the fraud-detection model.")

    parser.add_argument('--input', required=True, help="Path to a CSV with new transactions.")
    parser.add_argument('--output', default="predictions.csv", help="Path to a CSV with predictions.")

    args = parser.parse_args()

    model, metadata = load_model()

    dataset = pd.read_csv(args.input)

    result = predict(dataset, model, metadata)

    result.to_csv(args.output, index=False)

    print(f"Results written to {args.output}")

if __name__ == "__main__":
    main()