"""Create a reproducible sample dataset and train the house price model."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

FEATURES = ["area", "bedrooms", "age"]
MODEL_PATH = Path(__file__).resolve().with_name("house_price_model.pkl")


def create_sample_dataset():
    """Build a small synthetic dataset with prices expressed in lakhs."""
    areas = np.array([800, 1000, 1200, 1500, 1800, 2000, 2400, 2800, 3200])
    bedrooms = np.array([1, 2, 3, 4, 5])
    ages = np.array([0, 5, 10, 15, 20, 25, 30])

    # Each row represents one combination of area, bedroom count, and age.
    feature_rows = np.array(
        np.meshgrid(areas, bedrooms, ages, indexing="ij")
    ).reshape(3, -1).T
    dataset = pd.DataFrame(feature_rows, columns=FEATURES)
    dataset["price"] = (
        0.025 * dataset["area"]
        + 2.5 * dataset["bedrooms"]
        - 0.55 * dataset["age"]
        + 0.38
    )
    return dataset


def main():
    dataset = create_sample_dataset()
    model = LinearRegression()
    model.fit(dataset[FEATURES], dataset["price"])
    joblib.dump(model, MODEL_PATH)

    print(f"Trained model saved to: {MODEL_PATH}")
    print(f"Training rows: {len(dataset)}")
    print(f"Example prediction: {model.predict(pd.DataFrame([[2000, 3, 5]], columns=FEATURES))[0]:.2f} lakhs")


if __name__ == "__main__":
    main()
