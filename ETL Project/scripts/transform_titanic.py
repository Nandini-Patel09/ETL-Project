import os
import pandas as pd
from extract_titanic import extract_data    # make sure this returns raw CSV path

def transform_data(raw_path):
    # 0. Base path setup (same style as your iris ETL)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    staged_dir = os.path.join(base_dir, "data", "staged")
    os.makedirs(staged_dir, exist_ok=True)

    # 1. Load raw Titanic data
    df = pd.read_csv(raw_path)

    # 2. Handle Missing Values
    numeric_cols = ["age", "fare", "sibsp", "parch"]
    categorical_cols = ["sex", "embarked", "embark_town", "class", "who", "deck"]

    # Fill numeric NaNs with median
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical NaNs with mode
    for col in categorical_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # 3. Feature Engineering

    # Family size
    df["family_size"] = df["sibsp"] + df["parch"] + 1

    # Is alone
    df["is_alone"] = (df["family_size"] == 1).astype(int)

    # Age bins
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 12, 18, 35, 60, 90],
        labels=["child", "teen", "young_adult", "adult", "senior"],
        include_lowest=True
    )

    # Fare per person
    df["fare_per_person"] = df["fare"] / df["family_size"]

    # Encode sex (example)
    df["is_male"] = (df["sex"] == "male").astype(int)

    # 4. Drop unnecessary columns (purely optional)
    drop_cols = ["alive", "who", "adult_male", "alone", "class"]
    df.drop(columns=[c for c in drop_cols if c in df.columns],
            inplace=True, errors="ignore")

    # 5. Save transformed file
    staged_path = os.path.join(staged_dir, "titanic_transformed.csv")
    df.to_csv(staged_path, index=False)

    print(f"Data transformed and saved at: {staged_path}")
    return staged_path


if __name__ == "__main__":
    raw_path = extract_data()
    transform_data(raw_path)