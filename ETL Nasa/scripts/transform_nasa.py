import pandas as pd
import json
import glob
import os

def transform_apod_data():
    # Ensure staged folder exists
    os.makedirs("../data/staged", exist_ok=True)

    # Get most recent raw APOD file
    files = glob.glob("../data/raw/nasa_*.json")
    if not files:
        raise FileNotFoundError("No NASA raw files found in ../data/raw/")
    latest_file = sorted(files)[-1]

    with open(latest_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if isinstance(data, list):
        data = data[0]

    # Build DataFrame with a single row
    df = pd.DataFrame([{
        "date":            data.get("date"),
        "title":           data.get("title"),
        "explanation":     data.get("explanation"),
        "media_type":      data.get("media_type"),
        "url":             data.get("url"),
        "hdurl":           data.get("hdurl"),
        "service_version": data.get("service_version"),
    }])

    df["extracted_at"] = pd.Timestamp.now()

    output_path = "../data/staged/nasa_cleaned.csv"
    df.to_csv(output_path, index=False)
    print(f"Transformed {len(df)} NASA record(s) saved to: {output_path}")
    return df

if __name__ == "__main__":
    transform_apod_data()