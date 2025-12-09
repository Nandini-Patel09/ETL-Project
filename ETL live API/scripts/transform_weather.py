import pandas as pd
import json
import glob
import os

def transform_weather_data():
    os.makedirs("../data/staged",exist_ok=True)

    latest_file=sorted(glob.glob("../data/raw/weather_*.json"))[-1]

    with open(latest_file,"r")as f:
        data=json.load(f)

    hourly=data["hourly"]
    df=pd.DataFrame({
        "time" : hourly["time"],
        "temperature_C" : hourly["temperature_2m"],
        "humidity_percent" : hourly["relative_humidity_2m"],
        "wind_speed_kmph" : hourly["wind_speed_10m"]
    })

    df["city"]="Hyderabad"
    df["extracted_at"]=pd.Timestamp.now()
    output_path="../data/staged/weather_cleaned.csv"
    df.to_csv(output_path,index=False)
    print(f"Transformed {len(df)} weather records saved to : {output_path}")
    return df 

if __name__=="__main__":
    transform_weather_data()