# 4. load_weather.py
 
import os
import time
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
 
# Initialize the Supabase client
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
 
 
def load_to_supabase():
    # Load cleaned CSV
    csv_path = "../data/staged/weather_cleaned.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")
 
    df = pd.read_csv(csv_path)
 
    # Convert timestamps → strings
    df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    df["extracted_at"] = pd.to_datetime(df["extracted_at"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
 
    batch_size = 20
 
    # Batch Insert Loop
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size]
 
        # Replace NaN → None
        batch = batch_df.where(pd.notnull(batch_df), None).to_dict("records")
 
        # Build VALUES for SQL Insert
        values = [
            (
                f"('{r['time']}', "
                f"{r.get('temperature_c', 'NULL')}, "
                f"{r.get('humidity_percent', 'NULL')}, "
                f"{r.get('wind_speed_kmph', 'NULL')}, "
                f"'{r.get('city', 'Hyderabad')}', "
                f"'{r['extracted_at']}')"
            )
            for r in batch
        ]
 
        insert_sql = (
            "INSERT INTO weather_data "
            "(time, temperature_c, humidity_percent, wind_speed_kmph, city, extracted_at) "
            f"VALUES {','.join(values)};"
        )
 
        # Execute SQL via RPC
        supabase.rpc("execute_sql", {"query": insert_sql}).execute()
 
        print(f"Inserted rows {i + 1} → {min(i + batch_size, len(df))}")
        time.sleep(0.5)
 
    print("Finished loading weather data.")
 
 
if __name__ == "__main__":
    load_to_supabase()