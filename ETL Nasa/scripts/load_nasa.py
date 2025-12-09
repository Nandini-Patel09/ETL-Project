import os
import time
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Supabase client
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def _escape_sql_string(s):
    """
    Escape single quotes for SQL and wrap in quotes.
    If s is None or NaN, returns NULL (without quotes).
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return "NULL"
    return "'" + str(s).replace("'", "''") + "'"

def load_apod_to_supabase():
    csv_path = "../data/staged/nasa_cleaned.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path)

    # Ensure extracted_at is ISO formatted datetime string
    if "extracted_at" in df.columns:
        df["extracted_at"] = pd.to_datetime(df["extracted_at"]).dt.strftime("%Y-%m-%dT%H:%M:%S")

    batch_size = 20

    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i: i + batch_size]
        batch = batch_df.where(pd.notnull(batch_df), None).to_dict("records")

        values = []
        for r in batch:
            date_            = _escape_sql_string(r.get("date"))
            title            = _escape_sql_string(r.get("title"))
            explanation      = _escape_sql_string(r.get("explanation"))
            media_type       = _escape_sql_string(r.get("media_type"))
            url              = _escape_sql_string(r.get("url"))
            hdurl            = _escape_sql_string(r.get("hdurl"))
            service_version  = _escape_sql_string(r.get("service_version"))
            extracted_at     = _escape_sql_string(r.get("extracted_at"))

            values.append(
                f"({date_}, {title}, {explanation}, {media_type}, "
                f"{url}, {hdurl}, {service_version}, {extracted_at})"
            )

        insert_sql = (
            "INSERT INTO nasa_apod "
            "(date, title, explanation, media_type, url, hdurl, service_version, extracted_at) "
            f"VALUES {', '.join(values)};"
        )

        supabase.rpc("execute_sql", {"query": insert_sql}).execute()

        print(f"Inserted rows {i + 1} → {min(i + batch_size, len(df))}")
        time.sleep(0.5)

    print("Finished loading NASA data.")

if __name__ =="__main__" :
    load_apod_to_supabase()