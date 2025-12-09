import json
from pathlib import Path
from datetime import datetime
import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def extract_weather_data(lat=17.3850, lon=78.4867, days=1):
    url = "https://api.nasa.gov/planetary/apod?api_key=JdfeyJGItZnwv19K5trW3PEZtsJurxrtIit3d36V"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "forecast_days": days,
        "timezone": "auto"
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    filename = DATA_DIR / f"weather_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filename.write_text(json.dumps(data, indent=2))

    print(f"Extracted weather data saved to: {filename}")
    return data

if __name__ == "__main__":
    extract_weather_data()
