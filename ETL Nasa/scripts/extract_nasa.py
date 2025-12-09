import json
import os
from pathlib import Path
from datetime import datetime
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
IMAGE_DIR = Path(__file__).resolve().parents[1] / "data" / "images"

RAW_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

def extract_apod_data(date=None):
    """
    Extract NASA APOD data + download image if media type is 'image'
    """
    url = "https://api.nasa.gov/planetary/apod"
    api_key = os.getenv("NASA_KEY", "DEMO_KEY")

    params = {"api_key": api_key}
    if date:
        params["date"] = date

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    # Save JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RAW_DIR / f"nasa_{timestamp}.json"
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Extracted NASA APOD data saved to: {json_path}")

    # Download image if available
    if data.get("media_type") == "image":
        image_url = data.get("hdurl") or data.get("url") 
        if image_url:
            extension = image_url.split(".")[-1].split("?")[0]
            image_path = IMAGE_DIR / f"nasa_{timestamp}.{extension}"

            img_resp = requests.get(image_url)
            if img_resp.status_code == 200:
                image_path.write_bytes(img_resp.content)
                print(f"Downloaded image → {image_path}")
            else:
                print("Failed to download image.")
    else:
        print(" APOD media is not an image (maybe video). No download.")

    return data

if __name__ == "__main__":
    extract_apod_data()