"""This script will pull more images down from the GBIF dataset"""
import requests
from pathlib import Path
import json
import logging
import time
import os

USDA_JSON_PATH = Path(__file__).parent.parent.parent / "data" / "EOI_Data_Pulls" / "KB"

IMAGES_ROOT = Path(__file__).parent.parent.parent / "data" / "GBIF_Data_Pulls" / "images"

MAX_IMAGES = 120
MAX_OFFSET = 500
REQUEST_TIMEOUT = 5
REQUEST_DELAY = 0.3

GBIF_NAME_URL = "https://api.gbif.org/v1/species/match"
GBIF_OCCURRENCE_URL = "https://api.gbif.org/v1/occurrence/search"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
def get_taxon_key(scientific_name: str) -> int:

    r = requests.get(GBIF_NAME_URL, params={"name": scientific_name}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()

    if data.get("matchType") == "NONE" or "usageKey" not in data:
        raise ValueError(f"NO GBIF MATCH")

    return data['usageKey']

def is_good_license(license_str : str) -> bool:

    if not license_str:
        return False
    license_str = license_str.lower()

    if "publicdomain/zero" in license_str or 'cc0' in license_str:
        return True
    if "licenses/by/" in license_str and "nc" not in license_str and "nd" not in license_str:
        return True
    return False

def fetch_images(taxon_key: int):
    images = []
    offset = 0
    while len(images) < MAX_IMAGES and offset < MAX_OFFSET:
        params = {
            "taxonKey" : taxon_key,
            "mediaType" : "StillImage",
            "basisOfRecord": "HUMAN_OBSERVATION",
            "limit": MAX_IMAGES,
            "offset" : offset     
        }
        r = requests.get(GBIF_OCCURRENCE_URL, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
    
        if not results:
            break
    
        for record in results:
            record_license = record.get("license", "")
            for media in record.get("media", []):
                if media.get("type") != "StillImage":
                    continue
                url = media.get("identifier")
                if not url:
                    continue
    
                media_license = media.get("license", "") or record_license
                if not is_good_license(media_license):
                    continue
    
                images.append({"url": url, "license": media_license, "occurrence_key": record.get("key")})
    
                if len(images) >= MAX_IMAGES:
                    break
            if len(images) >= MAX_IMAGES:
                break
    
        if data.get("endOfRecords", True):
            break

        logger.info(f"page offset={offset}, {len(images)} useable images have been found")
        offset += MAX_IMAGES
        time.sleep(REQUEST_DELAY)
    if offset >= MAX_OFFSET:
        logger.info(f"max page offset has been reached, terminating")
    return images

def download_image(url: str ,dest_path:Path):
    r = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
    r.raise_for_status()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, 'wb') as f:
        f.write(r.content)

def get_extenstion(url: str) -> str:
    suffix = Path(url.split("?")[0]).suffix
    return suffix if suffix else ".jpg"

def load_plant(record: dict, failed_records: list):
    symbol = record.get("symbol")
    scientific_name = record.get("scientific_name")
    download = 0
    plant_dir = IMAGES_ROOT / symbol 
    if os.path.exists(plant_dir):
        return
    try:
        taxon_key = get_taxon_key(scientific_name=scientific_name)
    except(requests.RequestException, ValueError) as e:
        logger.warning(f"[{symbol}] taxonID search has failed {e}")
        failed_records.append({
            "symbol": symbol,
            "scientific_name": scientific_name
        })
        return

    time.sleep(REQUEST_DELAY)

    try:
        images = fetch_images(taxon_key=taxon_key)
    except requests.RequestException as e:
         logger.warning(f"[{symbol}] occurence search failed: {e}")
         failed_records.append({
                    "symbol": symbol,
                    "scientific_name": scientific_name
                })
         return


    for i, img in enumerate(images):
        ext = get_extenstion(img['url'])
        dest = plant_dir / f"{symbol}_{i}{ext}"
        try:
            download_image(img["url"], dest)
            download += 1
        except (requests.RequestException, OSError) as e:
            failed_records.append({
                                "symbol": symbol,
                                "scientific_name": scientific_name
                            })

        time.sleep(REQUEST_DELAY)
        logger.info(f"[{symbol}] downloaded {download} images")

def main():
    IMAGES_ROOT.mkdir(parents=True, exist_ok=True)

    failed_records = []
    json_files = sorted(p for p in USDA_JSON_PATH.iterdir() if p.is_file())

    logger.info(f"Found {len(json_files)}  plant files in {USDA_JSON_PATH}")
    count = 0
    for idx, path in enumerate(json_files, 1):
        try:
            with open(path, "r", encoding="utf-8") as f:
                record = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            continue

        symbol = record.get("symbol", path.stem)
        logger.info(f"[{idx}/{len(json_files)}] processing {symbol}")

        load_plant(record=record, failed_records=failed_records)

        

        



#cycle through collection of USDA plant pulls




if __name__ == "__main__":
    main()
