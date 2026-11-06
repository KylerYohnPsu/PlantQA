"""This script will pull more images down from the GBIF dataset"""
import requests
from pathlib import Path
import json
import logging
import time
import os

#Intilize constant variables as well as any important data paths
USDA_JSON_PATH = Path(__file__).parent.parent.parent / "data" / "EOI_Data_Pulls" / "KB"
IMAGES_ROOT = Path(__file__).parent.parent.parent / "data" / "GBIF_Data_Pulls" / "images"

#Intilize variables for pulling data down from GBIF site
MAX_IMAGES = 120
MAX_OFFSET = 500
REQUEST_TIMEOUT = 5
REQUEST_DELAY = 0.3

#intilize urls to collect data from
GBIF_NAME_URL = "https://api.gbif.org/v1/species/match"
GBIF_OCCURRENCE_URL = "https://api.gbif.org/v1/occurrence/search"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_taxon_key(scientific_name: str) -> int:
    '''
    Pull down the respective taxon identification key from the GBIF website
    Param: Scientific_Name: String (provided from USDA data)
    return: int -> Unique GBIF identification taxon key
    '''

    #utilize requests and pass a time out to not spam the website
    r = requests.get(GBIF_NAME_URL, params={"name": scientific_name}, timeout=REQUEST_TIMEOUT)
    #if error in process throw error
    r.raise_for_status()
    #read the data in
    data = r.json()
    #if no unique key identified GBIF does not have an images of the plant
    if data.get("matchType") == "NONE" or "usageKey" not in data:
        raise ValueError(f"NO GBIF MATCH")
    #Return the unique GBIF identifier to pull plants
    return data['usageKey']

def is_good_license(license_str : str) -> bool:

    '''
    Check if the license is acceptable for free and fair use (i.e do not pull content that has copyrights)
    param: license_str: str -> String supplied from GBIF image information
    return: bool -> If true license is ok to use
    '''
    #if there is not license string reject it
    if not license_str:
        return False
    license_str = license_str.lower()
    #check if image is in public domain or has a permissive license
    if "publicdomain/zero" in license_str or 'cc0' in license_str:
        return True
    if "licenses/by/" in license_str and "nc" not in license_str and "nd" not in license_str:
        return True
    return False

def fetch_images(taxon_key: int):
    '''
    find the URL paths for each unique plant image on GBIF
    param: Taxon_key: int -> Unique plant identifier utilized by GBIF
    return: list of dictionaries -> Dictionary contains Url, License, and Occurrence Key 
    '''
    images = []
    
    offset = 0
    #If the images is greater than a set Max images or the desired offset then terminate the loop
    while len(images) < MAX_IMAGES and offset < MAX_OFFSET:
        #initilize parameters to pass to the GBIF API
        params = {
            "taxonKey" : taxon_key,
            "mediaType" : "StillImage",
            "basisOfRecord": "HUMAN_OBSERVATION",
            "limit": MAX_IMAGES,
            "offset" : offset     
        }
        #call the URL to grab the image URL list from GBIF
        r = requests.get(GBIF_OCCURRENCE_URL, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        #extract the returned data from the returned json
        results = data.get("results", [])
        #if empty stop the loop
        if not results:
            break
        
        for record in results:
            #grab the license from the record
            record_license = record.get("license", "")
            for media in record.get("media", []):
                #make sure any image in the record is a still image
                if media.get("type") != "StillImage":
                    continue
                #grab the url to the image in the records
                url = media.get("identifier")
                if not url:
                    continue
                #grab the license from the individual media or default to the overall record license
                #Check if license is good/safe to use
                media_license = media.get("license", "") or record_license
                if not is_good_license(media_license):
                    continue
                #after verification append a dictionary of all images to the list
                images.append({"url": url, "license": media_license, "occurrence_key": record.get("key")})
        if data.get("endOfRecords", True):
            break

        #display how many images were found, and how many pages were generated
        logger.info(f"page offset={offset}, {len(images)} useable images have been found")
        offset += MAX_IMAGES
        #use sleep to not spam GBIF server
        time.sleep(REQUEST_DELAY)
    #If offset is reached before max images found, then alert the user
    if offset >= MAX_OFFSET:
        logger.info(f"max page offset has been reached, terminating")
    return images

def download_image(url: str ,dest_path:Path):
    '''
    Download the image from GBIF and save to Data path
    param: Url -> STR,
            dest_path ->str

    '''
    #request the URL
    r = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
    r.raise_for_status()
    #create a folder with the corresponding destination path if it does not exist
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    #write the image to the file
    with open(dest_path, 'wb') as f:
        f.write(r.content)

def get_extension(url: str) -> str:
    '''
    get file extension from the URL
    Param: URL -> STR
    Return: suffix -> Str
    '''
    #split the URL into before the query and after
    #grab the suffix of the url path
    suffix = Path(url.split("?")[0]).suffix
    return suffix if suffix else ".jpg"

def load_plant(record: dict, failed_records: list):
    '''
    take a plant record and extract information to pull the data
    params: Record -> dict : A dictionary containing plant information from the USDA
            failed_Records: list -> a List containing any failed plant downloads
    '''
    #grab symbol key and scientific name key from record dictionary
    symbol = record.get("symbol")
    scientific_name = record.get("scientific_name")
    #initialize total download count to 0
    download = 0
    plant_dir = IMAGES_ROOT / symbol 
    #if the path already exists, this plant was already generated, skip the plant
    if os.path.exists(plant_dir):
        return
    try:
        #get the GBIF Unique taxon key
        taxon_key = get_taxon_key(scientific_name=scientific_name)
    #if failure append to failed records
    except(requests.RequestException, ValueError) as e:
        logger.warning(f"[{symbol}] taxonID search has failed {e}")
        failed_records.append({
            "symbol": symbol,
            "scientific_name": scientific_name
        })
        return
    #implement a sleep to not flood GBIF servers
    time.sleep(REQUEST_DELAY)

    try:
        #try and get a list of image URLS from GBIF
        images = fetch_images(taxon_key=taxon_key)
        #if failure append to failed records
    except requests.RequestException as e:
         logger.warning(f"[{symbol}] occurence search failed: {e}")
         failed_records.append({
                    "symbol": symbol,
                    "scientific_name": scientific_name
                })
         return

    #for each image try to download and save the image
    for i, img in enumerate(images):
        #find the extension from the image url
        ext = get_extension(img['url'])
        #generate a destination save path
        dest = plant_dir / f"{symbol}_{i}{ext}"
        try:
            #try and save and download the image
            download_image(img["url"], dest)
            download += 1
            #append to a failed record
        except (requests.RequestException, OSError) as e:
            failed_records.append({
                                "symbol": symbol,
                                "scientific_name": scientific_name
                            })
        #add a sleep inbetween each download
        time.sleep(REQUEST_DELAY)
        #print to the user how many images were saved
        logger.info(f"[{symbol}] downloaded {download} images")

def main():
    #make the image root if it does not exist
    IMAGES_ROOT.mkdir(parents=True, exist_ok=True)

    failed_records = []
    #sort the USDA_JSON_PATH and grab only files
    json_files = sorted(p for p in USDA_JSON_PATH.iterdir() if p.is_file())

    logger.info(f"Found {len(json_files)}  plant files in {USDA_JSON_PATH}")
    count = 0
    #for the USDA plant in json_files, iterate through and begin pulling data from GBIF dataset
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
