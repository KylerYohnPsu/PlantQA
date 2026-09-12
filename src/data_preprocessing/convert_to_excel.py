import pandas as pd
import os
from pathlib import Path
import requests
import time
import tarfile
from plant_record import plant_record
import random
from urllib.parse import urlparse
"""
If you have not downloaded the USDA plant data, download that first and set up the file structure as follows:
data/
    EOI_Data_Pulls/
        taxon.xlsx
From the data set you only need the taxon file for this script to work.
https://zenodo.org/records/18945687 -- Link to usda images download
https://zenodo.org/records/20646743 -- link to usda Plants Structured data download
"""
#global access
plant_df = None
HEADERS = {'User-Agent': 'Pennsylvania State University Masters Students in artifical intelligence'}
##First Check that data exists in proper location
def extract_USDA_data(path_to_eol_data: str):
    #Path for EOL data
    assert os.path.exists(os.path.join(path_to_eol_data, "usda_plant_traits.tar.gz")), "usda_plant_traits.tar.gz not found in data/EOI_Data_Pulls. Please download from https://zenodo.org/records/18945687 and place in the correct location."
    assert os.path.exists(os.path.join(path_to_eol_data, "usda_plant_images.tar.gz")), "usda_plant_images.tar.gz not found in data/EOI_Data_Pulls. Please download from https://zenodo.org/records/18945687 and place in the correct location."

    if not os.path.exists(os.path.join(path_to_eol_data, "taxon.tab")) or not os.path.exists(os.path.join(path_to_eol_data, "media_resource.tab")) or not os.path.exists(os.path.join(path_to_eol_data, "taxon_images.tab")):
        print("Extracting tar.gz files...")
        #extract the tar.gz files
        with tarfile.open(os.path.join(path_to_eol_data, "usda_plant_traits.tar.gz"), "r:gz") as tar:
            tar.extract("./taxon.tab", path=path_to_eol_data)

        with tarfile.open(os.path.join(path_to_eol_data, "usda_plant_images.tar.gz"), "r:gz") as tar:
            tar.extract("./media_resource.tab", path=path_to_eol_data)

        with tarfile.open(os.path.join(path_to_eol_data, "usda_plant_images.tar.gz"), "r:gz") as tar:
            member = tar.getmember("./taxon.tab")
            member.name = "./taxon_images.tab"
            tar.extract(member, path=path_to_eol_data)


    #read in tab files
    taxon_df = pd.read_csv(os.path.join(path_to_eol_data, "taxon.tab"), sep="\t", low_memory=False)
    media_resource_df = pd.read_csv(os.path.join(path_to_eol_data, "media_resource.tab"), sep="\t", low_memory=False)
    taxon_images_df = pd.read_csv(os.path.join(path_to_eol_data, "taxon_images.tab"), sep="\t", low_memory=False)

    # Only grab species of plants
    taxon_df = taxon_df[taxon_df['taxonRank'] == 'species']
    #Grab important columns from all dfs
    taxon_df = taxon_df[['taxonID', 'scientificName']]
    media_resource_df = media_resource_df[['taxonID', 'accessURI']]
    taxon_images_df = taxon_images_df[taxon_images_df['taxonRank'] == 'species']
    taxon_images_df = taxon_images_df[['taxonID', 'scientificName', 'source']]

    #remove any non int taxonIDs from taxon_images_df
    taxon_images_df = taxon_images_df[taxon_images_df['taxonID'].apply(lambda x: str(x).isdigit())].astype({'taxonID': 'int'})

    #extract the symbol to be able to pull out data to match taxonId on taxon file
    taxon_images_df['symbol'] = taxon_images_df['source'].str.extract(r"symbol=([^&]+)$")


    media_taxon_df = pd.merge(media_resource_df, taxon_images_df, on='taxonID',how='inner')

    #Join media_taxon to taxon_df to get taxonID for USDA website search
    final_scrape_df = pd.merge(media_taxon_df, taxon_df, left_on='symbol', right_on='taxonID',how='inner')

    #save the final df for use later
    final_scrape_df.to_csv(os.path.join(path_to_eol_data, "final_scrape_df.csv"), index=False)
    return final_scrape_df

def pull_images(plant_name: str, image_uri: list, image_save_path: str):
    if not os.path.exists(os.path.join(image_save_path, plant_name)):
        os.mkdir(os.path.join(image_save_path, plant_name))

    counter = 1
    for i in image_uri:
        ext = os.path.splitext(urlparse(i).path)[1] or ".jpg"
        img_path = image_save_path / plant_name / f'{plant_name}_{counter}{ext}'
        try:
            r = requests.get(i)
            content_type = r.headers.get('Content-Type', "")
            if r.status_code == 200 and 'image' in content_type.lower():
                with open(img_path, "wb") as file:
                    file.write(r.content)
                counter += 1
        except requests.exceptions.RequestException as e:
            print(f'Failed to fetch {i}: {e}')
        time.sleep(1 + random.uniform(0,2))



def pull_docs(plant_name: str, plant_uri: list, doc_save_path: str):
     if not os.path.exists(os.path.join(doc_save_path, plant_name)):
            os.mkdir(os.path.join(doc_save_path, plant_name))
     counter = 1

     for i in plant_uri:
         ext = os.path.splitext(urlparse(i).path)[1] or ".pdf"
         pdf_path = doc_save_path / plant_name / f'{plant_name}_{counter}{ext}'
         r = requests.get(f'https://plants.sc.egov.usda.gov/{i}')
         if r.status_code == 200:
             with open(pdf_path, "wb") as file:
                 file.write(r.content)
             counter += 1
         time.sleep(1 + random.uniform(0,2))

def pull_text_data(plant_name : str, kb_path:str):
    '''
    Retrieve data from the USDA plant site
    '''
    try:
        r = requests.get(f"https://plantsservices.sc.egov.usda.gov/api/PlantProfile?symbol={plant_name}", headers=HEADERS)
    except requests.exceptions.RequestException as e:
        print(f"Request Failed for {plant_name}: {e}")
        return None

    if r.status_code != 200 or not r.text.strip():
        print(f"No valid response for {plant_name} (status {r.status_code})")
        return None
    try:
        plant_json = r.json()
    except requests.exceptions.JSONDecodeError:
        print(f"Bad JSON for {plant_name}")
        return None
    pr = plant_record(plant_json)
    plant_kb_path = kb_path / f'{plant_name}'
    pr.save(plant_kb_path)
    time.sleep(1 + random.uniform(0,0.5))
    return pr
    #grab as many images as possible associated with the text
 
def get_completed(kb_path: str) -> set[str]:
    return {p.stem for p in kb_path.iterdir()}


if __name__ == '__main__':
    path_to_eol_data = Path(__file__).parent.parent.parent / "data" / "EOI_Data_Pulls" 
    images_path = path_to_eol_data / "images"
    kb_path = path_to_eol_data/'KB'
    plant_sheet_path = path_to_eol_data/'plant_sheets'
    final_struct_df = Path(__file__).parent.parent.parent / "data" / "EOI_Data_Pulls" / "final_scrape_df.csv"
    
    if not os.path.exists(images_path):
        os.mkdir(images_path)

    if not os.path.exists(kb_path):
        os.mkdir(kb_path)

    if not os.path.exists(plant_sheet_path):
        os.mkdir(plant_sheet_path)

    if os.path.exists(final_struct_df):
        plant_df = pd.read_csv(final_struct_df)
    else:
        plant_df = extract_USDA_data(path_to_eol_data=path_to_eol_data)
    counter = 1
    completed = get_completed(kb_path)
    for plant in plant_df['symbol'].unique():
        if plant in completed:
            continue
        plant_obj = pull_text_data(plant_name=plant, kb_path=kb_path)

        if plant_obj is None:
            continue
        image_list = plant_df.loc[plant_df['symbol'] == plant, 'accessURI'].to_list()
        plant_guide_list = plant_obj.plant_guide_urls
        fact_sheet_list = plant_obj.fact_sheet_urls
        plant_uri_list = plant_guide_list + fact_sheet_list
        pull_images(plant_name=plant, image_uri=image_list, image_save_path=images_path)   

        if plant_obj.Has_documents:
            pull_docs(plant_name=plant, plant_uri=plant_uri_list, doc_save_path=plant_sheet_path)


