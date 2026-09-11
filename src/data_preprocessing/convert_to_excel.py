import pandas as pd
import os
from pathlib import Path
import requests
import time
import tarfile
"""
If you have not downloaded the USDA plant data, download that first and set up the file structure as follows:
data/
    EOI_Data_Pulls/
        taxon.xlsx
From the data set you only need the taxon file for this script to work.
https://zenodo.org/records/18945687 -- Link to usda images download
https://zenodo.org/records/20646743 -- link to usda Plants Structured data download
"""
##First Check that data exists in proper location
def extract_USDA_data():
    #Path for EOL data
    path_to_eol_data = Path(__file__).parent.parent.parent / "data" / "EOI_Data_Pulls" 

    if not os.path.exists(os.path.join(path_to_eol_data, "images")):
        images_path = path_to_eol_data / "images"
        os.mkdir(images_path)

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

def pull_images(plant_name: str, image_uri: list):
    path_to_eol_images = Path(__file__).parent.parent.parent / "data" / "EOI_Data_Pulls" / "images"

    #make a new path for the new incoming plant
    os.mkdir(os.join(path_to_eol_images, plant_name))



def pull_docs(plant_name: str, doc_uri: list):
    pass

def pull_text_data(plant_name: str):
    

if __name__ == '__main__':
    extract_USDA_data()
    pull_images()


