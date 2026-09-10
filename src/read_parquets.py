"""
Use this file to be able to read parquet files fownloaded from plant leaf
"""

import pandas as pd
import os 
from pathlib import Path
from PIL import Image
import io


# grab path to data folder
path_to_data = Path(__file__).parent.parent / "data" / "PlantLeafLet" / "data"
path_to_csv = Path(__file__).parent.parent / "data" / "PlantLeafLet" / "csv"
path_to_images = Path(__file__).parent.parent / "data" / "PlantLeafLet" / "images"

if not os.path.exists(path_to_images):
    os.makedirs(path_to_images)

if not os.path.exists(path_to_csv):
    os.makedirs(path_to_csv)
for file in os.listdir(path_to_data):
    file_name = file.split(".")[0]
    #read the parquet file into a data frame
    df = pd.read_parquet(path_to_data / file)

    for index, row in df.iterrows():
        image = Image.open(io.BytesIO(row['image']['bytes']))
        image.save(path_to_images / f"{file_name}_{index}.png")
        df.at[index, 'image'] = f"{file_name}_{index}.png"


    
    df.to_csv(path_to_csv / f"{file_name}.csv", index=False) 

