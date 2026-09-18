import src.data_preprocessing.image_preprocessing as ip
import os
from pathlib import Path
import PIL
import matplotlib.pyplot as plt
import cv2

PATH_TO_GBIF = Path(__file__).parent.parent.parent / 'data' / 'GBIF_Data_Pulls' /'images'
PATH_TO_CONVERTED_GBIF = Path(__file__).parent.parent.parent / 'data' / 'GBIF_Pre_Processed' 

##Convert to UINT-8
#Loop through folders within Path

for folder in PATH_TO_GBIF.iterdir():

    output_folder = PATH_TO_CONVERTED_GBIF / folder.name
    output_folder.mkdir(parents=True, exist_ok=True)
    
    #Loop through images in the folder
    for jpg in folder.iterdir():

        #load image
        #img = ip.load_image(jpg)
        save_path = output_folder / jpg.name

        if save_path.exists():
            continue
        #resize

        try:
            img = ip.preprocess_image(jpg)
            ip.save_image(img, save_path=save_path)
        except cv2.error as e:
             print(f"Error Converting image {save_path}: {e}")
        except Exception as e:
            print(f"Error Converting image {save_path}: {e}")
        

