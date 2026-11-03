from pathlib import Path
import json
import pandas as pd
import src.data_preprocessing.text_preprocessing as text_pre
import src.config as config

def generate_GBIF_CSV():
    #Get the path for the Knowledge Base
    USDA_JSON_PATH = Path(__file__).parent.parent.parent / "data" / "EOI_Data_Pulls" / "KB"
    
    #Get the Path for the Pre_Processed Images
    PATH_TO_CONVERTED_GBIF = Path(__file__).parent.parent.parent / 'data' / 'GBIF_Pre_Processed' 
    
    MAIN_DATA = pd.DataFrame(columns=["image_id", "image_path", "symbol", "group", "family", "genus", "crop", "disease", "category", "severity"])

    #Loop Through each and assign at random an image to Test, an image to Train, and an image to Validation
    #70% of images to Train
    #10% to validation
    #20% to Test
    #Assuming all images are healthy plants ---- Might need to revist this and add some type of CV check for ill plants?
    
    for folder in PATH_TO_CONVERTED_GBIF.iterdir():
    
        #plant abrev
        symbol = folder.name
        #json path
        kb_path = USDA_JSON_PATH / symbol
        try:
            with open(kb_path, "r", encoding="utf-8") as f:
                record = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            continue
        for file in folder.iterdir():
        
            img_id = file.name.split(".")[0]
            img_path = "/".join(file.parts[-4:])
            crop = record.get('scientific_name')
            family = record.get('family')
            genus = record.get('genus')
            symbol = record.get('symbol')
            group = record.get('group')
            ##Revist this -- Is there a way to diagnose the plants or generate a bool for disease presence and severity? -- Would unknown be better 
            disease = 'healthy'
            category = 'healthy'
            severity = 'healthy'
            MAIN_DATA.loc[len(MAIN_DATA)] = [img_id, img_path, symbol, group, family, genus, crop, disease, category, severity]
    
    project_root = Path(__file__).resolve().parent.parent.parent
    project_root = project_root / "data" / "GBIF"/ "GBIF_Data_set.csv"
    
    
    
    MAIN_DATA.to_csv(project_root, index=False)

def load_vqa_dataset():
    config_vqa = config.Data.PlantExpertVQA
    files = {"TRAIN": config_vqa.TRAIN_FILE,
             "TEST": config_vqa.TEST_FILE,
             "VALID": config_vqa.VALIDATION_FILE}
    datasets = {}
    for name, file_path in files.items():
        data = text_pre.load_csv(file_path)
        text_pre.preprocess_dataframe(
            data,
            config_vqa.TEXT_COLUMNS,
            config_vqa.COLUMNS_TO_REMOVE,
            config_vqa.NA_FILL,
            ["image_path"],
            config_vqa.ROOT,
        )
        datasets[name] = data

    return datasets["TRAIN"], datasets["TEST"], datasets["VALID"]
    
    
    