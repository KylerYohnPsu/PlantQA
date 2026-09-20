import numpy as np
from sentence_transformers import SentenceTransformer
import src.util.general as general_utils
from src.util.logger import Logger
from dataclasses import dataclass
from pathlib import Path
from . import text_preprocessing as text_pre
from . import TextEmbedder
import json
import pandas as pd

def embed_USDA_data(root_dir: str, embedder, chunk_method: str="word", chunk_size: int=64, overlap: int=16):
    all_records= []

    plant_sheets_records= embed_USDA_plant_sheets(root_dir, embedder, chunk_method, chunk_size, overlap)
    Logger.debug(f"Num plant sheet records: {len(plant_sheets_records)}")

    json_records= embed_USDA_json_data(root_dir, embedder)
    Logger.debug(f"Num json records: {len(json_records)}")

    all_records.extend(plant_sheets_records)
    all_records.extend(json_records)

    return all_records

def make_USDA_plant_sheets_dataframe(plant_sheet_dir: str):
    plant_sheets_data= load_USDA_plant_sheet_data(plant_sheet_dir)
    frame= pd.DataFrame(plant_sheets_data, columns=["text"])
    return frame

def load_USDA_plant_sheet_data(plant_sheet_dir: str):
    base_dir= Path(plant_sheet_dir)
    if not base_dir.is_dir():
        Logger.error(f"Provided path is not a directory: {base_dir}")
        return None
    
    plant_sheets_data= []

    for subdir in base_dir.iterdir():
        if not subdir.is_dir():
            continue # not a dir, skip

        for file in subdir.iterdir():
            if general_utils.is_pdf(file):
                # load the text data in the pdf
                text= text_pre.load_pdf_text_data(file)
            elif general_utils.is_docx(file):
                # load the text data in the docx file
                text= text_pre.load_docx_text_data(file)
            else:
                # invalid file, skip
                Logger.warning(f"Unexpected type found: {file}")
                continue

            plant_sheets_data.append(text)

    return plant_sheets_data




def embed_USDA_plant_sheets(root_dir: str, embedder, chunk_method: str="word", chunk_size: int=64, overlap: int=16):
    """
    Parse every single USDA plant sheet.  Chunk the sheets and embed them into vectors.
    Make a record of each chunk that tracks the chunk's text, embedded vector, and metadata
    Return all of the records in a dictionary
    """
    # Find the directory of all the plant sheets
    base_dir= Path(root_dir)
    base_dir= base_dir / "plant_sheets"
    if not base_dir.is_dir():
        Logger.error(f"Provided path is not a directory: {base_dir}")
        return None
    
    # This will store all of the records, this is returned at the end of the function
    all_records= []

    # Iterate over ever plant sheet subdir and parse the plant sheets inside
    for folder in base_dir.iterdir():
        if not folder.is_dir():
            continue # not a folder, skip

        # determine what to store the record under in the dictionary (either plant code or common name)
        plant_name= folder.name
        json_info= retrieve_USDA_json(root_dir, plant_name)
        metadata= generate_metadata(json_info)

        # process the plant sheets (chunk, vectorize, make records)
        records= process_USDA_sheets_sub_dir(folder, metadata, embedder, chunk_method, chunk_size, overlap)

        # put the records for this plant in the return dict
        all_records.extend(records)

    return all_records

def embed_USDA_json_data(root_dir: str, embedder):
    base_dir= Path(root_dir)
    json_dir= base_dir / "KB"
    if not json_dir.is_dir():
        Logger.error(f"Provided path is not a directory: {json_dir}")
        return None
    
    all_records= []

    for file in json_dir.iterdir():
        if not file.is_file():
            continue # not a file, skip

        plant_name= file.name
        json_info= retrieve_USDA_json(base_dir, plant_name)
        metadata= generate_metadata(json_info)

        json_metadata= metadata.copy()
        json_metadata["source file"]= plant_name
        embeddable_json_text= embedder.object_to_embeddable_string(json_info)
        json_record= embedder.encode_and_make_record(embeddable_json_text, json_metadata)
        all_records.append(json_record)

    return all_records

def retrieve_USDA_json(base_dir, code):
    """
    parse the metadata file for the specified plant code
    """
    metadata_file= base_dir / "KB" / code
    if not metadata_file.is_file():
        Logger.warning(f"Metadata file not found: {metadata_file}")
        return {}
    
    with open(metadata_file, 'r') as file:
        metadata= json.load(file)

    return metadata

def generate_metadata(data: dict):
    """
    return a metadata dict with key info in it
    """
    code= data["symbol"]
    name= data["common_name"]
    sci_name= data["scientific_name"]

    metadata= {
        "code": code,
        "common name": name,
        "scientific name": sci_name,
    }
    return metadata
    
def process_USDA_sheets_sub_dir(folder: Path, metadata: dict, embedder, chunk_method: str, chunk_size: int, overlap: int):
    """
    Process a single plant sheet subdir
    Process all sheets in the dir: chunking, embedding, record making
    Return the created records
    """
    plant_records= []

    # Iterate through every plant sheet in the subdir
    for file in folder.iterdir():
        if general_utils.is_pdf(file):
            # load the text data in the pdf
            text= text_pre.load_pdf_text_data(file)
        elif general_utils.is_docx(file):
            # load the text data in the docx file
            text= text_pre.load_docx_text_data(file)
        else:
            # invalid file, skip
            Logger.warning(f"Unexpected type found: {file}")
            continue

        # Remove urls and url symbols from the text data
        # We dont want to overly clean this data
        text= text_pre.replace_urls(text, "url")
        text= text.replace("<","")
        text= text.replace(">","")

        # update the metadata to have file specific info
        specific_metadata= metadata.copy()
        specific_metadata["source file"]= file.name


        # chunk, encode, and make records of the data
        records= embedder.chunk_and_encode(text, chunk_method, chunk_size, overlap, specific_metadata)

        # put the records in the return list
        plant_records.extend(records)

    return plant_records
