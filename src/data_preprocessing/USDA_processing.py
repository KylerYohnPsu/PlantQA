import numpy as np
from sentence_transformers import SentenceTransformer
import src.util.general as general_utils
from src.util.logger import Logger
from dataclasses import dataclass
from pathlib import Path
from . import text_preprocessing as text_pre
from . import TextEmbedder
import json

def process_USDA_plant_sheets(root_dir: str, embedder, chunk_method: str="word", chunk_size: int=64, overlap: int=16):
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
    all_records= {}

    # Iterate over ever plant sheet subdir and parse the plant sheets inside
    for folder in base_dir.iterdir():
        if not folder.is_dir():
            continue # not a folder, skip

        # determine what to store the record under in the dictionary (either plant code or common name)
        plant_code= folder.name
        metadata= retrieve_USDA_metadata(root_dir, plant_code)
        if "common_name" in metadata:
            plant_name= metadata["common_name"]
            if plant_name is None:
                Logger.warning("Plant name is none, using plant code instead")
                plant_name= plant_code
            else:
                plant_name= plant_name.lower()
        else:
            Logger.warning("No common name found, using plant code instead")
            plant_name= plant_code

        # make a blank records list that we will extend as we make the records
        all_records[plant_name]= []

        # process the plant sheets (chunk, vectorize, make records)
        records= process_USDA_sheets_sub_dir(folder, metadata, embedder, chunk_method, chunk_size, overlap)

        # put the records for this plant in the return dict
        all_records[plant_name].extend(records)

    return all_records

def retrieve_USDA_metadata(base_dir, code):
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

        # chunk, encode, and make records of the data
        records= embedder.chunk_and_encode(text, chunk_method, chunk_size, overlap, metadata)

        # put the records in the return list
        plant_records.extend(records)
        
    return plant_records
