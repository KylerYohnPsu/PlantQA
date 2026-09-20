"""
When adding new configuration, just make a class named whatever you want and define the variables and assign them values.
"""
from pathlib import Path

class General:
    PROJECT_ROOT= Path(__file__).resolve().parent.parent
    DATA_DIR= PROJECT_ROOT / "data"

class Logging:
    """
    Config for Logging
    """
    LEVEL= "DEBUG"
    PATTERN = "%(log_color)s%(asctime)s [%(levelname)-8s]: %(message)s%(reset)s"

    COLORS = {
        "DEBUG": "cyan",
        "INFO": "black",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    }

class Data:
    class PlantExpertVQA:
        """
        Config for PlantExpertVQA dataset management
        """
        ROOT= General.DATA_DIR /"PlantExpertVQA"
        DATA= ROOT / "data"
        TRAIN_FILE= DATA / "train.csv"
        TEST_FILE= DATA / "test.csv"
        VALIDATION_FILE= DATA / "val.csv"

        TEXT_COLUMNS= ["question_text"]
        COLUMNS_TO_REMOVE= ["dataset_source"]

        NA_FILL= {
            "answer_type": "unspecified",
            "question_text": "",
            "question_category": "unspecified",
            "cognitive_level": "unspecified",
        }

    class USDA:
        """
        Config for USDA dataset management
        """
        ROOT= General.DATA_DIR / "EOI_Data_Pulls"
        PLANT_SHEETS= ROOT / "plant_sheets"
        JSON_FILES= ROOT / "KB"


