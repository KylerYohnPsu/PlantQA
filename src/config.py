import logging

############ Plant Expert VQA Dataset Handling Configuration ############

# The path to the Plant Expert VQA data folder FROM THE root/data DIR
PEVQA_DATA_PATH= "PlantExpertVQA/data"

# The file names of the train/test/val data files in the PEVQA dataset
PEVQA_TRAIN_FILE= "train.csv"
PEVQA_TEST_FILE= "test.csv"
PEVQA_VAL_FILE= "val.csv"

# Some columns have NA values that we want to replace, rather than remove
# Each pair represents a column we want to edit and the new value to replace NA vals in that column with
# (col name, NA replace val)
PEVQA_NA_COLUMN_FILL_PAIRS= [
    ("answer_type", "unspecified"),
    ("question_text", ""),
    ("question_category", "unspecified"),
    ("cognitive_level", "unspecified"),
]

# Column names in the PEVQA dataset that have text that needs to be cleaned/normalized
PEVQA_TEXT_COLUMNS= ["question_text", "answer"]

# Columns in the PEVQA dataset that we want to remove
PEVQA_COLUMNS_TO_REMOVE= ["dataset_source"]

####### Logging Configuration ########

# The lowest level logs to show
# Options: [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
LOG_LEVEL= logging.DEBUG

# The format of logging outputs
LOG_PATTERN= "%(log_color)s%(asctime)s [%(levelname)-8s]: %(message)s%(reset)s"

# The color of each log type
LOG_COLORS= {
	"DEBUG": "cyan",
	"INFO": "black",
	"WARNING": "yellow",
	"ERROR": "red",
	"CRITICAL": "bold_red",
}

