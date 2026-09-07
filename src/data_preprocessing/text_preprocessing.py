from src.util.logger import Logger
import re
import nltk
import pandas as pd
import pathlib

# build the set of stopwords we will use to remove stop words from text
STOP_WORDS = set(nltk.corpus.stopwords.words("english"))


def load_csv(csv_path: str):
    """
    Load a specified csv file into a pandas dataframe
    PARAM:
        csv_path: str | The path to the csv file to load
    RETURN:
        dataframe: a pandas dataframe of the csv data
    """
    # verify the file exists
    path= pathlib.Path(csv_path)
    if not path.is_file():
        Logger.error(f"[load_csv] Unable to open {csv_path}.")
        return None
    
    # load and return the data
    data= pd.read_csv(csv_path)
    Logger.debug(f"[load_csv] Successfully loaded {csv_path} into dataframe")
    return data

def preprocess_dataframe(data: pd.DataFrame, cols_to_normalize: list[str], cols_to_remove: list[str], na_col_fill_pairs: list[tuple]):
    """
    Grand daddy preprocessing function.
    Just calls other preprocessing functions, that way we only have to call one function to do all the work
    PARAM:
        data: pd.DataFrame | The dataframe to preprocess
        cols_to_normalize: list[str] | The columns of text data to normalize
        cols_to_remove: list[str] | The columns to remove from the dataset
        na_col_fill_pairs: list[tuple] | (column_name, na_replacement_value) pairs
    """
    fill_na_values(data, na_col_fill_pairs)
    preprocess_text_columns(data, cols_to_normalize)
    clean_dataframe(data, cols_to_remove)

def preprocess_text_columns(data: pd.DataFrame, columns: list[str]):
    """
    Preprocess the text in a dataframe
    Process the specified columns
    PARAM:
        data: pd.DataFrame | The dataframe to edit
        columns: list[str] | The names of the columns to edit
    """
    for column in columns:
        # make sure the column is all strings to prevent errors
        data[column]= data[column].astype(str)

        # remove symbols, normalize whitespace, make lower, etc
        data[column]= data[column].apply(normalize_text)

        # remove stop words from the text
        data[column]= data[column].apply(remove_stop_words)

def fill_na_values(data: pd.DataFrame, column_and_fill: list):
    """
    Fill the N/A values in the specified columns with the specified values
    PARAM:
        data: pd.DataFrame | The data frame we want to edit
        column_and_fill: list | a list of tuple pairs where the first value is the column name and the second is the value to replace NA with
    """
    # Iterate through the specified columns and replace the NA values with the specified values
    for (column, fill_value) in column_and_fill:
        data[column]= data[column].fillna(fill_value)

def clean_dataframe(data: pd.DataFrame, columns_to_remove: list[str]):
    """
    Clean a dataframe of bad data
    PARAM:
        data: pd.DataFrame | The dataframe to clean
    """
    # drop unwanted columns
    data.drop(columns=columns_to_remove, inplace=True)

    # drop invalid rows
    data.dropna(inplace=True)

    # drop duplicate entries
    data.drop_duplicates(inplace=True)




        

##############################################################
########## INDIVIDUAL STRING EDITING FUNCTIONS ################
################################################################


def remove_symbols(text: str) -> str:
    """"
    Remove symbols from teh given string
    PARAM:
        text: str | The text to remove symbols from
    RETURN:
        str: The text without symbols
    """
    regex= r'[^\w\s]'
    no_symbols= re.sub(regex, '', text)
    return no_symbols

def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in the text.  Remove additional spacing and make all whitespace 1 space
    PARAM:
        text: str | the text to normalize
    RETURN:
        str: the normalized text
    """
    regex= r'\s+'
    normalized_text= re.sub(regex, " ", text)
    normalized_text= normalized_text.strip()
    return normalized_text

def normalize_text(text: str) -> str:
    """
    Normalize text by removing non-words/numbers, making everything lowercase, and removing whitespace
    PARAM:
        text: str | the text to normalize
    RETURN:
        str: the normalized text
    """
    text= text.lower()
    text= remove_symbols(text)
    text= normalize_whitespace(text)
    return text

###############
#NOTE: The nltk stop words may be too intensive, because they remove words like "what', "why", "how", etc., which will likely be important for question processing
# We may want to use a different library for stop words, or define our own set of stop words
def remove_stop_words(text: str):
    """
    Remove stopwords from a text
    PARAM:
        text: str | The text to remove stop words from
    RETURN:
        str: The text without stopwords
    """
    # break the text into a list for easy iteration
    text_words= text.split()

    acceptable_words= []
    for word in text_words:
        if word.lower() not in STOP_WORDS:
            acceptable_words.append(word)

    # put the acceptable words back into a single text "sentence"
    cleaned_text= " ".join(acceptable_words)
    return cleaned_text

def preprocess_text(text: str):
    """
    preprocess a single text entry
    normalize and clean the text
    PARAM:
        text: str | The text to preprocess
    RETURN:
        str: preprocessed text, subject change
    """

    #TODO: Do we want to do tokenization and whatnot hear, or do we only want to do text cleaning/normalization here

    normalized_text= normalize_text(text)
    simplified_text= remove_stop_words(normalized_text)

    return simplified_text