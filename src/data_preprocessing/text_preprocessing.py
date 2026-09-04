from src.util.logger import Logger
import re
import nltk
import pandas as pd
import pathlib

nltk.download('stopwords')

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

def remove_stop_words(text: str):
    """
    Remove stopwords from a text
    PARAM:
        text: str | The text to remove stop words from
    RETURN:
        str: The text without stopwords
    """
    # Get the stop words we want to remove
    stop_words= nltk.corpus.stopwords
    stop_words= set(stop_words.words("english"))

    # break the text into a list for easy iteration
    text_words= text.split()

    acceptable_words= []
    for word in text_words:
        if word.lower() not in stop_words:
            acceptable_words.append(word)

    # put the acceptable words back into a single text "sentence"
    cleaned_text= " ".join(acceptable_words)
    return cleaned_text

def preprocess_text(text: str):
    """
    preprocess a single text entry
    placeholder, remove/edit if you want to
    """
    return

