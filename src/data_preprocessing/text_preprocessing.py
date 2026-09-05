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

