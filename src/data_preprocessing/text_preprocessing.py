from src.util.logger import Logger
import re

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
