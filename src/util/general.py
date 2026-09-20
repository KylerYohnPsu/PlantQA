
from src.util.logger import Logger, bold, italic, underline
import pandas as pd
import re
from pathlib import Path

def is_csv(file: Path):
    if not isinstance(file, Path):
        raise TypeError("Only accepts Path objects")
    
    if file.is_file() and file.suffix.lower() == ".csv":
        return True
    else:
        return False

def is_pdf(file: Path):
    if not isinstance(file, Path):
        raise TypeError("Only accepts Path objects")
    
    if file.is_file() and file.suffix.lower() == ".pdf":
        return True
    else:
        return False
    
def is_docx(file: Path):
    if not isinstance(file, Path):
        raise TypeError("Only accepts Path objects")
    
    if file.is_file() and file.suffix.lower() == ".docx":
        return True
    else:
        return False

def split_text_by_sentence(text:str):
    sentences = re.split(r'(?<=[.!?])\s+', text)

    clean_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()

        if sentence:
            clean_sentences.append(sentence)

    return clean_sentences

def split_text_by_word(text: str):
    words= text.split()
    clean_words= []

    for word in words:
        word= word.strip()
        if word:
            clean_words.append(word)
    
    return clean_words

def parse_list_from_string(string: str) -> list[str]:
    substrings= string.split(",")
    return_list= []
    for element in substrings:
        return_list.append(element.strip())
    return return_list

def dictionary_pretty_string(d: dict) -> str:
    """
    Format a dictionary into a pretty string
    PARAM:
        d: dict | The dictionary to format
    RETURN:
        str: the pretty string with the dictionary's info in it
    """
    string= ""
    for key, val in d.items():
        string+= f"\t{bold(key)}: {val}\n"
    return string

def stringify_dataframe_entry(entry: pd.DataFrame) -> str:
    """
    Take a single dataframe entry and make it a string that 
    looks nice when logged
    PARAM:
        entry: pd.DataFrame | The frame entry to make pretty
    RETURN:
        str: The entry info in a pretty string
    """
    pretty_string= ""
    iterable_entry= entry.iloc[0]
    for col, val in iterable_entry.items():
        pretty_string+= f"\t{bold(col)}: {val}\n"
    return pretty_string

def stringify_list(l: list[any]) -> list[str]:
    """
    Convert every item in a list into a string
    PARAM:
        l: list[any] | A list of data
    RETURN:
        list[str]: The given list where every entry is a str
    """
    str_list= []
    for entry in l:
        str_entry= str(entry)
        str_list.append(str_entry)
    return str_list