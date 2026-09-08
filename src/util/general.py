from logger import Logger, bold, italic, underline
import pandas as pd


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