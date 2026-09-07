"""
Put code for exploring text data here
"""

import pandas as pd
import wordcloud
from src.util.logger import Logger, bold, italic, underline
import matplotlib.pyplot as plt

def explore_data(data: pd.DataFrame, name: str="Data"): 
    """ Explore a single pandas dataframe """

   
    info= data.info()
    Logger.info(f"{name} Info:\n{info}")

    print(f"\n\n")

    description= data.describe()
    Logger.info(f"{name} Description:\n{description}")

    print(f"\n\n")

    head= data.head(1)
    loggable_head= stringify_dataframe_entry(head)
    Logger.info(f"{name} Head:\n{loggable_head}")


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

def make_column_distribution_graph(data: pd.DataFrame, column_name: str, show_counts: bool= False, size:tuple[int, int]= (5,5)):
    """
    Make a distribution graph of the specified column to show how often each type appears in the data
    PARAM:
        data: pd.DataFrame | The dataframe the data lives in
        column: str | The specific column we want a distribution of
        show_counts: bool | whether or not to write counts over bars in the graph
        size: tuple[int, int] | The size of the graph
    """
    column= data[column_name]

    # this will help make the graph title look nicer
    pretty_column_name= column_name.replace("_", " ").title()

    # get how many times each type appears in the column
    counts= column.value_counts(dropna=False)

    plt.figure(figsize=size)

    # this is going to be a bar graph distribution
    bars= plt.bar(counts.index.astype(str), counts.values, color="blue")

    plt.title(f"Distribution of {pretty_column_name}")
    plt.xlabel(pretty_column_name)
    plt.ylabel("Number of Occurrences")
    plt.xticks(rotation=60)

    # show the number of entries above each bar
    if show_counts:
        for bar, count in zip(bars, counts.values):
            y_pos= bar.get_height()
            x_pos= bar.get_x() + bar.get_width() / 2
            plt.text( x_pos, y_pos, str(count), ha="center", va="bottom")

    plt.tight_layout()
    plt.show()
