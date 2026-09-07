"""
Put code for exploring text data here
"""

import pandas as pd
import wordcloud
from src.util.logger import Logger, bold, italic, underline
import matplotlib.pyplot as plt
import nltk

def explore_data(data: pd.DataFrame, col_dist_args: list[dict], wordcloud_cols: list[str],
                 word_freq_dist_cols: list[str], top_n_words: int=20, name: str="Data"): 
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

    print("\n\n\n\n\n")
    Logger.info(f"\n\n-------- {name} Class Frequency Distributions --------\n\n")

    make_column_distributions(data, col_dist_args)

    print("\n\n\n\n\n")
    Logger.info(f"\n\n-------- {name} WordClouds --------\n\n")
    make_wordclouds(data, wordcloud_cols)

    print("\n\n\n\n\n")
    Logger.info(f"\n\n-------- {name} Word Frequency Distributions --------\n\n")
    make_word_frequency_distributions(data, word_freq_dist_cols, top_n_words=top_n_words)

def make_word_frequency_distributions(data: pd.DataFrame, columns: list[str], top_n_words: int= 20):
    """
    Make top word frequency distributions for the specified columns
    PARAM:
        data: pd.DataFrame | The data
        columns: list[str] | The names of the columns in the data that we want to make distributions for
        top_n_words: int | The to N most frequent words to show in the distribution (i.e. top 20 words)
    """
    for col in columns:
        make_most_frequent_words_distribution(data, col, top_n_words=top_n_words)

def make_wordclouds(data: pd.DataFrame, wordcloud_cols: list[str]):
    """
    Make the wordclouds for the specified columns
    PARAM:
        data: pd.DataFrame | The data to make clouds out of
        wordcloud_cols: list[str] | The names of the columns to make wordclouds out of
    """
    for column in wordcloud_cols:
        make_wordcloud(data, column)


def make_column_distributions(data: pd.DataFrame, col_dist_args: list[tuple]):
    """
    Iterate through the given column distribution arguments and use them to make distributions
    PARAM:
        data: pd.DataFrame | The dataframe that holds the data
        col_dist_args: list[dict] | A list of dictionaries, where each dictionary contains the args for make_column_distribution_graph
    """
    for args in col_dist_args:
        col_name= args['column']
        fig_size= args['figure_size']
        show_counts= args['show_counts']
        make_column_distribution_graph(data, col_name, size=fig_size, show_counts=show_counts)


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

def make_wordcloud(data: pd.DataFrame, column_name: str): 
    """
    Make a wordcloud that shows the most frequent words in a dataframe column
    PARAM:
        data: pd.DataFrame | The dataframe that holds the column we want
        column_name: str | the column that holds the data we want
    """
    # Get the column we want and make sure all entries are valid for a wordcloud
    column= data[column_name]
    column= column.dropna()
    column= column.astype(str)

     # this will help make the graph title look nicer
    pretty_column_name= column_name.replace("_", " ").title()
    
    # put all the text into one big string
    all_text= " ".join(column)

    # make the wordcloud
    word_cloud= wordcloud.WordCloud(width=800, height=400, background_color="black").generate(all_text)

    # show the wordcloud
    plt.figure(figsize=(8,4))
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.title(f"Wordcloud of text from {pretty_column_name}")
    plt.show()

def make_most_frequent_words_distribution(data: pd.DataFrame, column_name: str, top_n_words: int=20):
    """
    Make a frequency distribution that shows the most common words in a column of data
    PARAM:
        data: pd.DataFrame | The dataframe the data lives in
        column_name: str | The name of the column we want to make a distribution of
        top_n_words: int | How many of the top occurring words to graph
    """
    # Get the column we want and make sure all entries are valid for a wordcloud
    column= data[column_name]
    column= column.dropna()
    column= column.astype(str)

    # this will help make the graph title look nicer
    pretty_column_name= column_name.replace("_", " ").title()

    # put all the words into a single string
    all_text= " ".join(column)

    # split the big string into a list, where each entry is a word
    all_words= all_text.split(" ")

    #create a distribution of the top n words in the column
    distribution= nltk.probability.FreqDist(all_words)

    # graph the distribution
    plt.figure()
    distribution.plot(top_n_words)
    plt.title(f"Top {top_n_words} words that appear in {pretty_column_name}")
    plt.show()
