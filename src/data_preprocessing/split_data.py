import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path

from build_training_sets import generate_GBIF_CSV

def build_train_test_split_data(data, path_to_save):

    train_df, test_df = train_test_split(data, test_size=0.2, stratify=data["crop"], random_state=42)

    train_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df["crop"], random_state=42)

    test_df.to_csv( path_to_save / "test_df.csv", index=False)
    train_df.to_csv(path_to_save / "train_df.csv", index=False)
    val_df.to_csv(path_to_save / "val_df.csv", index=False)


if __name__ == "__main__":

    #generate the maing GBIF csv
    generate_GBIF_CSV()
    #Read in the GBIF data
    path = Path(__file__).parent.parent.parent / 'data' / "GBIF" / 'GBIF_Data_set.csv'
    path_to_save = Path(__file__).parent.parent.parent / "data" / "GBIF"  
    gbif_df = pd.read_csv(path)
    build_train_test_split_data(gbif_df, path_to_save)