import pandas as pd
from transformers import AutoTokenizer
from src.util.logger import Logger
import src.util.general as general_util

class TextTokenizer:
    def __init__(self, tokenizer_model_name: str, max_length: int):
        self._modelName= tokenizer_model_name
        self._maxLength= max_length

        self._tokenizer= AutoTokenizer.from_pretrained(self._modelName)


    def encode_text(self, text: str | list | pd.DataFrame, encoding_type: str="np") -> str | None:
        """
        tokenize the given text data
        PARAM:
            text: str, list, pd.DataFrame | The text data (single string, list of data, or single pd.DataFrame col) to tokenize
            encoding_type: str | The encoding type
        RETURN:
            | None: None if invalid data
        """
        # To tokenize the data, it needs to be a list of strings
        # convert whatever was given into a list of strings
        if isinstance(text, str):
            texts= [text]
        elif isinstance(text, list):
            texts= general_util.stringify_list(text)
        elif isinstance(text, pd.DataFrame):
            # convert the dataframe into a list of strings
            texts= self._convert_dataframe_to_str_list(text)
            if texts is None:
                return None # invalid data give
        else:
            Logger.error(f"[encode_text] Invalid type received.  Accepted types are (str, list, pd.DataFrame).  Got {type(text)}")
            return None

        tokenized_texts= self._tokenizer(
            texts,
            padding= True,
            truncation= True,
            max_length= self._maxLength,
            return_tensors= encoding_type,
        )

        return tokenized_texts
        
    def _convert_dataframe_to_str_list(data: pd.DataFrame) -> list[str] | None:
        """
        Convert the given dataframe into a list of strings
        PARAM:
            data: pd.DataFrame | The frame to convert
        RETURN:
            list[str] | None: The dataframe as a list of strings, None if failure
        """
        invalid_dataframe_shape= data.shape[1] != 1
        if invalid_dataframe_shape:
            Logger.error(f"[_convert_dataframe_to_str_list] Failed to tokenize dataframe.  Expected a single column, got frame with {text.shape[1]} column(s)")
            return None

        # Get all data from the first col (should only be 1 col)
        data_col= data.iloc[:,0]
        # replace na vals with blank strs
        no_na_data= data_col.fillna("")
        # make every entry a str
        str_data= no_na_data.astype(str)
        # put all vals into a list
        str_list_data= str_data.tolist()
        return str_list_data
        
