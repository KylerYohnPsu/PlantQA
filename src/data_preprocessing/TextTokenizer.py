import pandas as pd
from transformers import AutoTokenizer, BatchEncoding
from src.util.logger import Logger
import src.util.general as general_util
import tensorflow as tf

class TextTokenizer:
    def __init__(self, tokenizer_model_name: str, max_length: int):
        self._modelName= tokenizer_model_name
        self._maxLength= max_length

        self._tokenizer= AutoTokenizer.from_pretrained(self._modelName)


    def encode_text(self, text: str | list | pd.DataFrame | pd.Series) -> str | None:
        """
        tokenize the given text data
        PARAM:
            text: str, list, pd.DataFrame | The text data (single string, list of data, or single pd.DataFrame col) to tokenize
        RETURN:
            | None: None if invalid data
        """
        texts= self._listify_text(text)
        if texts is None:
            return None

        tokenized_texts= self._tokenizer(
            texts,
            padding= True,
            truncation= True,
            max_length= self._maxLength,
            return_tensors= "np",
        )

        inputs, mask= self._convert_encoding_to_tf_tensors(tokenized_texts)

        return inputs, mask

    def _listify_text(self, text: str | list | pd.DataFrame | pd.Series) -> list[str] | None:
        if isinstance(text, str):
            return [text]
        elif isinstance(text, list):
            return general_util.stringify_list(text)
        elif isinstance(text, pd.Series):
            return self._convert_series_to_str_list(text)
        elif isinstance(text, pd.DataFrame):
            # convert the dataframe into a list of strings
            return self._convert_dataframe_to_str_list(text)
        else:
            Logger.error(f"[encode_text] Invalid type received.  Accepted types are (str, list, pd.DataFrame, pd.Series).  Got {type(text)}")
            return None

    def _convert_series_to_str_list(self, data: pd.Series) -> list[str]:
        """
        Convert the given series into a list of strings
        PARAM:
            data: pd.series | The series to convert
        RETURN:
            list[str] | None: The series as a list of strings, None if failure
        """
        no_na= data.fillna("")
        str_no_na= no_na.astype(str)
        str_list= str_no_na.tolist()
        return str_list

    def _convert_encoding_to_tf_tensors(self, encoding: BatchEncoding):
        input_ids= encoding["input_ids"]
        attention_mask= encoding["attention_mask"]

        tf_inputs= tf.convert_to_tensor(input_ids)
        tf_attention= tf.convert_to_tensor(attention_mask)

        return (tf_inputs, tf_attention)
        
    def _convert_dataframe_to_str_list(self, data: pd.DataFrame) -> list[str] | None:
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
        
