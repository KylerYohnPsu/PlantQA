import pandas as pd
from transformers import AutoTokenizer, BatchEncoding
from src.util.logger import Logger
import src.util.general as general_util
import tensorflow as tf
import numpy as np
import re
from . import text_preprocessing as text_pre

class TextTokenizer:
    def __init__(self, tokenizer_model_name: str, max_length: int):
        self._modelName= tokenizer_model_name
        self._maxLength= max_length

        self._tokenizer= AutoTokenizer.from_pretrained(self._modelName)

    
    def split_text_by_sentence(self, text:str):
        sentences = re.split(r'(?<=[.!?])\s+', text)

        clean_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()

            if sentence:
                clean_sentences.append(sentence)

        return clean_sentences
    
    def split_text_by_word(self, text: str):
        words= text.split()
        clean_words= []

        for word in words:
            word= word.strip()
            if word:
                clean_words.append(word)
        
        return clean_words

    def count_tokens(self, text: str) -> int:
        tokens= self._tokenizer.encode(text, add_special_tokens=False,)
        num_tokens= len(tokens)
        return num_tokens
    
    def split_text(self, text: str, method: str):
        match method.lower():
            case "sentence":
                return self.split_text_by_sentence(text)
            case "word":
                return self.split_text_by_word(text)
            case _:
                Logger.error(f"Unable to split text. Invalid method requested: {method}")
                return None


    def chunk_text(self, text: str, split_method: str) -> list[str]:

        text= self.split_text(text, split_method)
        if text is None:
            return None

        chunks= []
        current_chunk= []
        chunk_total_tokens= 0

        for sub_text in text:
            sub_text_tokens= self.count_tokens(sub_text)

            size_limit_reached= current_chunk != [] and chunk_total_tokens + sub_text_tokens > self._maxLength

            if size_limit_reached:

                chunk= " ".join(current_chunk)
                chunks.append(chunk)
                current_chunk= []
                chunk_total_tokens= 0

            current_chunk.append(sub_text)
            chunk_total_tokens+= sub_text_tokens

        # add any left over text
        if current_chunk:
            chunk= " ".join(current_chunk)
            chunks.append(chunk)
        
        return chunks




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

    def decode(self, ids: tf.Tensor | np.ndarray) -> str:
        """
        decode a single tensor into a string
        PARAM:
            ids: tf.Tensor | Tensor model outputs
        RETURN:
            str: The decoded tensor
        """
        if isinstance(ids, tf.Tensor):
            ids= ids.numpy() # Turn tensor into np array
        elif isinstance(ids, np.ndarray):
            pass # nothing to do
        else:
            Logger.error(f"[decode] Invalid ids received.  Accepts types (tf.Tensor, np.ndarray).  Got {type(ids)}")
            return None

        return_string= self._tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return return_string

    def batch_decode(self, ids) -> list[str]:
        """
        decode a batch of tensors of (size, length)
        PARAM:
            ids: tf.Tensor | The ids to batch decode
        RETURN:
            list[str] | None: the decoded strings in a list, or None if error
        """
        if isinstance(ids, tf.Tensor):
            ids= ids.numpy()
        elif isinstance(ids, np.ndarray):
            pass
        else:
            Logger.error(f"[batch_decode] Invalid ids received.  Accepts types (tf.Tensor, np.ndarray).  Got {type(ids)}")
            return None

        return_list= self._tokenizer.batch_decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return return_list

        
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
        
