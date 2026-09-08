import pandas as pd
from transformers import AutoTokenizer

class TextTokenizer:
    def __init__(self, tokenizer_model_name: str, max_length: int):
        self._modelName= tokenizer_model_name
        self._maxLength= max_length

        self._tokenizer= AutoTokenizer.from_pretrained(self._modelName)


    def encode_text(self, text: str, encoding_type: str="np"):
        # make sure text is a string
        # also put text into a list, so the tokenizer can take it in
        text= str(text)
        text= [text]

        encoding= self._tokenizer(
            list(text),
            padding=True,
            truncation=True,
            max_length= self._maxLength,
            return_tensors=encoding_type,
        )

        return encoding
