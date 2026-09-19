import numpy as np
from sentence_transformers import SentenceTransformer
import src.util.general as general_utils
from src.util.logger import Logger
from dataclasses import dataclass
from pathlib import Path
from . import text_preprocessing as text_pre
import json

@dataclass
class EmbeddingRecord:
    """
    A single record that contains the raw text data,
    the vector embedding, and the metadata for an
    encoded chunk
    """
    text: str
    embedding: np.ndarray
    metadata: dict

    def __str__(self):
        text= self.text[:50]
        if len(self.text) > 50:
            text += "..."

        embedding = self.embedding[:5].tolist()
        if len(self.embedding) > 5:
            embedding = embedding + ["..."]

        return (
            f"Text: {text}\n"
            f"Embedding: {embedding}\n"
            f"Metadata: {self.metadata}"
        )

class TextEmbedder:
    """
    Chunks and embeds text data
    """
    def __init__(self, model_name):
        self._modelName= model_name
        self._embedder= SentenceTransformer(model_name)

    ######## PRIMARY FUNCTIONS #########

    def encode(self, text: str | list[str]):
        """
        Create the embedded vector out of the text data (either single string, or list of strings)
        """
        embeddings= self._embedder.encode(text, convert_to_numpy=True)
        return embeddings

    def encode_and_make_record(self, text: str, metadata: dict={}):
        """
        encode a string and make a record for it
        """
        embedding= self.encode(text)
        record= EmbeddingRecord(text, embedding, metadata)
        return record
    
    def chunk(self, text: str, split_method: str, max_size: 
        int, overlap_amount: int) -> list[str]:
        """
        Break the given text data into a list of strings of a specified token size
        """
        # split the text string into a list of strings
        text= self.split_text(text, split_method)
        if text is None:
            return None

        chunks= []
        current_chunk= []
        chunk_total_tokens= 0

        # iterate through every string in the split text
        for sub_text in text:
            # Determine how many tokens are in the string
            sub_text_tokens= self.count_tokens(sub_text)

            # determine if adding this substring to the chunk will exceed the size limit
            size_limit_reached= self.size_limit_reached(
                current_chunk,
                sub_text_tokens, 
                chunk_total_tokens, 
                max_size
            )

            # if the chunk has reached the size limit, finish it off and store it for returning later
            if size_limit_reached:

                # We want a little bit of overlap between each chunk
                # This helps limit any info being cutoff between chunks
                overlap_chunk= []
                overlap_tokens= 0

                # look at the back of the chunk and get the last <overlap_amount> tokens
                reversed_chunk= reversed(current_chunk)
                for previous_text in reversed_chunk:
                    previous_tokens= self.count_tokens(previous_text)

                    if overlap_tokens + previous_tokens > overlap_amount:
                        break

                    overlap_chunk.insert(0, previous_text)
                    overlap_tokens+= previous_tokens

                # put the current chunk we have been working on into the return list
                self.add_to_chunks(chunks, current_chunk)

                # Start the next chunk with the little bit of overlap we just built
                current_chunk= overlap_chunk
                chunk_total_tokens= overlap_tokens

            # Put the text we are looking at into the chunk
            current_chunk.append(sub_text)
            chunk_total_tokens+= sub_text_tokens

        # add any left over chunks
        if current_chunk:
            self.add_to_chunks(chunks, current_chunk)
        
        return chunks
    
    def chunk_and_encode(self, text: str, chunk_method: str, chunk_size: int, overlap_amount: int, metadata={}):
        """
        Chunk and encode the string data
        """
        # chunk the text
        chunks= self.chunk(text, chunk_method, chunk_size, overlap_amount)

        # embed the chunks into vectors
        embeddings= self.encode(chunks)

        records= self.make_records(chunks, embeddings, metadata)

        return records

    def object_to_embeddable_string(self, obj: object):
        if isinstance(obj, dict):
            return self.dict_to_embeddable_string(obj)
        elif isinstance(obj, list):
            return self.list_to_embeddable_string(obj)
        elif isinstance(obj, str):
            return obj.replace("_", " ")
        else:
            return str(obj)


    ################# HELPER FUNCTIONS ######################
    def dict_to_embeddable_string(self, d: dict):
        embeddable_string= ""
        for key, value in d.items():
            key_string= self.object_to_embeddable_string(key)
            val_string= self.object_to_embeddable_string(value)
            substring= f"{key_string}: {val_string}"
            embeddable_string+= f"{substring}\n"
        return embeddable_string
    
    def list_to_embeddable_string(self, l: list):
        str_list= []
        for entry in l:
            str_entry= self.object_to_embeddable_string(entry)
            str_list.append(str_entry)
        
        embeddable_string= "\n\t".join(str_list)
        return embeddable_string

    def make_records(self, chunks: list[str], embeddings, metadata= {}):
        if len(chunks) != len(embeddings):
            Logger.error(f"Unable to make embedding records. len(chunks) {len(chunks)} != len(embeddings) {len(embeddings)}")
            return None
        
        records= []
        for chunk, embedding in zip(chunks, embeddings):
            record= EmbeddingRecord(chunk, embedding, metadata)
            records.append(record)
        
        return records

    def size_limit_reached(self, chunk: list[str], chunk_tokens: int, total_tokens: int, max_size: int):
        chunk_valid= chunk != []
        limit_reached= total_tokens + chunk_tokens > max_size
        valid_and_limit_reached= chunk_valid and limit_reached
        return valid_and_limit_reached

    def add_to_chunks(self, chunks: list[str], chunk: list[str]):
        """
        rebuild the chunk into a single string and add it to the chunks list
        """
        chunk= " ".join(chunk)
        chunks.append(chunk)

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text's encoding
        """
        tokens= self._embedder.tokenizer(text, add_special_tokens=False,)
        tokens= tokens["input_ids"]
        num_tokens= len(tokens)
        return num_tokens
    
    def split_text(self, text: str, method: str):
        """
        split the given text string using the specified method into a list of strings
        """
        match method.lower():
            case "sentence":
                return general_utils.split_text_by_sentence(text)
            case "word":
                return general_utils.split_text_by_word(text)
            case _:
                Logger.error(f"Unable to split text. Invalid method requested: {method}")
                return None
            
    