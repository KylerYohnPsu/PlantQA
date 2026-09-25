'''
Steps to Agent

1. Recieve question and Visual breakdown (either class or separated by features of a plant)

2. preprocess features of plant and get returned 
'''

"""Are we able to leverage the local host model"""
"""Add in crowd sourced data to the model"""

from dataclasses import dataclass
from typing import List, Tuple
from util.supabase_utils import query_embeddings
import numpy as np
from src.util.logger import Logger
from src.data_preprocessing.TextEmbedder import TextEmbedder, EmbeddingRecord
from src.data_preprocessing.TextTokenizer import TextTokenizer

class Retriever:
    def __init__(self, supabase_client, embedder: TextEmbedder):
        self.supabase_client = supabase_client
        self.embedder = embedder
    def retrieve(self, question, plant_code_filter = None, num_results = 5):
        question_embedding = self.embedder.encode(question)
        results = query_embeddings(
            self.supabase_client,
            question_embedding,
            plant_code = plant_code_filter,
            match_count=num_results,
            match_threshold=.6
        )

        return results
class ResponseModel:
    def __init__(self, tokenizer_model):
        self.model = None
        self.tokenizer = TextTokenizer(tokenizer_model, max_length = 512)

    def generate_answer(self, question, chunks):
        retrieved_context = "|".join([c["body"] for c in chunks])
        prompt = f"plant question and answer:\nquestion:{question}\nretrieved context: {retrieved_context}"

        inputs = self.tokenizer.encode_text(prompt)
        outputs = self.model.generate(inputs["input_ids"], max_length = 200)
        answer = self.tokenizer.decode(outputs[0])
        return answer

    def train_model(self, training_data, epochs = 10, batch_size = 32):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

class NLPAgent:
    def __init__(self, retriever: Retriever, response_model: ResponseModel):
        self.retriever = retriever
        self.response_model= response_model

    def ask_question(self, question, plant_code_filter = None, num_results = 5):
        question_embedding = self.retriever.retrieve(question, plant_code_filter)
        answer = self.response_model.generate_answer(question_embedding)
        return answer
