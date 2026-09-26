import data_preprocessing.TextEmbedder as TextEmbedder
from util.supabase_utils import query_embeddings


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