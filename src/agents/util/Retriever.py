import src.data_preprocessing.TextEmbedder as TextEmbedder
from src.util.supabase_utils import query_embeddings
import src.config as config

KNOWN_PLANT_CODES = {p.name for p in config.Data.USDA.JSON_FILES.iterdir()}

class Retriever:
    def __init__(self, supabase_client, embedder: TextEmbedder):
        self.supabase_client = supabase_client
        self.embedder = embedder
    def retrieve(self, question, plant_code_filter = None, genus_filter = None, num_results = 5):
        if plant_code_filter not in KNOWN_PLANT_CODES:
            plant_code_filter = None
        question_embedding = self.embedder.encode(question)
        results = query_embeddings(
            self.supabase_client,
            question_embedding,
            plant_code = plant_code_filter,
            genus = genus_filter,
            match_count=num_results,
            match_threshold=.6
        )

        return results