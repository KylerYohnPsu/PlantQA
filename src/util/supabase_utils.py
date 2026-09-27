
import supabase
import os
from dotenv import load_dotenv
def query_embeddings(supabase_client, query, plant_code = None, common_name = None, scientific_name = None, match_count = 10, match_threshold = .6):
    if hasattr(query, "tolist"):
        query = query.tolist()
    results = supabase_client.rpc("match_embeddings", {
        "query_embedding": query,
        "match_threshold": match_threshold,
        "match_count": match_count,
        "plant_code_filter": plant_code,
        "common_name_filter": common_name,
        "scientific_name_filter": scientific_name,
    }).execute()
    return results.data

def connect_supabase():
    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL")  # Get from .env
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase_client