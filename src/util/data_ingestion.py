import supabase
import os
from dotenv import load_dotenv
from src.util.logger import Logger
load_dotenv()
def connect_supabase():
    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL")  # Get from .env
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase_client

def ingest_to_supabase(records, supabase_client):
    batch_size = 1000
    total = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        rows = [
            {
                "title": record.metadata.get("code", "unknown"),
                "body": record.text[:10000],
                "embedding": record.embedding.tolist()
            }
            for record in batch
        ]
        try:
            response = supabase_client.table("embeddings").insert(rows).execute()
            total += len(rows)
            Logger.info(f"Inserted {total} embeddings to supabase")
        except Exception as e:
            print(f"Batch {i} Failed to insert {total} embeddings to supabase: {e}")
    Logger.info(f"Completed: Inserted {total} embeddings to supabase")

def add_metadata_tags(records, supabase_client):
    batch_size = 1000
    total = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        rows = [
            {
                "id": i + j,
                "plant_code": record.metadata.get("code", "unknown"),
                "common_name": record.metadata.get("common_name", "unknown"),
                "scientific_name": record.metadata.get("scientific name", "unknown"),
                "source_file": record.metadata.get("source file", "unknown"),
            }
            for j, record in enumerate(batch)
        ]
        result = supabase_client.table("embeddings").upsert(rows).execute()
        total += len(rows)
        Logger.info(f"Inserted {total} embeddings to supabase")
    Logger.info(f"Completed: Inserted {total} embeddings to supabase")
