import supabase
import os
from dotenv import load_dotenv
from src.util.logger import Logger
load_dotenv()
def ingest_to_supabase(records):
    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL")  # Get from .env
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)

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