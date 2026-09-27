import util.supabase_utils as supabase_utils
from src.util.logger import Logger
import config
import json



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
    seen_plants = set()
    for record in records:
        plant_code = record.metadata.get("code")
        if plant_code in seen_plants:
            continue
        seen_plants.add(plant_code)

        update_dict = {
            "common_name": record.metadata.get("common name"),
            "scientific_name": record.metadata.get("scientific name"),
            "genus": record.metadata.get("genus"),
            "family": record.metadata.get("family"),
            "plant_group": record.metadata.get("group"),
        }

        supabase_client.table("embeddings").update(update_dict).eq("plant_code", plant_code).execute()

    Logger.info(f"Updated {len(seen_plants)} plant codes")