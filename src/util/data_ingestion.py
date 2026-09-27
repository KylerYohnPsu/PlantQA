import util.supabase_utils as supabase_utils
from src.util.logger import Logger



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

def add_metadata_tags(records, supabase_client, start_count = 0,
                      column_names = None,
                      column_values = None):
    if column_names is None:
        column_names = ["title", "body", "embedding", "plant_code", "common_name", "scientific_name", "source_file"]
    if column_values is None:
        column_values = ["unknown", "unknown", "unknown", "unknown", "unknown", "unknown", "unknown"]

    batch_size = 1000
    total = 0
    start = start_count
    update_dict = dict(zip(column_names, column_values))
    for i in range(start, len(records), batch_size):
        batch = records[i:i + batch_size]
        for j, record in enumerate(batch):
            row_id = i + j

            supabase_client.table("embeddings").update(
                update_dict
            ).eq("id", row_id).execute()
            total += 1
        Logger.info(f"Inserted {total} embeddings to supabase")
    Logger.info(f"Completed: Inserted {total} embeddings to supabase")