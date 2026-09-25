import supabase

def query_embeddings(supabase_client, query, plant_code = None, common_name = None, scientific_name = None):
    results = supabase_client.rpc("match_embeddings", {
        "query_embedding": query,
        "match_threshold": 0.5,
        "match_count": 10,
        "plant_code_filter": plant_code,
        "common_name_filter": common_name,
        "scientific_name_filter": scientific_name,
    }).execute()
    return results.data