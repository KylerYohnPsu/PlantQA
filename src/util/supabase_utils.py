import supabase

def query_embeddings(supabase_client, query, plant_code = None, common_name = None, scientific_name = None, match_count = 10, match_threshold = .6):
    results = supabase_client.rpc("match_embeddings", {
        "query_embedding": query,
        "match_threshold": match_threshold,
        "match_count": match_count,
        "plant_code_filter": plant_code,
        "common_name_filter": common_name,
        "scientific_name_filter": scientific_name,
    }).execute()
    return results.data