from typing import List, Dict, Any

def reciprocal_rank_fusion(dense_ranks: List[Dict], sparse_ranks: List[Dict], k: int = 60) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) algorithm combining dense vector and sparse keyword search scores:
    RRF_Score = sum(1 / (k + rank))
    """
    scores = {}
    doc_map = {}

    # Dense scoring
    for rank, doc in enumerate(dense_ranks, 1):
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    # Sparse scoring
    for rank, doc in enumerate(sparse_ranks, 1):
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    # Sort descending by fused RRF score
    sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    
    results = []
    for doc_id, score in sorted_docs:
        doc = doc_map[doc_id].copy()
        doc["rrf_score"] = round(score, 5)
        results.append(doc)
        
    return results
