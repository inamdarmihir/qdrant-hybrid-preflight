"""Tests use the checksum-verified real BEIR SciFact files, not fabricated docs."""
from pathlib import Path
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from benchmark import load_data, tokenized_average
from preflight import preflight_hybrid_search
from sweep import validate_queries, ndcg, paired_interval


def test_real_dataset_and_qrels():
    docs, queries, qrels = load_data(Path('data'))
    assert len(docs) == 5183 and len(qrels) == 300
    corpus_ids = {d['_id'] for d in docs}
    assert all(q in queries for q in qrels)
    assert all(set(labels) <= corpus_ids for labels in qrels.values())
    for labels in qrels.values():
        ranked = sorted(labels, key=labels.get, reverse=True)
        assert ndcg(ranked, labels) == 1
    assert paired_interval([1.] * len(qrels), [1.] * len(qrels)) == [0., 0.]


def test_real_bm25_length_and_sparse_vectors():
    docs, queries, qrels = load_data(Path('data'))
    encoder = SparseTextEmbedding('Qdrant/bm25')
    texts = [f"{d['title']} {d['text']}".strip() for d in docs]
    avg = tokenized_average(encoder.model, texts)
    assert abs(avg - 151.38028169014083) < 1e-8
    vector = next(encoder.query_embed(queries[next(iter(qrels))]))
    assert len(vector.indices) == len(vector.values) > 0
    assert len(set(vector.indices)) == len(vector.indices)


def test_actual_collection_idf_and_production_request():
    encoder = SparseTextEmbedding('Qdrant/bm25')
    docs, queries, qrels = load_data(Path('data'))
    sparse = next(encoder.query_embed(queries[next(iter(qrels))]))
    c = QdrantClient(':memory:')
    try:
        c.create_collection('test', vectors_config={}, sparse_vectors_config={
            'bm25': models.SparseVectorParams(modifier=models.Modifier.IDF)})
        request = models.QueryRequest(query=models.FusionQuery(fusion=models.Fusion.DBSF),
            prefetch=[models.Prefetch(query=models.SparseVector(indices=sparse.indices.tolist(),
                values=sparse.values.tolist()), using='bm25', limit=50)])
        result = preflight_hybrid_search(c, 'test', 'bm25', request,
            151.38028169014083, 151.38028169014083, 150)
        assert all(f.level == 'info' for f in result)
        absent = preflight_hybrid_search(c, 'test', 'missing')
        assert 'sparse_missing' in {f.code for f in absent}
        inner = models.Prefetch(query=request.query, prefetch=request.prefetch)
        nested = models.QueryRequest(query=request.query, prefetch=inner)
        assert 'nested_fusion' in {f.code for f in preflight_hybrid_search(c,'test','bm25',nested)}
    finally:
        c.close()
