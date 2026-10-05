from types import SimpleNamespace as NS
import pytest
from qdrant_client import QdrantClient, models
from preflight import preflight_hybrid_search


def codes(**kwargs):
    client = NS(get_collection=lambda _: NS(config=NS(params=NS(
        sparse_vectors={'bm25': models.SparseVectorParams(modifier=models.Modifier.IDF)}))))
    return {f.code for f in preflight_hybrid_search(client, 'x', 'bm25', **kwargs)}


def test_unknown_inputs_are_not_passes():
    assert {'avg_len_unverified', 'query_unverified', 'labels_unverified'} <= codes()


def test_actual_config_idf_missing():
    c = QdrantClient(':memory:')
    try:
        c.create_collection('x', vectors_config={}, sparse_vectors_config={
            'bm25': models.SparseVectorParams()})
        assert 'idf_missing' in {f.code for f in preflight_hybrid_search(c, 'x', 'bm25')}
        assert 'sparse_missing' in {f.code for f in preflight_hybrid_search(c, 'x', 'other')}
    finally:
        c.close()


def test_drift_boundary():
    assert 'avg_len_checked' in codes(bm25_avg_len=115, measured_avg_len=100)
    assert 'avg_len_drift' in codes(bm25_avg_len=116, measured_avg_len=100)


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_invalid_average(value):
    with pytest.raises(ValueError):
        codes(bm25_avg_len=value)


def test_nested_nonfusion_is_not_flagged():
    q = models.QueryRequest(prefetch=models.Prefetch(prefetch=models.Prefetch(
        query=[1., 0.], using='dense'), query=[1., 0.], using='dense'),
        query=models.FusionQuery(fusion=models.Fusion.RRF))
    assert 'nested_fusion' not in codes(query_request=q)


def test_deep_nested_fusion_and_threshold():
    inner = models.Prefetch(query=models.FusionQuery(fusion=models.Fusion.DBSF),
        prefetch=models.Prefetch(query=[1., 0.], using='dense'), score_threshold=.8)
    q = models.QueryRequest(prefetch=models.Prefetch(prefetch=inner,
        query=[1., 0.], using='dense'), query=models.RrfQuery(rrf=models.Rrf(k=2)))
    assert {'nested_fusion', 'fusion_threshold'} <= codes(query_request=q)


def test_encoder_already_has_idf():
    assert 'idf_double_weight' in codes(expects_idf=False)


def test_counts_are_heuristics():
    assert 'few_labels' in codes(labeled_query_count=49)
    assert 'labels_counted' in codes(labeled_query_count=50)


@pytest.mark.parametrize('count', [-1, 1.5, True])
def test_invalid_counts(count):
    with pytest.raises(ValueError):
        codes(labeled_query_count=count)
