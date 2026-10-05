"""Offline unit tests: no downloads, no server, no embedding models.

They exercise the preflight rules and metrics against Qdrant's in-memory mode
and tiny hand-built inputs. The real-data experiment lives in test_benchmark.py.
"""
import pytest
from qdrant_client import QdrantClient, models

from qdrant_hybrid_preflight import ndcg, paired_interval, preflight_hybrid_search, validate_queries


@pytest.fixture()
def client():
    c = QdrantClient(':memory:')
    yield c
    c.close()


def make_collection(client, modifier):
    client.create_collection('t', vectors_config={}, sparse_vectors_config={
        'bm25': models.SparseVectorParams(modifier=modifier)})


def codes(findings):
    return {f.code for f in findings}


def test_missing_idf_is_an_error(client):
    make_collection(client, None)
    assert 'idf_missing' in codes(preflight_hybrid_search(client, 't', 'bm25'))


def test_idf_on_preweighted_encoder_warns(client):
    make_collection(client, models.Modifier.IDF)
    result = preflight_hybrid_search(client, 't', 'bm25', expects_idf=False)
    assert 'idf_double_weight' in codes(result)


def test_unknown_sparse_vector_is_an_error(client):
    make_collection(client, models.Modifier.IDF)
    assert 'sparse_missing' in codes(preflight_hybrid_search(client, 't', 'nope'))


def test_missing_inputs_stay_unverified_not_clean(client):
    make_collection(client, models.Modifier.IDF)
    result = preflight_hybrid_search(client, 't', 'bm25')
    assert {'avg_len_unverified', 'query_unverified', 'labels_unverified'} <= codes(result)


def test_avg_len_drift_and_few_labels(client):
    make_collection(client, models.Modifier.IDF)
    result = preflight_hybrid_search(client, 't', 'bm25', bm25_avg_len=100.0,
                                     measured_avg_len=150.0, labeled_query_count=10)
    assert {'avg_len_drift', 'few_labels'} <= codes(result)


def test_nested_fusion_and_threshold_are_flagged(client):
    make_collection(client, models.Modifier.IDF)
    fusion = models.FusionQuery(fusion=models.Fusion.DBSF)
    inner = models.Prefetch(query=fusion, score_threshold=0.5)
    request = models.QueryRequest(query=fusion, prefetch=inner)
    assert {'nested_fusion', 'fusion_threshold'} <= codes(
        preflight_hybrid_search(client, 't', 'bm25', query_request=request))


@pytest.mark.parametrize('kwargs', [
    {'drift_threshold': 1.0}, {'bm25_avg_len': -1.0}, {'labeled_query_count': True}])
def test_invalid_arguments_raise(client, kwargs):
    make_collection(client, models.Modifier.IDF)
    with pytest.raises(ValueError):
        preflight_hybrid_search(client, 't', 'bm25', **kwargs)


def test_ndcg_perfect_and_zero():
    qrels = {'a': 1, 'b': 1}
    assert ndcg(['a', 'b', 'c'], qrels) == pytest.approx(1.0)
    assert ndcg(['x', 'y'], qrels) == 0.0


def test_ndcg_orders_by_rank():
    qrels = {'a': 1}
    assert ndcg(['a', 'x'], qrels) > ndcg(['x', 'a'], qrels)


def test_paired_interval_brackets_a_constant_delta():
    low, high = paired_interval([0.6, 0.7, 0.8], [0.5, 0.6, 0.7])
    assert low == pytest.approx(0.1) and high == pytest.approx(0.1)


def valid_query(qid='q1'):
    return {'id': qid, 'dense': [0.1, 0.2], 'sparse': {'indices': [1, 2], 'values': [0.5, 0.5]},
            'qrels': {'1': 1}}


def test_validate_queries_accepts_good_input_and_rejects_bad():
    validate_queries([valid_query()], dense_size=2)
    with pytest.raises(ValueError):
        validate_queries([valid_query(), valid_query()])  # duplicate ids
    with pytest.raises(ValueError):
        validate_queries([valid_query()], dense_size=3)   # wrong dimension
    bad = valid_query()
    bad['qrels'] = {'1': 0}
    with pytest.raises(ValueError):
        validate_queries([bad])                           # no positive judgment
