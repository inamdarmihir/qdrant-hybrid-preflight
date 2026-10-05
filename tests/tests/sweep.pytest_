import pytest
from qdrant_client import QdrantClient
from demo import build_fixture, COLLECTION
from sweep import ndcg, paired_interval, run_sweep, validate_queries


def test_ndcg_ideal_and_unknown():
    assert ndcg(['1', '2'], {'1': 2, '2': 1}) == 1
    assert ndcg(['9'], {'1': 2}) == 0
    assert ndcg(['1'], {}) == 0
    assert 0 < ndcg(['2', '1'], {'1': 2, '2': 1}) < 1


def test_paired_bootstrap_reproducible():
    assert paired_interval([1, 1], [1, 1]) == [0, 0]
    assert paired_interval([.5, 1], [0, .5]) == [.5, .5]


def test_real_local_engine_sweep():
    c = QdrantClient(':memory:')
    try:
        q = build_fixture(c)
        result = run_sweep(c, COLLECTION, q, depths=(10,), ks=(2,))
        assert len(result['runs']) == 4
        assert result['query_count'] == 12
        assert all(0 <= x['mean_ndcg'] <= 1 for x in result['runs'])
        assert all(len(x['rankings']) == 12 for x in result['runs'])
        with pytest.raises(ValueError):
            run_sweep(c, COLLECTION, q, depths=(5,), limit=10)
    finally:
        c.close()


def test_empty_and_duplicate_queries_rejected():
    with pytest.raises(ValueError):
        validate_queries([])
    q = {'id': 'a', 'dense': [1.], 'sparse': {'indices': [0], 'values': [1.]}, 'qrels': {'1': 1}}
    with pytest.raises(ValueError):
        validate_queries([q, q])


@pytest.mark.parametrize('qrels', [{}, {'1': -1}, {'1': float('nan')}, {'1': 0}])
def test_invalid_labels_rejected(qrels):
    with pytest.raises(ValueError):
        validate_queries([{'id': 'q', 'dense': [1.], 'sparse': {'indices': [], 'values': []}, 'qrels': qrels}])
