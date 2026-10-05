"""Root-level fusion sweep over caller-supplied vectors and relevance labels."""
import math
import time
import numpy as np
from qdrant_client import models


def ndcg(ranked_ids, relevance, k=10):
    """Exponential-gain nDCG; missing judgments count as zero relevance."""
    if k <= 0:
        raise ValueError('k must be positive')
    def dcg(grades):
        return sum((2 ** float(g) - 1) / math.log2(i + 2)
                   for i, g in enumerate(grades))
    ideal = dcg(sorted(relevance.values(), reverse=True)[:k])
    return dcg([relevance.get(str(p), 0) for p in ranked_ids[:k]]) / ideal if ideal else 0.0


def paired_interval(candidate, baseline, seed=42, samples=2000):
    """Percentile paired bootstrap CI over per-query nDCG deltas."""
    delta = np.asarray(candidate) - np.asarray(baseline)
    if not len(delta):
        raise ValueError('At least one query is required')
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(delta, size=len(delta), replace=True).mean()
                      for _ in range(samples)])
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def validate_queries(queries, dense_size=None):
    if not queries:
        raise ValueError('queries must not be empty')
    seen = set()
    for q in queries:
        if q['id'] in seen:
            raise ValueError('Duplicate query IDs')
        seen.add(q['id'])
        dense = q['dense']
        sparse = q['sparse']
        if not dense or not all(math.isfinite(x) for x in dense):
            raise ValueError('Dense vectors must be nonempty and finite')
        if dense_size is not None and len(dense) != dense_size:
            raise ValueError('Dense vector dimension mismatch')
        indices, values = sparse['indices'], sparse['values']
        if len(indices) != len(values) or len(set(indices)) != len(indices):
            raise ValueError('Sparse indices must be unique and match values')
        if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices):
            raise ValueError('Sparse indices must be nonnegative integers')
        if not all(math.isfinite(x) for x in values):
            raise ValueError('Sparse values must be finite')
        if not q['qrels'] or not any(x > 0 for x in q['qrels'].values()):
            raise ValueError('Every query needs a positively judged document')
        if any(not math.isfinite(x) or x < 0 or x > 30 for x in q['qrels'].values()):
            raise ValueError('Relevance must be finite and between 0 and 30')


def run_sweep(client, collection, queries, dense_name='dense', sparse_name='bm25',
              depths=(20, 50), ks=(2, 5, 20, 61), weight_pairs=((1., 1.),), limit=10, config_names=None):
    """Execute dense/sparse baselines then RRF/DBSF. No embeddings are inferred.

    Client/server errors propagate. Latencies are serial request timings, not load
    tests. This is one joint tuning grid, not a held-out significance experiment.
    """
    if limit < 1 or any(x < limit for x in depths):
        raise ValueError('All candidate depths must be >= result limit > 0')
    if not depths or not ks or any(k <= 0 for k in ks):
        raise ValueError('Nonempty depths and positive RRF k values required')
    if not weight_pairs or any(len(w) != 2 or not all(math.isfinite(x) and x >= 0 for x in w) or not sum(w) for w in weight_pairs):
        raise ValueError('Each weight pair must have two finite nonnegative values and positive sum')
    validate_queries(queries)
    configs = [('dense', None, dense_name, None), ('sparse', None, sparse_name, None)]
    for depth in depths:
        configs.append((f'dbsf_depth{depth}', models.FusionQuery(fusion=models.Fusion.DBSF), None, depth))
        for k in ks:
            for weights in weight_pairs:
                configs.append((f'rrf_k{k}_w{weights[0]:g}-{weights[1]:g}_depth{depth}',
                    models.RrfQuery(rrf=models.Rrf(k=k, weights=list(weights))), None, depth))
    if config_names is not None:
        configs = [c for c in configs if c[0] in config_names]
        if not {'dense', 'sparse'} <= {c[0] for c in configs}:
            raise ValueError('Selected configs must include dense and sparse baselines')
    rows = []
    for name, fusion, using, depth in configs:
        scores, timings, rankings = [], [], []
        for q in queries:
            sparse = models.SparseVector(**q['sparse'])
            if fusion is None:
                request = dict(query=q['dense'] if using == dense_name else sparse, using=using, search_params=models.SearchParams(exact=True))
            else:
                request = dict(prefetch=[models.Prefetch(query=q['dense'], using=dense_name, limit=depth, params=models.SearchParams(exact=True)),
                    models.Prefetch(query=sparse, using=sparse_name, limit=depth)], query=fusion)
            start = time.perf_counter()
            points = client.query_points(collection_name=collection, limit=limit,
                with_payload=False, **request).points
            timings.append((time.perf_counter() - start) * 1000)
            ids = [str(p.id) for p in points]
            scores.append(ndcg(ids, q['qrels'], limit))
            rankings.append({'query_id': q['id'], 'ids': ids, 'ndcg': scores[-1]})
        rows.append({'name': name, 'mean_ndcg': float(np.mean(scores)),
            'median_ms': float(np.median(timings)), 'p95_ms': float(np.quantile(timings, .95)),
            'per_query': scores, 'rankings': rankings})
    baseline = max(rows[:2], key=lambda x: x['mean_ndcg'])
    for row in rows:
        row['delta_vs_best_single'] = row['mean_ndcg'] - baseline['mean_ndcg']
        row['delta_ci95'] = paired_interval(row['per_query'], baseline['per_query'])
    return {'query_count': len(queries), 'limit': limit, 'baseline': baseline['name'],
            'bootstrap_seed': 42, 'bootstrap_samples': 2000, 'runs': rows}
