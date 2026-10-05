"""Inspect the checkable subset of silent hybrid-search failures."""
from dataclasses import asdict, dataclass
import math
from qdrant_client import models


@dataclass(frozen=True)
class Finding:
    code: str
    level: str
    message: str

    def to_dict(self):
        return asdict(self)


def preflight_hybrid_search(client, collection_name, sparse_vector_name,
                            query_request=None, bm25_avg_len=None,
                            measured_avg_len=None, labeled_query_count=None,
                            expects_idf=True, drift_threshold=0.15):
    """Return findings, never claim missing metadata was verified.

    expects_idf=False supports encoders such as SPLADE that include term weights.
    Average lengths must use the encoder's tokenization, not raw word counts.
    A label-count warning is a heuristic, not a significance test.
    Connection errors propagate: an unavailable collection is not a clean bill.
    """
    if not 0 <= drift_threshold < 1:
        raise ValueError('drift_threshold must be in [0, 1)')
    for name, value in [('bm25_avg_len', bm25_avg_len),
                        ('measured_avg_len', measured_avg_len)]:
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f'{name} must be finite and positive')
    if labeled_query_count is not None and (isinstance(labeled_query_count, bool)
            or not isinstance(labeled_query_count, int) or labeled_query_count < 0):
        raise ValueError('labeled_query_count must be a nonnegative integer')
    findings = []
    def add(code, level, message):
        findings.append(Finding(code, level, message))
    info = client.get_collection(collection_name)
    sparse = info.config.params.sparse_vectors or {}
    params = sparse.get(sparse_vector_name)
    if params is None:
        add('sparse_missing', 'error', f'No sparse vector named {sparse_vector_name!r}.')
    elif expects_idf and params.modifier != models.Modifier.IDF:
        add('idf_missing', 'error', 'This encoder expects server-side IDF but it is unset.')
    elif not expects_idf and params.modifier == models.Modifier.IDF:
        add('idf_double_weight', 'warning', 'Server IDF is enabled for an encoder declared to include IDF.')
    else:
        add('idf_checked', 'info', 'Sparse IDF setting matches the declared encoder expectation.')
    if bm25_avg_len is None or measured_avg_len is None:
        add('avg_len_unverified', 'warning', 'Supply both the encoder avg_len and a measured tokenized corpus average; neither is readable from collection config.')
    else:
        drift = abs(bm25_avg_len - measured_avg_len) / measured_avg_len
        add('avg_len_drift' if drift > drift_threshold else 'avg_len_checked',
            'warning' if drift > drift_threshold else 'info',
            f'avg_len relative drift={drift:.2%}; threshold={drift_threshold:.0%}. Inputs supplied by caller.')
    if query_request is None:
        add('query_unverified', 'warning', 'No request supplied; fusion placement and threshold are unverified.')
    else:
        def walk(node, path, depth):
            query = getattr(node, 'query', None)
            fused = isinstance(query, (models.FusionQuery, models.RrfQuery))
            if fused and depth:
                add('nested_fusion', 'warning', f'{path}.query fuses inside prefetch; on a sharded server this runs per shard. May be intentional before rescoring.')
            if fused and getattr(node, 'score_threshold', None) is not None:
                add('fusion_threshold', 'warning', f'{path}.score_threshold applies to fused scores, not dense similarity; validate on labels.')
            children = getattr(node, 'prefetch', None) or []
            if not isinstance(children, list):
                children = [children]
            for index, child in enumerate(children):
                walk(child, f'{path}.prefetch[{index}]', depth + 1)
        walk(query_request, 'request', 0)
        add('query_checked', 'info', 'Inspected the supplied request recursively; this does not prove production sends the same request.')
    if labeled_query_count is None:
        add('labels_unverified', 'warning', 'Labeled query count was not supplied.')
    elif labeled_query_count < 50:
        add('few_labels', 'warning', f'Only {labeled_query_count} labeled queries. Small-sample heuristic only; use held-out queries and uncertainty intervals.')
    else:
        add('labels_counted', 'info', f'{labeled_query_count} labeled queries; count alone does not establish statistical power.')
    return findings
