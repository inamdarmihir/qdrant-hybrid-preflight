"""Deterministic toy fixture: executes Qdrant, but is not an embedding benchmark."""
import argparse
import json
import platform
from pathlib import Path
from qdrant_client import QdrantClient, models
from preflight import preflight_hybrid_search
from sweep import run_sweep

COLLECTION = 'hybrid_preflight_demo'


def build_fixture(client):
    """Six hand-authored documents and query vectors; no model downloads."""
    client.create_collection(COLLECTION,
        vectors_config={'dense': models.VectorParams(size=3, distance=models.Distance.COSINE)},
        sparse_vectors_config={'bm25': models.SparseVectorParams(modifier=models.Modifier.IDF)})
    vectors = [([1., .1, 0.], [0, 1]), ([.9, .2, .1], [0, 2]),
               ([.1, 1., 0.], [3, 4]), ([.2, .9, .1], [3, 5]),
               ([0., .1, 1.], [6, 7]), ([.1, .2, .9], [6, 8])]
    client.upsert(COLLECTION, points=[models.PointStruct(id=i + 1,
        vector={'dense': dense, 'bm25': models.SparseVector(indices=terms, values=[1., 1.])},
        payload={'fixture': True}) for i, (dense, terms) in enumerate(vectors)])
    queries = []
    # Intentional lexical/semantic disagreements exercise fusion, not accuracy.
    for i in range(12):
        group = i % 3
        target = group * 2 + 1
        dense = [0.1, 0.1, 0.1]
        dense[(group + (i // 3) % 2) % 3] = 1.
        queries.append({'id': f'q{i:02d}', 'dense': dense,
            'sparse': {'indices': [group * 3, group * 3 + 1 + (i % 2)], 'values': [1., 1.]},
            'qrels': {str(target): 2, str(target + 1): 1}})
    return queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', help='Dedicated local server, e.g. http://localhost:6333')
    parser.add_argument('--output', default='results/toy-local.json')
    args = parser.parse_args()
    client = QdrantClient(url=args.url) if args.url else QdrantClient(':memory:')
    # Never delete an existing collection. This demo is for a disposable instance.
    if client.collection_exists(COLLECTION):
        raise SystemExit(f'{COLLECTION} exists; refusing to overwrite it. Use a fresh instance.')
    queries = build_fixture(client)
    request = models.QueryRequest(prefetch=[
        models.Prefetch(query=queries[0]['dense'], using='dense', limit=10),
        models.Prefetch(query=models.SparseVector(**queries[0]['sparse']), using='bm25', limit=10)],
        query=models.RrfQuery(rrf=models.Rrf(k=2)), limit=10)
    report = run_sweep(client, COLLECTION, queries, depths=(10, 20),
        weight_pairs=((1., 1.), (2., 1.), (1., 2.)))
    report['fixture'] = '6 hand-authored vectors; 12 toy queries; not BM25-encoded text'
    report['engine'] = 'Qdrant server' if args.url else 'qdrant-client local in-memory'
    report['python'] = platform.python_version()
    report['client_version'] = '1.17.1'
    report['preflight'] = [f.to_dict() for f in preflight_hybrid_search(client, COLLECTION,
        'bm25', query_request=request, labeled_query_count=len(queries))]
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Wrote {path}: {len(report["runs"])} configurations, {len(queries)} queries')
    print('name,mean_ndcg,delta_vs_best_single,ci95')
    for row in report['runs']:
        print(f'{row["name"]},{row["mean_ndcg"]:.6f},{row["delta_vs_best_single"]:.6f},{row["delta_ci95"]}')
    client.close()


if __name__ == '__main__':
    main()
