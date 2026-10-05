"""Read-only preflight and query sweep against an existing Qdrant collection."""
import argparse
import json
import os
from pathlib import Path
from qdrant_client import QdrantClient, models
from .preflight import preflight_hybrid_search
from .sweep import run_sweep


def main(argv=None):
    p = argparse.ArgumentParser(prog='qdrant-hybrid-preflight', description=__doc__)
    p.add_argument('command', choices=['check', 'sweep'])
    p.add_argument('--url', required=True)
    p.add_argument('--collection', required=True)
    p.add_argument('--sparse-name', default='bm25')
    p.add_argument('--dense-name', default='dense')
    p.add_argument('--encoder-includes-idf', action='store_true')
    p.add_argument('--bm25-avg-len', type=float)
    p.add_argument('--measured-avg-len', type=float)
    p.add_argument('--labeled-query-count', type=int)
    p.add_argument('--request', help='JSON QueryRequest matching your production query')
    p.add_argument('--queries', help='JSON list with id, dense, sparse, qrels')
    p.add_argument('--depths', type=int, nargs='+', default=[20, 50, 200])
    p.add_argument('--ks', type=int, nargs='+', default=[2, 5, 20, 61])
    p.add_argument('--limit', type=int, default=10)
    p.add_argument('--output', required=True)
    args = p.parse_args(argv)
    client = QdrantClient(url=args.url, api_key=os.environ.get('QDRANT_API_KEY'))
    request = models.QueryRequest.model_validate_json(Path(args.request).read_text()) if args.request else None
    queries = json.loads(Path(args.queries).read_text()) if args.queries else None
    findings = preflight_hybrid_search(client, args.collection, args.sparse_name,
        query_request=request, bm25_avg_len=args.bm25_avg_len,
        measured_avg_len=args.measured_avg_len,
        labeled_query_count=len(queries) if queries is not None else args.labeled_query_count,
        expects_idf=not args.encoder_includes_idf)
    report = {'preflight': [x.to_dict() for x in findings]}
    if args.command == 'sweep':
        if queries is None:
            p.error('sweep requires --queries')
        if any(x.level == 'error' for x in findings):
            p.error('preflight errors must be fixed before sweep')
        report['sweep'] = run_sweep(client, args.collection, queries,
            args.dense_name, args.sparse_name, args.depths, args.ks, limit=args.limit)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    client.close()
    print(json.dumps(report['preflight'], indent=2))
    if any(x.level == 'error' for x in findings):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
