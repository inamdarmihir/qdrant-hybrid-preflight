"""Download, embed, index and evaluate BEIR SciFact on a disposable Qdrant server."""
import argparse
import csv
import hashlib
import numpy as np
import json
import os
import platform
import random
import time
import urllib.request
import zipfile
from pathlib import Path
import importlib.metadata as metadata
from fastembed import TextEmbedding, SparseTextEmbedding
from fastembed.sparse.bm25 import remove_non_alphanumeric
from qdrant_client import QdrantClient, models
from preflight import preflight_hybrid_search
from sweep import run_sweep

URL = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip'
MD5 = '5f7d1de60b170fc8027bb7898e2efca1'
MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def load_data(folder):
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / 'scifact.zip'
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    if hashlib.md5(archive.read_bytes()).hexdigest() != MD5:
        raise ValueError('SciFact archive checksum mismatch; remove the cached archive')
    with zipfile.ZipFile(archive) as z:
        # Do not trust archive member paths to escape the data folder.
        for name in z.namelist():
            if not (folder / name).resolve().is_relative_to(folder.resolve()):
                raise ValueError('Unsafe archive member')
        z.extractall(folder)
    root = folder / 'scifact'
    docs = sorted([json.loads(x) for x in (root / 'corpus.jsonl').read_text().splitlines()],
                  key=lambda d: d['_id'])
    query_texts = {x['_id']: x['text'] for x in
        (json.loads(line) for line in (root / 'queries.jsonl').read_text().splitlines())}
    qrels = {}
    with (root / 'qrels/test.tsv').open() as f:
        for row in csv.DictReader(f, delimiter='\t'):
            qrels.setdefault(row['query-id'], {})[row['corpus-id']] = int(row['score'])
    return docs, query_texts, qrels


def tokenized_average(encoder, texts):
    """Match pinned FastEmbed BM25's actual length, including stopword removal.

    Its public token_count counts BEFORE stemming/filtering, so it is unsuitable
    for the BM25 denominator. This adapter deliberately uses pinned internals.
    """
    return sum(len(encoder._stem(encoder.tokenizer.tokenize(remove_non_alphanumeric(t))))
               for t in texts) / len(texts)


def write_table(report, path):
    with path.open('w') as f:
        w = csv.writer(f)
        w.writerow(['configuration', 'ndcg_at_10', 'median_ms', 'p95_ms',
                    'delta_vs_best_single', 'ci95_low', 'ci95_high'])
        for r in report['runs']:
            w.writerow([r['name'], r['mean_ndcg'], r['median_ms'], r['p95_ms'],
                        r['delta_vs_best_single'], *r['delta_ci95']])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://localhost:6333')
    p.add_argument('--collection', default='scifact_hybrid_preflight')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--output-dir', default='results')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=32)
    args = p.parse_args()
    client = QdrantClient(url=args.url, api_key=os.environ.get('QDRANT_API_KEY'), timeout=120)
    if client.collection_exists(args.collection):
        p.error('Collection exists. Refusing to overwrite it; choose a fresh --collection.')
    started = time.time()
    docs, texts_by_query, qrels = load_data(Path(args.data_dir))
    if len(docs) != 5183 or len(qrels) != 300:
        raise ValueError('Unexpected SciFact corpus or labeled query count')
    texts = [f"{d['title']} {d['text']}".strip() for d in docs]
    sparse_model = SparseTextEmbedding('Qdrant/bm25')
    avg = tokenized_average(sparse_model.model, texts)
    sparse_model.model.avg_len = avg
    dense_model = TextEmbedding(MODEL, threads=args.threads)
    print(f'{len(docs)} documents; {len(qrels)} labeled queries; BM25 avg_len={avg:.6f}', flush=True)
    client.create_collection(args.collection, shard_number=1,
        vectors_config={'dense': models.VectorParams(size=384, distance=models.Distance.COSINE)},
        sparse_vectors_config={'bm25': models.SparseVectorParams(modifier=models.Modifier.IDF)})
    # SciFact's numeric document IDs are valid Qdrant unsigned integer point IDs.
    # Cache each encoded batch, keyed by corpus checksum and actual model bytes.
    artifact = Path(dense_model.model._model_dir) / 'model.onnx'
    model_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    cache_key = hashlib.sha256((MD5 + model_sha + metadata.version('fastembed') +
                               'title space abstract').encode()).hexdigest()[:16]
    cache = Path(args.data_dir) / ('dense-' + cache_key)
    cache.mkdir(exist_ok=True)
    def dense_batches():
        for offset in range(0, len(texts), args.batch_size):
            part = cache / f'{offset:05d}.npy'
            expected = min(args.batch_size, len(texts) - offset)
            if part.exists():
                vectors = np.load(part, allow_pickle=False)
            else:
                vectors = np.array(list(dense_model.embed(
                    texts[offset:offset + args.batch_size], batch_size=args.batch_size)))
                np.save(part, vectors)
            if vectors.shape != (expected, 384) or not np.isfinite(vectors).all():
                raise ValueError('Invalid embedding cache batch')
            yield from vectors
    dense_iter = dense_batches()
    sparse_iter = sparse_model.embed(texts, batch_size=args.batch_size)
    batch = []
    for doc, dense, sparse in zip(docs, dense_iter, sparse_iter):
        batch.append(models.PointStruct(id=int(doc['_id']), vector={
            'dense': dense.tolist(), 'bm25': models.SparseVector(
                indices=sparse.indices.tolist(), values=sparse.values.tolist())}))
        if len(batch) >= args.batch_size:
            client.upsert(args.collection, points=batch, wait=True); batch = []
    if batch:
        client.upsert(args.collection, points=batch, wait=True)
    if client.count(args.collection, exact=True).count != len(docs):
        raise RuntimeError('Indexed point count mismatch')
    print('Index populated; embedding queries', flush=True)
    ids = sorted(qrels)
    qtexts = [texts_by_query[q] for q in ids]
    queries = [{'id': qid, 'dense': dense.tolist(), 'sparse': {
        'indices': sparse.indices.tolist(), 'values': sparse.values.tolist()}, 'qrels': qrels[qid]}
        for qid, dense, sparse in zip(ids, dense_model.query_embed(qtexts, batch_size=args.batch_size),
                                      sparse_model.query_embed(qtexts))]
    random.Random(42).shuffle(queries)
    dev, test = queries[:150], queries[150:]
    request = models.QueryRequest(prefetch=[
        models.Prefetch(query=dev[0]['dense'], using='dense', limit=200),
        models.Prefetch(query=models.SparseVector(**dev[0]['sparse']), using='bm25', limit=200)],
        query=models.RrfQuery(rrf=models.Rrf(k=2)), limit=10)
    findings = preflight_hybrid_search(client, args.collection, 'bm25', request,
                                      avg, avg, len(dev))
    if any(f.level != 'info' for f in findings):
        raise RuntimeError(f'Preflight did not pass: {findings}')
    settings = dict(depths=(50, 200), ks=(2, 5, 20, 61),
                    weight_pairs=((1., 1.), (2., 1.), (1., 2.)))
    print('Running 28-config development grid', flush=True)
    development = run_sweep(client, args.collection, dev, **settings)
    chosen = max(development['runs'], key=lambda r: (r['mean_ndcg'], r['name']))['name']
    # Only the development winner and baselines touch the held-out half.
    print(f'Chosen on development queries: {chosen}; evaluating held-out queries', flush=True)
    heldout = run_sweep(client, args.collection, test, config_names={'dense', 'sparse', chosen}, **settings)
    cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                if line.startswith('model name')), platform.processor()) if Path('/proc/cpuinfo').exists() else platform.processor()
    info = {'dataset_url': URL, 'archive_md5': MD5, 'documents': len(docs),
        'dev_query_ids': [q['id'] for q in dev], 'test_query_ids': [q['id'] for q in test],
        'chosen_on_dev': chosen, 'split_seed': 42, 'bm25_avg_len': avg,
        'dense_model': MODEL, 'dense_max_tokens': 256, 'sparse_model': 'Qdrant/bm25',
        'bm25_k': 1.2, 'bm25_b': .75, 'text': 'title + space + abstract',
        'versions': {k: metadata.version(k) for k in ['qdrant-client','fastembed','numpy','onnxruntime','tokenizers']},
        'python': platform.python_version(), 'cpu': cpu, 'logical_cpus': os.cpu_count(),
        'platform': platform.platform(), 'embedding_threads': args.threads,
        'model_onnx_sha256': model_sha, 'dataset_sha256': hashlib.sha256((Path(args.data_dir) / 'scifact.zip').read_bytes()).hexdigest(),
        'server_version': client.info().version,
        'batch_size': args.batch_size, 'elapsed_seconds': time.time() - started,
        'search': 'exact dense retrieval, one server shard; no quantization',
        'preflight': [f.to_dict() for f in findings]}
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    for name, obj in [('development',development),('heldout',heldout),('metadata',info)]:
        (out / f'{name}.json').write_text(json.dumps(obj, indent=2) + '\n')
    write_table(development, out / 'development.csv'); write_table(heldout, out / 'heldout.csv')
    for r in heldout['runs']:
        print(f"{r['name']}: nDCG@10={r['mean_ndcg']:.6f}, delta={r['delta_vs_best_single']:.6f}, CI={r['delta_ci95']}", flush=True)
    print(f'Results: {out}, elapsed {info["elapsed_seconds"]:.1f}s', flush=True)
    client.close()


if __name__ == '__main__':
    main()
