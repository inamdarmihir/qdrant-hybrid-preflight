# Qdrant Hybrid Preflight

A runnable companion to [Your Hybrid Search Benchmark May Be Lying to You](https://mihirinamdar.substack.com/p/your-hybrid-search-benchmark-may). This repo expands the inline preflight check and adds a small sweep runner. The published article has not been changed.

It checks what the collection and supplied request actually expose. It says "unverified" when a fact lives outside Qdrant. It does not turn missing metadata into a pass.

## What it does

| Check | Evidence | Limit |
| --- | --- | --- |
| Sparse IDF modifier | Live collection config | Caller declares whether encoder expects IDF. BM25/miniCOIL usually do; SPLADE already weights terms. |
| BM25 average length | Supplied encoder value and measured tokenized corpus mean | Neither can be recovered from stored sparse vectors. Raw whitespace counts are not the encoder's token counts. |
| Fusion placement | Recursive walk of the supplied QueryRequest | Nested fusion can run per shard. It may be intentional before rescoring. This does not inspect a production client's real network traffic. |
| Fusion score threshold | Supplied root or nested fusion request | A dense similarity threshold is not a fused-score threshold. |
| Label count | Supplied count or evaluation query list | Below 50 is a warning heuristic, not a significance test. Above 50 is not proof of power. |

The sweep runs dense and sparse baselines, DBSF, and configurable RRF `k`, candidate depths and weight pairs. Fusion is always at the query root. Output includes per-query rankings, nDCG, serial median/p95 request time, and paired-bootstrap delta intervals against the better single retriever.

## Install and run

Python 3.10 or newer. Tested here on Python 3.10.12, qdrant-client 1.17.1 and Qdrant server 1.17.1. RRF parameters require Qdrant 1.17 or later. Start with the pinned versions before upgrading either side.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python demo.py
```

No API key, Docker, embedding downloads or remote service is needed for the local demo. It creates six points in an in-memory collection and executes 28 configurations over 12 queries. Each point and query uses hand-authored vectors. The sparse vector is called `bm25` for the API example, but is NOT produced by a BM25 encoder.

For a real server smoke test, start a disposable pinned container:

```bash
docker run --rm -p 6333:6333 qdrant/qdrant:v1.17.1
# in another terminal with the virtual environment active:
python demo.py --url http://localhost:6333 --output results/my-server-run.json
```

The demo writes a collection called `hybrid_preflight_demo`. It refuses to overwrite an existing collection. Use a fresh instance, not production. The CLI below only reads collection config and executes queries; it never creates, deletes or changes points.

## Check an existing collection

```bash
# Optional authentication: export QDRANT_API_KEY in your environment.
# Never put keys in a query file or commit them.
python cli.py check \
  --url http://localhost:6333 --collection my_collection \
  --sparse-name bm25 --request example-request.json \
  --bm25-avg-len 42 --measured-avg-len 40 \
  --labeled-query-count 120 --output check.json
```

The numbers 42, 40 and 120 are examples, not measurements. Replace them with your encoder configuration, mean token length after its exact stemming/stopword rules, and label count. Replace the request with your actual production request. Use `--encoder-includes-idf` for an encoder with built-in IDF; it warns if Qdrant would apply IDF again. A missing sparse vector or incompatible IDF setting returns exit code 2. Warnings remain in the report and are not silently treated as errors or passes. Network/collection errors propagate.

## Sweep your own labels

Supply a JSON list shaped like `example-queries.json`. Each query has a unique ID, its dense and sparse query vectors, and `qrels` mapping point IDs (as strings) to nonnegative relevance grades. Every query must have a positive judgment. Unknown returned documents count as relevance zero. Point IDs must match your collection's IDs, not a document title.

```bash
python cli.py sweep \
  --url http://localhost:6333 --collection my_collection \
  --dense-name dense --sparse-name bm25 \
  --queries my-heldout-queries.json --request my-request.json \
  --bm25-avg-len 42 --measured-avg-len 40 \
  --depths 20 50 200 --ks 2 5 20 61 --limit 10 \
  --output sweep.json
```

Generate query vectors with the SAME models, dimensions, token vocabulary and preprocessing that indexed the collection. This repo does not invent embeddings or corpus statistics. To sweep weights from Python:

```python
from sweep import run_sweep
report = run_sweep(client, "my_collection", queries,
                   depths=(20, 50, 200), ks=(2, 5, 20, 61),
                   weight_pairs=((1., 1.), (2., 1.), (1., 2.)))
```

Use a development set to choose settings, then rerun the chosen configuration on untouched test queries. The included grid reports exploratory intervals, not corrected significance after choosing the best of many trials. nDCG uses `(2**grade - 1) / log2(rank + 1)` with one-based rank. Bootstrap resamples paired per-query differences 2,000 times, seed 42. Repeated/paraphrased queries are not independent evidence.

## What actually ran

On 2026-10-05, the automated implementation run completed:

- 22 tests, all passing (see `RUN.md`).
- Local in-memory demo: 28 configurations, 12 queries, six hand-authored points.
- Real Qdrant 1.17.1 server binary: the same demo, plus a read-only CLI collection check.

| Toy run | Dense nDCG@10 | Sparse nDCG@10 | DBSF depth 10 | Equal-weight RRF k=2, depth 10 |
| --- | ---: | ---: | ---: | ---: |
| Local in-memory | 0.717046 | 0.898354 | 0.898354 | 0.896176 |
| Qdrant server | 0.717046 | 0.898354 | 0.898354 | 0.905191 |

These are a smoke test, NOT retrieval quality evidence. The fixture has deliberate score ties, six points and reused query patterns. Local/server tie order and even repeated server request order can change ranks. Do not read a winner or a production improvement into these numbers. Depths 10 and 20 both exceed the entire six-point corpus, so this fixture does not validate candidate truncation behavior.

Measured CSV summaries are in `toy-local.csv` and `toy-server.csv`. Running the demo writes full JSON with individual rankings. Timings are serial cold/warm mixed request timings from one sandbox, not concurrency or latency-budget benchmarks. Local mode is not a substitute for server sharding, HNSW/index behavior, or production data. No real corpus, tokenizer-based avg_len, BM25 text encoding or multi-shard experiment has been measured here.

## Files

- `preflight.py`: typed findings and request/config checks.
- `sweep.py`: validation, query grid, metric and paired intervals.
- `cli.py`: read-only commands for an existing collection.
- `demo.py`: disposable synthetic fixture.
- `example-queries.json`, `example-request.json`: toy query/request schemas.
- `test_preflight.py`, `test_sweep.py`: validation, recursive checks and local-engine integration.
- `RUN.md`, `toy-local.csv`, `toy-server.csv`: actual smoke-test summaries, not research claims.

## Sources and relationship to the article

The starting function is in [the AI Hive source](https://github.com/inamdarmihir/aihive/blob/da26c9b59ade8286de5997113367825f6bede662/content/posts/qdrant-hybrid-search-sweep-it-yourself.md). This version checks exact fusion types recursively, reports missing measured avg_len explicitly, validates impossible inputs, and supports encoders whose sparse weights already include IDF. It does not reproduce or claim ownership of Qdrant's published benchmark numbers.

- [Qdrant: How to Tune Hybrid Search](https://qdrant.tech/documentation/search-tuning/how-to-tune-hybrid-search/)
- [Qdrant: What to Check Before Tuning a Collection](https://qdrant.tech/documentation/search-tuning/before-tuning-a-qdrant-collection/)
- [Qdrant Query API and hybrid queries](https://qdrant.tech/documentation/concepts/hybrid-queries/)

No license has been chosen yet. This repo is not a Qdrant-endorsed benchmark.
