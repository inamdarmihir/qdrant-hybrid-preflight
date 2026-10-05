# Qdrant Hybrid Preflight

A runnable companion to [Your Hybrid Search Benchmark May Be Lying to You](https://mihirinamdar.substack.com/p/your-hybrid-search-benchmark-may). It checks the settings a hybrid benchmark depends on, then runs real retrieval on BEIR SciFact. The published article is unchanged.

No mock documents, fake embeddings or invented relevance labels. The default command downloads all 5,183 scientific abstracts and the 300 BEIR test queries with real judgments. It embeds title + abstract with MiniLM, encodes BM25 with FastEmbed, indexes Qdrant, checks the setup, tunes on 150 queries, and evaluates the chosen configuration on the other 150.

## Run it

Python 3.10 or newer, Docker, CPU, internet for the first dataset/model downloads, and a few GB of disk space. On the measured 2-CPU machine, embedding the corpus took several minutes. Later runs cache dense batches by dataset checksum and actual ONNX model hash. Network/download time varies.

Start a disposable server in one terminal:

```bash
docker run --rm -p 6333:6333 qdrant/qdrant:v1.17.1
```

In another terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python benchmark.py
```

`python benchmark.py` is the single download-index-check-sweep command. It needs the running server, not a Qdrant cloud account or API key. It writes `results/development.csv`, `heldout.csv`, full per-query JSON rankings, and metadata with the exact split IDs and model hash. The first run downloads the pinned SciFact archive and model files.

The collection name is `scifact_hybrid_preflight`. The command REFUSES to overwrite an existing collection. For another run use a fresh disposable server or a new name:

```bash
python benchmark.py --collection scifact_second_run
python -m pytest -q test_benchmark.py
```

The tests also read the checksum-verified SciFact data and real sparse query vectors. No dummy fixture is required. Dataset files and model caches are not committed.

## What the measured run says

Run on 2026-10-05 against Qdrant server 1.17.1. The development set chose DBSF with 50 candidates from each retriever. Its held-out result was higher than either individual retriever in this setup.

| Configuration | Development nDCG@10, 150 queries | Held-out nDCG@10, 150 queries |
| --- | ---: | ---: |
| Dense MiniLM | 0.601115 | 0.689049 |
| BM25 sparse | 0.651940 | 0.725251 |
| DBSF, depth 50 | 0.688005 | 0.771831 |

On the held-out queries DBSF beat sparse by **0.046580 absolute nDCG@10**. The paired percentile-bootstrap 95% interval for that delta was **[0.020206, 0.073512]**, 2,000 resamples, seed 42. That interval excludes zero for this fixed chosen configuration and this sample. It is not a universal claim that DBSF wins, and it does not establish causality for the preflight checks.

`development.csv` contains the complete 28-configuration tuning grid: dense, sparse, DBSF at depths 50/200, and RRF at k=2/5/20/61 with weights 1:1/2:1/1:2 at both depths. `heldout.csv` contains only the two baselines and the development winner. A second execution with the same real embeddings reproduced these quality scores. Captured latency columns are serial HTTP request times on this machine, not load-test or production p95 claims.

## Reproduction details

- Corpus: BEIR SciFact, archive MD5 `5f7d1de60b170fc8027bb7898e2efca1`; URL and SHA256 in `metadata.json`.
- All 5,183 documents; title + one space + abstract, no chunking.
- All 300 judged BEIR test queries. Sort query IDs lexicographically, shuffle with Python `random.Random(42)`, first 150 for development, remaining 150 held out. This is a custom split of BEIR's test set, NOT the official BEIR train/test protocol. Exact IDs are recorded in `metadata.json`.
- Dense: `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, cosine, FastEmbed ONNX, 256-token truncation. Actual ONNX SHA256 recorded in `metadata.json`.
- Sparse: FastEmbed `Qdrant/bm25`, English stemming/stopword filtering, k=1.2, b=0.75, server-side IDF.
- BM25 avg_len: **151.38028169014083**, measured AFTER the same tokenization, filtering and stemming used for encoding. FastEmbed's public `token_count` counts before filtering, so the pinned adapter uses its internal tokenizer/stemmer. This should be rechecked when upgrading FastEmbed.
- Qdrant 1.17.1, one shard, exact dense search, no quantization or reranker. This isolates fusion from approximate-neighbor recall.
- Python 3.10.12; qdrant-client 1.17.1; fastembed 0.7.4; numpy 2.2.6; onnxruntime 1.23.2; tokenizers 0.23.2.
- Linux x86_64; Intel Xeon @ 2.60GHz, 2 logical CPUs, approximately 2 GB RAM. Embedding threads=2, batch size=32. CPU only.
- The complete corpus was really embedded. The final repeat used that keyed dense cache; its roughly 30-second elapsed time is NOT first-run runtime. `metadata.json` records the repeat's actual timing.

nDCG uses exponential gains `(2**grade - 1)` and logarithmic rank discount. SciFact judgments here are binary, so exponential and linear gains agree. Unjudged returned documents count as zero relevance. The bootstrap resamples per-query paired score deltas. Baseline selection is the better single retriever on each reported split; sparse is that baseline in this run.

## What preflight can and cannot check

| Check | Evidence | Limit |
| --- | --- | --- |
| Sparse IDF modifier | Live collection config | Caller declares encoder expectations. BM25/miniCOIL need server IDF; SPLADE already weights terms. |
| BM25 avg_len | Actual encoder value + measured tokenized corpus mean | Cannot recover these from stored sparse vectors. |
| Fusion placement | Recursive walk of the QueryRequest | Nested fusion runs per shard on a sharded server; may be intentional before rescoring. |
| Fusion score threshold | Request root/nested fusion | Dense similarity thresholds are not fused-score thresholds. |
| Label count | Actual evaluation query count | Small-sample heuristic, not a power or significance guarantee. |

The pipeline supplies the real encoder value, measured average, request and count. Missing inputs in the general CLI stay "unverified", never silently pass. A connection failure propagates instead of producing an empty clean report.

## Use your own collection

The preflight/sweep CLI is read-only: no point writes or collection deletion. Supply vectors made with the SAME encoders as your indexed data and relevance judgments keyed by Qdrant point ID strings. Each query JSON entry has `id`, `dense`, `sparse` (`indices` + `values`), and `qrels` (point ID to grade). Query IDs must be unique. Every query needs a positive judgment.

```bash
# Optional: set QDRANT_API_KEY in your environment, never in a committed file.
python cli.py check --url http://localhost:6333 --collection my_collection \
  --sparse-name bm25 --request my-production-request.json \
  --bm25-avg-len 151.38 --measured-avg-len 151.38 \
  --labeled-query-count 150 --output check.json

python cli.py sweep --url http://localhost:6333 --collection my_collection \
  --dense-name dense --sparse-name bm25 --queries my-real-queries.json \
  --request my-production-request.json \
  --bm25-avg-len 151.38 --measured-avg-len 151.38 \
  --depths 50 200 --ks 2 5 20 61 --limit 10 --output sweep.json
```

Replace those values with YOUR measured corpus and encoder values, not SciFact's. `--encoder-includes-idf` handles already-weighted sparse encoders. Errors stop the sweep; warnings remain visible. The generic CLI evaluates the grid on the supplied labels only. Use separate tuning and held-out files yourself, or use the SciFact pipeline's split as the pattern.

## Limits

One dataset, one model pair, one fixed query split. Query-level bootstrap assumes independent queries; related claims can weaken that assumption. Selection happened only on development, but this is not a preregistered experiment. A statistically positive delta here does not predict another corpus.

Dense abstracts can be truncated at 256 tokens. BM25 sees full text. No multi-shard experiment, no ANN-recall evaluation, no online traffic, no concurrency test, no alternative encoder sweep. The preflight checks were satisfied in the measured run; there is no controlled broken-vs-fixed ablation proving how much each check helps.

## Files and sources

`benchmark.py`: end-to-end real-data pipeline. `preflight.py`: checks. `sweep.py`: metric, grid and intervals. `cli.py`: existing-collection adapter. `test_benchmark.py`: real-data tests. `development.csv`, `heldout.csv`, `metadata.json`, `RUN.md`: measured results/provenance.

Starting function: [AI Hive source](https://github.com/inamdarmihir/aihive/blob/da26c9b59ade8286de5997113367825f6bede662/content/posts/qdrant-hybrid-search-sweep-it-yourself.md).

- [BEIR dataset list](https://github.com/beir-cellar/beir/wiki/Datasets-available)
- [SciFact dataset card](https://huggingface.co/datasets/BeIR/scifact), source data CC-BY-SA-4.0. Dataset is downloaded, not redistributed here.
- [Qdrant hybrid tuning](https://qdrant.tech/documentation/search-tuning/how-to-tune-hybrid-search/)
- [Qdrant pre-tuning checks](https://qdrant.tech/documentation/search-tuning/before-tuning-a-qdrant-collection/)

Code license not chosen yet. This is not a Qdrant-endorsed benchmark.
