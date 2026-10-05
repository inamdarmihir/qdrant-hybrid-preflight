# 🧪 Qdrant Hybrid Preflight: Check the Setup Before You Trust a Hybrid-Search Result

> **A read-only preflight for existing Qdrant collections, plus a runnable SciFact experiment with a development split and a held-out split. Every number below comes from the committed run.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)
[![Tested with Qdrant 1.17.1](https://img.shields.io/badge/tested%20with-Qdrant%201.17.1-dc244c.svg)](RUN.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Last commit](https://img.shields.io/github/last-commit/inamdarmihir/qdrant-hybrid-preflight)](https://github.com/inamdarmihir/qdrant-hybrid-preflight/commits/main)

---

## 🚀 What Is This?

This repo combines a read-only preflight check for existing Qdrant collections with a runnable SciFact experiment. It checks encoder assumptions and query structure, sweeps fusion settings on a development split, then evaluates the selected configuration on held-out queries.

The companion [article](https://mihirinamdar.substack.com/p/your-hybrid-search-benchmark-may) predates this repo. The benchmark adds measured evidence without changing the published article.

- **🔍 Read-only**: `cli.py check` and `cli.py sweep` inspect a collection and never write to it
- **📊 Measured**: SciFact, 5,183 documents, tuned on 150 queries and scored on 150 held-out queries
- **🧾 Inspectable**: CSV/JSON results, `metadata.json` and `RUN.md` are committed next to the code
- **⚖️ Honest**: one dataset, one encoder pair, one split; the limits are listed below

[Quickstart](#-quick-start) · [Results](#-results) · [Preflight checks](#-preflight-checks) · [Your collection](#-your-collection) · [Limits](#-limits)

---

## ⚡ Quick Start

Python 3.10+, Docker, CPU, internet for the first data/model downloads, and several GB of disk space. Use a disposable Qdrant server:

```bash
git clone https://github.com/inamdarmihir/qdrant-hybrid-preflight.git
cd qdrant-hybrid-preflight
docker run --rm -p 127.0.0.1:6333:6333 qdrant/qdrant:v1.17.1
```

In a second terminal:

```bash
cd qdrant-hybrid-preflight
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python benchmark.py
```

This downloads checksum-verified SciFact data, embeds all 5,183 abstracts, indexes dense and BM25 vectors, runs preflight, tunes on 150 queries and evaluates the winner on the other 150. There are no synthetic documents or invented relevance labels in this experiment.

It refuses to overwrite its existing collection. For another run, use a fresh server or collection name:

```bash
python benchmark.py --collection scifact_second_run
python -m pytest -q test_benchmark.py
```

Generated outputs go to `results/`: development and held-out CSV/JSON, per-query rankings and metadata. Dense embeddings are cached using dataset and model provenance. Cached runtime is not first-run runtime.

---

## 📊 Results

Committed run: October 5, 2026, Qdrant 1.17.1. Development selected **DBSF with 50 candidates from each retriever** out of a 28-configuration grid.

| Configuration | Development nDCG@10 (150 queries) | Held-out nDCG@10 (150 queries) |
| --- | ---: | ---: |
| Dense MiniLM | 0.601115 | 0.689049 |
| BM25 sparse | 0.651940 | 0.725251 |
| DBSF, depth 50 | 0.688005 | 0.771831 |

Held-out delta vs sparse: **+0.046580 nDCG@10**. Paired percentile-bootstrap 95% interval: **[0.020206, 0.073512]**, 2,000 resamples, seed 42.

This interval describes the fixed chosen configuration on this sample. It is not evidence that DBSF always wins, or that the preflight checks caused the gain. Query independence is a bootstrap assumption that related scientific claims may weaken.

Evidence: [`development.csv`](development.csv), [`heldout.csv`](heldout.csv), [`metadata.json`](metadata.json) and [`RUN.md`](RUN.md). The committed evidence is at the repository root; new benchmark runs write under `results/` by default. Serial HTTP timings are not concurrent-load latency measurements.

---

## 🛠️ Preflight Checks

| Check | What it inspects | What it cannot establish |
| --- | --- | --- |
| Sparse IDF | Live collection config and declared encoder expectation | Whether an unknown encoder already weighted terms |
| BM25 average length | Supplied encoder value vs measured tokenized corpus mean | Recovering tokenizer statistics from stored vectors |
| Fusion placement | Recursive walk of the QueryRequest | Whether nested, shard-local fusion was intentional |
| Score thresholds | Root and nested fusion requests | Equivalence between dense similarity and fused scores |
| Label count | Supplied labeled-query count | Statistical power or significance |

Missing inputs remain **unverified**, not a clean pass. Connection failures propagate instead of becoming empty reports.

```text
real corpus + encoders + Qdrant config + request
                       |
                    preflight
                       |
             development fusion sweep
                       |
             selected configuration only
                       |
              held-out paired evaluation
```

---

## 🔬 Reproduction Details

- BEIR SciFact: all 5,183 documents, title + space + abstract, no chunking. Archive MD5 `5f7d1de60b170fc8027bb7898e2efca1`; download URL and SHA256 recorded in metadata.
- All 300 judged test queries: sort IDs lexicographically, shuffle with `random.Random(42)`, split 150/150. **This is a custom split of BEIR's test set**, not the official BEIR train/test protocol. Exact IDs are recorded.
- Dense: FastEmbed ONNX `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, cosine, 256-token truncation. Actual model hash is recorded.
- Sparse: FastEmbed `Qdrant/bm25`, English stemming and stopword filtering, k=1.2, b=0.75, server-side IDF. Measured post-filter/stem average length: 151.38028169014083. The adapter uses internal tokenizer APIs; recheck on upgrades.
- One shard, exact dense search, no quantization or reranker. Development grid: dense/sparse, DBSF depths 50/200, and RRF k=2/5/20/61 with 1:1, 2:1 and 1:2 weights at both depths.
- Python 3.10.12, qdrant-client 1.17.1, FastEmbed 0.7.4. Recorded environment: Linux x86_64, Intel Xeon, two logical CPUs, about 2 GB RAM; threads 2, batch size 32.
- nDCG uses exponential gains and logarithmic discount; binary judgments make linear and exponential gains agree here. Unjudged documents score zero.

---

## 🧩 Your Collection

The generic `cli.py check` and `cli.py sweep` paths are read-only. The SciFact `benchmark.py` path creates a collection and writes points; do not confuse the two.

Use vectors from the same encoders as your indexed data, and judgments keyed by Qdrant point-ID strings. Each query needs a unique `id`, `dense`, `sparse` (`indices` and `values`) and `qrels` (point ID to grade), including a positive judgment.

```bash
# Optional authentication: set QDRANT_API_KEY in the environment.
python cli.py check --url http://localhost:6333 --collection my_collection \
  --sparse-name bm25 --request my-request.json \
  --bm25-avg-len 151.38 --measured-avg-len 151.38 \
  --labeled-query-count 150 --output check.json

python cli.py sweep --url http://localhost:6333 --collection my_collection \
  --dense-name dense --sparse-name bm25 --queries my-queries.json \
  --request my-request.json \
  --bm25-avg-len 151.38 --measured-avg-len 151.38 \
  --depths 50 200 --ks 2 5 20 61 --limit 10 --output sweep.json
```

The average lengths and query count above are examples. Replace them with your own measurements. `--encoder-includes-idf` handles already-weighted sparse encoders. Errors stop the sweep; warnings remain visible.

The generic CLI evaluates on the supplied labels only. Keep tuning and evaluation inputs separate yourself; it does not create a held-out split for you.

---

## 🗂️ Project Map

| File | Purpose |
| --- | --- |
| `benchmark.py` | Download, embed, index, tune and evaluate SciFact |
| `preflight.py` | Collection and request checks |
| `sweep.py` | Metrics, fusion grid and paired intervals |
| `cli.py` | Existing-collection interface |
| `test_benchmark.py` | Tests using real SciFact data and sparse vectors |
| CSV/JSON and `RUN.md` | Committed result evidence and provenance |

---

## ⚠️ Limits

One dataset, one encoder pair and one fixed split. Not preregistered. Dense text is truncated; BM25 sees full text. No multi-shard experiment, ANN-recall measurement, online traffic, concurrency test or alternative encoder sweep. No controlled broken-vs-fixed ablation isolates each preflight check's effect.

---

## 📚 Sources and License

- [BEIR dataset list](https://github.com/beir-cellar/beir/wiki/Datasets-available)
- [SciFact dataset card](https://huggingface.co/datasets/BeIR/scifact), CC-BY-SA-4.0; downloaded, not redistributed here
- [Qdrant hybrid tuning](https://qdrant.tech/documentation/search-tuning/how-to-tune-hybrid-search/)
- [Qdrant pre-tuning checks](https://qdrant.tech/documentation/search-tuning/before-tuning-a-qdrant-collection/)

No code license has been chosen yet. This is not a Qdrant-endorsed benchmark.
