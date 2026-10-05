<p align="center">
  <a href="https://github.com/inamdarmihir/qdrant-hybrid-preflight">
    <img src="docs/assets/banner.svg" width="800px" alt="Qdrant Hybrid Preflight: check the setup before you trust a hybrid-search result">
  </a>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a>
  ·
  <a href="#results">Results</a>
  ·
  <a href="#preflight-checks">Preflight checks</a>
  ·
  <a href="#use-it-on-your-collection">Your collection</a>
  ·
  <a href="#limits">Limits</a>
  ·
  <a href="https://mihirinamdar.substack.com/p/your-hybrid-search-benchmark-may">Article</a>
</p>

<p align="center">
  <a href="https://github.com/inamdarmihir/qdrant-hybrid-preflight/actions/workflows/ci.yml">
    <img src="https://github.com/inamdarmihir/qdrant-hybrid-preflight/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="pyproject.toml">
    <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  </a>
  <a href="RUN.md">
    <img src="https://img.shields.io/badge/tested%20with-Qdrant%201.17.1-dc244c.svg" alt="Tested with Qdrant 1.17.1">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
  <a href="https://github.com/inamdarmihir/qdrant-hybrid-preflight/commits/main">
    <img src="https://img.shields.io/github/last-commit/inamdarmihir/qdrant-hybrid-preflight" alt="Last commit">
  </a>
</p>

# Qdrant Hybrid Preflight

Hybrid search can look fine and still be misconfigured: sparse IDF switched off, a stale BM25 `avg_len`, fusion running per shard, a score threshold applied to fused scores. **Hybrid Preflight inspects an existing Qdrant collection and query for these silent failures, then sweeps fusion settings so you can pick one on evidence.**

The repo also ships a runnable experiment on the full BEIR SciFact corpus: tune on a development split, then evaluate the single chosen configuration on held-out queries. Every number below comes from the committed run.

| | |
| --- | --- |
| **Read-only** | `check` and `sweep` inspect a collection and never write to it. |
| **Honest by default** | Missing inputs are reported as *unverified*, never as a clean pass. Connection errors propagate. |
| **Measured** | SciFact, 5,183 documents, tuned on 150 queries, scored on 150 held-out queries. |
| **Inspectable** | CSV/JSON results, `metadata.json` and [`RUN.md`](RUN.md) are committed next to the code. |

## Contents

- [Quickstart](#quickstart)
- [Data and ground truth](#data-and-ground-truth)
- [Results](#results)
- [Preflight checks](#preflight-checks)
- [Use it on your collection](#use-it-on-your-collection)
- [Reproduction details](#reproduction-details)
- [Repository layout](#repository-layout)
- [Limits](#limits)
- [Development](#development)
- [Sources and license](#sources-and-license)

## Quickstart

Requires Python 3.10+, Docker, a CPU, internet for the first data and model downloads, and several GB of disk. Use a disposable Qdrant server.

```bash
git clone https://github.com/inamdarmihir/qdrant-hybrid-preflight.git
cd qdrant-hybrid-preflight
docker run --rm -p 127.0.0.1:6333:6333 qdrant/qdrant:v1.17.1
```

In a second terminal:

```bash
cd qdrant-hybrid-preflight
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[benchmark]"
python -m qdrant_hybrid_preflight.benchmark
```

This downloads checksum-verified SciFact data, embeds all 5,183 abstracts, indexes dense and BM25 vectors, runs the preflight, tunes on 150 queries and evaluates the winner on the other 150. There are no synthetic documents or invented relevance labels.

The benchmark refuses to overwrite an existing collection. For another run, use a fresh server or collection name:

```bash
python -m qdrant_hybrid_preflight.benchmark --collection scifact_second_run
```

Outputs go to `results/`: development and held-out CSV/JSON, per-query rankings and metadata. Dense embeddings are cached using dataset and model provenance, so cached runtime is not first-run runtime.

## Data and ground truth

Nothing in the results below is synthetic. Documents, queries and relevance judgments all come from a published benchmark; this repo only embeds, indexes and scores them.

| Item | Source | Notes |
| --- | --- | --- |
| Corpus (5,183 abstracts) and 300 test queries | **BEIR SciFact**, [download](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip) · [dataset card](https://huggingface.co/datasets/BeIR/scifact) · [BEIR repo](https://github.com/beir-cellar/beir) | Archive MD5 `5f7d1de60b170fc8027bb7898e2efca1` and SHA256 `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165` are checked or recorded in [`metadata.json`](metadata.json). CC-BY-SA-4.0; downloaded at run time, not redistributed. |
| Ground truth (relevance judgments) | `qrels/test.tsv` shipped inside that archive | Human-annotated judgments from the original SciFact dataset. We do not generate, edit or infer any label. Unjudged documents score zero. |
| Original dataset | Wadden et al., [*Fact or Fiction: Verifying Scientific Claims*](https://arxiv.org/abs/2004.14974) (EMNLP 2020) · [allenai/scifact](https://github.com/allenai/scifact) | The source of the claims and evidence annotations. |
| BEIR packaging | Thakur et al., [*BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models*](https://arxiv.org/abs/2104.08663) (2021) | Defines the corpus/queries/qrels format used here. |
| Dense encoder | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) via FastEmbed | ONNX weight hash recorded in `metadata.json`. |
| Sparse encoder | [`Qdrant/bm25`](https://huggingface.co/Qdrant/bm25) via FastEmbed | k=1.2, b=0.75, measured average length recorded. |

**What we add, and what it means.** The 150/150 development/held-out split is **our own** (seed 42) over BEIR's 300 test queries, so it is not the official BEIR protocol and the numbers are not comparable to BEIR leaderboard entries. The query IDs for both halves are recorded in `metadata.json`.

**Not real data.** `tests/test_offline.py` uses tiny hand-built inputs to unit-test the checks. They never feed any reported number.

## Results

Committed run: October 5, 2026, Qdrant 1.17.1. Development selected **DBSF with 50 candidates from each retriever** out of a 28-configuration grid.

| Configuration | Development nDCG@10 (150 queries) | Held-out nDCG@10 (150 queries) |
| --- | ---: | ---: |
| Dense MiniLM | 0.601115 | 0.689049 |
| BM25 sparse | 0.651940 | 0.725251 |
| DBSF, depth 50 | 0.688005 | 0.771831 |

Held-out delta vs sparse: **+0.046580 nDCG@10**. Paired percentile-bootstrap 95% interval: **[0.020206, 0.073512]**, 2,000 resamples, seed 42.

This interval describes the fixed chosen configuration on this sample. It is not evidence that DBSF always wins, or that the preflight checks caused the gain. Query independence is a bootstrap assumption that related scientific claims may weaken.

Evidence: [`development.csv`](development.csv), [`heldout.csv`](heldout.csv), [`metadata.json`](metadata.json) and [`RUN.md`](RUN.md). The committed evidence is at the repository root; new benchmark runs write under `results/` by default. Serial HTTP timings are not concurrent-load latency measurements.

## Preflight checks

| Check | What it inspects | What it cannot establish |
| --- | --- | --- |
| Sparse IDF | Live collection config and declared encoder expectation | Whether an unknown encoder already weighted terms |
| BM25 average length | Supplied encoder value vs measured tokenized corpus mean | Recovering tokenizer statistics from stored vectors |
| Fusion placement | Recursive walk of the `QueryRequest` | Whether nested, shard-local fusion was intentional |
| Score thresholds | Root and nested fusion requests | Equivalence between dense similarity and fused scores |
| Label count | Supplied labeled-query count | Statistical power or significance |

Each finding has a `code`, a `level` (`info`, `warning`, `error`) and a message. Missing inputs remain **unverified**, not a clean pass.

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

## Use it on your collection

The `check` and `sweep` commands are read-only. The SciFact benchmark creates a collection and writes points; do not confuse the two.

Use vectors from the same encoders as your indexed data, and judgments keyed by Qdrant point-ID strings. Each query needs a unique `id`, `dense`, `sparse` (`indices` and `values`) and `qrels` (point ID to grade), including a positive judgment.

```bash
pip install -e .          # core install; no embedding models needed
# Optional authentication: set QDRANT_API_KEY in the environment.

qdrant-hybrid-preflight check --url http://localhost:6333 --collection my_collection \
  --sparse-name bm25 --request my-request.json \
  --bm25-avg-len 151.38 --measured-avg-len 151.38 \
  --labeled-query-count 150 --output check.json

qdrant-hybrid-preflight sweep --url http://localhost:6333 --collection my_collection \
  --dense-name dense --sparse-name bm25 --queries my-queries.json \
  --request my-request.json \
  --bm25-avg-len 151.38 --measured-avg-len 151.38 \
  --depths 50 200 --ks 2 5 20 61 --limit 10 --output sweep.json
```

The average lengths and query count above are examples. Replace them with your own measurements. `--encoder-includes-idf` handles already-weighted sparse encoders such as SPLADE. Preflight errors stop the sweep and set exit status 2; warnings stay visible.

From Python:

```python
from qdrant_client import QdrantClient
from qdrant_hybrid_preflight import preflight_hybrid_search

client = QdrantClient(url="http://localhost:6333")
for finding in preflight_hybrid_search(client, "my_collection", "bm25",
                                       bm25_avg_len=151.4, measured_avg_len=151.4,
                                       labeled_query_count=150):
    print(finding.level, finding.code, finding.message)
```

The CLI evaluates on the supplied labels only. Keep tuning and evaluation inputs separate yourself; it does not create a held-out split for you.

## Reproduction details

- BEIR SciFact: all 5,183 documents, title + space + abstract, no chunking. Archive MD5 `5f7d1de60b170fc8027bb7898e2efca1`; download URL and SHA256 recorded in metadata.
- All 300 judged test queries: sort IDs lexicographically, shuffle with `random.Random(42)`, split 150/150. **This is a custom split of BEIR's test set**, not the official BEIR train/test protocol. Exact IDs are recorded.
- Dense: FastEmbed ONNX `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, cosine, 256-token truncation. Actual model hash is recorded.
- Sparse: FastEmbed `Qdrant/bm25`, English stemming and stopword filtering, k=1.2, b=0.75, server-side IDF. Measured post-filter/stem average length: 151.38028169014083. The adapter uses internal tokenizer APIs; recheck on upgrades.
- One shard, exact dense search, no quantization or reranker. Development grid: dense/sparse, DBSF depths 50/200, and RRF k=2/5/20/61 with 1:1, 2:1 and 1:2 weights at both depths.
- Python 3.10.12, qdrant-client 1.17.1, FastEmbed 0.7.4. Recorded environment: Linux x86_64, Intel Xeon, two logical CPUs, about 2 GB RAM; threads 2, batch size 32.
- nDCG uses exponential gains and logarithmic discount; binary judgments make linear and exponential gains agree here. Unjudged documents score zero.

## Repository layout

| Path | Purpose |
| --- | --- |
| `qdrant_hybrid_preflight/preflight.py` | Collection and request checks |
| `qdrant_hybrid_preflight/sweep.py` | Metrics, fusion grid and paired intervals |
| `qdrant_hybrid_preflight/cli.py` | `check` / `sweep` interface for existing collections |
| `qdrant_hybrid_preflight/benchmark.py` | Download, embed, index, tune and evaluate SciFact |
| `tests/test_offline.py` | Offline unit tests (no downloads or server) |
| `tests/test_benchmark.py` | Tests on the real SciFact data and sparse vectors |
| `development.csv`, `heldout.csv`, `metadata.json`, `RUN.md` | Committed result evidence and provenance |

## Limits

One dataset, one encoder pair and one fixed split. Not preregistered. Dense text is truncated; BM25 sees full text. No multi-shard experiment, ANN-recall measurement, online traffic, concurrency test or alternative encoder sweep. No controlled broken-vs-fixed ablation isolates each preflight check's effect.

## Development

```bash
pip install -e ".[dev]"
pytest -q -m "not realdata"      # offline, seconds

pip install -e ".[benchmark,dev]"
pytest -q -m realdata            # downloads SciFact, loads FastEmbed models
```

## Sources and license

- [BEIR dataset list](https://github.com/beir-cellar/beir/wiki/Datasets-available)
- [SciFact dataset card](https://huggingface.co/datasets/BeIR/scifact), CC-BY-SA-4.0; downloaded, not redistributed here
- [Qdrant hybrid tuning](https://qdrant.tech/documentation/search-tuning/how-to-tune-hybrid-search/)
- [Qdrant pre-tuning checks](https://qdrant.tech/documentation/search-tuning/before-tuning-a-qdrant-collection/)

Code: [MIT](LICENSE). This is not a Qdrant-endorsed benchmark.
