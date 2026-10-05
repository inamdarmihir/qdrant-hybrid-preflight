# Real-data run, 2026-10-05

Full BEIR SciFact: 5,183 documents; 300 queries with real qrels. No dummy documents or vectors. Dense FastEmbed all-MiniLM-L6-v2; sparse FastEmbed Qdrant/bm25. BM25 measured filtered/stemmed average length 151.38028169014083.

Qdrant server 1.17.1, official x86_64 musl binary; exact dense search, one shard. Python 3.10.12, 2 logical Xeon CPUs at 2.60GHz, about 2GB RAM. Dependencies and ONNX hash in metadata.json.

Development: 150 queries, split seed 42. 28 configurations. Winner DBSF depth50, development nDCG@10 0.688005.
Held-out: 150 queries. Dense 0.689049; sparse 0.725251; selected DBSF depth50 0.771831. Delta over sparse 0.046580; paired percentile 95% bootstrap interval [0.020206,0.073512], 2,000 resamples, seed42.

Two complete server executions produced the same reported quality scores. The final pipeline execution reused cached actual dense embeddings keyed by corpus checksum and model hash. Corpus encoding ran beforehand in resumable batches; final roughly 30 seconds excludes that first encoding/download time and must not be described as full fresh runtime.

3 real-data tests passed. They load the checksum-verified SciFact files, test real qrels, measure actual BM25 tokenized lengths and use a real sparse query vector in collection/request checks. No smoke-test numbers from the superseded synthetic fixture remain in the current tree.
