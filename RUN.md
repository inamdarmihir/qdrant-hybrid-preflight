# Smoke test run, 2026-10-05

Python 3.10.12; qdrant-client 1.17.1; numpy 2.2.6.
Qdrant server 1.17.1, commit eabee371fda447974a94d29fbaa675a6a596cc7b.
Server binary: qdrant-x86_64-unknown-linux-musl.tar.gz from the official v1.17.1 release.

22 tests passed in 1.04 seconds. Both local and server demos completed 28 configurations over 12 synthetic queries. The read-only CLI `check` ran against the server fixture and reported IDF checked, avg_len unverified, request unverified and labels unverified when those inputs were omitted.

The CSVs contain the measured summary. Dense/sparse ties occur in this tiny fixture, and different request paths can order them differently. These scores are not a reproducible real-data benchmark or evidence of a fusion improvement. Bootstrap intervals on deliberately repeated toy query patterns are demonstration output, not valid independent-query confidence claims.
