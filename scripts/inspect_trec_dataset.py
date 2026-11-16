
import gzip
from collections import Counter

from huggingface_hub import hf_hub_download


REPO = "trec-product-search/product-recommendation-2025"

files = [
    "initial/train-esci.qrels.gz",
    "initial/train-esci.queries.gz",
]

paths = {}

for filename in files:
    paths[filename] = hf_hub_download(
        repo_id=REPO,
        repo_type="dataset",
        filename=filename,
    )

print("\nFIRST 10 RELEVANCE JUDGMENTS")
with gzip.open(
    paths["initial/train-esci.qrels.gz"],
    "rt",
    encoding="utf-8",
) as f:
    for _, line in zip(range(10), f):
        print(line.rstrip())

print("\nFIRST 10 REFERENCE PRODUCTS")
with gzip.open(
    paths["initial/train-esci.queries.gz"],
    "rt",
    encoding="utf-8",
) as f:
    for _, line in zip(range(10), f):
        print(line.rstrip())

print("\nRELEVANCE LABEL DISTRIBUTION")
labels = Counter()

with gzip.open(
    paths["initial/train-esci.qrels.gz"],
    "rt",
    encoding="utf-8",
) as f:
    for line in f:
        columns = line.split()
        if len(columns) >= 4:
            labels[columns[3]] += 1

print(labels)
