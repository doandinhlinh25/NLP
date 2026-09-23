# TF-IDF search, evaluation, and error analysis

## Search setup

The search engine uses Pipeline B (lowercase, Unicode accent normalization, punctuation exclusion, English stopword removal, word TF-IDF with L2 normalization). Query and document vectors share the same fitted vocabulary; their dot product is therefore cosine similarity. The manually labeled relevant identifiers are recorded in `run_search_evaluation.py`, and all top-five retrieved documents are in `results.csv`.

## Evaluation at K = 5

| Query | Precision@5 | Recall@5 | Reciprocal rank |
| --- | ---: | ---: | ---: |
| barbecue cooking class | 0.000 | 0.000 | 0.000 |
| mac disk utility | 0.200 | 1.000 | 0.250 |
| natural language processing | 0.200 | 1.000 | 0.333 |
| python programming tutorial | 0.200 | 1.000 | 1.000 |
| japanese encephalitis vaccine | 0.200 | 1.000 | 1.000 |
| football match | 0.400 | 1.000 | 0.333 |
| climate change | 0.600 | 1.000 | 0.500 |
| **Mean** | **0.257** | **0.857** | **0.488** |

Thus the final aggregate measures are **P@5 = 0.257**, **Recall@5 = 0.857**, and **MRR = 0.488**. The set is intentionally small and manually judged; it demonstrates the evaluation procedure rather than claiming a benchmark-quality score.

## Error analysis

### Good case: `japanese encephalitis vaccine`

The labeled document is retrieved because the rare medical terms have strong lexical overlap with the query. These terms receive higher IDF than generic words such as `vaccine`, so the query-document cosine score is dominated by highly discriminative shared vocabulary.

### Good case: `climate change`

Several labeled documents occur in the top five. The repeated two-word phrase has direct lexical overlap with the documents and is widespread enough to retrieve multiple on-topic sources. This is a situation where a lexical model behaves reliably.

### Weak case: `barbecue cooking class`

Document 0 is the intended relevant document, but the query says `barbecue` while the document uses `BBQ`. Pipeline B does not know that these are equivalent, so it retrieves documents matching the generic terms `cooking` and `class` instead. The failure is primarily lexical matching and vocabulary representation, not a cosine-similarity bug.

### Weak case: `mac disk utility`

The labeled Mac OS X document appears below a generic Disk Drill download page. Both contain overlapping words, but TF-IDF cannot judge whether the full intent is troubleshooting the macOS utility. It favors lexical weight rather than task-level relevance.

## Most important failure and next representation

The `barbecue`/`BBQ` case is the clearest failure: two expressions refer to the same concept but have no shared word token. A word-embedding or contextual dense representation could place such expressions near one another from similar usage contexts; a hybrid lexical+dense retriever could preserve exact-match strengths while addressing this semantic gap.
