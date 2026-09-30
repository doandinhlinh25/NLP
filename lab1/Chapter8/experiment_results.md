# Experiment 2 — Preprocessing ablation

## Corpus and sparse representation

- Documents: **30,000**
- Pipeline A matrix shape: **(30,000, 233,866)**
- Pipeline A non-zero values: **5,080,798**
- Pipeline A sparsity: **99.927582%**

Every document vector has one position for every corpus vocabulary item so that all document vectors inhabit the same feature space. It remains sparse because any individual document contains only a small subset of those items.

## Pipeline comparison

| Pipeline | Vocabulary size | Avg. non-zero features/document | nnz | Matrix sparsity | Mean query-feature OOV | Mean P@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A_minimal | 233,866 | 169.36 | 5,080,798 | 99.927582% | 6.25% | 0.229 |
| B_normalized | 174,270 | 118.39 | 3,551,639 | 99.932066% | 6.25% | 0.257 |
| C_subword | 100,000 | 1820.04 | 54,601,221 | 98.179959% | 1.25% | 0.143 |

Pipeline A is lowercase plus word tokenization. Pipeline B additionally normalizes accents, excludes punctuation through its word-token rule, and removes the scikit-learn English stopword list. Pipeline C uses character 3–5 grams (subword features), with `min_df=3` and `max_features=100,000` to keep the 30K-document experiment feasible. Its OOV rate is measured over character n-gram features, so it is not directly identical to word-token OOV; it is included to show the subword robustness effect. Mean P@5 uses the same seven manually labeled queries as Exercise 10, so it is a small diagnostic comparison rather than a general benchmark.

## Pipeline A vocabulary inspection

### 20 highest document-frequency terms

- `the`: df = 27,890
- `and`: df = 27,418
- `to`: df = 26,665
- `of`: df = 26,013
- `a`: df = 25,880
- `in`: df = 25,129
- `for`: df = 23,643
- `is`: df = 22,738
- `with`: df = 21,403
- `on`: df = 20,095
- `that`: df = 18,274
- `this`: df = 17,837
- `are`: df = 17,594
- `it`: df = 16,762
- `as`: df = 16,454
- `at`: df = 16,326
- `from`: df = 16,310
- `be`: df = 16,146
- `you`: df = 15,933
- `by`: df = 15,072

### 20 highest-IDF terms

- `0-position`: idf = 10.6158
- `0-rc`: idf = 10.6158
- `0-release`: idf = 10.6158
- `ﬁxes`: idf = 10.6158
- `0-200`: idf = 10.6158
- `0-255`: idf = 10.6158
- `0-262-69339-9`: idf = 10.6158
- `0-27`: idf = 10.6158
- `ﬁg`: idf = 10.6158
- `0-3-0`: idf = 10.6158
- `흐름을`: idf = 10.6158
- `힘써온`: idf = 10.6158
- `ﬂuid`: idf = 10.6158
- `ospina's`: idf = 10.6158
- `ospmi`: idf = 10.6158
- `osr`: idf = 10.6158
- `활동을`: idf = 10.6158
- `활발한`: idf = 10.6158
- `황미은`: idf = 10.6158
- `회화사의`: idf = 10.6158

### 20 highest TF-IDF terms in document 0

Document 0 preview: Beginners BBQ Class Taking Place in Missoula! Do you want to get better at making delicious BBQ? You will have the opportunity, put this on your calendar now. Thursday, September 2

- `bbq`: tf-idf = 0.4583
- `class`: tf-idf = 0.2840
- `meat`: tf-idf = 0.1934
- `balay`: tf-idf = 0.1777
- `kcbs`: tf-idf = 0.1709
- `lonestar`: tf-idf = 0.1661
- `will`: tf-idf = 0.1533
- `missoula`: tf-idf = 0.1492
- `smoker`: tf-idf = 0.1429
- `apron`: tf-idf = 0.1400
- `you`: tf-idf = 0.1367
- `timelines`: tf-idf = 0.1313
- `cost`: tf-idf = 0.1297
- `trimming`: tf-idf = 0.1293
- `spectators`: tf-idf = 0.1272
- `rangers`: tf-idf = 0.1238
- `22nd`: tf-idf = 0.1184
- `beginner`: tf-idf = 0.1168
- `t-shirt`: tf-idf = 0.1158
- `beginners`: tf-idf = 0.1154

## Interpretation

A high-document-frequency term is not necessarily high TF-IDF: its IDF is low because it is shared by many documents. A high-IDF term is also not automatically high TF-IDF in every document, because the term must occur in the selected document and have enough local TF. Lowercasing merges capitalization variants and therefore reduces duplicate vocabulary entries. Stopword removal reduces frequent function words, but can remove useful short query terms or phrases; it is a modeling choice, not an automatic improvement. Removing punctuation can also discard information such as `C++`, version strings, URLs, emoticons, decimal numbers, and hyphenated entities. In the measured table, Pipeline B is the sparsest and Pipeline C is the least sparse because each document produces many overlapping character n-grams. The highest Mean P@5 should be chosen for this small labeled set; it does not automatically establish the best preprocessing policy for every retrieval task.
