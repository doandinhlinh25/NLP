# Experiment 1 and preprocessing ablation

## Corpus and sparse representation

- Documents: **30,000**
- Pipeline A matrix shape: **(30,000, 193,837)**
- Pipeline A non-zero values: **5,104,560**
- Pipeline A sparsity: **99.912219%**

Every document vector has one position for every corpus vocabulary item so that all document vectors inhabit the same feature space. It remains sparse because any individual document contains only a small subset of those items.

## Pipeline comparison

| Pipeline | Vocabulary size | Avg. non-zero features/document | nnz | Matrix sparsity | Mean query-feature OOV | Mean P@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A_minimal | 193,837 | 170.15 | 5,104,560 | 99.912219% | 6.25% | 0.257 |
| B_normalized | 174,270 | 118.39 | 3,551,639 | 99.932066% | 6.25% | 0.257 |
| C_subword | 100,000 | 1820.04 | 54,601,221 | 98.179959% | 1.25% | 0.143 |

Pipeline A is lowercase plus word tokenization. Pipeline B additionally normalizes accents, excludes punctuation through its word-token rule, and removes the scikit-learn English stopword list. Pipeline C uses character 3–5 grams (subword features), with `min_df=3` and `max_features=100,000` to keep the 30K-document experiment feasible. Its OOV rate is measured over character n-gram features, so it is not directly identical to word-token OOV; it is included to show the subword robustness effect. Mean P@5 uses the same seven manually labeled queries as Exercise 10, so it is a small diagnostic comparison rather than a general benchmark.

## Pipeline A vocabulary inspection

### 20 highest document-frequency terms

- `the`: df = 27,893
- `and`: df = 27,423
- `to`: df = 26,689
- `of`: df = 26,031
- `a`: df = 25,905
- `in`: df = 25,224
- `for`: df = 23,651
- `is`: df = 22,739
- `with`: df = 21,405
- `on`: df = 20,262
- `that`: df = 18,370
- `this`: df = 17,840
- `are`: df = 17,594
- `it`: df = 17,168
- `s`: df = 16,959
- `as`: df = 16,467
- `at`: df = 16,347
- `from`: df = 16,316
- `be`: df = 16,153
- `you`: df = 16,094

### 20 highest-IDF terms

- `ﬁt`: idf = 10.6158
- `ﬁnancial`: idf = 10.6158
- `ﬁlter`: idf = 10.6158
- `ﬁlm`: idf = 10.6158
- `ﬁg`: idf = 10.6158
- `ﬁelds`: idf = 10.6158
- `000016`: idf = 10.6158
- `00001888`: idf = 10.6158
- `00002`: idf = 10.6158
- `00000`: idf = 10.6158
- `000000`: idf = 10.6158
- `00000000`: idf = 10.6158
- `0000000000000965`: idf = 10.6158
- `00000001`: idf = 10.6158
- `00000048`: idf = 10.6158
- `000002`: idf = 10.6158
- `0006`: idf = 10.6158
- `00060`: idf = 10.6158
- `000651`: idf = 10.6158
- `회화성을`: idf = 10.6158

### 20 highest TF-IDF terms in document 0

Document 0 preview: Beginners BBQ Class Taking Place in Missoula! Do you want to get better at making delicious BBQ? You will have the opportunity, put this on your calendar now. Thursday, September 2

- `bbq`: tf-idf = 0.4607
- `class`: tf-idf = 0.2752
- `meat`: tf-idf = 0.1937
- `balay`: tf-idf = 0.1790
- `kcbs`: tf-idf = 0.1721
- `lonestar`: tf-idf = 0.1673
- `will`: tf-idf = 0.1543
- `missoula`: tf-idf = 0.1502
- `apron`: tf-idf = 0.1410
- `smoker`: tf-idf = 0.1402
- `you`: tf-idf = 0.1368
- `timelines`: tf-idf = 0.1322
- `trimming`: tf-idf = 0.1302
- `spectators`: tf-idf = 0.1281
- `cost`: tf-idf = 0.1276
- `rangers`: tf-idf = 0.1247
- `22nd`: tf-idf = 0.1193
- `beginner`: tf-idf = 0.1170
- `beginners`: tf-idf = 0.1162
- `culinary`: tf-idf = 0.1158

## Interpretation

A high-document-frequency term is not necessarily high TF-IDF: its IDF is low because it is shared by many documents. A high-IDF term is also not automatically high TF-IDF in every document, because the term must occur in the selected document and have enough local TF. Lowercasing merges capitalization variants and therefore reduces duplicate vocabulary entries. Stopword removal reduces frequent function words, but can remove useful short query terms or phrases; it is a modeling choice, not an automatic improvement. Removing punctuation can also discard information such as `C++`, version strings, URLs, emoticons, decimal numbers, and hyphenated entities. In the measured table, Pipeline B is the sparsest and Pipeline C is the least sparse because each document produces many overlapping character n-grams. The highest Mean P@5 should be chosen for this small labeled set; it does not automatically establish the best preprocessing policy for every retrieval task.
