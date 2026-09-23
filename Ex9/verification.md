# Verification of the core implementation

Run with:

```powershell
python Ex9/implementation.py
```

Output: `All unit tests passed.`

Each required function is covered by an assertion using the three-document corpus from the hand-calculation exercise:

- `build_vocabulary()` verifies deterministic alphabetical ordering.
- `compute_counts()` verifies all three count vectors.
- `compute_tf()` verifies `TF(cat, D1) = 1/3` and that TF sums to one.
- `compute_idf()` verifies `IDF(cat) = log(3/2)` and `IDF(fish) = 0`.
- `compute_tfidf()` verifies the `cat` and `fish` weights in D1.
- `cosine_similarity()` verifies `cos([1,1,1], [1,1,0]) = 2/sqrt(6)`.

The final verification uses scikit-learn's `CountVectorizer` and `TfidfTransformer(norm=None, smooth_idf=False)`. Scikit-learn defines this unsmoothed IDF as `log(N / df) + 1`; the verification removes its added `1` before comparing it with the lab convention `log(N / df)`. It also applies the same document-length TF normalization as the custom code. The adjusted reference IDF and TF-IDF values match within `1e-9`.
