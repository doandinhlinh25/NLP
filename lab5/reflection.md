# LAB 05 - Reflection

## Dataset and sequence length

The IMDb CSV has 50,000 reviews, evenly split between positive and negative. There are 418 exact duplicate review strings and no missing text or labels. After the tokenizer removes HTML markup and splits words, lengths range from 6 to 2,477 tokens (median 173, mean 231.3, 75th percentile 280, 90th percentile 452). `sequence_lengths.png` shows the long right tail. Duplicates are removed before splitting; the experiment then takes balanced subsets of 5,000 training, 1,000 validation, and 1,000 test reviews. This is a compute-limited subset, so LAB 04's full-data score is not a direct benchmark.

Maximum length should follow the length distribution, how much meaningful content is retained, where sentiment cues tend to occur, and the available compute. Padding adds no language information; packed sequences prevent the final state from advancing through PAD tokens. Truncation discards the tail permanently. In the training subset, lengths 50, 100, and 200 retain 21.5%, 41.6%, and 68.9% of tokens, respectively. A limit of 30 could omit the conclusion or a late contrast after "but".

## Baseline and architectures

All rows below use the same held-out 1,000-review test subset. Precision, recall, and F1 refer to positive sentiment. Training time excludes preprocessing and evaluation.

| Model | Length | Parameters | Train seconds | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | all | 12,001 | 2.9 | 0.854 | 0.843 | 0.870 | 0.856 |
| RNN | 100 | 776,385 | 17.2 | 0.494 | 0.493 | 0.430 | 0.459 |
| LSTM | 100 | 801,345 | 31.9 | 0.553 | 0.556 | 0.526 | 0.541 |
| GRU | 100 | 793,025 | 12.2 | 0.573 | 0.564 | 0.642 | 0.601 |

The prediction that LSTM can handle long dependencies better than a plain RNN is supported by its higher F1 here, but the experiment does not isolate long-distance reasoning. GRU has the highest F1 among recurrent models at length 100. The TF-IDF baseline is much stronger under this small-data, two-epoch setup. Therefore the results do not support a claim that recurrent models are always more accurate. The parameter counts include the embedding layer; GRU has fewer recurrent parameters than LSTM at the same dimensions.

The order probe holds word counts fixed. TF-IDF gives exactly the same positive probability (0.4769) for "this movie is not good" and "this movie is good not"; GRU gives 0.4383 and 0.4469. For "the film is not bad" and "the film is bad not", TF-IDF gives 0.0284 for both, while GRU gives 0.4926 and 0.4822. Thus the trained GRU is order-sensitive in this small probe, as its computation predicts. The scores are close and do not show reliable understanding of negation. Both probes contain unnatural wording and cannot establish broad performance.

## Length and cost

With GRU fixed, F1 changes from 0.565 at 50 tokens to 0.601 at 100 and 0.654 at 200. Training times are 6.7, 12.2, and 167.6 seconds. Validation losses at epoch 2 are 0.694, 0.689, and 0.677. More context helped F1 in these three runs, while the longest run was much costlier. This is a measured trade-off for this CPU and configuration, not a guarantee that increasing length always helps. Memory consumption was not measured.

## Learning curves and errors

At length 100, RNN train loss decreased from 0.699 to 0.684 while validation loss moved from 0.699 to 0.700, an early possible overfitting signal. LSTM and GRU train and validation losses both decreased across the two epochs. Two points are too few to diagnose their later behavior. The GRU confusion matrix is `[[252, 248], [179, 321]]` with negative then positive rows and columns. Its false positives and false negatives both matter; accuracy alone hides this distribution.

`error_analysis.md` lists three correct and three incorrect held-out reviews with probabilities, lengths, and possible contributing factors. The first false negative starts with explicit praise ("wonderful", "terrific") and is 175 tokens long. Since those cues occur before the 100-token cutoff, truncation alone is not an adequate explanation. The second false negative is 719 tokens and mixes favorable film language with words about loss and grief; lexical polarity and lost later context are both plausible. The third false negative is 909 tokens and has a long plot description; a 100-token prefix can omit the final judgment. These are hypotheses, not causal diagnoses. Vocabulary coverage, a short training budget, label ambiguity, and optimization may also matter.

## Application and long-range probes

The saved GRU predicts positive probabilities 0.482 for the short positive review, 0.427 for the long mixed positive review, 0.456 for a negated negative review, and 0.420 for the "terrible, but surprisingly good" contrast review. At a 0.5 threshold it calls all four negative, including three positive examples. The likely influential contexts are "enjoyed", "powerful ... wonderful", "not like", and "terrible ... but ... good"; the outputs do not prove which words the model used. The long review is below the 100-token cutoff, so its error cannot be attributed to truncation.

For the synthetic position probes, changing "excellent" to "terrible" barely changes the GRU's probability at any position (differences under 0.01). These examples show weak sensitivity under this training budget. They do not prove vanishing gradients: that mechanism follows from repeated Jacobian products, while the measured behavior could also come from weak training, vocabulary, or distribution shift.

## Link to LAB 04 and attention

TF-IDF is sparse and ignores exact order in this unigram baseline. Average Word2Vec is dense but averages away order. RNN, LSTM and GRU produce context-dependent states in sequence order; LSTM and GRU add gates that can help preserve information across steps. Recurrent models still compress a growing prefix into limited state, read positions sequentially, and make distant interactions traverse many steps. Attention provides more direct access to relevant positions and motivates the next lab. Better architecture alone does not replace sufficient data, careful evaluation, and error analysis.
