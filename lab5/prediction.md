# Predictions recorded before training

| Question | Prediction | Reason | Confidence |
|---|---|---|---|
| TF-IDF + LR versus RNN + Embedding: direct order handling | RNN | Its state is updated in token order; unigram TF-IDF counts terms without positions. | High |
| RNN versus LSTM for distant dependencies | LSTM | A gated cell state can preserve information across many steps, reducing dependence on repeated tanh transitions. | High |
| RNN length 50 to 200 | Slower training; improvement is uncertain | More recurrent steps cost time and lengthen gradient paths while preserving more context. | High for time, medium for quality |
| Truncate every review at 30 tokens | Accuracy may drop | The verdict, reversal, or negation can appear after the first 30 tokens. | High |
| Must LSTM always exceed RNN accuracy? | No | Dataset, training budget, variance, and task length influence performance. | High |
