# Individual Learning Check

1. **Pretraining vs fine-tuning:** Pretraining learns reusable parameters from a large corpus using a general objective; fine-tuning continues updating those parameters on labeled downstream examples and task loss.
2. **MLM:** Mask selected input tokens and train the model to predict the original vocabulary items from surrounding context.
3. **Tokenizer:** Maps raw strings to the model's known token IDs and defines the vocabulary boundaries expected by its embeddings.
4. **Subwords:** Let the vocabulary represent rare/unseen words as reusable pieces rather than requiring a separate ID for every whole word.
5. **[CLS]:** A special first token whose contextual output is often used as the sequence representation for classification.
6. **Attention mask:** Marks real tokens versus padding so padding positions do not contribute as content.
7. **Frozen encoder:** Encoder weights are excluded from gradient updates; a newly trained task head can still learn from its output features.
8. **Fine-tuning parameters:** Gradients can update the pretrained encoder and task head; the exact updated subset depends on the training configuration.
9. **Small downstream data:** Pretrained representations provide a useful initialization learned from much broader text, reducing the labeled data needed for a useful task mapping.
10. **Domain shift:** No. Pretraining does not guarantee the target domain's vocabulary, style, or label relationship matches its training data.
11. **Sequence length:** Full self-attention compares token pairs; the attention computation/memory grows approximately quadratically with sequence length.
12. **Confidence:** No. A high softmax maximum is model certainty, not a guarantee of correctness; inspect confidence against held-out outcomes.
