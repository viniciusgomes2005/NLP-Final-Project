# Short Paper — Template (NLP Deep Learning Researcher)

## 1. Introduction
We study multilingual hate/offense detection with an emphasis on **Portuguese** performance and **cross-lingual few-shot transfer**.

## 2. Related Work
Briefly review: bag-of-words baselines, multilingual transformers (e.g., XLM-R), few-shot protocols (k-shot), Macro-F1 as the primary metric under imbalance.

## 3. Methods
- **Monolingual PT baseline**: TF‑IDF + LinearSVC (ngram 1–2, C tuned on val).
- **Cross-lingual**: fine-tune XLM‑R on ENG + PT k‑shot; evaluate on PT test.
- Splits: stratified; seed fixed; labels in `lang_type` format.

## 4. Experiments
- Dataset: CSV from Kaggle (≈279k full or 35k balanced). Filtering language via `label.split('_')[0]`.
- **Goals:** PT Macro-F1 ≥ 0.80 (baseline); cross-lingual Macro-F1 ≥ 0.65 at k=32 (XLM-R).
- Protocols: k-shot (k ∈ {4,8,16,32}); 2 epochs for XLM-R; batch 16; lr 2e-5.

## 5. Results
- Table: Macro-F1/Accuracy for PT baseline (val/test).
- Curves: Macro-F1 vs k (4/8/16/32) for cross-lingual XLM‑R.
- Confusions: common confusions among label types.

## 6. Discussion
- Trade-offs: computation vs. accuracy; robustness to imbalance.
- Error analysis: inspect top false positives/negatives.

## 7. Conclusion
Did you reach the targets? Next steps: domain adaptation, class rebalancing, prompt-based zero-shot.
