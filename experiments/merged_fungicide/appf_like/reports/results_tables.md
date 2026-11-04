# APPf-like results

## CatBoost across scaffold splits

| feature set | test AUC | test PR-AUC | test F1 | test MCC | val AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| EXT | 0.8717 ± 0.0182 | 0.7696 ± 0.0323 | 0.7441 ± 0.0122 | 0.5812 ± 0.0414 | 0.9136 |
| EXT_OLD_ONLY | 0.8281 ± 0.0453 | 0.7212 ± 0.0525 | 0.6627 ± 0.0257 | 0.4633 ± 0.0630 | 0.8839 |
| P24 | 0.7370 ± 0.0616 | 0.6043 ± 0.0393 | 0.6395 ± 0.0488 | 0.3736 ± 0.1325 | 0.7941 |

## Split-42 selected EXT model

| metric | value |
| --- | ---: |
| threshold | 0.4615 |
| ROC-AUC | 0.8526 |
| PR-AUC | 0.7492 |
| F1 | 0.7300 |
| precision | 0.7111 |
| recall | 0.7500 |
| MCC | 0.5342 |
| Brier | 0.1583 |
| ECE (10 bins) | 0.0891 |

## Source shift (old-only -> new-only positives)

| split | ROC-AUC | PR-AUC | F1 | MCC | top-10% precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.8017 | 0.7298 | 0.6997 | 0.4644 | 0.8507 |
| 43 | 0.8492 | 0.8027 | 0.7320 | 0.5379 | 0.9412 |
| 44 | 0.8107 | 0.7551 | 0.6389 | 0.4548 | 0.9130 |

## Train/validation/test fit (split 42)

- train AUC: 0.9928
- validation AUC: 0.9110
- test AUC: 0.8526
- train minus test: 0.1401

## Label permutation

- observed fixed-iteration AUC: 0.8513
- null mean: 0.5017 ± 0.0491
- empirical p: 0.0196 (n=50)
