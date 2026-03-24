# ML Coursework 2: TypiClust ($TPC_{RP}$) Implementation
```bash
git clone https://github.com/jonny246810/ML_Coursework_2
cd ML_Coursework_2
pip install -r requirements.txt
```

# Original implementation
```bash
Execute src/original_implementation.ipynb
```

# Modified implementation
```bash
Execute src/modified_implementation.ipynb
```

# Hyperparameter Compliance

## SimCLR Backbone Training
| Hyperparameter | Paper | Implementation | Match |
|---|---|---|---|
| Batch size | 512 | 512 | ✓ |
| Learning rate | 0.4 | 0.4 | ✓ |
| Momentum | 0.9 | 0.9 | ✓ |
| Weight decay | 0.0001 | 0.0001 | ✓ |
| Scheduler | Cosine | Cosine | ✓ |
| Augmentations | Random crop, flip, colour jitter, grayscale | Identical | ✓ |
| Embedding | 512-dim L2 normalised penultimate layer | Identical | ✓ |
| Epochs | 500 | 50 | ✗ — hardware constraint |

## Fully Supervised Evaluation
| Hyperparameter | Paper | Implementation | Match |
|---|---|---|---|
| Architecture | ResNet-18 | ResNet-18 | ✓ |
| Optimiser | SGD, momentum 0.9, Nesterov | Identical | ✓ |
| Learning rate | 0.025 | 0.025 | ✓ |
| Scheduler | Cosine | Cosine | ✓ |
| Augmentations | Random crop, horizontal flip | Identical | ✓ |
| Epochs | 200 | 200 | ✓ |

## Self-Supervised (Linear) Evaluation
| Hyperparameter | Paper | Implementation | Match |
|---|---|---|---|
| Architecture | Single linear layer d×C | Identical | ✓ |
| Optimiser | SGD, momentum 0.9 | Identical | ✓ |
| Learning rate | 2.5 | 2.5 | ✓ |
| Scheduler | Cosine | Cosine | ✓ |
| Epochs | 200 | 200 | ✓ |

## Semi-Supervised Evaluation
| Hyperparameter | Paper | Implementation | Match |
|---|---|---|---|
| Architecture | WideResNet-28, widen=2 | Identical | ✓ |
| Optimiser | SGD, momentum 0.9 | Identical | ✓ |
| Learning rate | 0.03 | 0.03 | ✓ |
| Weight decay | 0.0005 | 0.0005 | ✓ |
| Batch size | 64 | 64 | ✓ |
| Leaky slope | 0.1 | 0.1 | ✓ |
| Dropout | None | None | ✓ |
| Weak augmentations | Random crop, horizontal flip | Identical | ✓ |
| Strong augmentations | RandAugment | Identical | ✓ |
| Repetitions | 3× averaged | 3× averaged | ✓ |
| Iterations | 400,000 | 1,000 | ✗ — hardware constraint |

# Note on Compute Limits
Due to hardware constraints, two deviations from the paper were necessary:
- SimCLR backbone trained for 50 epochs instead of 500
- Semi-supervised evaluation run for 1,000 iterations instead of 400,000

All other hyperparameters match the paper exactly.