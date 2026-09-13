# Linear Regression from Scratch (NumPy)

## Overview
[1-2 sentences: what this repo does — linear regression implemented from scratch,
validated on California housing data]

## Math

### Cost Function
[Explain: MSE measures average squared distance between predictions and actual
values. Write out J(θ) formula. Explain why we square errors (penalizes big
misses more, always positive).]

### Gradient Derivation
[Explain: gradient tells us the direction that increases cost fastest, so we
move opposite to it. Write out the gradient formula. Explain why it's
Xᵀ(predictions - actual) / m in vectorized form — this is the part you should
be able to explain to your teaching-mode understanding from Step 5.]

### Why Subtract the Gradient?
[Your own explanation from earlier in this chat]

## Results

- Dataset: California Housing (20,640 rows, 8 features)
- Train/test split: 80/20
- Learning rate: 0.1, Iterations: 1000
- Cost converged smoothly (see plot below) — no oscillation

| Metric | From Scratch | Scikit-learn |
|---|---|---|
| Intercept | 2.0719 | 2.0719 |
| MSE (test) | 0.5560 | — |
| R² (test) | 0.5757 | 0.5758 |

[Insert your cost convergence plot image here]

## Conclusion
[1-2 sentences: coefficients and R² match sklearn closely, proving the
from-scratch implementation is correct]
