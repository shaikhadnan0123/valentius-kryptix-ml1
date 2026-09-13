# Linear Regression from Scratch (NumPy)

## Overview
This project implements linear regression entirely from scratch using NumPy — no `sklearn.linear_model` — to understand the math behind the hypothesis function, cost function, and gradient descent that ML libraries normally hide. The implementation is validated against scikit-learn's `LinearRegression` on the California Housing dataset.

## Math

### Cost Function
The cost function measures how far off the model's predictions are from the actual values, averaged across all training examples. It's defined as:

J(θ) = (1 / 2m) * Σ (h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²

where `m` is the number of training examples, `h(x)` is the model's prediction, and `y` is the actual value. Errors are squared so that larger mistakes are penalized more heavily than small ones, and so the cost is always non-negative regardless of whether a prediction is too high or too low. The 1/2 factor is a convention that cancels out cleanly when the derivative (gradient) is taken.

### Gradient Derivation
The gradient tells us, for each parameter θⱼ, the direction in which the cost function increases fastest. Taking the partial derivative of J(θ) with respect to θⱼ gives:

∂J/∂θⱼ = (1/m) * Σ (h(x⁽ⁱ⁾) - y⁽ⁱ⁾) * xⱼ⁽ⁱ⁾

This says: for each feature, look at how strongly it correlates with the current prediction errors. Computing this for every parameter at once, in vectorized matrix form, becomes:

∇J(θ) = (1/m) * Xᵀ(Xθ - y)

which avoids looping over each of the 9 parameters (8 features + bias) manually.

### Why Subtract the Gradient?
The gradient points in the direction of steepest increase of the cost function. Since the goal is to minimize cost, each update moves θ in the opposite direction:

θ := θ - α * ∇J(θ)

where α is the learning rate. Subtracting moves downhill toward the minimum; adding would move uphill and increase the cost every iteration, causing it to diverge instead of converge.

## Results

- Dataset: California Housing (20,640 rows, 8 features)
- Train/test split: 80/20
- Learning rate: 0.1, Iterations: 1000
- Cost decreased smoothly from ~2.34 to ~0.26 over training, with no oscillation — confirming gradient descent converged correctly

| Metric | From Scratch | Scikit-learn |
|---|---|---|
| Intercept | 2.0719 | 2.0719 |
| MSE (test) | 0.5560 | — |
| R² (test) | 0.5757 | 0.5758 |

<img width="567" height="454" alt="image" src="https://github.com/user-attachments/assets/73644e1b-50f5-41b0-bb86-f29f8525dbb6" />


## Conclusion
The from-scratch implementation's coefficients, intercept, and R² closely match scikit-learn's `LinearRegression` (differences in the 3rd–4th decimal place), confirming the hand-built gradient descent implementation is mathematically correct.
