# Chapter 4 - Neural Networks Basics

> The building blocks of deep learning, without the hype. You will understand what a neural network does, when to use one, and when a Random Forest is the better choice.

> Full runnable code for all examples is in `examples/chapter-04/`.

```{admonition} Who is this for?
If you finished Chapters 1, 2, and 3 (Python/Pandas, Census APIs, record linkage), you are ready.
By the end of this chapter you will have seen four model families (logistic regression, decision tree, Random Forest, neural network) all applied to the same survey dataset. Model selection — choosing which one to use — is the real skill this chapter builds.
```

## Learning goals

- Describe what a neuron, a layer, and an activation function do.
- Explain backpropagation conceptually without needing to implement it.
- Train an MLP (multilayer perceptron) with scikit-learn for both classification and regression.
- Read a training curve and identify overfitting.
- Tune hidden layer sizes, learning rate, and regularization.
- Compare neural network performance to Random Forest and logistic regression on the same data.
- State clearly when a neural network is and is not the right tool for federal survey work.

```{admonition} Why this matters for federal statistics
:class: tip
Neural networks power many AI products that federal agencies are evaluating or procuring: natural language processing for open-ended survey responses, image classification for form processing, and anomaly detection in administrative records. Understanding the basics lets you:

- Ask the right questions when a vendor proposes a neural network solution.
- Recognize when the interpretability cost is worth paying.
- Know that for most tabular survey prediction tasks, simpler models are competitive and easier to defend.
- Brief leadership on neural networks without resorting to either hype or dismissal.
```

---

## 1. What is a neural network?

### 1.1 The neuron

A single artificial neuron takes a weighted sum of its inputs and passes it through a nonlinear function called an **activation function**:

$$a = \sigma\!\left(\sum_{j} w_j x_j + b\right)$$

- $x_j$ are the inputs (feature values).
- $w_j$ are the learned weights (how much to trust each input).
- $b$ is a bias term (shifts the activation threshold).
- $\sigma$ is the activation function. Common choices: ReLU ($\max(0, z)$), sigmoid ($1/(1+e^{-z})$), or tanh.

A single neuron with a sigmoid activation *is* logistic regression. The power of neural networks comes from stacking many neurons in layers.

### 1.2 Layers

A multilayer perceptron (MLP) stacks neurons into three types of layers:

- *Input layer*: one node per feature. No computation — it simply passes feature values forward.
- *Hidden layers*: each unit computes a weighted sum of the previous layer's outputs, then applies an activation function. This is where the learning happens.
- *Output layer*: one unit for regression (raw value) or binary classification (sigmoid probability), or one unit per class for multi-class (softmax).

The architecture diagram in `examples/chapter-04/01_dataset_setup.py` visualises a network with 5 inputs, two hidden layers of 6 units each, and one binary output. Every node in each layer connects to every node in the next layer — a *fully connected* network. Those connections carry the learned weights.

### 1.3 Activation functions

Without nonlinear activations, stacking layers is mathematically equivalent to a single linear transformation — no more powerful than logistic regression. Activation functions break the linearity, allowing the network to learn complex patterns.

- *ReLU* (`max(0, z)`): default for hidden layers. Computationally cheap, avoids vanishing gradients in deep networks.
- *Sigmoid*: output layer for binary classification. Squashes the output to (0, 1), interpretable as a probability.
- *Tanh*: alternative to sigmoid; output in (-1, 1), zero-centered. Sometimes preferred in shallow networks.

The choice of activation function rarely matters more than a few tenths of a percent on tabular survey data. ReLU is the right default.

### 1.4 How training works

Training adjusts all the weights $w_j$ to minimize a *loss function* (prediction error). The process:

1. *Forward pass*: feed a batch of training records through the network, compute predicted outputs.
2. *Loss*: compute the error between predictions and labels. Classification uses cross-entropy loss. Regression uses mean squared error.
3. *Backpropagation*: compute the gradient of the loss with respect to every weight using the chain rule of calculus. This tells each weight "if you increase, does the loss go up or down?"
4. *Gradient descent*: nudge every weight slightly in the direction that reduces loss. The *learning rate* controls how big each nudge is.
5. Repeat for many passes through the training data (*epochs*).

You do not need to implement backpropagation. Scikit-learn does it automatically. What you do need to understand is the training curve: a plot of loss vs. epoch. A well-behaved curve shows loss decreasing and then flattening as the model converges. Loss still falling steeply at the last epoch means the model is undertrained (increase `max_iter`). Loss bouncing erratically means the learning rate is too high.

The gradient descent illustration in `examples/chapter-04/01_dataset_setup.py` shows a toy one-dimensional loss surface. Notice that a learning rate that is too large causes the optimizer to overshoot the minimum and oscillate — the exact pattern you see when an MLP training curve never settles.

---

## 2. Setup: the same survey dataset

We use the same synthetic dataset from Chapters 1-3. This is the final model comparison point for Part I.

The dataset has n=1,200 synthetic survey respondents with five classification features (`age`, `education_years`, `urban`, `contact_attempts`, `prior_response`) and a binary nonresponse outcome, plus four regression features for income prediction.

```{admonition} Standardization is required for neural networks
:class: warning
Logistic regression and Random Forests can work with raw feature values. Neural networks cannot. Gradient descent is sensitive to feature scale — a feature measured in dollars (10,000–250,000) will dominate a feature measured in years (18–80) unless both are standardized. Always use `StandardScaler` before training an MLP. Fit the scaler on train, transform both train and test. The regression target (income) should also be standardized to match the weight initialization scale; de-standardize predictions before computing error metrics. See `examples/chapter-04/01_dataset_setup.py` for the complete setup.
```

---

## 3. MLP for classification: nonresponse prediction

The MLP classifier in `examples/chapter-04/02_mlp_classification.py` uses two hidden layers of 100 and 50 units, ReLU activations, Adam optimizer, and early stopping. Early stopping monitors the held-out validation loss and halts training when it stops improving — the simplest overfitting defense in scikit-learn's MLP.

The training curve is the primary convergence diagnostic. Read it before trusting any metrics. A curve that is still falling steeply when training ends means the model was stopped too early; a curve that oscillates without settling means the learning rate is too high.

```{admonition} Reading a training curve
:class: note
A well-behaved training curve shows cross-entropy loss decreasing and then flattening. Watch for:

- Loss still falling steeply at the end: not enough epochs. Increase `max_iter`.
- Loss bouncing erratically: learning rate too high. Reduce `learning_rate_init`.
- Training loss falls but validation loss rises: overfitting. Add regularization (`alpha`) or rely on early stopping.
```

---

## 4. MLP for regression: income prediction

`examples/chapter-04/03_mlp_regression.py` demonstrates regression on the income target. Two additions beyond the classification setup:

1. The target is standardized before training (mean 0, std 1). Without this, the output layer weights must span a $10,000–$250,000 range, which conflicts with the small random values used at initialization.
2. Predictions are de-standardized after training before computing MAE and R².

The parity plot (actual vs. predicted income) is the standard regression diagnostic. Systematic below-diagonal bias in the high-income range would indicate the model is not capturing the upper tail — a common pattern when a log-income distribution is modeled on limited data.

---

## 5. Hyperparameters: what to tune

Neural networks have more tunable hyperparameters than logistic regression or decision trees. The key ones for scikit-learn's `MLPClassifier`:

| Parameter | What it controls | Typical range |
|-----------|-----------------|---------------|
| `hidden_layer_sizes` | Network depth and width | `(64,)`, `(64,64)`, `(128,64)` |
| `activation` | Nonlinearity in hidden layers | `"relu"` (default), `"tanh"` |
| `learning_rate_init` | Step size for gradient descent | 0.0001 to 0.01 |
| `alpha` | L2 regularization (controls overfitting) | 0.0001 to 1.0 |
| `max_iter` | Maximum training epochs | 200 to 1,000 |
| `early_stopping` | Stop when validation loss plateaus | `True` (recommended) |

### 5.1 Architecture search

`examples/chapter-04/04_architecture_search.py` benchmarks six configurations from a single hidden layer of 50 units to a three-layer pyramid (100, 50, 25). The result on n=1,200 survey records is almost always the same: the AUC spread across all configurations is less than one percentage point. Larger architectures have more parameters, take longer to converge, and are more sensitive to the learning rate — without offering a measurable accuracy advantage.

The lesson is not to find the optimal architecture. The lesson is that architecture tuning is largely wasted effort on modest tabular datasets. If the architecture spread is 0.5 AUC points and the RF-to-MLP gap in Section 6 is also 0.5 AUC points, the "neural network improvement" disappears into tuning noise.

### 5.2 Regularization with alpha (L2 penalty)

`examples/chapter-04/05_regularization.py` sweeps alpha across five orders of magnitude. The diagnostic output is the *train-test AUC gap*:

- Large gap (training AUC >> test AUC): the model is memorizing the training set. Increase alpha.
- Near-zero gap with low test AUC: the model is too constrained to learn. Decrease alpha.
- A gap larger than five AUC points in a vendor presentation should prompt immediate questions about whether the reported metric is training accuracy or held-out accuracy.

---

## 6. The full four-model comparison

`examples/chapter-04/06_four_model_comparison.py` refits all four model families on the same training split and evaluates them on the same held-out test set. The results below are representative of what this script produces on the n=1,200 synthetic dataset:

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.754 | 0.771 | 0.881 | 0.822 | 0.782 |
| Decision Tree (depth 3) | 0.742 | 0.758 | 0.878 | 0.814 | 0.759 |
| Random Forest (200 trees) | 0.771 | 0.784 | 0.893 | 0.835 | 0.813 |
| MLP (100, 50) | 0.768 | 0.781 | 0.887 | 0.831 | 0.809 |

```{admonition} What the comparison shows
:class: tip
On this synthetic tabular dataset, all four models perform within a few points of each other. This is the norm in federal statistics:

- Logistic regression and Random Forest are frequently competitive with neural networks on tabular survey data.
- The interpretability cost of a neural network (no printable rules, no coefficients) is real.
- Neural networks earn their keep on *large* datasets (millions of records) or *unstructured* data (text, images, audio).
- For a 1,200-record survey file, a logistic regression is probably the right call.
```

---

## 7. The interpretability cost

`examples/chapter-04/07_interpretability.py` demonstrates the contrast directly. The output below is representative:

```{code-block} text
Logistic Regression — interpretable coefficients:
  prior_response:    -1.18  (strongest negative predictor of nonresponse)
  contact_attempts:  +0.24  (more attempts → more likely to not respond)
  urban:             -0.29  (urban respondents slightly more likely to respond)
  age:               +0.01  (small positive effect)
  education_years:   -0.02  (negligible)

Neural Network (100,50) — weight matrix shapes:
  Layer 1 weights: (5, 100)   — 500 parameters
  Layer 1 bias:    (100,)     — 100 parameters
  Layer 2 weights: (100, 50)  — 5,000 parameters
  Layer 2 bias:    (50,)      — 50 parameters
  Output weights:  (50, 1)    — 50 parameters
  Total trainable parameters: 5,700
  → No coefficient you can print and explain
```

The logistic regression output is a methodology table. The MLP output is a weight matrix — 5,700 numbers that do not translate into decision rules or odds ratios.

Partial dependence plots (PDPs) provide the best available aggregate explanation for a neural network. They show the marginal effect of one feature on predictions, averaged over all other features. The PDP for `prior_response` will tell you "as prior_response increases from 0 to 1, the predicted nonresponse probability drops by X points." That is useful. But a PDP cannot tell you why the model predicted 0.72 nonresponse probability for household 1042 specifically. For individual-case explanation, you need SHAP values (covered in later chapters) or a simpler model.

### 7.1 The complexity burden

Interpretability is one dimension of a broader cost structure. In federal environments, each additional dimension matters:

*Interpretability cost*: cannot print decision rules; hard to explain a specific decision in response to a respondent inquiry or IG question.

*Deployment cost*: scikit-learn's `MLPClassifier` runs on CPU and requires only numpy and scipy — it will work in Colab or a standard agency Python environment. PyTorch and TensorFlow may require additional IT approval, GPU allocation, or cloud infrastructure that is not in the current ATO.

*Maintenance cost*: retraining pipelines, monitoring for distribution shift, version control for model weights. A logistic regression retrained quarterly is a spreadsheet operation. An MLP retraining pipeline is a software engineering project.

*Approval cost*: OMB review for new statistical methodology, ATO process for new infrastructure, and potentially a Data Governance Board sign-off. The ATO timeline for new infrastructure can exceed the shelf life of the model.

*Auditability cost*: harder to defend in response to FOIA requests, IG audits, or congressional inquiries. "The algorithm determined it" is not an answer when the agency is required to explain individual decisions under the Privacy Act or agency program rules.

These are real costs in federal environments. A one-percent AUC improvement rarely covers them.

---

## 8. When to use a neural network

The model selection guide below summarises the decision for federal survey work. References to "Chapter 11" (Transformers) and "Chapter 12" (LLMs and language models) indicate where unstructured-data applications are covered.

| Situation | Recommended model | Reason |
|-----------|-------------------|--------|
| Small tabular dataset (< 10K records) | Logistic regression or Random Forest | NN overfits easily; simpler models generalize better |
| Medium tabular dataset (10K–1M records) | Random Forest or gradient boosting | Strong performance; interpretable feature importance |
| Large tabular dataset (> 1M records) | Neural network or gradient boosting | NN can learn complex interactions at scale |
| Text data (survey open-ends) | Fine-tuned language model (Chapter 12) | NNs dominate unstructured text |
| Image data (form processing) | CNN (covered in later chapters on language models) | Spatial hierarchy requires NNs |
| Need printable decision rules | Decision tree (shallow) | Rules are auditable and attachable to methodology reports |
| Need coefficients for methodology | Logistic regression | Direct odds-ratio interpretation |
| Constrained IT environment / no GPU | Logistic regression or Random Forest | sklearn MLP uses CPU; PyTorch/TF may require ATO |

```{admonition} The federal IT constraint
:class: warning
Many federal agencies operate in environments with restricted internet access, approved software lists, and no GPU allocation. Scikit-learn's `MLPClassifier` runs on CPU and requires only numpy and scipy — it will work in Colab or a standard Python environment. PyTorch and TensorFlow may not be approved in all agency environments. When evaluating AI tools for production survey use, always check the IT authorization baseline.
```

### 8.1 When neural networks earn their keep

To be specific about the cases where the complexity is justified:

- *Text classification of open-ended survey responses*: a transformer-based classifier (Chapter 11) will substantially outperform any tabular method on free-text fields. This is the primary legitimate use case in survey work.
- *Image processing for form digitization or signature detection*: convolutional networks are the right tool when the input is a scanned document image.
- *Large-scale administrative record linkage*: at tens of millions of records, Random Forests hit computational limits. A well-tuned MLP can be faster and comparably accurate at that scale.
- *NOT*: standard tabular prediction on survey files with fewer than 100K records.
- *NOT*: when the method must be explainable record-by-record to respondents or auditors.

### 8.2 Questions to ask when a vendor proposes a neural network

Before accepting a vendor's claim that a neural network outperforms existing methods on your data, ask these seven questions:

1. *How much training data was used?* Neural networks need far more data than tabular methods. Under 100,000 records, simpler models usually win. Under 10,000, the neural network is almost certainly overfit.
2. *What is the baseline comparison?* Did they compare to a Random Forest on the same data, with the same train-test split, evaluated on the same metric?
3. *What is the performance gap?* If the improvement is less than one to two AUC points, is the additional complexity justified by the agency's actual decision requirements?
4. *How is the model explained?* PDPs? SHAP? Or "trust the system"? For federal programs, "trust the system" is not an acceptable methodology defense.
5. *What is the deployment environment?* GPU required? Cloud dependency? Is it on the approved software list? What does the ATO timeline look like?
6. *What is the retraining cadence?* Neural networks can degrade faster than simpler models when the data distribution shifts (survey population changes, operational procedure changes). Who owns the retraining pipeline?
7. *What happens if the model fails?* Is there a fallback strategy? Can the agency revert to a rule-based system or a logistic regression while the neural network is retrained or audited?

---

## 9. In-class activity

You are evaluating four modeling approaches for a nonresponse prediction task at a regional office. Your office has the same 300-tract dataset used throughout Part I. The following pre-computed results table represents what a full comparison produces on this dataset:

| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Logistic Regression | (run the script) | (run the script) | (run the script) |
| Decision Tree (depth 3) | — | — | — |
| Random Forest (100 trees) | — | — | — |
| MLP (64, 64) | — | — | — |

**Exercise questions:**

1. Run `examples/chapter-04/08_exercises.py` and record the results table. Which model would you recommend deploying at this regional office? Write a one-paragraph justification that cites specific evidence from the metrics.

2. A vendor proposes replacing all four models with a deep neural network. Using the checklist in Section 8.2, write out all seven questions as they apply to this specific tract-level prediction task.

3. The IT department says PyTorch is not on the approved software list. What are your options? (Hint: scikit-learn's `MLPClassifier` uses numpy and scipy, not PyTorch. What does that tell you about the approval question for the sklearn MLP specifically? What are the remaining questions you would still need to answer?)

4. If the MLP achieves 0.815 AUC vs. the Random Forest's 0.813 AUC on the tract dataset, would you recommend the switch? Identify at least three factors from Section 7.1 (The complexity burden) that govern the answer.

5. Optional: modify the solution in `08_exercises.py` to add a fifth model (gradient boosting via `sklearn.ensemble.GradientBoostingClassifier`). Does it change the recommendation?

---

## Key takeaways for survey methodology

- *Neural networks are not magic*. On tabular survey data (hundreds to thousands of records), Random Forests and logistic regression are usually competitive. Use a neural network when you have large datasets, complex nonlinear interactions, or unstructured data (text, images).
- *Standardization is required* for neural networks. Always fit `StandardScaler` on the training set and transform both train and test. Standardize the regression target too.
- *Early stopping prevents overfitting* without manual epoch tuning. Set `early_stopping=True` in sklearn.
- *Partial dependence plots* provide limited explainability for neural networks. They show marginal effects but cannot explain individual predictions.
- *The interpretability cost is real*. A neural network cannot produce a printed decision rule or a coefficient table. In federal programs subject to external audit, this is a legitimate risk.
- *The complexity burden is broader than interpretability*. Deployment, maintenance, approval, and auditability costs all compound. A one-percent AUC improvement rarely covers them.
- *Model selection across four chapters*: the right model depends on data size, interpretability requirements, IT constraints, and performance gaps. Rarely is a neural network the obvious choice for standard survey prediction tasks.

```{admonition} How to explain these methods to leadership
:class: tip
*What is a neural network?* Think of it as a very large system of interconnected equations, each learning to detect a pattern in the data. It stacks these detectors in layers: the first layer might learn simple patterns (higher contact attempts = lower response), the second learns combinations, and so on.

*How is it different from the other models?* It can theoretically learn more complex patterns with enough data. But it requires more data to train well, is harder to explain, and needs careful tuning. On a survey dataset with a few thousand records, it often does not outperform a Random Forest.

*Why would we ever use one?* For natural language processing (analyzing open-ended survey responses at scale), image processing (reading paper forms), or very large administrative record linkage tasks where the relationship is genuinely complex. These are specific use cases, not a general replacement for classical methods.

*What does "black box" mean?* We can tell you what the model predicts, and we can show which features matter overall (via partial dependence). We cannot show a decision tree or a coefficient for each feature. If your program requires individual-case explainability ("why did we impute this value for this household?"), a simpler model is more defensible.
```

---

## Part I summary

You have now seen four model families applied to the same federal survey prediction task. For most survey prediction tasks on tabular data, logistic regression or a Random Forest is the right starting point. Use a decision tree when you need printable rules that can be attached to a methodology report. Consider a neural network only when the data is very large, unstructured, or when simpler models demonstrably and substantially underperform on a well-designed benchmark.

Part II introduces methods for specific federal data challenges: record linkage, dimension reduction, and imputation — problems where the right algorithm choice depends on data structure and agency context, not on maximizing AUC on a standard benchmark.
