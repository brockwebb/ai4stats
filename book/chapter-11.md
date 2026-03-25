# Chapter 11 - Transformers for Survey Text Classification

> The architecture behind modern language models, applied to the federal statistics problem of coding open-ended survey responses.

```{admonition} Who is this for?
If you have completed Chapter 4 (Neural Networks Basics), you are ready for the concepts here. The example code is more involved than earlier chapters -- focus on understanding what each component does rather than memorizing syntax. You do not need to implement a transformer from scratch; you need to understand how to use one and when it is appropriate.
```

```{admonition} Why this matters for federal statistics
:class: tip
Statistical agencies code millions of open-ended text responses every year:

- *Industry coding*: "I work at a dental office" gets coded to NAICS 6212.
- *Occupation coding*: "high school math teacher" gets coded to SOC 25-2022.
- *Cause-of-death*: free-text death certificate descriptions get coded to ICD-10.
- *Survey write-ins*: "Other (please specify)" fields require classification.

Historically this required armies of human coders or complex rule-based systems. Transformers have changed this: modern pre-trained language models, fine-tuned on agency-specific data, now achieve near-human accuracy on many coding tasks. Understanding the architecture lets you evaluate these tools, oversee their deployment, and ask the right questions about training data, error patterns, and governance.
```

---

## Learning goals

- Explain the survey text coding problem and why it requires sequence models.
- Describe tokenization: converting text to token IDs.
- Explain embeddings: representing tokens as dense vectors.
- Understand sinusoidal positional encodings and why they are needed.
- Interpret scaled dot-product attention and read an attention weight matrix.
- Describe multi-head attention and why multiple heads are useful.
- Explain the transformer encoder block structure.
- Connect mini-transformers to BERT, GPT, and modern large language models.
- Apply the evaluation checklist before recommending any NLP coding deployment.

---

## 1. Setup

This chapter uses a synthetic industry coding dataset and a small transformer encoder. All code is in `examples/chapter-11/`. Scripts 01-05 require only numpy; scripts 06-10 require PyTorch.

See `examples/chapter-11/01_dataset.py` for dataset generation.

---

## 2. The survey text coding problem

Federal surveys collect free-text responses that must be assigned standardized codes. A respondent who writes "I install electrical wiring in houses" should be coded to SOC 47-2111 (Electricians). A respondent who writes "I work for a software startup" should be coded to NAICS 5415 (Computer Systems Design).

This is a *sequence classification* task: the input is a variable-length text string and the output is a category label (the industry or occupation code).

### Why rule-based systems struggle

Rule-based systems require manually written patterns for every variant. "dental office" and "dentist's office" and "work at the dentist" all mean the same thing. Abbreviations compound the problem: "HR dept at mfg plant" needs to map to two different codes for industry and occupation. Misspellings are common in free-text fields: "resturant server" still means food service. A classifier trained on data learns these patterns automatically.

The scale of the problem makes manual coding infeasible at speed. The American Community Survey collects millions of industry and occupation write-ins per year. The National Death Index processes hundreds of thousands of cause-of-death descriptions annually. Transformers -- and the specialized systems built on them -- are the production solution at this scale.

### NIOCCS: the reference system

The NIOSH Industry and Occupation Computerized Coding System (NIOCCS) is the federal production system for this task. It is discussed in full in Section 11 of this chapter. Understanding its real-world performance gives concrete targets for any agency exploring this approach.

---

## 3. Synthetic industry coding dataset

To illustrate the concepts, we use a small synthetic dataset of short industry descriptions. Each description is 3-8 words drawn from industry-specific vocabulary. The dataset has 150 examples across 6 categories.

**Class distribution:**

| Category | Examples | Label |
|---|---|---|
| agriculture | 25 | 0 |
| construction | 25 | 1 |
| healthcare | 25 | 2 |
| manufacturing | 25 | 3 |
| retail | 25 | 4 |
| technology | 25 | 5 |

**Example descriptions:**

```{code-block} text
[agriculture  ] farm crop dairy worker
[construction ] building concrete site manager
[healthcare   ] hospital patient clinical analyst
[manufacturing] factory assembly production specialist
[retail       ] store customer sales coordinator
[technology   ] software computer programming developer
```

**Train/test split (80/20 stratified):**

| Split | Count | Per class |
|---|---|---|
| Training | 120 | 20 |
| Test | 30 | 5 |

See `examples/chapter-11/01_dataset.py` to generate and inspect the full dataset.

---

## 4. Tokenization: turning text into numbers

Before a neural network can process text, it must convert characters or words to integer indices. We use *character-level tokenization* for this demonstration: each character becomes a separate token.

Character-level tokenization is simple to understand and handles misspellings and abbreviations naturally. Production systems use *subword tokenization* (BPE or WordPiece; Sennrich, Haddow & Birch, 2016), which splits words like "restaurant" into "rest" + "aurant." This middle ground handles rare words without an explosion in vocabulary size.

**Character token example:**

| Text | Token IDs |
|---|---|
| `farm` | `[8, 5, 24, 19]` |
| `crop` | `[9, 24, 21, 22]` |
| `farm crop` | `[8, 5, 24, 19, 2, 9, 24, 21, 22]` |

The vocabulary has two special tokens plus one entry per unique character:
- Index 0: `<pad>` (padding for variable-length sequences in a batch)
- Index 1: `<unk>` (unknown characters not in the training vocabulary)
- Index 2+: one entry per unique character seen in training

```{admonition} Character vs. word tokenization
:class: note
Character tokenization treats each letter individually (vocabulary ~ 30-100 tokens). Word tokenization splits on whitespace (vocabulary ~ 10,000-100,000 tokens). Large language models use subword tokenization (BPE or WordPiece), which splits "restaurant" into "rest" + "aurant" -- a middle ground that handles rare words without an explosion of vocabulary size. For this demo, character tokenization keeps the vocabulary small and the code simple.
```

See `examples/chapter-11/02_tokenization.py` for vocabulary construction, encoding, decoding, and coverage statistics.

---

## 5. Embeddings and positional encoding

### Embeddings

Each token ID is mapped to a dense vector (the *embedding*). The embedding layer is a learned lookup table of shape `(vocab_size, d_model)`. At initialization the embeddings are random. Through training, tokens appearing in similar contexts get similar vectors. After training, "server" and "cashier" are likely closer in embedding space than "server" and "engineer."

### Positional encoding

Self-attention (described next) has no built-in notion of position -- it treats the sequence as a set, not an ordered list. We add a *positional encoding* to each token embedding so the model knows where each token appears in the sequence.

The sinusoidal encoding uses alternating sine and cosine functions at different frequencies:

```{code-block} text
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

This produces a unique vector for each position. Lower-indexed dimensions oscillate rapidly (high frequency, encoding fine position distinctions); higher-indexed dimensions oscillate slowly (low frequency, encoding coarser position information). The pattern ensures that every position has a unique encoding and that nearby positions have similar encodings.

**Example: positional encoding for 5 positions, d_model=8**

```{code-block} text
pos | dim00   dim01   dim02   dim03   dim04   dim05   dim06   dim07
----+----------------------------------------------------------
  0 | +0.000  +1.000  +0.000  +1.000  +0.000  +1.000  +0.000  +1.000
  1 | +0.841  +0.540  +0.100  +0.995  +0.010  +1.000  +0.001  +1.000
  2 | +0.909  -0.416  +0.199  +0.980  +0.020  +1.000  +0.002  +1.000
  3 | +0.141  -0.990  +0.296  +0.955  +0.030  +1.000  +0.003  +1.000
  4 | -0.757  -0.654  +0.389  +0.921  +0.040  +1.000  +0.004  +1.000
```

The positional encoding is added to the token embedding: `H_t = E_t + PE_t`. The addition preserves embedding information while injecting position information.

See `examples/chapter-11/03_embeddings_positional.py` for the full pattern across 64 positions and 64 dimensions, including cosine similarity between adjacent positions.

---

## 6. Scaled dot-product attention

Attention is the core innovation of transformers (Vaswani et al., 2017). It lets each token in a sequence "look at" other tokens and decide how much to weight their information when computing its own representation.

Given three matrices Q (queries), K (keys), V (values):

```{code-block} text
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

- *Q* (query): what this token is looking for.
- *K* (key): what each token offers.
- *V* (value): what each token contributes if attended to.
- The dot product `QK^T` measures how well each query matches each key.
- Dividing by `sqrt(d_k)` prevents the scores from growing too large as dimension increases.
- Softmax converts scores to weights that sum to 1.
- The output is a weighted sum of the value vectors.

In plain language: each token looks at all other tokens, assigns attention weights based on similarity, and takes a weighted average of their information. A token in "dairy farm worker" will attend strongly to other tokens that appear together in similar contexts.

**Example: attention weight matrix for "farm crop dairy worker"**

The matrix below shows which token (row) attends to which other token (column). Values sum to 1.0 across each row. This example uses random weights before training -- a trained model produces more semantically meaningful patterns.

```{code-block} text
         farm     crop    dairy   worker
farm   | 0.2541  0.2483  0.2488  0.2488
crop   | 0.2491  0.2507  0.2499  0.2503
dairy  | 0.2491  0.2513  0.2497  0.2499
worker | 0.2496  0.2511  0.2492  0.2501
```

```{admonition} Reading an attention heatmap
:class: note
Each row is a query token (the token that is "asking"). Each column is a key token (the token being "asked"). A high value at position (row i, col j) means the i-th token is strongly attending to the j-th token. In a trained model for industry coding, "dairy" might attend strongly to "farm" -- their co-occurrence defines the agriculture category.
```

See `examples/chapter-11/04_attention_numpy.py` for a full worked example with annotated matrix printouts.

---

## 7. Multi-head attention

Instead of one set of Q/K/V projections, *multi-head attention* runs `h` parallel attention operations ("heads"), each with separate projection matrices. The outputs are concatenated and projected back:

```{code-block} text
MHA(X) = Concat(head_1, ..., head_h) W_O
head_j  = Attention(X W_j^Q,  X W_j^K,  X W_j^V)
```

Different heads can learn different relationships. One head might attend to adjacent characters (local syntax). Another might attend to the first word in the description (often the strongest category signal). A third might pick up suffix patterns: "-er", "-ist", "-or" are common occupational role indicators. Multi-head attention lets all of these specializations develop simultaneously within a single layer.

**Example: two heads on "farm crop dairy worker"**

Head 1 might concentrate attention on "farm" and "dairy" (content words). Head 2 might distribute attention more evenly or focus on "worker" (the noise word appended to all descriptions). After training, the specialization becomes more pronounced.

See `examples/chapter-11/05_multihead_attention.py` for a side-by-side comparison of both attention matrices.

---

## 8. The Transformer Encoder block

A full Transformer Encoder block wraps multi-head attention with residual connections, layer normalization, and a feed-forward sub-layer:

```{code-block} text
H'  = LayerNorm(H  + MHA(H))
H'' = LayerNorm(H' + FFN(H'))
```

The *residual connection* (He et al., 2016) helps gradients flow during training -- it ensures that even if the attention sub-layer learns nothing initially, the signal still passes through. *Layer normalization* (Ba, Kiros & Hinton, 2016) stabilizes activations by normalizing across the embedding dimension. The *feed-forward network* (FFN) applies a pointwise nonlinear transformation to each position independently after attention.

For classification, the per-position representations are averaged over all non-padding tokens (*mean pooling*). This single vector is then passed through a linear classification head to produce logits over the output categories.

**Architecture diagram:**

```{code-block} text
Input text
  -> character tokenization
  -> embedding  (vocab_size -> d_model=64)
  -> + sinusoidal positional encoding
  -> multi-head self-attention  (4 heads, d_k=16 each)
  -> residual connection + LayerNorm
  -> feed-forward network  (64 -> 128 -> 64, ReLU)
  -> residual connection + LayerNorm
  -> mean pooling over non-padding positions
  -> classifier  (64 -> 64 -> 6 classes)
```

See `examples/chapter-11/06_model.py` for the full architecture, parameter breakdown, and a forward pass test.

---

## 9. Case study: training a mini transformer

The model in `examples/chapter-11/06_model.py` is trained for 30 epochs on the 120-example training set using cross-entropy loss and the Adam optimizer (Kingma & Ba, 2015).

**Training results (representative run):**

| Epoch | Train loss | Train acc | Test loss | Test acc |
|---|---|---|---|---|
| 5 | 1.62 | 0.325 | 1.70 | 0.267 |
| 10 | 1.41 | 0.508 | 1.52 | 0.433 |
| 15 | 1.17 | 0.658 | 1.31 | 0.567 |
| 20 | 0.93 | 0.775 | 1.08 | 0.667 |
| 25 | 0.71 | 0.867 | 0.87 | 0.733 |
| 30 | 0.54 | 0.917 | 0.72 | 0.800 |

**Per-class classification report (test set, representative run):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| agriculture | 0.83 | 0.80 | 0.81 | 5 |
| construction | 0.80 | 0.80 | 0.80 | 5 |
| healthcare | 0.83 | 1.00 | 0.91 | 5 |
| manufacturing | 0.75 | 0.60 | 0.67 | 5 |
| retail | 0.80 | 0.80 | 0.80 | 5 |
| technology | 0.80 | 0.80 | 0.80 | 5 |
| macro avg | 0.80 | 0.80 | 0.80 | 30 |

With 150 training examples and a model trained from scratch, ~80% test accuracy is a reasonable result. Production systems fine-tune large pre-trained models (BERT; Devlin et al., 2019) on thousands of labeled examples, achieving accuracy in the 90-95% range.

See `examples/chapter-11/07_training.py` for the full training loop and `examples/chapter-11/08_evaluation.py` for the evaluation report and confusion matrix.

---

## 10. Attention visualization on real predictions

After training, attention weights reflect what the model has learned. Tokens that are informative for the correct category receive higher attention weights than noise words.

**Example: attention by word for "hospital patient clinical analyst"**

```{code-block} text
Word           Avg attention (head 0)
-----------    ----------------------
hospital       0.2941  ###########
patient        0.2712  ##########
clinical       0.2688  ##########
analyst        0.1659  ######
```

The content words ("hospital", "patient", "clinical") receive more attention than the generic noise word "analyst." This is the learned pattern: the model has associated clinical vocabulary with the healthcare category.

**Confidence contrast: correct vs. incorrect predictions**

A useful operational diagnostic: correct predictions tend to have higher confidence than incorrect ones. If the average confidence on correct predictions is substantially higher than on incorrect predictions, the confidence score is a reliable signal for routing.

See `examples/chapter-11/09_attention_visualization.py` for top-3 attended tokens per example and the confidence contrast analysis.

---

## 11. NIOCCS: automated coding in federal production

The concepts in this chapter are not theoretical. NIOSH's NIOCCS (National Industry and Occupation Computerized Coding System) is a production system that has been coding industry and occupation text to NAICS and SOC codes at scale for over a decade. Understanding its real-world performance grounds the chapter's architecture and human-in-the-loop design choices in operational reality.

Federal statisticians working on survey text classification do not need to start from scratch. NIOCCS is available as a free public API and represents the baseline against which any proposed alternative must be compared.

The ICD-10 cause-of-death context adds a second production data point from a different agency, demonstrating that the approach generalizes across coding schemes and statistical domains.

Both examples also illustrate that human-in-the-loop design is not a theoretical nicety but an engineering requirement in current production systems.

### What NIOCCS is

NIOCCS is a free web-based application that assigns NAICS 2017 industry codes and SOC 2018 occupation codes to free-text descriptions (NIOSH, 2024). It is available via a public web API maintained by CDC/NIOSH. The system adopted machine learning in 2021 and has since processed over 100 million records from federal and academic researchers (CDC/NIOSH, 2022).

The scope of the coding task is substantial: 365 NAICS 2017 codes for industry, 808 SOC 2018 codes for occupation. This is a much harder problem than the 6-class demo in this chapter. Confusable categories abound -- "dental assistant" and "dental hygienist" are different SOC codes, but the free-text descriptions respondents provide often do not contain enough information to distinguish them reliably.

### Performance reality check

Published evaluations report kappa statistics around 70% for detailed NAICS codes and around 80% agreement on broader code categories. These are respectable numbers for a 365-class problem, but they are not perfect.

The autocoding rate -- the fraction of records the system codes without requiring human review -- is 60-72% in practice, depending on input quality and threshold settings. Human professional coders achieve 95-100% completion rates. The gap is not a failure of the ML approach; it reflects the system's design. NIOCCS routes uncertain cases to a human review queue. The combination of automated coding for high-confidence cases and human review for borderline cases produces final outputs with quality approaching manual coding, at a fraction of the cost and time.

This validates the confidence-routing approach modeled in `examples/chapter-11/10_new_descriptions.py`: auto-code confident predictions, send uncertain ones to human reviewers.

### Input quality sensitivity

One of the most consequential findings from NIOCCS evaluations is the sensitivity of ML coding to input quality. One study found 53.6% discordance with raw, unprocessed inputs, dropping to 5.0% with refined inputs (Friesen et al., 2022). The study's conclusion is directly applicable to survey design: "Machine learning algorithms can systematically and consistently classify data but are highly dependent on the quality and amount of input data."

The implication for federal survey methodology is significant. Survey instrument design affects NLP performance downstream. Open-ended fields that elicit vague or terse responses ("works with computers") produce more ambiguous input than fields that prompt for specificity ("describe your main job duties"). A field that collected more structured information would yield cleaner inputs and higher autocoding rates -- even with the same underlying model.

This is a reason to involve data scientists early in survey questionnaire design, not just in post-collection analysis.

### ICD-10 cause-of-death coding

NIOCCS codes industry and occupation. The National Center for Health Statistics uses a different production system for a related task: the ACME (Automated Classification of Medical Entities) system, which has been in production since the late 1960s and assigns ICD-10 codes to cause-of-death free-text narratives.

Recent research has explored applying deep learning models to this task. The results illustrate an important general principle: purpose-trained models consistently outperform general-purpose LLMs on structured coding tasks with well-defined coding schemes. Pedersen et al. (2023) found that GPT-4 correctly assigned full ICD-10 codes to 75% of archaic causes of death and 90% of current causes, while Coutinho & Martins (2024) showed that fine-tuned encoder models matched or exceeded generative LLMs on Portuguese death certificate coding. For structured classification tasks with abundant labeled data, a smaller purpose-trained model often outperforms a general-purpose LLM.

This is not an argument against LLMs in federal statistics. It is an argument for purpose-trained models when labeled data is available and the coding task is well-defined.

### SDL implications

Models trained on respondent free-text have statistical disclosure limitation (SDL) implications. A model trained on confidential survey write-ins encodes information from those responses in its weights. This is relevant regardless of whether the model itself is released. Cross-reference Chapter 10 for the SDL framework and the concept of "confidential models."

---

## 12. Choosing a model: cost, accuracy, and the shifting landscape

```{admonition} This landscape changes weekly
:class: warning
Every specific model name, price point, and benchmark score in this section will be outdated by the time you read it. The value here is not the snapshot — it is the *evaluation process*. The section below shows what that process looks like using early 2026 as an illustration. The framework (define task requirements, identify model class, evaluate cost-performance tradeoff, assess supply chain risk, plan for retraining) is stable. The specific answers are not.
```

### The cost-performance frontier has collapsed

The mini-transformer trained in Section 9 has approximately 40,000 parameters. Modern large language models have tens of billions. The architecture is the same; the scale differs by orders of magnitude. But scale alone is not the decision criterion — cost and task fit are.

GPT-4-class performance cost approximately 30 USD per million tokens in 2023. By early 2026, equivalent quality is available for under 1 USD per million tokens, with mid-tier 8-30B parameter models clustering at 0.03-0.10 USD/M tokens. For a task like NAICS coding — bounded classification, short text, well-defined categories — you do not need a frontier model. The cost-performance decision should be driven by your task requirements, not by marketing.

**Decision framework (snapshot, early 2026):**

| Model class | Typical size | Approx. cost (2026) | Best for | Watch out for |
|---|---|---|---|---|
| Fine-tuned SLM (Phi, Gemma, etc.) | 0.5-7B | Self-hosted or very low API cost | Bounded classification, NAICS/SOC coding | Needs agency-specific fine-tuning; limited on open-ended tasks |
| Mid-tier instruction model | 8-30B | 0.03-0.10 USD/M tokens | General text tasks, moderate reasoning, zero-shot classification | May not beat a fine-tuned SLM on your specific coding scheme |
| Frontier model (GPT-4 class) | 70B-1T+ | 0.50-3.00+ USD/M tokens | Complex reasoning, open-ended generation, few-shot on novel tasks | Overkill and expensive for bounded classification; API dependency |
| Non-autoregressive (Mercury, etc.) | Varies | Emerging, pricing TBD | High-throughput batch processing, latency-sensitive applications | New; limited production track record for federal statistical use |

*Specific models and prices change rapidly. The decision framework — match model class to task requirements, budget, and supply chain risk tolerance — is stable even as the options shift.*

### Small, fine-tuned models beat large general models on bounded tasks

For a federal agency coding 500,000 survey write-ins, a fine-tuned 3B model running on agency hardware may be cheaper, faster, and more controllable than an API call to a 70B+ model. Small language models (SLMs), when quantized for deployment, consistently outperform general-purpose large models on structured classification tasks — including the kind of text-to-code assignment this chapter covers.

Sreenivas et al. (2025) found that AWQ-style quantization consistently preserves model fidelity and reasoning accuracy on SLMs across downstream classification tasks, outperforming pruning as a compression strategy. This makes SLMs viable for on-premises agency deployment without API dependencies.

```{admonition} What changes in large language models
:class: note
- *Architecture*: essentially the same — embedding, positional encoding, stacked transformer blocks, classification or generation head.
- *Scale*: many more layers, much larger embeddings, many more attention heads.
- *Pretraining*: LLMs are pretrained on massive text corpora (predict the next token). This gives them broad language understanding before any task-specific fine-tuning.
- *Fine-tuning*: for industry coding, take a pretrained model (BERT or a federal-approved equivalent) and fine-tune it on labeled occupational descriptions. The pretrained model already understands language; fine-tuning teaches it the specific coding task.
- *The coding workflow*: agency text + labels → fine-tune → deploy → human review of low-confidence predictions.
```

### The architecture landscape is shifting — plan accordingly

The transformer attention mechanism taught in this chapter is the foundation of every production system today. But the field is moving. Diffusion language models (dLLMs) generate and refine entire sequences in parallel rather than predicting tokens one at a time (Tong et al., 2025). Early results claim substantial throughput gains over autoregressive models, though reported magnitudes vary by benchmark and implementation. This does not change how you evaluate a model for your task — it changes which models are on the frontier.

Any deployment plan should include a retraining and re-evaluation schedule, not an assumption that today's model or architecture is permanent.

### Model supply chain security

Before deploying any model on federal data, evaluate the model supply chain with the same rigor you apply to any software acquisition.

**Model provenance matters.** A model's training data, training process, and organizational governance are not always transparent. For federal statistical work, provenance is not optional.

**Model poisoning is a real threat.** Training data can be deliberately manipulated to introduce backdoors, biases, or targeted misclassification behaviors that are invisible during standard evaluation but activate on specific inputs. This is an active area of adversarial ML research, not a theoretical concern.

**Apply existing federal supply chain frameworks.** NIST SP 800-218 (NIST, 2022) provides a starting point. The questions are the same: Who trained it? On what data? Under what governance? Is the training process auditable?

**Practical guidance for agencies:**
- Prefer models with documented training data provenance and published model cards
- Evaluate whether the model's training governance meets your agency's risk tolerance
- Test for targeted misclassification on your specific coding scheme before deployment
- Consider fine-tuning a trusted base model on agency data rather than deploying an unfamiliar model directly
- Treat model selection as a procurement decision with the same supply chain rigor as any other software acquisition

Chapter 12 builds on this foundation by examining how large language models are used in federal contexts, including the alignment step that converts a next-token predictor into a helpful assistant.

---

## 13. When transformers are and are not appropriate

Transformers are the right tool for text classification tasks with enough labeled data. They are not the right tool for tabular prediction, numeric imputation, or any task where interpretability is more important than accuracy.

**Decision guide:**

| Task | Use transformer? | Reason |
|---|---|---|
| Classify 50K industry write-ins to 20 NAICS sectors | Yes | Core use case; fine-tuned BERT achieves near-human accuracy |
| Tabular nonresponse prediction (age, income, contacts) | No | Random forest or logistic regression is faster and more interpretable |
| Sentiment in open-ended satisfaction responses | Yes | Text classification; pre-trained model + fine-tuning works well |
| Impute missing numeric income values | No | Hot-deck or regression imputation; transformers add no value here |
| Extract dates from free-text administrative records | Maybe | Rule-based regex may suffice; transformer only if many edge cases |
| Generate synthetic survey responses for testing | Yes (with caution) | LLMs can generate plausible text, but validate statistical properties |
| Small dataset: 200 labeled descriptions | Maybe | Fine-tune a pre-trained model (BERT); training from scratch will underfit |
| Need to audit every classification decision | No alone | Combine with confidence scores and human review queue |

The evaluation checklist in Section 14 provides a structured way to work through these questions before recommending any deployment.

---

## 14. Evaluation checklist for NLP coding deployments

Before recommending or approving any transformer-based text coding system for production use, work through these questions. The checklist is organized into three phases.

### Before deployment

- What is the target coding scheme (NAICS, SOC, ICD-10, custom)? How many categories?
- How many labeled training examples exist per category? Is there enough data to fine-tune a pre-trained model, or will the system be trained from scratch?
- Was the model trained on your agency's data? A model fine-tuned on ACS occupation write-ins will not generalize well to CPS write-ins without re-training on CPS data.
- Is there an existing production system (NIOCCS, ACME) that already handles this task? What is its published performance baseline? Are there agency-specific requirements that existing systems cannot meet?

### Model evaluation

- What is the autocoding rate at your chosen confidence threshold? Is that threshold justified by the per-class accuracy at that level of confidence?
- What are the per-class accuracy rates? Are there specific categories with systematically low accuracy? (Categories with few training examples or high vocabulary overlap with other categories are common failure modes.)
- How does performance degrade with messy input? Test on descriptions with common abbreviations, misspellings, and incomplete information. The NIOCCS finding of 53.6% discordance with raw inputs is a benchmark to test against.
- Has the model been tested on data from a different time period or survey wave than the training set? Temporal generalization is often overlooked and is essential for production systems.

### Deployment and governance

- Is there a human review queue for low-confidence predictions? What is the workflow for human reviewers, and how are their decisions fed back to improve the model?
- How will the model handle new categories that did not exist in the training data? For example, new NAICS 2027 codes will be completely unknown to a model trained on NAICS 2017. The model will assign every new-category record to the closest existing category. This is a systematic error, not random noise.
- What is the retraining schedule? NAICS, SOC, and ICD-10 codes are revised on regular cycles. A model trained on 2017 codes needs a documented plan for 2027.
- Are there SDL considerations if the model was trained on confidential respondent text? (Cross-reference Chapter 10.) Does the model need disclosure review before release or external API access?

---

## 15. Exercises

**Exercise 1 (provided).** A model was run on 20 new industry descriptions. The table below shows a subset of predictions and confidence scores.

| Description | Predicted | Confidence |
|---|---|---|
| farm livestock grain technician | agriculture | 0.91 |
| hospital nursing treatment lead | healthcare | 0.88 |
| store checkout inventory analyst | retail | 0.84 |
| assembly factory worker | manufacturing | 0.61 |
| building concrete site | construction | 0.72 |
| software system data | technology | 0.55 |

**Exercise 2 (analysis).** Which categories had the lowest confidence in the table above? Propose two explanations for why those descriptions produced lower-confidence predictions. Consider vocabulary overlap, description length, and the information content of the words used.

**Exercise 3 (governance).** Your agency wants to deploy this model for NAICS coding of 500,000 survey write-ins per year. Using the evaluation checklist in Section 14, identify three questions that must be answered before production deployment. For each question, explain what evidence would satisfy it.

**Exercise 4 (design).** The 2027 NAICS revision will add new industry categories not present in the training data. Propose a workflow for handling records that fall into categories the model has never seen. Your workflow should specify: how unknown-category records are identified, how they are routed, and how the model is updated over time.

**Exercise 5 (optional coding).** Modify the confidence threshold in `examples/chapter-11/10_new_descriptions.py` by changing the `AUTOCODE_THRESHOLD` variable. Run the script at thresholds of 0.60, 0.70, 0.80, 0.90, and 0.95. Record the autocoding rate at each threshold. What is the tradeoff? At what threshold would you operate this model in production, and why?

---

## Key takeaways

- *Survey text coding is a high-volume, high-stakes classification task.* Transformers achieve near-human accuracy on industry and occupation coding when fine-tuned on agency-labeled data.
- *Attention is the key innovation.* Each token "attends to" other tokens to build context-aware representations. The attention weight matrix shows which tokens the model found most relevant for each prediction.
- *Character-level tokenization works for demonstrations* but subword tokenization (BPE, WordPiece) is standard in production because it handles the full vocabulary of English more efficiently.
- *You do not need to train from scratch.* Fine-tune a pre-trained model (BERT, RoBERTa, or a federal-approved equivalent) on labeled descriptions. The pre-trained model already understands language; fine-tuning teaches it the specific coding schema.
- *Confidence scores enable human-in-the-loop workflows.* Route high-confidence predictions automatically; flag low-confidence cases for human review. This hybrid approach is how NIOCCS and similar production systems operate.
- *NIOCCS demonstrates that ML-based text coding is already in federal production at scale* (100M+ records). Performance is real but bounded: approximately 70% kappa for detailed NAICS codes, 60-72% autocoding rate. Knowing this baseline is essential for evaluating any proposed replacement or extension.
- *Input quality is a first-order concern.* Survey instrument design affects NLP performance downstream. One study found that refining raw inputs reduced NIOCCS discordance from 53.6% to 5.0%. Messy free-text fields produce degraded coding results regardless of model sophistication.
- *Simpler purpose-trained models sometimes outperform general-purpose LLMs* on structured classification tasks. For ICD-10 cause-of-death coding, purpose-trained models matched or exceeded general-purpose LLMs on structured coding tasks. The right model for a task depends on data availability and task specificity, not model size alone.
- *Retraining plans are essential.* NAICS, SOC, and ICD-10 codes are revised on regular cycles. A model trained on 2017 NAICS will produce systematic errors on records belonging to 2027 NAICS categories it has never seen.
- *The same architecture powers ChatGPT and Claude.* GPT and Claude are transformers pretrained on enormous text corpora with an additional alignment step. The mini-encoder trained in this chapter is the same concept at approximately 1/1,000,000th the scale.

```{admonition} How to explain these methods to leadership
:class: tip
**What problem are we solving?** We have 500,000 write-in industry descriptions from the survey and we need to assign each one a NAICS code. Human coding takes months and costs millions. An automated classifier can do it in hours, with human review of the uncertain cases.

**What is a transformer?** It is a type of neural network that processes text by having each word or character "look at" all the other words to understand context. "Server" in "restaurant server" means something different from "server" in "database server" -- the model learns this from the surrounding words.

**Is this proven?** Yes. NIOSH's NIOCCS system has coded over 100 million records using this approach. Published performance is approximately 70% kappa for detailed codes. It reduces coder workload substantially while routing uncertain cases to human review.

**How confident should we be in the results?** The model produces a confidence score for each prediction. Predictions with confidence above 80% are typically reliable. Predictions below 60% should go to human reviewers. This gives us a defensible hybrid process.

**What about bias and errors?** The model reflects patterns in the training data. If training labels have systematic errors, the model learns them. Human review of a random sample -- stratified by category and confidence level -- is essential before operationalizing any coding model.

**Is this interpretable?** More than a black box, less than a decision tree. Attention weights show which words most influenced each prediction. But unlike a rule-based system, you cannot fully enumerate the logic. That is why human review and ongoing error analysis are non-negotiable in federal production use.
```
