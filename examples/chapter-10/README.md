# Chapter 10 Examples — Statistical Disclosure Limitation in the Age of AI

These scripts demonstrate the concrete SDL risks introduced by AI systems trained on confidential statistical data. They are designed for SDL reviewers and data governance practitioners who need to understand what membership inference attacks look like in practice and how to quickly triage the risk level of a proposed AI deployment.

Neither script requires advanced machine learning knowledge. The goal is to make abstract governance concepts observable and actionable.

---

## Scripts

### `01_membership_inference_demo.py`

**What it demonstrates:** That a trained classifier assigns systematically higher prediction confidence to records it was trained on versus records it has never seen. This confidence gap is the membership inference attack signal.

The script:
1. Generates 200 synthetic records with 5 features and a binary label
2. Splits them into 150 training records and 50 holdout records
3. Trains a RandomForestClassifier
4. Prints the mean max-confidence score for training records vs. holdout records
5. Runs a naive membership inference attack (AUC-ROC of using confidence score as the attack signal)

**Why it matters for SDL reviewers:** Any API that exposes prediction probabilities from a model trained on confidential data provides an adversary with the signal needed to run this attack. This script makes the disclosure risk visible on a simple example. The same signal — typically stronger — exists on high-capacity models trained on rare or unique records.

---

### `02_sdl_risk_classifier.py`

**What it demonstrates:** An explicit, auditable decision-rule system for triaging the SDL risk of a proposed AI deployment.

The classifier takes four inputs:
- `data_sensitivity`: public / restricted / confidential
- `model_capacity`: simple / moderate / high
- `access_mode`: internal_only / rate_limited_api / unrestricted_api
- `dp_applied`: True / False

And returns a risk level (Low / Medium / High / Critical) with a recommended review action and a one-sentence rationale.

The script runs four example scenarios:
1. Confidential data, high-capacity model, unrestricted API, no DP → **Critical**
2. Confidential data, moderate model, internal only, DP applied → **Medium**
3. Restricted data, simple model, rate-limited API, no DP → **Medium**
4. Public data, any model, any access → **Low**

**Why it matters for SDL reviewers:** The SDL evaluation checklist in Chapter 10 Section 8 covers every relevant question. This classifier operationalizes the most consequential risk factors into a quick triage tool for initial deployment screening. All High and Critical results require a full disclosure impact assessment and review board sign-off.

---

## Requirements

- Python 3.9+
- numpy
- scikit-learn

Install dependencies:

```
pip install numpy scikit-learn
```

Both scripts are self-contained and can be run directly:

```
python 01_membership_inference_demo.py
python 02_sdl_risk_classifier.py
```

No data files or external resources are required.

---

## Connection to Chapter 10

These scripts accompany Chapter 10 of the AI4STATS handbook. Chapter 10 covers:
- Why trained models are disclosure channels, not just IT assets
- Membership inference and model inversion attacks (plain language, no math)
- Three structural gaps in federal SDL governance (specification, enforcement, impact)
- Model weights as restricted statistical data (a normative proposal)
- Whether APIs count as "releases" under current SDL definitions
- The SDL evaluation checklist for AI deployments

For the mechanics of differential privacy and synthetic data that underlie the mitigations referenced in these scripts, see the Chapter 9 examples.
