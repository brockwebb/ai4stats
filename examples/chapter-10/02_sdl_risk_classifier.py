"""
Chapter 10 — SDL Risk Classifier for AI Deployments
=====================================================

Purpose
-------
This script implements a decision-rule-based risk classifier for evaluating
the SDL (Statistical Disclosure Limitation) risk of a proposed AI deployment
involving confidential statistical data.

It is not a machine learning model. It is an explicit, auditable rule system —
deliberately simple so that reviewers can trace every classification back to
a specific policy rationale. The goal is to help SDL practitioners quickly
diagnose which deployments require the most intensive review.

The classifier takes four inputs that characterize a proposed deployment:
  - data_sensitivity: how sensitive is the training data?
  - model_capacity: how capable is the model of memorizing rare records?
  - access_mode: who can query the model, and how?
  - dp_applied: was differential privacy used during training?

And returns:
  - A risk level: Low, Medium, High, or Critical
  - A recommended review action

Why this matters for SDL reviewers
------------------------------------
The SDL evaluation checklist in Section 8 of Chapter 10 covers many questions.
This classifier operationalizes the most consequential risk factors into a quick
triage tool. It should be used as a starting point for review conversations, not
as a final determination. All High and Critical deployments require full disclosure
impact assessment and sign-off from the appropriate review authority.

Usage
-----
    python 02_sdl_risk_classifier.py

Requirements: Python 3.9+, numpy (for future extensions; not strictly required
by this version of the script)
"""


def classify_sdl_risk(
    data_sensitivity: str,
    model_capacity: str,
    access_mode: str,
    dp_applied: bool,
) -> dict:
    """
    Classify the SDL risk of a proposed AI deployment.

    Parameters
    ----------
    data_sensitivity : str
        Sensitivity of the training data. One of:
        - "public"       : data already publicly released; no confidentiality obligation
        - "restricted"   : licensed microdata or data with access restrictions
        - "confidential" : microdata subject to statutory confidentiality protections
                           (Title 13, Title 26, CIPSEA, etc.)

    model_capacity : str
        The model's capacity to memorize rare records. One of:
        - "simple"    : linear models, shallow decision trees, logistic regression
                        — limited memorization capacity
        - "moderate"  : random forests, XGBoost, gradient boosting
                        — moderate memorization; meaningful membership inference risk
        - "high"      : deep neural networks, large language models, foundation models
                        — high memorization capacity; strongest membership inference
                          and model inversion risk

    access_mode : str
        Who can query the deployed model and under what constraints. One of:
        - "internal_only"    : accessible only to cleared agency staff within
                               a secure enclave or internal network; no external access
        - "rate_limited_api" : accessible externally but with documented rate limits,
                               query logging, and output auditing
        - "unrestricted_api" : accessible externally with no meaningful per-user
                               limits on query volume

    dp_applied : bool
        Whether differential privacy (DP-SGD or equivalent) was applied during
        training with a documented privacy budget (epsilon).

    Returns
    -------
    dict with keys:
        - "risk_level"        : str — "Low", "Medium", "High", or "Critical"
        - "recommended_review" : str — brief description of required review action
        - "rationale"         : str — one-sentence explanation of the primary risk driver
    """
    # Validate inputs
    valid_sensitivity = {"public", "restricted", "confidential"}
    valid_capacity = {"simple", "moderate", "high"}
    valid_access = {"internal_only", "rate_limited_api", "unrestricted_api"}

    if data_sensitivity not in valid_sensitivity:
        raise ValueError(
            f"data_sensitivity must be one of {valid_sensitivity}, "
            f"got '{data_sensitivity}'"
        )
    if model_capacity not in valid_capacity:
        raise ValueError(
            f"model_capacity must be one of {valid_capacity}, "
            f"got '{model_capacity}'"
        )
    if access_mode not in valid_access:
        raise ValueError(
            f"access_mode must be one of {valid_access}, "
            f"got '{access_mode}'"
        )

    # --- Classification logic ---
    # Rule 1: Public training data is Low risk regardless of other factors.
    # The confidentiality obligation exists because of the data, not the model.
    # A model trained on already-public data cannot disclose what was already public.
    if data_sensitivity == "public":
        return {
            "risk_level": "Low",
            "recommended_review": (
                "Standard IT security review. No SDL disclosure review required. "
                "Confirm that training data classification is accurate."
            ),
            "rationale": (
                "Training data is public; no statutory confidentiality obligation applies "
                "to the model regardless of capacity or access mode."
            ),
        }

    # Rule 2: Confidential data + unrestricted API + no DP = Critical.
    # This is the worst-case combination: maximum disclosure obligation, maximum
    # access surface, no technical mitigation. Membership inference and model
    # inversion attacks are both feasible and unconstrained.
    if (
        data_sensitivity == "confidential"
        and access_mode == "unrestricted_api"
        and not dp_applied
    ):
        return {
            "risk_level": "Critical",
            "recommended_review": (
                "Deployment should not proceed without full SDL disclosure impact "
                "assessment, membership inference testing, model inversion testing, "
                "and explicit sign-off from the agency disclosure review authority. "
                "Strongly consider requiring DP training or restricting access mode "
                "before proceeding."
            ),
            "rationale": (
                "Confidential training data, unrestricted external API access, and "
                "no differential privacy applied: this combination maximizes both "
                "the disclosure obligation and the adversary's attack surface with "
                "no technical mitigation in place."
            ),
        }

    # Rule 3: Confidential data + high-capacity model + unrestricted API (even with DP)
    # is still High. DP reduces risk but does not eliminate it for this combination.
    if (
        data_sensitivity == "confidential"
        and model_capacity == "high"
        and access_mode == "unrestricted_api"
    ):
        return {
            "risk_level": "High",
            "recommended_review": (
                "Full SDL disclosure impact assessment required, including explicit "
                "membership inference and model inversion testing. Documented epsilon "
                "required if DP is claimed as a mitigation. Review board sign-off "
                "required before deployment. Consider restricting access mode."
            ),
            "rationale": (
                "High-capacity model trained on confidential data with unrestricted "
                "external API access: even with DP, the attack surface is broad and "
                "the memorization risk from a high-capacity model warrants intensive review."
            ),
        }

    # Rule 4: Confidential data with any high-capacity model (internal or rate-limited)
    # is High without DP, Medium with DP.
    if data_sensitivity == "confidential" and model_capacity == "high":
        if not dp_applied:
            return {
                "risk_level": "High",
                "recommended_review": (
                    "Full SDL disclosure impact assessment required. Membership inference "
                    "testing required. Documented privacy budget (epsilon) strongly "
                    "recommended. Review board sign-off before deployment."
                ),
                "rationale": (
                    "High-capacity model trained on confidential data without differential "
                    "privacy: memorization risk is material and no technical mitigation "
                    "is in place."
                ),
            }
        else:
            return {
                "risk_level": "Medium",
                "recommended_review": (
                    "SDL disclosure impact assessment required. Verify documented epsilon "
                    "and confirm it falls within agency-approved privacy budget. "
                    "Membership inference testing recommended."
                ),
                "rationale": (
                    "High-capacity model on confidential data, but DP applied: risk "
                    "is materially reduced. Documentation and verification of privacy "
                    "budget required."
                ),
            }

    # Rule 5: Confidential data + moderate model + internal only + DP = Medium.
    # This is the controlled-research scenario: strong access controls and DP applied,
    # but confidentiality obligation is real and review is still required.
    if (
        data_sensitivity == "confidential"
        and model_capacity == "moderate"
        and access_mode == "internal_only"
        and dp_applied
    ):
        return {
            "risk_level": "Medium",
            "recommended_review": (
                "SDL disclosure review required. Verify documented epsilon and confirm "
                "internal access controls are enforced. Membership inference testing "
                "recommended as part of standard review."
            ),
            "rationale": (
                "Confidential data and moderate model capacity, but internal-only "
                "access and DP applied significantly reduce the attack surface."
            ),
        }

    # Rule 6: Confidential data + moderate model + any external access without DP = High.
    if (
        data_sensitivity == "confidential"
        and model_capacity == "moderate"
        and access_mode != "internal_only"
        and not dp_applied
    ):
        return {
            "risk_level": "High",
            "recommended_review": (
                "Full SDL disclosure impact assessment required. No DP means no "
                "documented technical mitigation for membership inference risk. "
                "Membership inference and model inversion testing required before "
                "deployment. Review board sign-off required."
            ),
            "rationale": (
                "Confidential data with external API access and no DP: moderate "
                "model capacity is sufficient to produce measurable membership "
                "inference signal, and no technical mitigation is documented."
            ),
        }

    # Rule 7: Restricted data + simple model + rate-limited API + no DP = Medium.
    # Access restrictions reduce risk, but no DP means no documented mitigation.
    if (
        data_sensitivity == "restricted"
        and model_capacity == "simple"
        and access_mode == "rate_limited_api"
        and not dp_applied
    ):
        return {
            "risk_level": "Medium",
            "recommended_review": (
                "SDL disclosure review required. Simple model limits memorization "
                "risk but rate-limited external access without DP still warrants "
                "review. Document access control parameters and query logging."
            ),
            "rationale": (
                "Restricted data with rate-limited external access: simple model "
                "capacity limits memorization risk, but absence of DP means "
                "the risk is not formally bounded."
            ),
        }

    # Rule 8: Restricted data + simple model + internal only = Low-to-Medium.
    # Simple model on restricted data with only internal access is the most
    # controlled non-public case; review is still required but depth is lower.
    if (
        data_sensitivity == "restricted"
        and model_capacity == "simple"
        and access_mode == "internal_only"
    ):
        return {
            "risk_level": "Medium",
            "recommended_review": (
                "Standard disclosure review required. Confirm internal access controls. "
                "Simple model on restricted data is lower-risk but not exempt from review."
            ),
            "rationale": (
                "Restricted data with simple model and internal-only access: "
                "limited attack surface, but confidentiality obligations remain."
            ),
        }

    # Default: all other confidential or restricted combinations not explicitly
    # covered above are treated as High until a more specific assessment is done.
    # This conservative default prevents gaps in coverage from becoming loopholes.
    risk_level = "High" if data_sensitivity == "confidential" else "Medium"
    return {
        "risk_level": risk_level,
        "recommended_review": (
            "SDL disclosure impact assessment required. This combination of "
            "data sensitivity, model capacity, and access mode did not match "
            "a specific rule; a conservative classification is applied. "
            "Consult agency disclosure review authority for guidance."
        ),
        "rationale": (
            "Conservative default classification: data sensitivity and model "
            "characteristics require formal review. See SDL evaluation checklist "
            "in Chapter 10, Section 8 for full assessment guidance."
        ),
    }


def print_scenario(label: str, result: dict) -> None:
    """Print a formatted scenario result."""
    print(f"  Scenario: {label}")
    print(f"  Risk level:         {result['risk_level']}")
    print(f"  Recommended review: {result['recommended_review']}")
    print(f"  Rationale:          {result['rationale']}")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("Chapter 10 — SDL Risk Classifier for AI Deployments")
    print("=" * 70)
    print()
    print(
        "This classifier evaluates four factors that most strongly determine "
        "SDL risk for a proposed AI deployment:\n"
        "  - data_sensitivity: public / restricted / confidential\n"
        "  - model_capacity:   simple / moderate / high\n"
        "  - access_mode:      internal_only / rate_limited_api / unrestricted_api\n"
        "  - dp_applied:       True / False\n"
    )
    print("-" * 70)
    print()

    # Scenario 1: Worst case — confidential + high capacity + unrestricted + no DP
    print_scenario(
        "Confidential data, high-capacity model, unrestricted API, no DP",
        classify_sdl_risk(
            data_sensitivity="confidential",
            model_capacity="high",
            access_mode="unrestricted_api",
            dp_applied=False,
        ),
    )

    # Scenario 2: Controlled research case — confidential + moderate + internal + DP
    print_scenario(
        "Confidential data, moderate model, internal only, DP applied",
        classify_sdl_risk(
            data_sensitivity="confidential",
            model_capacity="moderate",
            access_mode="internal_only",
            dp_applied=True,
        ),
    )

    # Scenario 3: External access without mitigation — restricted + simple + rate-limited + no DP
    print_scenario(
        "Restricted data, simple model, rate-limited API, no DP",
        classify_sdl_risk(
            data_sensitivity="restricted",
            model_capacity="simple",
            access_mode="rate_limited_api",
            dp_applied=False,
        ),
    )

    # Scenario 4: Public data — low risk regardless of other factors
    print_scenario(
        "Public data, high-capacity model, unrestricted API",
        classify_sdl_risk(
            data_sensitivity="public",
            model_capacity="high",
            access_mode="unrestricted_api",
            dp_applied=False,
        ),
    )

    print("-" * 70)
    print(
        "Note: This classifier is a triage tool, not a substitute for full "
        "SDL disclosure impact assessment. All High and Critical classifications "
        "require formal review by the agency disclosure review authority. "
        "See Chapter 10, Section 8 for the complete SDL evaluation checklist."
    )
