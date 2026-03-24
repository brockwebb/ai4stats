"""
01_coding_problem.py -- Chapter 12: The Scale of Federal Text Coding

Illustrates the scale of the automated text coding challenge in federal
statistical programs. Shows the coding systems in use (NAICS, SOC, ICD-10),
the annual volumes involved, and examples of the ambiguity that makes
the problem non-trivial.

This script:
- Prints a table of major federal coding tasks with approximate volumes
- Shows concrete industry coding examples with ambiguity annotations
- Explains why rule-based systems leave a residual human review burden

Standalone: no external data files required. Run with Python 3.9+.
No LLM API calls are made.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Federal coding volumes at a glance
# ---------------------------------------------------------------------------

CODING_TASKS = pd.DataFrame({
    "Survey / Program": [
        "Current Population Survey",
        "American Community Survey",
        "Economic Census",
        "National Death Index",
        "Occupational Employment and Wage Statistics",
        "National Health Interview Survey",
    ],
    "Variable coded": [
        "Industry and occupation",
        "Industry and occupation",
        "Industry description",
        "Cause of death",
        "Occupation description",
        "Health conditions",
    ],
    "Classification system": [
        "NAICS / SOC",
        "NAICS / SOC",
        "NAICS",
        "ICD-10",
        "SOC",
        "ICD-10",
    ],
    "Annual volume (approx)": [
        "3 million responses",
        "3 million responses",
        "7 million establishments",
        "2.8 million deaths",
        "1.2 million establishments",
        "35,000 interviews",
    ],
})

# Annotation: coding systems used
SYSTEM_NOTES = {
    "NAICS": (
        "North American Industry Classification System. "
        "20 top-level sectors, 1,057 6-digit industries."
    ),
    "SOC": (
        "Standard Occupational Classification. "
        "23 major groups, 867 detailed occupations."
    ),
    "ICD-10": (
        "International Classification of Diseases, 10th revision. "
        "Over 70,000 codes."
    ),
}

# ---------------------------------------------------------------------------
# Concrete industry coding examples showing the ambiguity problem
# ---------------------------------------------------------------------------

EXAMPLES = [
    (
        "I work at a small bakery downtown making bread and pastries",
        "311811",
        "Retail Bakeries",
        "CLEAR -- specific product and retail setting point directly to NAICS 311811",
    ),
    (
        "I'm a nurse at a clinic",
        "621111",
        "Offices of Physicians (or 622110 General Medical Hospital)",
        (
            "AMBIGUOUS -- 'clinic' spans outpatient offices (621) "
            "and hospital outpatient depts (622). "
            "Requires follow-up question or coder judgment."
        ),
    ),
    (
        "I do IT for a bank",
        "522110",
        "Commercial Banking",
        (
            "NAICS RULE -- NAICS codes the employer's industry, not the occupation. "
            "The worker does IT but the industry is banking."
        ),
    ),
    (
        "I drive for a company that delivers packages",
        "492110",
        "Couriers and Express Delivery Services",
        "CLEAR -- delivery industry is unambiguous",
    ),
    (
        "I work from home for myself",
        "999990",
        "Unclassifiable",
        (
            "MNAR -- self-employed with no industry information provided. "
            "Cannot code without additional follow-up."
        ),
    ),
    (
        "Administrative assistant at a nonprofit that helps veterans",
        "813110",
        "Religious, Civic, Professional, and Similar Organizations",
        (
            "AMBIGUOUS -- 'nonprofit helping veterans' could be "
            "813 (civic) or 624 (social assistance). "
            "Type of nonprofit matters."
        ),
    ),
    (
        "I fix computers",
        "811212",
        "Computer and Office Machine Repair (or 541512 Computer Systems Design)",
        (
            "AMBIGUOUS -- independent repair shop is 811; "
            "IT services firm doing repair work is 541. "
            "Business type matters."
        ),
    ),
    (
        "Teacher at a charter school",
        "611110",
        "Elementary and Secondary Schools",
        "CLEAR -- charter schools are classified with other K-12 schools",
    ),
]


def print_coding_scale_table(df):
    """Print the federal coding burden summary table."""
    print("=" * 75)
    print("FEDERAL TEXT CODING BURDEN: ANNUAL VOLUMES")
    print("=" * 75)
    print(df.to_string(index=False))
    print()
    print("Human coder productivity:     ~400-800 descriptions per coder per day")
    print("Rule-based autocoding rate:   60-80% automation; remainder to human review")
    print()
    print("Classification systems in use:")
    for name, note in SYSTEM_NOTES.items():
        print(f"  {name}: {note}")


def print_coding_examples(examples):
    """Print industry coding examples with ambiguity annotations."""
    print()
    print("=" * 75)
    print("INDUSTRY CODING EXAMPLES: TEXT -> NAICS")
    print("Each example illustrates a distinct coding challenge")
    print("=" * 75)
    for text, code, label, note in examples:
        print()
        print(f"  Description: '{text}'")
        print(f"  Code:        {code}")
        print(f"  Category:    {label}")
        print(f"  Coding note: {note}")


if __name__ == "__main__":
    print_coding_scale_table(CODING_TASKS)
    print_coding_examples(EXAMPLES)

    print()
    print("=" * 75)
    print("WHY AUTOMATED CODING IS HARD")
    print("=" * 75)
    ambiguous = [ex for ex in EXAMPLES if "AMBIGUOUS" in ex[3] or "MNAR" in ex[3]]
    clear = [ex for ex in EXAMPLES if "CLEAR" in ex[3]]
    print(f"  Clear cases in this sample:     {len(clear)} / {len(EXAMPLES)}")
    print(f"  Ambiguous/unclear in sample:    {len(ambiguous)} / {len(EXAMPLES)}")
    print()
    print("  In practice, ambiguous cases cluster around:")
    print("    - Dual-industry employers (hospital IT, law firm admin)")
    print("    - Self-employment with vague descriptions")
    print("    - Adjacent sectors (health vs. social assistance)")
    print("    - Occupation-sounding descriptions (NAICS codes the INDUSTRY, not the job)")
