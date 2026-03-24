"""
03_evaluation_dataset.py -- Chapter 12: Simulated LLM Evaluation Dataset

Constructs a 200-record simulated evaluation dataset for LLM industry coding.
Simulates LLM responses with realistic error patterns observed in published
studies of GPT-4 class models on NAICS coding tasks.

This script:
- Generates 200 records (10 NAICS sectors, 20 descriptions each)
- Simulates LLM responses with adjacent-sector confusion and occasional refusals
- Uses a fixed seed (2025) for reproducibility
- Prints a summary of the dataset

Standalone: no external data files required. Run with Python 3.9+.
No LLM API calls are made. All responses are simulated.
"""

import numpy as np
import pandas as pd

# Fixed seed for reproducibility across all Chapter 12 scripts
RANDOM_SEED = 2025

# ---------------------------------------------------------------------------
# NAICS sector definitions (2-digit, 10 sectors used in this simulation)
# ---------------------------------------------------------------------------

SECTORS = {
    "44-45": "Retail Trade",
    "62":    "Health Care and Social Assistance",
    "61":    "Educational Services",
    "54":    "Professional, Scientific, and Technical Services",
    "72":    "Accommodation and Food Services",
    "23":    "Construction",
    "52":    "Finance and Insurance",
    "51":    "Information",
    "81":    "Other Services",
    "92":    "Public Administration",
}

SECTOR_CODES = list(SECTORS.keys())

# ---------------------------------------------------------------------------
# Representative text descriptions by sector (20 per sector)
# ---------------------------------------------------------------------------

DESCRIPTIONS_BY_SECTOR = {
    "44-45": [
        "I work at a department store as a sales associate",
        "Cashier at a grocery store",
        "I manage a clothing boutique downtown",
        "Works at an auto parts store",
        "Retail manager at a big box store",
        "I sell electronics at a consumer electronics retailer",
        "Bookstore employee",
        "Hardware store sales associate",
        "Pet supply store manager",
        "Flower shop owner",
        "Furniture store sales representative",
        "I stock shelves at a drugstore",
        "Assistant manager at a sporting goods store",
        "Used car lot salesperson",
        "I work the register at a convenience store",
        "Toy store employee during the holiday season",
        "Shoe store manager",
        "I work in the appliance department of a home improvement store",
        "Office supply store supervisor",
        "Gift shop attendant at a museum gift shop",
    ],
    "62": [
        "Registered nurse at a hospital",
        "I work at a dental office as a hygienist",
        "Medical billing specialist at a health clinic",
        "Home health aide helping elderly clients",
        "Pharmacist at a community pharmacy",
        "Physical therapist at a rehabilitation center",
        "Receptionist at a doctor's office",
        "I work in the emergency room",
        "Mental health counselor at a treatment facility",
        "Medical laboratory technician",
        "Occupational therapist at a children's health center",
        "Radiologist at a diagnostic imaging center",
        "Case manager at a social services agency",
        "Substance abuse counselor at a residential facility",
        "Speech-language pathologist at a clinic",
        "Medical coder for a hospital system",
        "I provide in-home care for people with disabilities",
        "Phlebotomist at a blood draw center",
        "Nursing home activities coordinator",
        "Health educator at a community health center",
    ],
    "61": [
        "High school math teacher",
        "I teach kindergarten at a public school",
        "College professor in the biology department",
        "School librarian",
        "Special education teacher",
        "University administrator in the financial aid office",
        "Tutor at a learning center",
        "Vocational education instructor at a technical school",
        "School principal at an elementary school",
        "Instructor at a community college",
        "Preschool teacher at a private daycare center",
        "Instructional coach for K-12 teachers",
        "Academic advisor at a four-year university",
        "Test prep instructor at a tutoring company",
        "School counselor at a middle school",
        "ESL instructor at a community learning center",
        "Online course developer for a university system",
        "Swim instructor at a school district facility",
        "I teach adult literacy classes at a nonprofit",
        "Dean of students at a private college",
    ],
    "54": [
        "Software developer at a technology consulting firm",
        "I'm an accountant at a CPA firm",
        "Attorney at a law firm specializing in corporate law",
        "Management consultant",
        "Civil engineer at an engineering firm",
        "I do market research for a consulting company",
        "Graphic designer at a design agency",
        "Architect at a small architecture firm",
        "HR consultant for small businesses",
        "Data analyst at a research and consulting firm",
        "Environmental scientist at an environmental consulting firm",
        "Patent attorney at an intellectual property firm",
        "Business analyst at a strategy consulting firm",
        "Technical writer for a professional services firm",
        "Veterinarian at a private veterinary clinic",
        "I provide PR and communications services to corporate clients",
        "Landscape architect at a planning firm",
        "Tax advisor at a financial planning firm",
        "Quality assurance consultant",
        "Translation and interpretation services provider",
    ],
    "72": [
        "I cook at a restaurant downtown",
        "Server at a hotel restaurant",
        "Hotel front desk manager",
        "Barista at a coffee shop",
        "Line cook at a fast food restaurant",
        "Catering manager for event services",
        "I run a food truck",
        "Housekeeper at a hotel",
        "Restaurant owner",
        "Banquet coordinator at a conference center",
        "Bartender at a sports bar",
        "Front of house manager at a fine dining restaurant",
        "Shift supervisor at a quick service restaurant",
        "Dishwasher at a hotel kitchen",
        "Pastry chef at a bakery-cafe",
        "Room service attendant at a full-service hotel",
        "Campground host at a private campsite",
        "Concierge at a luxury hotel",
        "Pizza delivery driver for a pizzeria",
        "Drive-through attendant at a fast food chain",
    ],
    "23": [
        "Construction worker building houses",
        "I'm a plumber working for a contracting company",
        "Electrician doing commercial wiring",
        "Site superintendent for a general contractor",
        "Roofer for a roofing company",
        "Painting contractor",
        "HVAC technician for a mechanical contractor",
        "Concrete finisher",
        "I install drywall for a construction company",
        "Project manager at a construction firm",
        "Heavy equipment operator at a road construction site",
        "Carpenter for a residential builder",
        "Tile setter for a flooring contractor",
        "Masonry worker for a commercial construction company",
        "Ironworker at a steel frame construction site",
        "Safety officer for a general contracting company",
        "Estimator at a construction company",
        "Demolition crew member",
        "Insulation installer",
        "Swimming pool contractor",
    ],
    "52": [
        "Loan officer at a bank",
        "I work in claims processing at an insurance company",
        "Financial advisor at a wealth management firm",
        "Bank teller",
        "Underwriter at an insurance company",
        "I process mortgage applications for a lending company",
        "Investment analyst at a brokerage",
        "Credit analyst at a regional bank",
        "Insurance agent",
        "Financial analyst at a brokerage firm",
        "Compliance officer at a commercial bank",
        "Securities trader at an investment bank",
        "Actuary at a life insurance company",
        "Branch manager at a credit union",
        "Foreign exchange dealer at a currency exchange firm",
        "Risk analyst at a financial services company",
        "Mortgage broker",
        "Trust officer at a bank",
        "Insurance claims adjuster",
        "I manage mutual fund accounts at an asset management firm",
    ],
    "51": [
        "Software engineer at a tech startup",
        "I work in network administration for an IT company",
        "Data scientist at a technology firm",
        "I manage social media for a digital marketing company",
        "Game developer at a video game studio",
        "Cybersecurity analyst at a cybersecurity firm",
        "I produce content for an online media company",
        "Database administrator for a software company",
        "I work in cloud infrastructure at a tech company",
        "UX designer at a software firm",
        "Product manager at a SaaS company",
        "I work in technical support for a software vendor",
        "AI researcher at a technology company",
        "DevOps engineer at a cloud services firm",
        "Content moderator for a social media platform",
        "Podcast producer for a digital media company",
        "I write code for a company that makes mobile apps",
        "Data engineer at an analytics software firm",
        "Streaming platform content curator",
        "I manage digital advertising campaigns for an online platform",
    ],
    "81": [
        "Auto mechanic at a repair shop",
        "Hairdresser at a salon",
        "I fix appliances for a home services company",
        "Cemetery groundskeeper",
        "Pet groomer",
        "Dry cleaner",
        "Funeral director",
        "I repair shoes at a cobbler shop",
        "Car wash attendant",
        "Laundromat owner",
        "Tattoo artist at a parlor",
        "Nail technician at a nail salon",
        "I detail cars at an auto detailing shop",
        "Seamstress at an alterations shop",
        "I do oil changes at a quick lube shop",
        "Massage therapist at a day spa",
        "Locksmith",
        "Upholstery repair technician",
        "I groom dogs at a pet boarding facility",
        "Photo developer at a print shop",
    ],
    "92": [
        "Police officer for the city",
        "I work in records management for a county government",
        "Firefighter for the fire department",
        "Postal worker",
        "I process benefits applications for the state",
        "Court clerk",
        "DMV employee",
        "City council staffer",
        "Social worker at a county agency",
        "Active duty military service member",
        "Budget analyst for a federal agency",
        "I work in IT for a state department of transportation",
        "Public defender in the county public defender's office",
        "Environmental inspector for a state regulatory agency",
        "Census field interviewer",
        "Corrections officer at a state prison",
        "Highway patrol officer",
        "City park ranger",
        "Procurement officer for a federal agency",
        "Elections administrator for the county",
    ],
}


# ---------------------------------------------------------------------------
# LLM response simulation
# ---------------------------------------------------------------------------

# Per-sector accuracy rates calibrated to published benchmarks
# (GPT-4 class models, NAICS 2-digit, studies circa 2023-2025)
SECTOR_ACCURACY = {
    "44-45": 0.92,
    "62":    0.88,
    "61":    0.90,
    "54":    0.78,
    "72":    0.88,
    "23":    0.82,
    "52":    0.84,
    "51":    0.80,
    "81":    0.74,
    "92":    0.86,
}

# When the model is wrong, which sectors does it confuse each sector with?
CONFUSION_PATTERNS = {
    "54": ["51", "52", "81"],
    "51": ["54", "52"],
    "62": ["81", "61"],
    "81": ["62", "54", "23"],
    "92": ["61", "62"],
    "52": ["54", "44-45"],
    "23": ["81", "44-45"],
}


def simulate_llm_responses(df, human_col, sector_codes,
                            sector_accuracy, confusion_patterns,
                            seed=RANDOM_SEED):
    """
    Simulate LLM coding responses with realistic error patterns.

    Error model:
    - 2% chance of refusal/unclear response
    - Per-sector accuracy drawn from sector_accuracy dict
    - Errors route to confusion_patterns if defined, else random wrong sector
    """
    rng = np.random.default_rng(seed)
    responses = []

    for _, row in df.iterrows():
        true_code = row[human_col]
        acc = sector_accuracy.get(true_code, 0.82)

        # 2% chance of refusal
        if rng.random() < 0.02:
            responses.append("UNCLEAR")
            continue

        if rng.random() < acc:
            responses.append(true_code)
        else:
            if true_code in confusion_patterns:
                wrong = rng.choice(confusion_patterns[true_code])
            else:
                candidates = [c for c in sector_codes if c != true_code]
                wrong = rng.choice(candidates)
            responses.append(wrong)

    return responses


if __name__ == "__main__":
    # Build base dataset
    records = []
    for sector_code, descs in DESCRIPTIONS_BY_SECTOR.items():
        for desc in descs:
            records.append({
                "description": desc,
                "human_sector": sector_code,
                "sector_name": SECTORS[sector_code],
            })

    df_eval = (
        pd.DataFrame(records)
        .sample(frac=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )

    # Simulate LLM responses
    df_eval["llm_sector"] = simulate_llm_responses(
        df_eval, "human_sector", SECTOR_CODES,
        SECTOR_ACCURACY, CONFUSION_PATTERNS,
        seed=RANDOM_SEED,
    )

    # Summary
    print("=" * 65)
    print("SIMULATED LLM EVALUATION DATASET -- CHAPTER 12")
    print("=" * 65)
    print(f"\nTotal records:        {len(df_eval)}")
    print(f"Sectors represented:  {df_eval['human_sector'].nunique()}")
    print(f"Random seed:          {RANDOM_SEED}")
    print()
    print("Distribution by sector:")
    print(f"  {'Code':<8} {'Sector Name':<44} {'N':>4}  {'LLM Correct':>11}")
    print("  " + "-" * 70)
    for code in SECTOR_CODES:
        mask = df_eval["human_sector"] == code
        n = mask.sum()
        correct = (
            df_eval.loc[mask, "llm_sector"] == code
        ).sum()
        pct = correct / n if n > 0 else 0.0
        print(f"  {code:<8} {SECTORS[code]:<44} {n:>4}  {correct:>5} ({pct:.0%})")

    print()
    n_unclear = (df_eval["llm_sector"] == "UNCLEAR").sum()
    n_correct = (df_eval["human_sector"] == df_eval["llm_sector"]).sum()
    overall_acc = n_correct / len(df_eval)
    print(f"Overall accuracy:  {n_correct}/{len(df_eval)} = {overall_acc:.1%}")
    print(f"Refusals (UNCLEAR): {n_unclear} ({n_unclear/len(df_eval):.1%})")

    print()
    print("Sample agreements (human == LLM):")
    agree = df_eval[df_eval["human_sector"] == df_eval["llm_sector"]].head(4)
    for _, row in agree.iterrows():
        print(f"  [{row['human_sector']}] '{row['description']}'")

    print()
    print("Sample disagreements (human != LLM, excluding UNCLEAR):")
    disagree = df_eval[
        (df_eval["human_sector"] != df_eval["llm_sector"]) &
        (df_eval["llm_sector"] != "UNCLEAR")
    ].head(4)
    for _, row in disagree.iterrows():
        print(f"  Human: [{row['human_sector']:6s}] LLM: [{row['llm_sector']:6s}] "
              f"'{row['description']}'")
