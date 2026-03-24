"""
_shared_data.py -- Chapter 12: Shared dataset definitions

Internal module used by scripts 04-09. Contains the sector definitions,
descriptions, accuracy rates, and confusion patterns from 03_evaluation_dataset.py
so they do not need to be duplicated in every script.

Not intended to be run directly.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 2025

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

SECTOR_ACCURACY = {
    "44-45": 0.92, "62": 0.88, "61": 0.90, "54": 0.78,
    "72": 0.88, "23": 0.82, "52": 0.84, "51": 0.80,
    "81": 0.74, "92": 0.86,
}

CONFUSION_PATTERNS = {
    "54": ["51", "52", "81"],
    "51": ["54", "52"],
    "62": ["81", "61"],
    "81": ["62", "54", "23"],
    "92": ["61", "62"],
    "52": ["54", "44-45"],
    "23": ["81", "44-45"],
}

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


def build_eval_df(seed=RANDOM_SEED):
    """Build and return the base evaluation DataFrame (no LLM column)."""
    records = []
    for code, descs in DESCRIPTIONS_BY_SECTOR.items():
        for desc in descs:
            records.append({
                "description": desc,
                "human_sector": code,
                "sector_name": SECTORS[code],
            })
    return (
        pd.DataFrame(records)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def simulate_llm(df, seed=RANDOM_SEED):
    """Simulate LLM coding responses with realistic error patterns."""
    rng = np.random.default_rng(seed)
    responses = []
    for _, row in df.iterrows():
        true_code = row["human_sector"]
        acc = SECTOR_ACCURACY.get(true_code, 0.82)
        if rng.random() < 0.02:
            responses.append("UNCLEAR")
            continue
        if rng.random() < acc:
            responses.append(true_code)
        else:
            if true_code in CONFUSION_PATTERNS:
                wrong = rng.choice(CONFUSION_PATTERNS[true_code])
            else:
                candidates = [c for c in SECTOR_CODES if c != true_code]
                wrong = rng.choice(candidates)
            responses.append(wrong)
    return responses


def simulate_confidence(human_code, llm_code, rng):
    """Simulate LLM confidence score (max softmax probability)."""
    if human_code == llm_code:
        return float(rng.beta(8, 2))
    else:
        if rng.random() < 0.45:
            return float(rng.beta(7, 2.5))
        else:
            return float(rng.beta(3, 5))


def get_full_eval_df(seed=RANDOM_SEED):
    """Return evaluation DataFrame with llm_sector and confidence columns."""
    df = build_eval_df(seed=seed)
    df["llm_sector"] = simulate_llm(df, seed=seed)
    rng = np.random.default_rng(seed)
    df["confidence"] = [
        simulate_confidence(row.human_sector, row.llm_sector, rng)
        for row in df.itertuples()
    ]
    return df
