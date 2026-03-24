# Chapter 1 Code Examples: Regression and Classification for Survey Data

Run in order:

1. `01_generate_survey_data.py` -- creates the synthetic dataset
2. `02_regression_income.py` -- linear regression for income prediction
3. `03_classification_nonresponse.py` -- logistic regression for nonresponse
4. `04_income_brackets.py` -- multi-class classification
5. `05_exercises.py` -- activity solutions

## Requirements

Python 3.9+, numpy, pandas, matplotlib, scikit-learn

## Quick start

```bash
pip install numpy pandas matplotlib scikit-learn
python 01_generate_survey_data.py
python 02_regression_income.py
```

## What the dataset contains

`01_generate_survey_data.py` creates `synthetic_acs_survey.csv` with 1,200 records
and these columns:

| Column | Description |
|--------|-------------|
| `state` | One of five states (California, Texas, New York, Florida, Illinois) |
| `age` | Age in years, 18-80 |
| `education_years` | Years of education (9, 12, 14, 16, or 18) |
| `hours_per_week` | Hours worked per week, 0-80 |
| `urban` | 1 = urban tract, 0 = rural |
| `contact_attempts` | Number of survey contact attempts, 1-7 |
| `prior_response` | 1 = responded in prior survey cycle |
| `income` | Annual income in dollars |
| `responded` | 1 = responded to this survey, 0 = did not respond |

Records are synthetic and do not represent real individuals. Income is drawn
from a log-normal distribution; nonresponse probability is modeled as a
logistic function of contact history and prior response.
