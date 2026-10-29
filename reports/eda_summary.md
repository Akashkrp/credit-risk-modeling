# EDA summary

Rows: 1000  |  Default rate: 30.0%

## Categorical features (chi-square test vs default)

| Feature | chi2 | p-value | Highest-risk group (default rate) |
|---|---|---|---|
| sex | 5.3 | 0.0207 | female (35%) |
| job | 1.9 | 0.5966 | 3 (34%) |
| housing | 18.2 | 0.0001 | free (41%) |
| saving_accounts | 36.1 | 0.0000 | little (36%) |
| checking_account | 123.7 | 0.0000 | little (49%) |
| purpose | 13.6 | 0.0579 | vacation/others (42%) |
| age_band | 18.9 | 0.0003 | <25 (41%) |

## Numeric features (Mann-Whitney U, default vs non-default)

| Feature | median good | median bad | p-value |
|---|---|---|---|
| age | 34.0 | 31.0 | 0.0004 |
| credit_amount | 2244.0 | 2574.5 | 0.0059 |
| duration | 18.0 | 24.0 | 0.0000 |
| monthly_burden | 134.9 | 121.1 | 0.0030 |