# Model Interpretation Report

## Project objective

This project analyzes household receipt of government benefits using public ENIGH 2022 microdata from INEGI. The main objective is to identify socioeconomic, demographic and territorial factors associated with whether a household reports income from government benefits.

The analysis is based on a household-level modeling dataset prepared from the `concentradohogar.csv` file.

## Research question

What household characteristics are associated with receiving government benefits in Mexico according to ENIGH 2022?

## Data source

The project uses public microdata from the National Household Income and Expenditure Survey, ENIGH 2022, published by INEGI.

The main input file is:

- `concentradohogar.csv`

This table contains household-level information, including income components, household composition, socioeconomic stratum, locality size and survey design variables.

## Unit of analysis

The unit of analysis is the household.

Each row in the modeling dataset represents one household.

## Target variable

The target variable is:

- `received_government_benefits`

It was constructed as follows:

- `1` if `government_benefits_income > 0`
- `0` if `government_benefits_income = 0`

This variable identifies whether a household reports income from government benefits.

## Methodology

The project follows a reproducible data analysis workflow:

1. Data preparation from ENIGH 2022 household-level microdata.
2. Construction of a binary target variable for government benefit receipt.
3. Exploratory analysis by socioeconomic stratum, locality size and income per capita.
4. Logistic regression modeling.
5. Model comparison across alternative specifications.
6. Threshold analysis to evaluate precision, recall, false positives and false negatives.

The main model is a binary logistic regression because the outcome variable is binary: the household either reports receiving government benefits or does not.

## Exploratory findings

The exploratory analysis shows several clear patterns.

First, government benefit receipt is higher among households in lower socioeconomic strata. The weighted share of households receiving government benefits is highest in the low stratum and decreases as socioeconomic stratum increases.

Second, benefit receipt is more common in smaller localities. Households in localities with fewer than 2,500 inhabitants show the highest weighted share of government benefit receipt, while households in localities with 100,000 or more inhabitants show the lowest share.

Third, households receiving government benefits tend to have lower income per capita. However, the income distributions of recipient and non-recipient households overlap substantially, suggesting that income alone does not fully explain benefit receipt.

## Logistic regression results

The base logistic regression model shows that the strongest predictor of government benefit receipt is whether the household has at least one elderly member.

Households with elderly members show substantially higher odds of receiving government benefits, holding other variables constant. This result is consistent with the relevance of public transfers associated with older adults.

The model also shows that low socioeconomic stratum and smaller locality size are associated with higher odds of receiving benefits. In contrast, higher income per capita, more employed household members and the presence of minor members are associated with lower odds of receiving benefits, holding other variables constant.

## Model comparison

Three logistic regression specifications were compared:

1. Minimal model
2. Base model
3. Economic-composition model

The minimal model uses five core conceptual predictors: income per capita, household size, elderly presence, locality size and socioeconomic stratum.

The base model adds additional demographic and household composition variables, including head of household age, presence of minors, number of employed members, food spending share and sex of the household head.

The economic-composition model adds labor income share to the base model.

The economic-composition model achieved the highest predictive performance, with the best ROC AUC, accuracy and F1 score. However, the improvement over the base model was small. The minimal model also performed well, suggesting that much of the predictive signal is captured by a compact set of household, socioeconomic and territorial variables.

The base model is retained as the main specification because it provides a strong balance between predictive performance, interpretability and methodological coherence.

## Threshold analysis

The base model was evaluated under different classification thresholds from 0.20 to 0.80.

At the default threshold of 0.50, the model achieved the highest accuracy. However, this threshold produced more false negatives than false positives, meaning that the model was relatively conservative when classifying households as benefit recipients.

A threshold of 0.35 maximized the F1 score. Compared with the 0.50 threshold, the 0.35 threshold increased recall and reduced false negatives, but also increased false positives.

This shows that threshold selection depends on the analytical or policy objective. If the priority is to reduce exclusion errors and identify more recipient households, a lower threshold may be preferable. If the priority is to avoid overclassifying households as recipients, the default 0.50 threshold is more conservative.

## Limitations

This analysis has several limitations.

First, the model estimates associations, not causal effects. The results should not be interpreted as evidence that a given variable causes benefit receipt.

Second, the model uses household-level information from ENIGH 2022 and does not identify eligibility rules for each specific government program.

Third, the target variable is based on reported income from government benefits. Measurement error, reporting differences or aggregation across programs may affect the observed outcome.

Fourth, the current model uses a standard train-test split and does not implement cross-validation or complex survey design estimation. Survey expansion factors are used in descriptive analysis, but the logistic regression is estimated as an applied predictive model.

## Conclusion

The analysis suggests that government benefit receipt is strongly associated with household demographic composition, socioeconomic conditions and territorial context.

Households with elderly members, households in lower socioeconomic strata and households in smaller localities show higher odds of receiving government benefits. Higher income per capita is associated with lower odds of benefit receipt.

The model comparison shows that a relatively compact logistic regression specification can achieve strong predictive performance while remaining interpretable. The threshold analysis further shows that classification decisions involve a trade-off between false positives and false negatives.

Overall, this project demonstrates a reproducible workflow for preparing public microdata, building interpretable statistical models, evaluating classification performance and communicating results in an applied social analysis context.