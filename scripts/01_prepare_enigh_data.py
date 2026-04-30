import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------
# 01_prepare_enigh_data.py
# Purpose:
# Prepare a household-level modeling dataset using ENIGH 2022
# CONCENTRADOHOGAR public microdata from INEGI.
#
# Main objective:
# Model whether a household receives government benefits
# using socioeconomic and demographic predictors.
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = BASE_DIR / "data" / "raw" / "concentradohogar.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DOCS_DIR = BASE_DIR / "docs"
TABLES_DIR = BASE_DIR / "outputs" / "tables"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = PROCESSED_DIR / "modeling_dataset.csv"
DICTIONARY_PATH = DOCS_DIR / "modeling_variables_dictionary.csv"
SUMMARY_PATH = TABLES_DIR / "modeling_dataset_summary.csv"

# Columns that must be read as text to preserve leading zeros
TEXT_COLUMNS = {
    "folioviv": "string",
    "foliohog": "string",
    "ubica_geo": "string",
    "tam_loc": "string",
    "est_socio": "string",
    "est_dis": "string",
    "upm": "string",
    "clase_hog": "string",
    "sexo_jefe": "string",
    "educa_jefe": "string"
}

# Read raw ENIGH concentradohogar file
df = pd.read_csv(RAW_PATH, dtype=TEXT_COLUMNS)

# Variables selected for modeling
selected_columns = [
    # Identifiers and survey design
    "folioviv",
    "foliohog",
    "ubica_geo",
    "tam_loc",
    "est_socio",
    "est_dis",
    "upm",
    "factor",

    # Household structure and head of household
    "clase_hog",
    "sexo_jefe",
    "edad_jefe",
    "educa_jefe",
    "tot_integ",
    "hombres",
    "mujeres",
    "mayores",
    "menores",
    "p12_64",
    "p65mas",
    "ocupados",
    "percep_ing",
    "perc_ocupa",

    # Income components
    "ing_cor",
    "ingtrab",
    "trabajo",
    "negocio",
    "otros_trab",
    "rentas",
    "transfer",
    "jubilacion",
    "becas",
    "donativos",
    "remesas",
    "bene_gob",
    "transf_hog",
    "trans_inst",
    "estim_alqu",
    "otros_ing",

    # Spending variables
    "gasto_mon",
    "alimentos",
    "vivienda",
    "salud",
    "transporte",
    "educa_espa",
    "personales"
]

missing_columns = [col for col in selected_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"The following expected columns are missing: {missing_columns}")

model_df = df[selected_columns].copy()

# Rename variables to clearer English names for the modeling dataset
rename_map = {
    "folioviv": "dwelling_id",
    "foliohog": "household_id",
    "ubica_geo": "geo_code",
    "tam_loc": "locality_size",
    "est_socio": "socioeconomic_stratum",
    "est_dis": "design_stratum",
    "upm": "primary_sampling_unit",
    "factor": "expansion_factor",
    "clase_hog": "household_type",
    "sexo_jefe": "head_sex",
    "edad_jefe": "head_age",
    "educa_jefe": "head_education",
    "tot_integ": "household_members",
    "hombres": "male_members",
    "mujeres": "female_members",
    "mayores": "adult_members",
    "menores": "minor_members",
    "p12_64": "members_12_64",
    "p65mas": "members_65_plus",
    "ocupados": "employed_members",
    "percep_ing": "income_recipients",
    "perc_ocupa": "employed_income_recipients",
    "ing_cor": "current_income",
    "ingtrab": "labor_income",
    "trabajo": "subordinate_labor_income",
    "negocio": "independent_business_income",
    "otros_trab": "other_labor_income",
    "rentas": "property_income",
    "transfer": "transfers_income",
    "jubilacion": "pensions_income",
    "becas": "scholarships_income",
    "donativos": "donations_income",
    "remesas": "remittances_income",
    "bene_gob": "government_benefits_income",
    "transf_hog": "household_transfers_income",
    "trans_inst": "institution_transfers_income",
    "estim_alqu": "imputed_rent_income",
    "otros_ing": "other_current_income",
    "gasto_mon": "monetary_spending",
    "alimentos": "food_spending",
    "vivienda": "housing_spending",
    "salud": "health_spending",
    "transporte": "transport_spending",
    "educa_espa": "education_recreation_spending",
    "personales": "personal_care_spending"
}

model_df = model_df.rename(columns=rename_map)

# Convert selected numeric variables
numeric_columns = [
    "expansion_factor",
    "head_age",
    "household_members",
    "male_members",
    "female_members",
    "adult_members",
    "minor_members",
    "members_12_64",
    "members_65_plus",
    "employed_members",
    "income_recipients",
    "employed_income_recipients",
    "current_income",
    "labor_income",
    "subordinate_labor_income",
    "independent_business_income",
    "other_labor_income",
    "property_income",
    "transfers_income",
    "pensions_income",
    "scholarships_income",
    "donations_income",
    "remittances_income",
    "government_benefits_income",
    "household_transfers_income",
    "institution_transfers_income",
    "imputed_rent_income",
    "other_current_income",
    "monetary_spending",
    "food_spending",
    "housing_spending",
    "health_spending",
    "transport_spending",
    "education_recreation_spending",
    "personal_care_spending"
]

for col in numeric_columns:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

# Create target variable
model_df["received_government_benefits"] = np.where(
    model_df["government_benefits_income"] > 0,
    1,
    0
)

# Create additional derived variables
model_df["entity_code"] = model_df["geo_code"].str.slice(0, 2)

model_df["income_per_capita"] = np.where(
    model_df["household_members"] > 0,
    model_df["current_income"] / model_df["household_members"],
    np.nan
)

model_df["labor_income_share"] = np.where(
    model_df["current_income"] > 0,
    model_df["labor_income"] / model_df["current_income"],
    np.nan
)

model_df["government_benefits_share"] = np.where(
    model_df["current_income"] > 0,
    model_df["government_benefits_income"] / model_df["current_income"],
    np.nan
)

model_df["food_spending_share"] = np.where(
    model_df["monetary_spending"] > 0,
    model_df["food_spending"] / model_df["monetary_spending"],
    np.nan
)

model_df["has_elderly_member"] = np.where(
    model_df["members_65_plus"] > 0,
    1,
    0
)

model_df["has_minor_member"] = np.where(
    model_df["minor_members"] > 0,
    1,
    0
)

# Label selected categorical variables
locality_size_map = {
    "1": "100,000 or more inhabitants",
    "2": "15,000 to 99,999 inhabitants",
    "3": "2,500 to 14,999 inhabitants",
    "4": "Less than 2,500 inhabitants"
}

socioeconomic_stratum_map = {
    "1": "Low",
    "2": "Lower-middle",
    "3": "Upper-middle",
    "4": "High"
}

head_sex_map = {
    "1": "Male",
    "2": "Female"
}

model_df["locality_size_label"] = model_df["locality_size"].map(locality_size_map)
model_df["socioeconomic_stratum_label"] = model_df["socioeconomic_stratum"].map(socioeconomic_stratum_map)
model_df["head_sex_label"] = model_df["head_sex"].map(head_sex_map)

# Reorder columns
ordered_columns = [
    "dwelling_id",
    "household_id",
    "geo_code",
    "entity_code",
    "locality_size",
    "locality_size_label",
    "socioeconomic_stratum",
    "socioeconomic_stratum_label",
    "design_stratum",
    "primary_sampling_unit",
    "expansion_factor",
    "household_type",
    "head_sex",
    "head_sex_label",
    "head_age",
    "head_education",
    "household_members",
    "male_members",
    "female_members",
    "adult_members",
    "minor_members",
    "members_12_64",
    "members_65_plus",
    "has_elderly_member",
    "has_minor_member",
    "employed_members",
    "income_recipients",
    "employed_income_recipients",
    "current_income",
    "income_per_capita",
    "labor_income",
    "labor_income_share",
    "transfers_income",
    "pensions_income",
    "scholarships_income",
    "donations_income",
    "remittances_income",
    "government_benefits_income",
    "government_benefits_share",
    "household_transfers_income",
    "institution_transfers_income",
    "imputed_rent_income",
    "other_current_income",
    "monetary_spending",
    "food_spending",
    "food_spending_share",
    "housing_spending",
    "health_spending",
    "transport_spending",
    "education_recreation_spending",
    "personal_care_spending",
    "received_government_benefits"
]

model_df = model_df[ordered_columns]

# Save modeling dataset
model_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

# Create a compact data dictionary
dictionary_rows = [
    ["dwelling_id", "Dwelling identifier from ENIGH.", "Text", "Original variable: folioviv"],
    ["household_id", "Household identifier within dwelling.", "Text", "Original variable: foliohog"],
    ["geo_code", "Geographic code. First two digits identify the state.", "Text", "Original variable: ubica_geo"],
    ["entity_code", "Two-digit state code derived from geo_code.", "Text", "Derived variable"],
    ["locality_size_label", "Locality size category.", "Categorical", "Derived label from tam_loc"],
    ["socioeconomic_stratum_label", "Socioeconomic stratum category.", "Categorical", "Derived label from est_socio"],
    ["expansion_factor", "Survey expansion factor.", "Numeric", "Original variable: factor"],
    ["design_stratum", "Survey design stratum.", "Text", "Original variable: est_dis"],
    ["primary_sampling_unit", "Primary sampling unit.", "Text", "Original variable: upm"],
    ["head_sex_label", "Sex of the household head.", "Categorical", "Derived label from sexo_jefe"],
    ["head_age", "Age of the household head.", "Numeric", "Original variable: edad_jefe"],
    ["head_education", "Education level code of the household head.", "Text", "Original variable: educa_jefe"],
    ["household_members", "Total number of household members.", "Numeric", "Original variable: tot_integ"],
    ["members_65_plus", "Number of household members aged 65 or older.", "Numeric", "Original variable: p65mas"],
    ["has_elderly_member", "Indicator for households with at least one member aged 65 or older.", "Binary", "Derived variable"],
    ["has_minor_member", "Indicator for households with at least one minor member.", "Binary", "Derived variable"],
    ["current_income", "Current quarterly household income.", "Numeric", "Original variable: ing_cor"],
    ["income_per_capita", "Current income divided by household members.", "Numeric", "Derived variable"],
    ["labor_income", "Quarterly household labor income.", "Numeric", "Original variable: ingtrab"],
    ["labor_income_share", "Labor income as a share of current income.", "Numeric", "Derived variable"],
    ["government_benefits_income", "Quarterly income from government benefits.", "Numeric", "Original variable: bene_gob"],
    ["government_benefits_share", "Government benefits as a share of current income.", "Numeric", "Derived variable"],
    ["food_spending_share", "Food spending as a share of monetary spending.", "Numeric", "Derived variable"],
    ["received_government_benefits", "Target variable: 1 if government_benefits_income > 0, otherwise 0.", "Binary", "Derived target"]
]

dictionary = pd.DataFrame(
    dictionary_rows,
    columns=["variable", "description", "type", "notes"]
)

dictionary.to_csv(DICTIONARY_PATH, index=False, encoding="utf-8-sig")

# Create summary table
summary = pd.DataFrame({
    "metric": [
        "raw_rows",
        "modeling_rows",
        "modeling_columns",
        "households_with_government_benefits",
        "households_without_government_benefits",
        "share_with_government_benefits_unweighted",
        "mean_current_income",
        "median_current_income",
        "mean_government_benefits_income_among_recipients"
    ],
    "value": [
        len(df),
        len(model_df),
        model_df.shape[1],
        int(model_df["received_government_benefits"].sum()),
        int((model_df["received_government_benefits"] == 0).sum()),
        model_df["received_government_benefits"].mean(),
        model_df["current_income"].mean(),
        model_df["current_income"].median(),
        model_df.loc[
            model_df["received_government_benefits"] == 1,
            "government_benefits_income"
        ].mean()
    ]
})

summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

print(f"Raw rows: {len(df)}")
print(f"Modeling dataset created at: {OUTPUT_PATH}")
print(f"Rows: {model_df.shape[0]}")
print(f"Columns: {model_df.shape[1]}")
print(f"Data dictionary created at: {DICTIONARY_PATH}")
print(f"Summary table created at: {SUMMARY_PATH}")
print("\nTarget variable distribution:")
print(model_df["received_government_benefits"].value_counts(dropna=False))
print("\nUnweighted share with government benefits:")
print(round(model_df["received_government_benefits"].mean(), 4))