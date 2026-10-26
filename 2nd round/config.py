"""Settings of the 2nd round (emerging market): files, data dictionary, codes and thresholds.

Everything that is a decision about this round's data lives here, so the three steps
(01_data_reading, 02_data_cleaning, 03_feature_engineering) contain no hidden constants.
Table and column names are exactly those of the case study files.
"""
from pathlib import Path

ROUND = "2nd round"
MARKET = "emerging market"
COMPETITION = "IAA AI-Assisted Hackathon 2026 - Part 2"

# --- folders -------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW, CLEAN, FEATURES = DATA / "raw", DATA / "clean", DATA / "features"
REPORTS = ROOT / "reports"

# --- dates ---------------------------------------------------------------------------------------
VALUATION_DATE = "2024-12-31"          # claims are valued at this date
TRAIN_YEARS = (2023, 2024)            # the insurer writes this market from 2023
TEST_YEAR = 2025
# Tests of each feature's value (03). Two training years allow one out-of-time fold only, too
# little to judge weak features, so the tests cross-fit by customer instead: learn on half of the
# customers (both years), score the other half, and swap. Every row is scored once; year effects
# are part of the baseline, so a column cannot look useful just by telling the years apart.
# Each fold: (fit years, test years, customer half used for fitting or None).
SIGNAL_FOLDS = (((2023, 2024), (2023, 2024), 0), ((2023, 2024), (2023, 2024), 1))

# --- business lines and perils -------------------------------------------------------------------
LINES = ("motor", "buildings", "contents")
LINE_PREFIX = {"motor": "M", "buildings": "B", "contents": "C"}
PERILS = {
    "motor": ("own_damage", "third_party_prop", "bodily_injury", "theft", "glass_hail"),
    "buildings": ("storm", "flood", "escape_of_water", "subsidence", "fire", "accidental_damage"),
    "contents": ("theft", "escape_of_water", "accidental_damage", "flood"),
}
# Perils a motor cover does not include (a claim of that peril on that cover is out of cover).
PERILS_OUTSIDE_COVER = {
    "third_party_only": ("own_damage", "theft", "glass_hail"),
    "third_party_fire_theft": ("own_damage", "glass_hail"),
}

# --- files of the case study ---------------------------------------------------------------------
TABLES = (
    "customers", "addresses", "weather", "claims",
    "policies_motor", "policies_buildings", "policies_contents",
    "test_motor", "test_buildings", "test_contents", "sample_submission",
)
OPTIONAL_TABLES = ("expenses", "reference_policies")   # 2nd round only: for the office premium
POLICY_TABLES = ("policies_motor", "policies_buildings", "policies_contents",
                 "test_motor", "test_buildings", "test_contents")
PRIMARY_KEYS = {
    "customers": ["customer_id"], "addresses": ["address_id"], "weather": ["address_id", "year"],
    "claims": ["claim_id"], "sample_submission": ["row_id"],
    **{t: ["row_id"] for t in POLICY_TABLES},
}
ID_COLUMNS = ("row_id", "policy_id", "customer_id", "address_id", "claim_id", "repairer_id",
              "third_party_id")
ID_WIDTH = {"customer_id": 8}            # canonical width of all-digit identifiers
DATE_COLUMNS = ("period_start", "period_end", "loss_date", "report_date", "date_of_birth")

# --- data dictionary: (column, type, meaning) ----------------------------------------------------
# type: id | date | category | ordinal | boolean | integer | numeric | text | outcome
_POLICY_COMMON = [
    ("row_id", "id", "one policy row: a policy over one calendar-year period, or part of one"),
    ("policy_id", "id", "the policy; constant across its years and mid-term changes"),
    ("customer_id", "id", "the policyholder (customers table)"),
    ("address_id", "id", "the insured or garaging address (addresses table)"),
    ("period_start", "date", "first day on cover in this row"),
    ("period_end", "date", "last day on cover in this row (inclusive)"),
    ("exposure", "numeric", "years on cover: (period_end - period_start + 1) / 365"),
]
_POLICY_ADMIN = [
    ("voluntary_excess", "integer", "excess chosen by the customer: 0, 250, 500 or 1000"),
    ("compulsory_excess", "integer", "excess set by the insurer"),
    ("channel", "category", "sales channel: direct, comparison_site, broker"),
    ("payment_method", "category", "annual or monthly"),
    ("missed_payments_12m", "integer", "premium payments missed in the last 12 months"),
    ("new_business", "boolean", "first year with the insurer"),
    ("renewal_count", "integer", "number of renewals so far"),
    ("days_quote_to_inception", "integer", "days between quote and start of cover"),
    ("discount_applied", "numeric", "discount on the premium (share)"),
    ("written_premium", "numeric", "annual premium charged by the current tariff"),
    ("customer_service_contacts", "integer", "contacts with customer service during the period"),
]
_RENEWAL = [("renewal_decision", "outcome", "what happened at the end of the period (training files only)")]

DICTIONARY = {
    "customers": [
        ("customer_id", "id", "customer identifier (8 digits)"),
        ("surname", "text", "surname (personal data)"),
        ("gender", "category", "f or m (a permitted rating factor in this market)"),
        ("occupation", "category", "occupation (free text; may start with 'self-employed')"),
        ("marital_status", "category", "single, married, divorced, widowed"),
        ("household_size", "integer", "people in the household"),
        ("prior_claims_5y", "integer", "claims declared for the last 5 years (any insurer)"),
        ("licence_points", "integer", "penalty points on the driving licence"),
        ("years_at_address", "integer", "years lived at the current address"),
        ("customer_since_year", "integer", "first year with the insurer"),
        ("email_domain", "category", "e-mail provider"),
        ("marketing_consent", "boolean", "agreed to receive marketing"),
        ("document_preference", "category", "email, post or app"),
        ("previous_insurer", "category", "insurer before joining"),
        ("campaign_code", "category", "marketing campaign that acquired the customer (missing: none)"),
        ("branch_id", "category", "servicing branch"),
        ("marketing_segment", "category", "segment assigned by marketing"),
    ],
    "addresses": [
        ("address_id", "id", "address identifier"),
        ("region", "category", "region E01-E06"),
        ("urban_density", "category", "rural, suburban, urban, dense_urban"),
        ("flood_zone", "ordinal", "flood zone 1 (lowest risk) to 5"),
        ("distance_to_river_km", "numeric", "distance to the nearest river"),
        ("elevation_m", "numeric", "elevation above sea level"),
        ("soil_type", "category", "clay, sand, loam, chalk, rock"),
        ("year_built", "integer", "year the property was built"),
        ("property_type", "category", "detached, semi_detached, terraced, flat"),
        ("roof_type", "category", "pitched or flat"),
        ("basement", "boolean", "has a basement"),
        ("trees_within_5m", "boolean", "trees within 5 metres of the building"),
        ("listed_building", "boolean", "protected (listed) building"),
        ("bedrooms", "integer", "number of bedrooms"),
        ("energy_rating", "ordinal", "energy rating A (best) to G"),
        ("solar_panels", "boolean", "has solar panels"),
    ],
    "weather": [
        ("address_id", "id", "address identifier"),
        ("year", "integer", "calendar year (2023-2025)"),
        ("hail_days", "numeric", "days with hail"),
        ("wind_exposure_index", "numeric", "wind exposure index 0-10"),
        ("freezing_days", "numeric", "days below freezing"),
        ("dry_summer_index", "numeric", "summer dryness index 0-1"),
        ("wildfire_index", "numeric", "wildfire index 0-1"),
        ("rainfall_mm", "numeric", "annual rainfall"),
        ("mean_temperature_c", "numeric", "annual mean temperature"),
    ],
    "claims": [
        ("claim_id", "id", "claim identifier"),
        ("policy_id", "id", "policy that made the claim"),
        ("line", "category", "motor, buildings or contents"),
        ("peril", "category", "cause of loss"),
        ("loss_date", "date", "date of the loss"),
        ("report_date", "date", "date the claim was reported"),
        ("status", "category", "closed, open or repudiated (declined)"),
        ("paid_to_date", "numeric", "paid so far, net of excess"),
        ("incurred_to_date", "numeric", "paid + case reserve, net of excess"),
        ("recoveries", "numeric", "amounts recovered (salvage, third parties)"),
        ("repairer_id", "id", "repairer (motor)"),
        ("third_party_id", "id", "third party (liability perils)"),
    ],
    "policies_motor": _POLICY_COMMON + [
        ("driver_age_band", "ordinal", "age band of the main driver: 17-24, 25-34, 35-49, 50-64, 65+"),
        ("years_licensed", "integer", "years since the main driver passed the test"),
        ("ncb_years", "integer", "no-claims bonus years (0-15)"),
        ("vehicle_value_group", "ordinal", "vehicle value group 1-20"),
        ("vehicle_age", "integer", "vehicle age in years"),
        ("make_model", "category", "vehicle make and model code (300 codes)"),
        ("engine_size_cc", "numeric", "engine size"),
        ("body_type", "category", "hatchback, saloon, estate, suv, coupe"),
        ("transmission", "category", "manual or automatic"),
        ("seats", "integer", "number of seats"),
        ("vehicle_colour", "category", "colour"),
        ("fuel_type", "category", "petrol, diesel, hybrid, electric"),
        ("driver_assist", "boolean", "has driver-assistance systems"),
        ("cover_type", "category", "comprehensive, third_party_fire_theft, third_party_only"),
        ("overnight_parking", "category", "garage, driveway, street"),
        ("vehicle_use", "category", "social, commuting, business"),
        ("named_young_driver", "boolean", "a young driver is named on the policy"),
    ] + _POLICY_ADMIN + _RENEWAL,
    "policies_buildings": _POLICY_COMMON + [
        ("policyholder_age_band", "ordinal", "age band of the policyholder: 17-24, 25-34, 35-49, 50-64, 65+"),
        ("sum_insured", "numeric", "rebuilding sum insured"),
        ("construction_type", "category", "brick, stone, timber, non_standard"),
        ("occupancy", "category", "owner_occupied or let"),
        ("days_unoccupied", "numeric", "days the property is unoccupied in a year"),
        ("smoke_alarms", "boolean", "smoke alarms fitted"),
        ("accidental_damage_cover", "boolean", "accidental damage cover bought"),
        ("mortgage", "boolean", "property is mortgaged"),
    ] + _POLICY_ADMIN + _RENEWAL,
    "policies_contents": _POLICY_COMMON + [
        ("policyholder_age_band", "ordinal", "age band of the policyholder: 17-24, 25-34, 35-49, 50-64, 65+"),
        ("sum_insured", "numeric", "contents sum insured"),
        ("tenure", "category", "owner or renter"),
        ("security_level", "ordinal", "standard, locks, alarm_and_locks"),
        ("high_value_items", "integer", "declared high-value items (0-5)"),
        ("accidental_damage_cover", "boolean", "accidental damage cover bought"),
        ("works_from_home", "boolean", "policyholder works from home"),
        ("lodgers", "boolean", "lodgers in the home"),
        ("bicycle_cover", "boolean", "bicycle cover bought"),
    ] + _POLICY_ADMIN + _RENEWAL,
    "sample_submission": [
        ("row_id", "id", "row to predict (the 2025 test rows, in submission order)"),
        ("prediction", "numeric", "expected ultimate claim cost of the row"),
    ],
}
for _line in LINES:   # the 2025 test files have the training layout without renewal_decision
    DICTIONARY[f"test_{_line}"] = [c for c in DICTIONARY[f"policies_{_line}"] if c[0] != "renewal_decision"]
DICTIONARY["expenses"] = [
    ("item", "category", "expense item"),
    ("developed", "numeric", "value in the developed market"),
    ("emerging", "numeric", "value in this (emerging) market"),
    ("unit", "category", "CU per policy, or a share of premium"),
]
DICTIONARY["reference_policies"] = list({c[0]: c for c in (
    DICTIONARY["test_motor"] + [("line", "category", "line of the reference policy")]
    + DICTIONARY["test_buildings"] + DICTIONARY["test_contents"])}.values())

# --- category codes ------------------------------------------------------------------------------
UPPER_CASE_CODES = ("energy_rating", "region", "make_model", "previous_insurer", "campaign_code")
CODE_MAPS = {   # after trimming and lower-casing
    "construction_type": {"w": "timber", "t": "timber", "wood": "timber", "brk": "brick",
                          "stn": "stone", "wod": "timber", "mix": "non_standard",
                          "non-standard": "non_standard", "nonstandard": "non_standard"},
}
DOMAINS = {
    "marital_status": ("single", "married", "divorced", "widowed"),
    "document_preference": ("email", "post", "app"),
    "region": ("E01", "E02", "E03", "E04", "E05", "E06"),
    "gender": ("f", "m"),
    "driver_age_band": ("17-24", "25-34", "35-49", "50-64", "65+"),
    "policyholder_age_band": ("17-24", "25-34", "35-49", "50-64", "65+"),
    "urban_density": ("rural", "suburban", "urban", "dense_urban"),
    "soil_type": ("clay", "sand", "loam", "chalk", "rock"),
    "property_type": ("detached", "semi_detached", "terraced", "flat"),
    "roof_type": ("pitched", "flat"),
    "energy_rating": ("A", "B", "C", "D", "E", "F", "G"),
    "channel": ("direct", "comparison_site", "broker"),
    "payment_method": ("annual", "monthly"),
    "body_type": ("hatchback", "saloon", "estate", "suv", "coupe"),
    "transmission": ("manual", "automatic"),
    "fuel_type": ("petrol", "diesel", "hybrid", "electric"),
    "cover_type": ("comprehensive", "third_party_fire_theft", "third_party_only"),
    "overnight_parking": ("garage", "driveway", "street"),
    "vehicle_use": ("social", "commuting", "business"),
    "construction_type": ("brick", "stone", "timber", "non_standard"),
    "occupancy": ("owner_occupied", "let"),
    "tenure": ("owner", "renter"),
    "security_level": ("standard", "locks", "alarm_and_locks"),
    "status": ("closed", "open", "repudiated"),
    "line": LINES,
}
ORDINAL_CODES = {
    "energy_rating": {k: i for i, k in enumerate("ABCDEFG", start=1)},
    "security_level": {"standard": 0, "locks": 1, "alarm_and_locks": 2},
}

# Age bands are ordered: each becomes its mid-point age (a number keeps the order for any model).
AGE_BANDS = {c: {"17-24": 21.0, "25-34": 30.0, "35-49": 42.0, "50-64": 57.0, "65+": 72.0}
             for c in ("driver_age_band", "policyholder_age_band")}

# --- cleaning thresholds (each one justified in reports/02_cleaning) ----------------------------
SI_UNIT_ERROR_RATIO = 100      # sum insured > 100 x the table median: recorded in units, not thousands
SI_UNIT_SCALE = 1000           # ... so it is divided by 1,000 (genuine values stay below 10 x median)
MIN_LICENCE_AGE = 17           # nobody can hold a licence before 17 (1st percentile of the data: 17)
SUSPECT_CLAIM_RATIO = 5        # an open claim > 5 x the next largest of its peril: reserve entry error
EXPOSURE_DAYS_IN_YEAR = 365    # the case study's exposure basis (a full 2024 row is 1 Jan - 30 Dec)

# --- columns that never become model features (reason in reports/03_features) -------------------
LEAKAGE = {
    "renewal_decision": "outcome after the period; absent from the 2025 files",
    "customer_service_contacts": "driven by the claims of the same period (0 contacts on 0.2% of "
                                 "rows with a claim vs 67% without); the 2025 file looks like no-claim rows",
}
PERSONAL_DATA = {
    "surname": "personal identifier, no risk meaning",
}
GOVERNANCE = {
    "written_premium": "output of the current tariff: a model using it would copy the old price",
    "discount_applied": "commercial decision, not risk (and fully set by channel x new business)",
    "marketing_segment": "assigned by marketing (age x affluence proxy)",
    "campaign_code": "acquisition attribute (customer-cohort proxy), not a risk characteristic",
    "email_domain": "non-risk marketing attribute (regulatory precedent of unfair proxy pricing)",
    "previous_insurer": "commercial attribute",
    "branch_id": "servicing attribute",
    "document_preference": "service attribute",
    "marketing_consent": "service attribute",
}

WEATHER_VARS = ("hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index",
                "wildfire_index", "rainfall_mm", "mean_temperature_c")
HISTORY_YEARS = 3              # look-back of the claims-history features (data exist from 2023)
LARGE_LOSS_QUANTILE = 0.995    # used only to cap claims inside the history features
SEED = 20260515

# --- modelling (04_developed_prior.py, 05_lightgbm_model.py) ------------------------------------
# Settings of the transfer model. They come from the emerging-market settings of the pricing
# repository (iaa-pricing-part2, configs/emerging.yaml) and its experiment log (MODEL_EXPERIMENTS.md).
DEVELOPED = ROOT.parent / "1st round"   # the developed market: its settings, clean data and features
HOLDOUT_YEAR = 2024            # out-of-time check: fit on 2023, score 2024
# Trend rates carried over from the developed market: two years cannot measure a trend, and the
# 2023 -> 2024 changes here are within about two standard errors of these rates.
TREND = {
    "motor": {"frequency": 0.0,
              "severity": {"own_damage": 0.035, "third_party_prop": 0.076, "bodily_injury": 0.0,
                           "theft": 0.034, "glass_hail": 0.079}},
    "buildings": {"frequency": 0.0, "severity": 0.060},
    "contents": {"frequency": 0.014, "severity": 0.034},
}
LEVEL_SHIFT = None             # the 1st round's bodily injury shift is about 2021-22, before this market
SI_INDEX = {"buildings": 0.03, "contents": 0.02}   # yearly growth of sums insured (renewals)
CAP_QUANTILE = 0.995           # large-loss cap on training row costs (the excess is loaded back)
# The 2025 weather lies far outside the two training years (rainfall anomaly -128 mm against
# +-3 mm, hail -4.1 days against +-1.6), so every 2025 row would land in the edge leaves of the
# trees. Only the address climate (wx_<x>_mean) is kept. In the pricing repository this raised
# the out-of-time D^2 of buildings from 0.056 to 0.065.
DROP_YEAR_WEATHER = True
ES_SHARE = 0.2                 # share of policies used only to choose the number of trees
REFIT_FACTOR = 1.1             # the refit on all rows gets 10% more trees

# The developed prior (04_developed_prior.py): a LightGBM fitted on every 1st-round row with the
# factors that mean the same thing in both markets. A fixed 200 trees on all rows beat a sampled,
# early-stopped prior for buildings and contents (MODEL_EXPERIMENTS.md, section 14).
PRIOR_TREES = 200
PRIOR_NOT_TRANSFERRED = {      # factors the prior must not carry
    "region": "the regions of the two markets are different places",
}
# Factors the prior gets even where the 1st round's step 3 dropped them. New business costs x1.10
# per year on cover in developed buildings (x1.45 motor), but step 3 dropped renewal_count for
# buildings because the claims-history features already told new customers apart (they have no
# history yet). The prior leaves those history features out, so it needs the flag itself.
PRIOR_ADD = {
    "renewal_count": "new business: its developed-market effect was hidden by the history features",
}
PRIOR_NOT_TRANSFERRED_PREFIXES = {
    "wx_": "a different climate: the 2nd-round model learns the weather itself",
    "hist_": "claims history with a different insurer",
    "holds_": "the other lines a customer holds depend on each insurer's products",
}

# The transfer model (05_lightgbm_model.py): smaller trees than in the 1st round for 20-50
# thousand rows a line; motor and contents more regularised (grouped cross-validation in the
# pricing repository; buildings kept the shared settings).
LGBM_PARAMS = {
    "learning_rate": 0.03, "num_leaves": 7, "min_data_in_leaf": 400, "min_sum_hessian_in_leaf": 1.0,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0,
    "max_bin": 255, "cat_smooth": 50, "cat_l2": 10, "min_data_per_group": 200, "max_cat_to_onehot": 32,
}
_SMALL = {"num_leaves": 5, "min_data_in_leaf": 700, "lambda_l2": 30.0, "feature_fraction": 0.9,
          "bagging_fraction": 0.9, "learning_rate": 0.025}
LGBM_PER_LINE = {"motor": _SMALL, "contents": _SMALL}

# Business adjustments: the new-business and gender effects the model still misses. They are
# measured on out-of-fold predictions of the training years, year by year, pooled over the two
# years and shrunk towards "no effect". The prior standard deviations are judgements, as in the
# pricing repository. New business is 76% of the 2025 book against 12-15% of 2023-24.
OOF_FOLDS = 5                  # policy-grouped folds for the out-of-fold predictions
N_BOOT = 200                   # customer-clustered bootstrap replicates for the standard errors
NB_SHRINK = {"freq": 0.15, "size": 0.10}   # prior sd of the log effect: frequency, claim size
NB_WITH_SIZE = ("motor",)      # motor new-business claims were larger in both years; home sizes are noise
GENDER_SHRINK = 0.30           # gender effects of +-30% are plausible in insurance
GENDER_LINES = ("motor", "contents")   # buildings was not credible in the pricing repository

# --- the 1st round's winning ensemble, transferred (06_geo_priors.py, 07_geo_experts.py,
#     08_geo_ensemble.py) -------------------------------------------------------------------------
# Every expert of the 1st round's best ensemble (geo_experts.py) starts here from a developed-
# market prior of its own component: claim frequency, claim size, and cost, frequency and size of
# each peril, all fitted on the 1st round with the factors both markets share (as the step-4
# prior). Two training years give one forward back-test only, too little to set a 2025 level, so
# the ensemble's recalibration learns from cross-validation (folds of customers over 2023-2024)
# and the forward back-test 2023 -> 2024 is kept as a check.
GEO_SETS = {"ho2024": 2024, "cv": TEST_YEAR, "final": TEST_YEAR}   # fit set: the cost level it prices
GEO_EXPERT_STEPS = ("total", "glm", "peril", "fs", "fs_peril")
GEO_MEMBERS = ("total", "fs_gamma", "fs_lognorm", "peril", "fs_peril", "glm")
CV_FOLDS = 5                   # folds of customers: every training row is predicted once, out of fold
DEV_MATURE_YEARS = (2023,)     # open claims to ultimate (recoveries, later repudiation)
DEV_IMMATURE_YEARS = (2024,)
REPUDIATION_MIN_Z = 2.0
MIN_FREQ_CLAIMS = 50           # fewer claims of a peril: its developed prior alone, re-levelled
MIN_SIZE_CLAIMS = 300          # fewer claims of a peril: its developed claim-size prior, re-levelled
GEO_SEVERITY = {"num_leaves": 7, "min_data_in_leaf": 100, "learning_rate": 0.03}   # few claims here
GEO_GLM = {"n_bins": 10, "min_exposure": 50.0}                                    # alpha: tuned below
GEO_POOL_ALPHA = 1e-5
# Tuning by cross-validation (pooled out-of-fold D^2 of the line). LightGBM settings of the
# frequency and cost experts, tried on the ``total`` expert; the GLM's ridge (with the prior in the
# offset, a strong ridge means "stay close to the developed market").
_BASE = {"min_sum_hessian_in_leaf": 1.0, "bagging_freq": 1, "max_bin": 255, "cat_smooth": 50, "cat_l2": 10,
         "min_data_per_group": 200, "max_cat_to_onehot": 32}
TUNE_GRID = (
    {**_BASE, "num_leaves": 7, "min_data_in_leaf": 400, "lambda_l2": 10.0, "learning_rate": 0.03,
     "feature_fraction": 0.8, "bagging_fraction": 0.8},                   # step 5 shared settings
    {**_BASE, "num_leaves": 5, "min_data_in_leaf": 700, "lambda_l2": 30.0, "learning_rate": 0.025,
     "feature_fraction": 0.9, "bagging_fraction": 0.9},                   # step 5 motor / contents
    {**_BASE, "num_leaves": 3, "min_data_in_leaf": 300, "lambda_l2": 10.0, "learning_rate": 0.03,
     "feature_fraction": 0.8, "bagging_fraction": 0.8},                   # very shallow
    {**_BASE, "num_leaves": 15, "min_data_in_leaf": 200, "lambda_l2": 10.0, "learning_rate": 0.03,
     "feature_fraction": 0.8, "bagging_fraction": 0.8},                   # deeper
    {**_BASE, "num_leaves": 7, "min_data_in_leaf": 1000, "lambda_l2": 100.0, "learning_rate": 0.03,
     "feature_fraction": 0.8, "bagging_fraction": 0.8},                   # heavily regularised
    {**_BASE, "num_leaves": 3, "min_data_in_leaf": 700, "lambda_l2": 30.0, "learning_rate": 0.03,
     "feature_fraction": 0.8, "bagging_fraction": 0.8},                   # shallow and regularised
    {**_BASE, "num_leaves": 2, "min_data_in_leaf": 300, "lambda_l2": 10.0, "learning_rate": 0.03,
     "feature_fraction": 0.8, "bagging_fraction": 0.8},                   # stumps: one split per tree
)
# Contents chose the strongest ridge of a first, narrower grid (1e-4 to 1), so the grid goes higher.
GLM_ALPHA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0)
GEO_MONOTONE = {
    "motor": {"prior_claims_5y": +1, "voluntary_excess": -1, "ncb_years": -1, "hist_pol_claims_1y": +1,
              "hist_pol_cost_3y": +1, "hist_cust_claims_3y": +1},
    "buildings": {"prior_claims_5y": +1, "voluntary_excess": -1, "sum_insured": +1, "flood_zone": +1,
                  "hist_addr_claims_3y": +1},
    "contents": {"prior_claims_5y": +1, "voluntary_excess": -1, "sum_insured": +1, "high_value_items": +1},
}
# The recalibration's slope b1, tested on the cross-validation predictions (08_geo_ensemble.py,
# reports/08_geo/slope_check.csv): fitted on one year's folds, with each line's level known, it
# changes the other year's D^2 by -0.003 to +0.004 with no consistent sign (motor about 0). Two
# years cannot support a slope, so it stays at 1 and only the level b0 is fitted.
GEO_FIT_SLOPE = False
