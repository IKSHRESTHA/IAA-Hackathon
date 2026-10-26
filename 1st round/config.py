"""Settings of the 1st round (developed market): files, data dictionary, codes and thresholds.

Everything that is a decision about this round's data lives here, so the three steps
(01_data_reading, 02_data_cleaning, 03_feature_engineering) contain no hidden constants.
Table and column names are exactly those of the case study files.
"""
from pathlib import Path

ROUND = "1st round"
MARKET = "developed market"
COMPETITION = "IAA AI-Assisted Hackathon 2026 - Part 1"

# --- folders -------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW, CLEAN, FEATURES = DATA / "raw", DATA / "clean", DATA / "features"
REPORTS = ROOT / "reports"

# --- dates ---------------------------------------------------------------------------------------
VALUATION_DATE = "2024-12-31"          # claims are valued at this date
TRAIN_YEARS = (2021, 2022, 2023, 2024)
TEST_YEAR = 2025
# Out-of-time tests of each feature's value (03): learn on the earlier years, score the next one,
# for every training year after the first (expanding window); the folds are pooled. 2024 claims
# are not all reported yet: its expected claims include the reporting completeness.
# Each fold: (fit years, test years, customer half used for fitting or None).
SIGNAL_FOLDS = (((2021,), (2022,), None), ((2021, 2022), (2023,), None), ((2021, 2022, 2023), (2024,), None))

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
        ("date_of_birth", "date", "date of birth"),
        ("occupation", "category", "occupation"),
        ("marital_status", "category", "single, married, divorced, widowed"),
        ("household_size", "integer", "people in the household"),
        ("credit_band", "ordinal", "credit score band, A (best) to E"),
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
        ("region", "category", "region R01-R12"),
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
        ("year", "integer", "calendar year (2021-2025)"),
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
        ("driver_age", "integer", "age of the main driver at the start of the row"),
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
        ("annual_mileage", "numeric", "declared annual mileage"),
        ("overnight_parking", "category", "garage, driveway, street"),
        ("vehicle_use", "category", "social, commuting, business"),
        ("named_young_driver", "boolean", "a young driver is named on the policy"),
        ("telematics_opt_in", "boolean", "telematics box fitted"),
        ("telematics_score", "numeric", "telematics driving score 1-100 (opted-in only)"),
    ] + _POLICY_ADMIN + _RENEWAL,
    "policies_buildings": _POLICY_COMMON + [
        ("policyholder_age", "integer", "age of the policyholder at the start of the row"),
        ("sum_insured", "numeric", "rebuilding sum insured"),
        ("construction_type", "category", "brick, stone, timber, non_standard"),
        ("occupancy", "category", "owner_occupied or let"),
        ("days_unoccupied", "numeric", "days the property is unoccupied in a year"),
        ("smoke_alarms", "boolean", "smoke alarms fitted"),
        ("accidental_damage_cover", "boolean", "accidental damage cover bought"),
        ("mortgage", "boolean", "property is mortgaged"),
    ] + _POLICY_ADMIN + _RENEWAL,
    "policies_contents": _POLICY_COMMON + [
        ("policyholder_age", "integer", "age of the policyholder at the start of the row"),
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

# --- category codes ------------------------------------------------------------------------------
UPPER_CASE_CODES = ("credit_band", "energy_rating", "region", "make_model", "previous_insurer",
                    "campaign_code")
CODE_MAPS = {   # after trimming and lower-casing
    "construction_type": {"w": "timber", "t": "timber", "wood": "timber", "brk": "brick",
                          "stn": "stone", "wod": "timber", "mix": "non_standard",
                          "non-standard": "non_standard", "nonstandard": "non_standard"},
}
DOMAINS = {
    "marital_status": ("single", "married", "divorced", "widowed"),
    "credit_band": ("A", "B", "C", "D", "E"),
    "document_preference": ("email", "post", "app"),
    "region": tuple(f"R{i:02d}" for i in range(1, 13)),
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
    "credit_band": {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5},
    "energy_rating": {k: i for i, k in enumerate("ABCDEFG", start=1)},
    "security_level": {"standard": 0, "locks": 1, "alarm_and_locks": 2},
}

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
    "date_of_birth": "personal data; the age at the start of each row is kept instead",
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
HISTORY_YEARS = 3              # look-back of the claims-history features (3 years of earlier data)
LARGE_LOSS_QUANTILE = 0.995    # used only to cap claims inside the history features
SEED = 20260515

# --- modelling (04_lightgbm_model.py) ------------------------------------------------------------
# Settings of the LightGBM model. They come from the developed-market settings of the pricing
# repository (iaa-pricing-part2, configs/developed.yaml), tuned there on a 2024 back-test.
HOLDOUT_YEAR = 2024            # out-of-time check: fit on the earlier years, score this one
# Annual trend rates from the reserving analysis of the developed market. A year's claims are
# projected to the year priced with ((1 + frequency) x (1 + severity)) ** (years ahead).
TREND = {
    "motor": {"frequency": 0.0,              # the 2023 step down is a change of mix, not a trend
              "severity": {"own_damage": 0.035, "third_party_prop": 0.076, "bodily_injury": 0.0,
                           "theft": 0.034, "glass_hail": 0.079}},
    "buildings": {"frequency": 0.0, "severity": 0.060},    # frequency is left to the weather
    "contents": {"frequency": 0.014, "severity": 0.034},
}
# Bodily injury claims of 2021-22 were settled at a higher level than from 2023 on: their cost is
# multiplied by 0.82 to put them on the 2023+ basis.
LEVEL_SHIFT = {"line": "motor", "peril": "bodily_injury", "years": (2021, 2022), "factor": 0.82}
SI_INDEX = {"buildings": 0.03, "contents": 0.02}   # yearly growth of sums insured (renewals)
CAP_QUANTILE = 0.995           # large-loss cap on training row costs (the excess is loaded back)
# Weather of the row's own year. Four training years give the trees a usable range, so it stays;
# the 2nd round drops it (two years cannot cover the 2025 values).
DROP_YEAR_WEATHER = False
ES_SHARE = 0.2                 # share of policies used only to choose the number of trees
REFIT_FACTOR = 1.1             # the refit on all rows gets 10% more trees (25% more data)
# Shallow trees and large leaves stop the model from fitting noise in rare, heavy-tailed claims.
# The training power equals the metric power (1.3 and 1.7 were worse in the back-test).
LGBM_PARAMS = {
    "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 2000, "min_sum_hessian_in_leaf": 1.0,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0,
    "max_bin": 255, "cat_smooth": 200, "cat_l2": 10, "min_data_per_group": 1000, "max_cat_to_onehot": 32,
}
LGBM_PER_LINE: dict = {}       # per-line overrides of LGBM_PARAMS (none needed so far)

# --- the 1st round's winning ensemble (05_geo_experts.py, 06_geo_ensemble.py) ------------------
# Six experts and their geometric mean, recalibrated per line: the method of the best private
# score of the 1st round (IAA_Part1_geo_models, private D^2 0.13830). Every expert is fitted on
# three fit sets: two back-tests, which teach the recalibration, and the final fit.
GEO_SETS = {"ho2023": 2023, "ho2024": 2024, "final": TEST_YEAR}     # fit set: the year it prices
GEO_EXPERT_STEPS = ("total", "glm", "peril", "fs", "fs_peril")      # "fs" = fs_gamma and fs_lognorm
GEO_MEMBERS = ("total", "fs_gamma", "fs_lognorm", "peril", "fs_peril", "glm")   # "v4" of the original
# Open claims to ultimate. Recovery ratio: recoveries / incurred on closed claims. Repudiation:
# applied only if the recent years show significantly fewer repudiations (z >= 2) than the
# mature years' rate predicts.
DEV_MATURE_YEARS = (2021, 2022)
DEV_IMMATURE_YEARS = (2023, 2024)
REPUDIATION_MIN_Z = 2.0
# make_model has 300 codes. Step 3 dropped it as a random label, but its 2021-22 claim frequency
# by code predicts 2023-24 (correlation 0.22 against 0.02 with shuffled labels), so the
# ensemble uses it as a credibility-weighted relativity (50% credibility at 100 claims).
CREDIBILITY_ENCODE = {"motor": ("make_model",)}
CREDIBILITY_K = 100
# Monotone constraints of the claim-frequency and cost experts (not of the claim-size ones):
# more past claims or penalty points never lower the price, a higher excess or more no-claims
# years never raise it, a larger sum insured never lowers it.
_HISTORY_UP = {c: +1 for c in ("hist_pol_claims_1y", "hist_pol_claims_3y", "hist_pol_cost_3y",
                               "hist_cust_claims_3y", "hist_addr_claims_3y", "hist_addr_weather_claims_3y",
                               "hist_pol_escape_of_water_3y")}
GEO_MONOTONE = {
    "motor": {"prior_claims_5y": +1, "credit_band": +1, "voluntary_excess": -1, "ncb_years": -1,
              "licence_points": +1, **_HISTORY_UP},
    "buildings": {"prior_claims_5y": +1, "credit_band": +1, "voluntary_excess": -1, "sum_insured": +1,
                  "flood_zone": +1, **_HISTORY_UP},
    "contents": {"prior_claims_5y": +1, "credit_band": +1, "voluntary_excess": -1, "sum_insured": +1,
                 "high_value_items": +1, **_HISTORY_UP},
}
GEO_SEVERITY = {"num_leaves": 15, "min_data_in_leaf": 300, "learning_rate": 0.03}   # claim-size models
GEO_GLM = {"alpha": 5e-4, "n_bins": 10, "min_exposure": 200.0}                     # the GLM expert
GEO_POOL_ALPHA = 1e-5          # ridge of the per-line recalibration (b0, b1)
