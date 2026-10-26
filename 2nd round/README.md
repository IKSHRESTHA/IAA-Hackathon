# 2nd round: emerging market

Reading, cleaning and feature engineering for *IAA AI-Assisted Hackathon 2026 - Part 2*: two training years (2023-2024), a 2025 book that is 76% new business (209,063 rows to price), and 20 reference policies to quote from the `expenses` table. Run the steps from this folder: `python 01_data_reading.py`, `python 02_data_cleaning.py`, `python 03_feature_engineering.py` (or `python run_all.py` one level up). The notebooks show the same steps with their evidence.

## Data

| Table | Raw rows | Clean rows | Clean columns |
|---|---:|---:|---:|
| `addresses` | 136,914 | 136,914 | 16 |
| `claims` | 9,258 | 9,258 | 19 |
| `customers` | 135,856 | 135,856 | 18 |
| `expenses` | 8 | 8 | 4 |
| `policies_buildings` | 27,447 | 27,447 | 29 |
| `policies_contents` | 34,566 | 34,566 | 30 |
| `policies_motor` | 47,283 | 47,283 | 38 |
| `reference_policies` | 20 | 20 | 53 |
| `sample_submission` | 209,063 | 209,063 | 2 |
| `test_buildings` | 52,295 | 52,295 | 28 |
| `test_contents` | 66,247 | 66,247 | 29 |
| `test_motor` | 90,521 | 90,521 | 37 |
| `weather` | 410,742 | 410,742 | 9 |

Model-ready tables (`data/features/`):

| Table | Rows | Columns | Features |
|---|---:|---:|---:|
| `policies_motor` | 47,283 | 56 | 28 + exposure |
| `test_motor` | 90,521 | 36 | 28 + exposure |
| `policies_buildings` | 27,447 | 45 | 14 + exposure |
| `test_buildings` | 52,295 | 22 | 14 + exposure |
| `policies_contents` | 34,566 | 34 | 9 + exposure |
| `test_contents` | 66,247 | 17 | 9 + exposure |

## Cleaning (from `reports/02_cleaning/cleaning_log.csv`)

| Rule | Table | Column | Rows | Action | Evidence |
|---|---|---|---:|---|---|
| C01 | `customers` | `gender` | 135,856 | recoded | 'F'->'f' (67,973); 'M'->'m' (67,883) |
| C01 | `customers` | `occupation` | 89,397 | recoded | 'Retired'->'retired' (9,218); 'Student'->'student' (6,445); 'Retired '->'retired' (2,736); 'Care worker'->'care worker' (2,694); 'Nurse'->'nurse' (2,648); 'Teac |
| C02 | `customers` | `occupation` | 11,975 | new column self_employed | 20 base occupations remain |
| C10 | `weather` | `*` | 0 | none found |  |
| C04 | `policies_motor` | `exposure` | 249 | recomputed | recorded in days: 47; on a 360-day basis: 202; other: 0 |
| C04 | `policies_buildings` | `exposure` | 152 | recomputed | recorded in days: 33; on a 360-day basis: 119; other: 0 |
| C05 | `policies_buildings` | `sum_insured` | 53 | rescaled | after the fix: 73,000 to 470,000 (median 229,000) |
| C01 | `policies_buildings` | `construction_type` | 27,447 | recoded | 'BRK'->'brick' (19,194); 'WOD'->'timber' (3,302); 'MIX'->'non_standard' (2,781); 'STN'->'stone' (2,170) |
| C04 | `policies_contents` | `exposure` | 194 | recomputed | recorded in days: 34; on a 360-day basis: 160; other: 0 |
| C05 | `policies_contents` | `sum_insured` | 57 | rescaled | after the fix: 7,600 to 131,600 (median 39,100) |
| C05 | `test_buildings` | `sum_insured` | 104 | rescaled | after the fix: 60,000 to 568,000 (median 236,000) |
| C01 | `test_buildings` | `construction_type` | 52,295 | recoded | 'BRK'->'brick' (36,709); 'WOD'->'timber' (6,255); 'MIX'->'non_standard' (5,214); 'STN'->'stone' (4,117) |
| C05 | `test_contents` | `sum_insured` | 135 | rescaled | after the fix: 9,200 to 98,100 (median 33,000) |
| C08 | `policies_motor` | `ncb_years` | 7,756 | kept as declared, flagged | too common to be typing errors (mostly young drivers); the declaration itself carries risk information |
| C08 | `test_motor` | `ncb_years` | 20,051 | kept as declared, flagged | too common to be typing errors (mostly young drivers); the declaration itself carries risk information |
| K03 | `claims` | `net_incurred` | 9,258 | new column | repudiated: 120 |
| K04 | `claims` | `row_id` | 36 | unmatched kept, flagged (excluded from targets) | status counts: {'matched': 9222, 'before_first_row': 36} |
| K05 | `claims` | `peril` | 0 | kept, flagged | {} |
| K06 | `claims` | `peril` | 74 | kept, flagged | 16% of home AD claims; mean size 1.03x the covered ones: plausibly basic AD cover |
| C11 | `reference_policies` | `*` | 20 | replaced by the cleaned row | values identical to the raw 2025 rows before cleaning |
| C12 | `expenses` | `emerging` | 1 | kept missing (no assumption made here) | items: ['commission_comparison_site'] |

No policy, customer, address or weather row is removed; the only rows removed are exact duplicate claims. Missing values are not imputed (step 3 adds a `_missing` flag where missingness predicts claims).

## Feature selection (from `reports/03_features/feature_register.csv`)

Every column of the joined table is either kept or dropped with a reason. The out-of-time tests (alone and inside a throwaway LightGBM, against 20 random probe columns) are pooled over two folds that cross-fit by customer (learn on half of the customers, score the other half); year levels are part of the baseline.

| Reason | motor | buildings | contents |
|---|---:|---:|---:|
| adds value | 28 | 14 | 9 |
| base | 1 | 1 | 1 |
| constant | 1 | 4 | 3 |
| data-quality flag | 1 | 2 | 2 |
| governance | 9 | 9 | 9 |
| leakage | 2 | 2 | 2 |
| near-duplicate | 2 | 1 | 2 |
| no value | 43 | 52 | 57 |
| not an identifier | 1 | 0 | 0 |
| not testable | 1 | 1 | 1 |
| personal data | 1 | 1 | 1 |
| redundant | 3 | 3 | 3 |

Pass mark per line (lowest z at which at most 1 of 20 probes would pass): motor z >= 1.5, buildings z >= 2, contents z >= 2.

### motor: 28 features + exposure

| Feature | Meaning | Why kept |
|---|---|---|
| `exposure` | years on cover in the row (days / 365) | base |
| `years_licensed` | years since the main driver passed the test | frequency_oneway, frequency_model |
| `ncb_years` | no-claims bonus years (0-15) | frequency_oneway, frequency_model |
| `vehicle_value_group` | vehicle value group 1-20 | severity_oneway |
| `vehicle_age` | vehicle age in years | severity_model |
| `body_type` | hatchback, saloon, estate, suv, coupe | frequency_oneway |
| `cover_type` | comprehensive, third_party_fire_theft, third_party_only | frequency_oneway, frequency_model |
| `overnight_parking` | garage, driveway, street | theft_oneway |
| `vehicle_use` | social, commuting, business | frequency_oneway |
| `voluntary_excess` | excess chosen by the customer: 0, 250, 500 or 1000 | frequency_oneway, frequency_model |
| `payment_method` | annual or monthly | frequency_oneway |
| `missed_payments_12m` | premium payments missed in the last 12 months | frequency_oneway |
| `renewal_count` | number of renewals so far | frequency_oneway |
| `dq_ncb_exceeds_licence` | no-claims bonus longer than the licence (as declared) | frequency_oneway |
| `gender` | f or m (a permitted rating factor in this market) | frequency_oneway, frequency_model |
| `occupation` | occupation (free text; may start with 'self-employed') | frequency_oneway |
| `marital_status` | single, married, divorced, widowed | frequency_oneway |
| `prior_claims_5y` | claims declared for the last 5 years (any insurer) | frequency_oneway |
| `years_at_address` | years lived at the current address | frequency_oneway |
| `urban_density` | rural, suburban, urban, dense_urban | frequency_oneway, theft_oneway |
| `start_month` | month the row starts | frequency_oneway |
| `starts_after_jan1` | row starts after 1 January (new business or a mid-term change) | frequency_oneway |
| `ends_before_year_end` | row ends before the end of the policy year | severity_oneway |
| `wx_hail_days` | hail_days at the address in the row's year | frequency_oneway, glass_hail_oneway |
| `wx_hail_days_mean` | address's climate: mean hail_days over the training years | glass_hail_oneway |
| `hist_pol_claims_1y` | claims of this policy in the previous calendar year (known at 1 January) | frequency_oneway |
| `hist_pol_cost_3y` | cost of this policy's claims in the previous 3 years (each capped at the line's 99.5% claim quantile) | frequency_oneway |
| `hist_pol_years_since_claim` | years since this policy's last claim (missing: none in 3 years) | frequency_oneway |
| `hist_cust_claims_3y` | claims of the customer on every line in the previous 3 years | frequency_oneway |

Dropped (64): constant: `compulsory_excess`; data-quality flag: `dq_exposure_fixed`; governance: `discount_applied`, `written_premium`, `email_domain`, `marketing_consent`, `document_preference`, `previous_insurer`, `campaign_code`, `branch_id`, `marketing_segment`; leakage: `customer_service_contacts`, `renewal_decision`; near-duplicate: `driver_age_band`, `hist_pol_claims_3y`; no value: `engine_size_cc`, `transmission`, `seats`, `vehicle_colour`, `fuel_type`, `driver_assist`, `named_young_driver`, `channel`, `days_quote_to_inception`, `household_size`, `licence_points`, `self_employed`, `region`, `flood_zone`, `distance_to_river_km`, `elevation_m`, `soil_type`, `property_type`, `roof_type`, `basement`, `trees_within_5m`, `listed_building`, `bedrooms`, `energy_rating`, `solar_panels`, `winter_fraction`, `property_age`, `wx_wind_exposure_index`, `wx_freezing_days`, `wx_dry_summer_index`, `wx_wildfire_index`, `wx_rainfall_mm`, `wx_mean_temperature_c`, `wx_wind_exposure_index_mean`, `wx_freezing_days_mean`, `wx_dry_summer_index_mean`, `wx_wildfire_index_mean`, `wx_rainfall_mm_mean`, `wx_mean_temperature_c_mean`, `hist_pol_exposure_3y`, `hist_cust_claims_other_lines_3y`, `holds_buildings`, `holds_contents`; not an identifier: `make_model`; not testable: `hist_years_available`; personal data: `surname`; redundant: `new_business`, `customer_since_year`, `year_built`

### buildings: 14 features + exposure

| Feature | Meaning | Why kept |
|---|---|---|
| `exposure` | years on cover in the row (days / 365) | base |
| `sum_insured` | rebuilding sum insured | frequency_oneway |
| `construction_type` | brick, stone, timber, non_standard | frequency_oneway |
| `accidental_damage_cover` | accidental damage cover bought | accidental_damage_oneway |
| `voluntary_excess` | excess chosen by the customer: 0, 250, 500 or 1000 | frequency_oneway |
| `payment_method` | annual or monthly | frequency_oneway |
| `missed_payments_12m` | premium payments missed in the last 12 months | frequency_oneway |
| `flood_zone` | flood zone 1 (lowest risk) to 5 | frequency_oneway, flood_oneway |
| `distance_to_river_km` | distance to the nearest river | flood_oneway |
| `elevation_m` | elevation above sea level | flood_oneway |
| `energy_rating` | energy rating as a number, A = 1 ... G = 7 | frequency_oneway |
| `property_age` | year - year_built | frequency_oneway, frequency_model |
| `wx_wind_exposure_index_mean` | address's climate: mean wind_exposure_index over the training years | storm_oneway |
| `wx_rainfall_mm_mean` | address's climate: mean rainfall_mm over the training years | flood_oneway |
| `hist_addr_claims_3y` | home claims at the address (any customer) in the previous 3 years | frequency_oneway |

Dropped (75): constant: `compulsory_excess`, `hist_pol_flood_3y`, `hist_pol_subsidence_3y`, `hist_pol_escape_of_water_3y`; data-quality flag: `dq_exposure_fixed`, `dq_sum_insured_fixed`; governance: `discount_applied`, `written_premium`, `email_domain`, `marketing_consent`, `document_preference`, `previous_insurer`, `campaign_code`, `branch_id`, `marketing_segment`; leakage: `customer_service_contacts`, `renewal_decision`; near-duplicate: `wx_rainfall_mm`; no value: `policyholder_age_band`, `occupancy`, `days_unoccupied`, `smoke_alarms`, `mortgage`, `channel`, `renewal_count`, `days_quote_to_inception`, `gender`, `occupation`, `marital_status`, `household_size`, `prior_claims_5y`, `licence_points`, `years_at_address`, `self_employed`, `region`, `urban_density`, `soil_type`, `property_type`, `roof_type`, `basement`, `trees_within_5m`, `listed_building`, `bedrooms`, `solar_panels`, `start_month`, `winter_fraction`, `starts_after_jan1`, `ends_before_year_end`, `si_per_bedroom`, `wx_hail_days`, `wx_wind_exposure_index`, `wx_freezing_days`, `wx_dry_summer_index`, `wx_wildfire_index`, `wx_mean_temperature_c`, `wx_hail_days_mean`, `wx_freezing_days_mean`, `wx_dry_summer_index_mean`, `wx_wildfire_index_mean`, `wx_mean_temperature_c_mean`, `hist_pol_claims_1y`, `hist_pol_claims_3y`, `hist_pol_cost_3y`, `hist_pol_years_since_claim`, `hist_pol_exposure_3y`, `hist_cust_claims_3y`, `hist_cust_claims_other_lines_3y`, `hist_addr_weather_claims_3y`, `holds_motor`, `holds_contents`; not testable: `hist_years_available`; personal data: `surname`; redundant: `new_business`, `customer_since_year`, `year_built`

### contents: 9 features + exposure

| Feature | Meaning | Why kept |
|---|---|---|
| `exposure` | years on cover in the row (days / 365) | base |
| `sum_insured` | rebuilding sum insured | severity_oneway |
| `tenure` | owner or renter | severity_oneway |
| `accidental_damage_cover` | accidental damage cover bought | frequency_oneway, accidental_damage_oneway |
| `voluntary_excess` | excess chosen by the customer: 0, 250, 500 or 1000 | frequency_oneway |
| `flood_zone` | flood zone 1 (lowest risk) to 5 | frequency_oneway, flood_oneway |
| `distance_to_river_km` | distance to the nearest river | flood_oneway |
| `elevation_m` | elevation above sea level | flood_oneway |
| `winter_fraction` | share of the covered days in December-February | frequency_oneway |
| `wx_rainfall_mm` | rainfall_mm at the address in the row's year | flood_oneway |

Dropped (80): constant: `compulsory_excess`, `hist_pol_flood_3y`, `hist_pol_escape_of_water_3y`; data-quality flag: `dq_exposure_fixed`, `dq_sum_insured_fixed`; governance: `discount_applied`, `written_premium`, `email_domain`, `marketing_consent`, `document_preference`, `previous_insurer`, `campaign_code`, `branch_id`, `marketing_segment`; leakage: `customer_service_contacts`, `renewal_decision`; near-duplicate: `wx_rainfall_mm_mean`, `holds_buildings`; no value: `policyholder_age_band`, `security_level`, `high_value_items`, `works_from_home`, `lodgers`, `bicycle_cover`, `channel`, `payment_method`, `missed_payments_12m`, `renewal_count`, `days_quote_to_inception`, `gender`, `occupation`, `marital_status`, `household_size`, `prior_claims_5y`, `licence_points`, `years_at_address`, `self_employed`, `region`, `urban_density`, `soil_type`, `property_type`, `roof_type`, `basement`, `trees_within_5m`, `listed_building`, `bedrooms`, `energy_rating`, `solar_panels`, `start_month`, `starts_after_jan1`, `ends_before_year_end`, `property_age`, `si_per_bedroom`, `wx_hail_days`, `wx_wind_exposure_index`, `wx_freezing_days`, `wx_dry_summer_index`, `wx_wildfire_index`, `wx_mean_temperature_c`, `wx_hail_days_mean`, `wx_wind_exposure_index_mean`, `wx_freezing_days_mean`, `wx_dry_summer_index_mean`, `wx_wildfire_index_mean`, `wx_mean_temperature_c_mean`, `hist_pol_claims_1y`, `hist_pol_claims_3y`, `hist_pol_cost_3y`, `hist_pol_years_since_claim`, `hist_pol_exposure_3y`, `hist_cust_claims_3y`, `hist_cust_claims_other_lines_3y`, `hist_addr_claims_3y`, `hist_addr_weather_claims_3y`, `holds_motor`; not testable: `hist_years_available`; personal data: `surname`; redundant: `new_business`, `customer_since_year`, `year_built`

## Targets in the training tables

* `n_claims`, `claim_cost`: claims whose loss date is in the row's period, cost = incurred - recoveries (net of the excess), repudiated claims excluded, valued at 31 December 2024; `n_open`: of which open.
* `n_<peril>`, `cost_<peril>`: the same by peril.
* `comp_count`, `comp_cost`: share of the row's claims (count, cost) already reported at 31 December 2024, from reporting-delay patterns corrected for truncation; `comp_<peril>` by peril. Use `log(exposure x comp_count)` or `log(exposure x comp_cost)` as the offset.
