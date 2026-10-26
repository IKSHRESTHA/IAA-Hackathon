# 1st round: developed market

Reading, cleaning and feature engineering for *IAA AI-Assisted Hackathon 2026 - Part 1*: four training years (2021-2024) and 626,639 rows to price for 2025. Run the steps from this folder: `python 01_data_reading.py`, `python 02_data_cleaning.py`, `python 03_feature_engineering.py` (or `python run_all.py` one level up). The notebooks show the same steps with their evidence.

## Data

| Table | Raw rows | Clean rows | Clean columns |
|---|---:|---:|---:|
| `addresses` | 575,983 | 575,983 | 16 |
| `claims` | 163,150 | 162,546 | 19 |
| `customers` | 545,240 | 545,240 | 18 |
| `policies_buildings` | 707,726 | 707,726 | 29 |
| `policies_contents` | 721,617 | 721,617 | 30 |
| `policies_motor` | 979,968 | 979,968 | 43 |
| `sample_submission` | 626,639 | 626,639 | 2 |
| `test_buildings` | 184,155 | 184,155 | 28 |
| `test_contents` | 187,498 | 187,498 | 29 |
| `test_motor` | 254,986 | 254,986 | 42 |
| `weather` | 2,879,915 | 2,879,915 | 9 |

Model-ready tables (`data/features/`):

| Table | Rows | Columns | Features |
|---|---:|---:|---:|
| `policies_motor` | 979,968 | 81 | 53 + exposure |
| `test_motor` | 254,986 | 61 | 53 + exposure |
| `policies_buildings` | 707,726 | 75 | 44 + exposure |
| `test_buildings` | 184,155 | 52 | 44 + exposure |
| `policies_contents` | 721,617 | 74 | 49 + exposure |
| `test_contents` | 187,498 | 57 | 49 + exposure |

## Cleaning (from `reports/02_cleaning/cleaning_log.csv`)

| Rule | Table | Column | Rows | Action | Evidence |
|---|---|---|---:|---|---|
| C01 | `customers` | `occupation` | 531,778 | recoded | 'Retired'->'retired' (98,624); 'Student'->'student' (40,063); 'Hairdresser'->'hairdresser' (23,869); 'Care worker'->'care worker' (23,383); 'Nurse'->'nurse' (23 |
| C10 | `weather` | `*` | 0 | none found |  |
| C03 | `policies_motor` | `customer_id` | 98,110 | zero-padded | lengths before: {2: 5, 3: 11, 4: 79, 5: 927, 6: 8958, 7: 88130}; every padded id is found in customers |
| C04 | `policies_motor` | `exposure` | 5,596 | recomputed | recorded in days: 934; on a 360-day basis: 4,662; other: 0 |
| C06 | `policies_motor` | `annual_mileage` | 11,928 | flag dq_mileage_zero | value kept: these cars claim a little less than the lowest band |
| C04 | `policies_buildings` | `exposure` | 4,078 | recomputed | recorded in days: 656; on a 360-day basis: 3,422; other: 0 |
| C05 | `policies_buildings` | `sum_insured` | 1,395 | rescaled | after the fix: 52,000 to 594,000 (median 222,000) |
| C01 | `policies_buildings` | `construction_type` | 560,124 | recoded | 'Brick'->'brick' (456,072); 'W'->'timber' (42,319); 'T'->'timber' (38,937); 'BRICK'->'brick' (14,709); 'w'->'timber' (2,243); 'NON_STANDARD'->'non_standard' (2, |
| C04 | `policies_contents` | `exposure` | 4,206 | recomputed | recorded in days: 729; on a 360-day basis: 3,477; other: 0 |
| C05 | `policies_contents` | `sum_insured` | 1,402 | rescaled | after the fix: 8,200 to 152,600 (median 37,900) |
| C03 | `test_motor` | `customer_id` | 25,554 | zero-padded | lengths before: {2: 1, 3: 3, 4: 18, 5: 241, 6: 2316, 7: 22975}; every padded id is found in customers |
| C06 | `test_motor` | `annual_mileage` | 3,108 | flag dq_mileage_zero | value kept: these cars claim a little less than the lowest band |
| C05 | `test_buildings` | `sum_insured` | 370 | rescaled | after the fix: 79,000 to 577,000 (median 243,500) |
| C01 | `test_buildings` | `construction_type` | 145,548 | recoded | 'Brick'->'brick' (118,393); 'W'->'timber' (21,099); 'BRICK'->'brick' (3,917); 'w'->'timber' (1,162); 'NON_STANDARD'->'non_standard' (552); 'STONE'->'stone' (425 |
| C05 | `test_contents` | `sum_insured` | 360 | rescaled | after the fix: 9,500 to 146,800 (median 41,850) |
| C07 | `policies_motor` | `years_licensed` | 2,962 | restored from the policy's other rows | 2,962 restored from the same policy's age at first licence, 0 capped at age - 17 (no other row) |
| C07 | `test_motor` | `years_licensed` | 755 | restored from the policy's other rows | 670 restored from the same policy's age at first licence, 85 capped at age - 17 (no other row) |
| C08 | `policies_motor` | `ncb_years` | 127,410 | kept as declared, flagged | too common to be typing errors (mostly young drivers); the declaration itself carries risk information |
| C08 | `test_motor` | `ncb_years` | 34,873 | kept as declared, flagged | too common to be typing errors (mostly young drivers); the declaration itself carries risk information |
| K01 | `claims` | `claim_id` | 604 | duplicate removed | removed ids have {11: 604} characters (normal: 10); loss years {2020: 1, 2021: 3, 2022: 600} |
| K02 | `claims` | `incurred_to_date` | 1 | divided by 10^1 | paid 738,769; corrected incurred 1,189,428 (a reserve typed with 1 extra digit(s)) |
| K03 | `claims` | `net_incurred` | 162,546 | new column | repudiated: 1,705 |
| K04 | `claims` | `row_id` | 413 | unmatched kept, flagged (excluded from targets) | status counts: {'matched': 162133, 'before_first_row': 413} |
| K05 | `claims` | `peril` | 4 | kept, flagged | {'own_damage': 4} |
| K06 | `claims` | `peril` | 1,653 | kept, flagged | 13% of home AD claims; mean size 1.04x the covered ones: plausibly basic AD cover |

No policy, customer, address or weather row is removed; the only rows removed are exact duplicate claims. Missing values are not imputed (step 3 adds a `_missing` flag where missingness predicts claims).

## Feature selection (from `reports/03_features/feature_register.csv`)

Every column of the joined table is either kept or dropped with a reason. The out-of-time tests (alone and inside a throwaway LightGBM, against 20 random probe columns) are pooled over three folds: learn on the earlier years, score the next one (2022, 2023, 2024).

| Reason | motor | buildings | contents |
|---|---:|---:|---:|
| adds value | 53 | 43 | 48 |
| base | 1 | 1 | 1 |
| constant | 1 | 3 | 2 |
| data-quality flag | 2 | 2 | 2 |
| governance | 9 | 9 | 9 |
| informative missingness | 0 | 1 | 1 |
| leakage | 2 | 2 | 2 |
| near-duplicate | 2 | 2 | 2 |
| no value | 22 | 22 | 18 |
| not an identifier | 1 | 0 | 0 |
| not testable | 1 | 1 | 1 |
| personal data | 2 | 2 | 2 |
| redundant | 3 | 3 | 3 |

Pass mark per line (lowest z at which at most 1 of 20 probes would pass): motor z >= 1, buildings z >= 1.5, contents z >= 1.

### motor: 53 features + exposure

| Feature | Meaning | Why kept |
|---|---|---|
| `exposure` | years on cover in the row (days / 365) | base |
| `driver_age` | age of the main driver at the start of the row | frequency_oneway, severity_oneway, frequency_model |
| `ncb_years` | no-claims bonus years (0-15) | frequency_oneway, frequency_model |
| `vehicle_value_group` | vehicle value group 1-20 | frequency_oneway, severity_oneway, severity_model |
| `vehicle_age` | vehicle age in years | frequency_oneway, severity_oneway, severity_model |
| `engine_size_cc` | engine size | frequency_oneway, severity_oneway, severity_model |
| `body_type` | hatchback, saloon, estate, suv, coupe | frequency_oneway |
| `vehicle_colour` | colour | frequency_oneway |
| `fuel_type` | petrol, diesel, hybrid, electric | severity_oneway |
| `driver_assist` | has driver-assistance systems | frequency_oneway, severity_oneway, severity_model |
| `cover_type` | comprehensive, third_party_fire_theft, third_party_only | frequency_oneway, frequency_model |
| `annual_mileage` | declared annual mileage | frequency_oneway, frequency_model |
| `overnight_parking` | garage, driveway, street | frequency_oneway, theft_oneway |
| `vehicle_use` | social, commuting, business | frequency_oneway, frequency_model |
| `telematics_opt_in` | telematics box fitted | frequency_oneway, severity_oneway |
| `telematics_score` | telematics driving score 1-100 (opted-in only) | frequency_oneway, frequency_model |
| `voluntary_excess` | excess chosen by the customer: 0, 250, 500 or 1000 | frequency_oneway, frequency_model |
| `channel` | sales channel: direct, comparison_site, broker | frequency_oneway, frequency_model |
| `payment_method` | annual or monthly | frequency_oneway, frequency_model |
| `missed_payments_12m` | premium payments missed in the last 12 months | frequency_oneway, frequency_model |
| `renewal_count` | number of renewals so far | frequency_oneway, frequency_model |
| `days_quote_to_inception` | days between quote and start of cover | frequency_oneway |
| `dq_mileage_zero` | declared annual mileage of 0 | frequency_oneway |
| `occupation` | occupation | frequency_oneway, frequency_model |
| `marital_status` | single, married, divorced, widowed | frequency_oneway, severity_oneway |
| `household_size` | people in the household | frequency_oneway |
| `credit_band` | credit band as a number, A = 1 ... E = 5 | frequency_oneway, severity_oneway, frequency_model |
| `prior_claims_5y` | claims declared for the last 5 years (any insurer) | frequency_oneway, frequency_model |
| `licence_points` | penalty points on the driving licence | frequency_oneway |
| `years_at_address` | years lived at the current address | frequency_oneway |
| `region` | region R01-R12 | frequency_oneway, frequency_model |
| `urban_density` | rural, suburban, urban, dense_urban | frequency_oneway, severity_oneway, frequency_model, severity_model, theft_oneway |
| `property_type` | detached, semi_detached, terraced, flat | frequency_oneway |
| `start_month` | month the row starts | frequency_oneway |
| `winter_fraction` | share of the covered days in December-February | frequency_oneway |
| `starts_after_jan1` | row starts after 1 January (new business or a mid-term change) | frequency_oneway |
| `age_first_licensed` | driver_age - years_licensed | frequency_oneway |
| `wx_hail_days` | hail_days at the address in the row's year | glass_hail_oneway |
| `wx_freezing_days` | freezing_days at the address in the row's year | frequency_oneway |
| `wx_dry_summer_index` | dry_summer_index at the address in the row's year | frequency_oneway |
| `wx_mean_temperature_c` | mean_temperature_c at the address in the row's year | frequency_oneway |
| `wx_hail_days_mean` | address's climate: mean hail_days over the training years | frequency_oneway, glass_hail_oneway |
| `wx_wind_exposure_index_mean` | address's climate: mean wind_exposure_index over the training years | frequency_oneway |
| `wx_dry_summer_index_mean` | address's climate: mean dry_summer_index over the training years | frequency_oneway |
| `wx_wildfire_index_mean` | address's climate: mean wildfire_index over the training years | frequency_oneway |
| `wx_mean_temperature_c_mean` | address's climate: mean mean_temperature_c over the training years | frequency_oneway |
| `hist_pol_claims_1y` | claims of this policy in the previous calendar year (known at 1 January) | frequency_oneway, severity_oneway |
| `hist_pol_claims_3y` | claims of this policy in the previous 3 calendar years | frequency_oneway, severity_oneway, frequency_model |
| `hist_pol_cost_3y` | cost of this policy's claims in the previous 3 years (each capped at the line's 99.5% claim quantile) | frequency_oneway, frequency_model |
| `hist_pol_years_since_claim` | years since this policy's last claim (missing: none in 3 years) | frequency_oneway, frequency_model |
| `hist_pol_exposure_3y` | years this policy was on cover in the previous 3 years | frequency_oneway |
| `hist_cust_claims_3y` | claims of the customer on every line in the previous 3 years | frequency_oneway, severity_oneway |
| `holds_buildings` | the customer holds a buildings policy the same year | frequency_oneway |
| `holds_contents` | the customer holds a contents policy the same year | frequency_oneway |

Dropped (45): constant: `compulsory_excess`; data-quality flag: `dq_exposure_fixed`, `dq_years_licensed_fixed`; governance: `discount_applied`, `written_premium`, `email_domain`, `marketing_consent`, `document_preference`, `previous_insurer`, `campaign_code`, `branch_id`, `marketing_segment`; leakage: `customer_service_contacts`, `renewal_decision`; near-duplicate: `years_licensed`, `wx_freezing_days_mean`; no value: `transmission`, `seats`, `named_young_driver`, `dq_ncb_exceeds_licence`, `flood_zone`, `distance_to_river_km`, `elevation_m`, `soil_type`, `roof_type`, `basement`, `trees_within_5m`, `listed_building`, `bedrooms`, `energy_rating`, `solar_panels`, `ends_before_year_end`, `property_age`, `wx_wind_exposure_index`, `wx_wildfire_index`, `wx_rainfall_mm`, `wx_rainfall_mm_mean`, `hist_cust_claims_other_lines_3y`; not an identifier: `make_model`; not testable: `hist_years_available`; personal data: `surname`, `date_of_birth`; redundant: `new_business`, `customer_since_year`, `year_built`

### buildings: 44 features + exposure

| Feature | Meaning | Why kept |
|---|---|---|
| `exposure` | years on cover in the row (days / 365) | base |
| `sum_insured` | rebuilding sum insured | frequency_oneway, severity_oneway, frequency_model, severity_model |
| `construction_type` | brick, stone, timber, non_standard | frequency_oneway, frequency_model, fire_oneway |
| `occupancy` | owner_occupied or let | frequency_oneway |
| `days_unoccupied` | days the property is unoccupied in a year | frequency_oneway, frequency_model, escape_of_water_oneway |
| `accidental_damage_cover` | accidental damage cover bought | frequency_oneway, frequency_model, accidental_damage_oneway |
| `voluntary_excess` | excess chosen by the customer: 0, 250, 500 or 1000 | frequency_oneway, frequency_model |
| `channel` | sales channel: direct, comparison_site, broker | frequency_oneway, frequency_model |
| `payment_method` | annual or monthly | frequency_oneway, frequency_model |
| `missed_payments_12m` | premium payments missed in the last 12 months | frequency_oneway, frequency_model |
| `credit_band` | credit band as a number, A = 1 ... E = 5 | frequency_oneway |
| `prior_claims_5y` | claims declared for the last 5 years (any insurer) | frequency_oneway, frequency_model |
| `region` | region R01-R12 | frequency_oneway, frequency_model |
| `flood_zone` | flood zone 1 (lowest risk) to 5 | frequency_oneway, frequency_model, flood_oneway |
| `distance_to_river_km` | distance to the nearest river | frequency_oneway, flood_oneway |
| `elevation_m` | elevation above sea level | frequency_oneway, frequency_model, flood_oneway |
| `soil_type` | clay, sand, loam, chalk, rock | subsidence_oneway |
| `property_type` | detached, semi_detached, terraced, flat | frequency_oneway, severity_oneway, storm_oneway |
| `roof_type` | pitched or flat | frequency_oneway, storm_oneway |
| `trees_within_5m` | trees within 5 metres of the building | subsidence_oneway |
| `listed_building` | protected (listed) building | severity_oneway |
| `bedrooms` | number of bedrooms | frequency_oneway, severity_oneway, severity_model |
| `energy_rating` | energy rating as a number, A = 1 ... G = 7 | frequency_oneway, frequency_model |
| `property_age` | year - year_built | frequency_oneway, frequency_model, escape_of_water_oneway |
| `si_per_bedroom` | sum insured / bedrooms | severity_oneway, severity_model |
| `wx_wind_exposure_index` | wind_exposure_index at the address in the row's year | frequency_oneway, storm_oneway |
| `wx_freezing_days` | freezing_days at the address in the row's year | escape_of_water_oneway |
| `wx_dry_summer_index` | dry_summer_index at the address in the row's year | subsidence_oneway |
| `wx_wildfire_index` | wildfire_index at the address in the row's year | fire_oneway |
| `wx_rainfall_mm` | rainfall_mm at the address in the row's year | flood_oneway |
| `wx_mean_temperature_c` | mean_temperature_c at the address in the row's year | escape_of_water_oneway |
| `wx_freezing_days_mean` | address's climate: mean freezing_days over the training years | escape_of_water_oneway |
| `wx_dry_summer_index_mean` | address's climate: mean dry_summer_index over the training years | subsidence_oneway |
| `wx_wildfire_index_mean` | address's climate: mean wildfire_index over the training years | fire_oneway |
| `wx_rainfall_mm_mean` | address's climate: mean rainfall_mm over the training years | frequency_oneway, flood_oneway |
| `hist_pol_claims_1y` | claims of this policy in the previous calendar year (known at 1 January) | frequency_oneway |
| `hist_pol_claims_3y` | claims of this policy in the previous 3 calendar years | frequency_oneway |
| `hist_pol_cost_3y` | cost of this policy's claims in the previous 3 years (each capped at the line's 99.5% claim quantile) | frequency_oneway |
| `hist_pol_years_since_claim` | years since this policy's last claim (missing: none in 3 years) | frequency_oneway, frequency_model |
| `hist_cust_claims_3y` | claims of the customer on every line in the previous 3 years | frequency_oneway |
| `hist_addr_claims_3y` | home claims at the address (any customer) in the previous 3 years | frequency_oneway |
| `hist_addr_weather_claims_3y` | storm, flood, subsidence and escape-of-water claims at the address | frequency_oneway |
| `hist_pol_escape_of_water_3y` | escape-of-water claims of this policy in the previous 3 years | frequency_oneway |
| `holds_motor` | the customer holds a motor policy the same year | frequency_oneway |
| `flood_zone_missing` | flood zone missing (such addresses claim more) | informative missingness |

Dropped (46): constant: `compulsory_excess`, `hist_pol_flood_3y`, `hist_pol_subsidence_3y`; data-quality flag: `dq_exposure_fixed`, `dq_sum_insured_fixed`; governance: `discount_applied`, `written_premium`, `email_domain`, `marketing_consent`, `document_preference`, `previous_insurer`, `campaign_code`, `branch_id`, `marketing_segment`; leakage: `customer_service_contacts`, `renewal_decision`; near-duplicate: `wx_wind_exposure_index_mean`, `wx_mean_temperature_c_mean`; no value: `policyholder_age`, `smoke_alarms`, `mortgage`, `renewal_count`, `days_quote_to_inception`, `occupation`, `marital_status`, `household_size`, `licence_points`, `years_at_address`, `urban_density`, `basement`, `solar_panels`, `start_month`, `winter_fraction`, `starts_after_jan1`, `ends_before_year_end`, `wx_hail_days`, `wx_hail_days_mean`, `hist_pol_exposure_3y`, `hist_cust_claims_other_lines_3y`, `holds_contents`; not testable: `hist_years_available`; personal data: `surname`, `date_of_birth`; redundant: `new_business`, `customer_since_year`, `year_built`

### contents: 49 features + exposure

| Feature | Meaning | Why kept |
|---|---|---|
| `exposure` | years on cover in the row (days / 365) | base |
| `policyholder_age` | age of the policyholder at the start of the row | frequency_oneway, frequency_model |
| `sum_insured` | rebuilding sum insured | frequency_oneway, severity_oneway, severity_model |
| `tenure` | owner or renter | severity_oneway |
| `security_level` | 0 standard, 1 locks, 2 alarm and locks | theft_oneway |
| `high_value_items` | declared high-value items (0-5) | severity_oneway |
| `accidental_damage_cover` | accidental damage cover bought | frequency_oneway, severity_oneway, frequency_model, accidental_damage_oneway |
| `works_from_home` | policyholder works from home | frequency_oneway |
| `lodgers` | lodgers in the home | frequency_oneway, accidental_damage_oneway |
| `voluntary_excess` | excess chosen by the customer: 0, 250, 500 or 1000 | frequency_oneway, frequency_model |
| `channel` | sales channel: direct, comparison_site, broker | frequency_oneway, frequency_model |
| `payment_method` | annual or monthly | frequency_oneway |
| `missed_payments_12m` | premium payments missed in the last 12 months | frequency_oneway, frequency_model |
| `renewal_count` | number of renewals so far | frequency_oneway |
| `days_quote_to_inception` | days between quote and start of cover | frequency_oneway |
| `occupation` | occupation | frequency_oneway |
| `marital_status` | single, married, divorced, widowed | frequency_oneway |
| `household_size` | people in the household | accidental_damage_oneway |
| `credit_band` | credit band as a number, A = 1 ... E = 5 | frequency_oneway |
| `prior_claims_5y` | claims declared for the last 5 years (any insurer) | frequency_oneway, frequency_model |
| `region` | region R01-R12 | frequency_oneway, frequency_model |
| `urban_density` | rural, suburban, urban, dense_urban | frequency_oneway, theft_oneway |
| `flood_zone` | flood zone 1 (lowest risk) to 5 | frequency_oneway, frequency_model, flood_oneway |
| `distance_to_river_km` | distance to the nearest river | frequency_oneway, flood_oneway |
| `elevation_m` | elevation above sea level | frequency_oneway, frequency_model, flood_oneway |
| `start_month` | month the row starts | frequency_oneway |
| `starts_after_jan1` | row starts after 1 January (new business or a mid-term change) | frequency_oneway |
| `si_per_bedroom` | sum insured / bedrooms | frequency_oneway, severity_oneway, severity_model |
| `wx_freezing_days` | freezing_days at the address in the row's year | frequency_oneway, escape_of_water_oneway |
| `wx_dry_summer_index` | dry_summer_index at the address in the row's year | frequency_oneway |
| `wx_wildfire_index` | wildfire_index at the address in the row's year | frequency_oneway |
| `wx_rainfall_mm` | rainfall_mm at the address in the row's year | flood_oneway |
| `wx_mean_temperature_c` | mean_temperature_c at the address in the row's year | frequency_oneway, escape_of_water_oneway |
| `wx_hail_days_mean` | address's climate: mean hail_days over the training years | frequency_oneway |
| `wx_wind_exposure_index_mean` | address's climate: mean wind_exposure_index over the training years | frequency_oneway |
| `wx_freezing_days_mean` | address's climate: mean freezing_days over the training years | frequency_oneway, escape_of_water_oneway |
| `wx_dry_summer_index_mean` | address's climate: mean dry_summer_index over the training years | frequency_oneway |
| `wx_wildfire_index_mean` | address's climate: mean wildfire_index over the training years | frequency_oneway |
| `wx_rainfall_mm_mean` | address's climate: mean rainfall_mm over the training years | flood_oneway |
| `hist_pol_claims_1y` | claims of this policy in the previous calendar year (known at 1 January) | frequency_oneway, severity_oneway |
| `hist_pol_claims_3y` | claims of this policy in the previous 3 calendar years | frequency_oneway |
| `hist_pol_cost_3y` | cost of this policy's claims in the previous 3 years (each capped at the line's 99.5% claim quantile) | frequency_oneway |
| `hist_pol_years_since_claim` | years since this policy's last claim (missing: none in 3 years) | frequency_oneway |
| `hist_pol_exposure_3y` | years this policy was on cover in the previous 3 years | frequency_oneway |
| `hist_cust_claims_3y` | claims of the customer on every line in the previous 3 years | frequency_oneway |
| `hist_addr_claims_3y` | home claims at the address (any customer) in the previous 3 years | frequency_oneway, severity_oneway |
| `hist_addr_weather_claims_3y` | storm, flood, subsidence and escape-of-water claims at the address | frequency_oneway, severity_oneway |
| `hist_pol_escape_of_water_3y` | escape-of-water claims of this policy in the previous 3 years | frequency_oneway |
| `holds_motor` | the customer holds a motor policy the same year | frequency_oneway |
| `flood_zone_missing` | flood zone missing (such addresses claim more) | informative missingness |

Dropped (41): constant: `compulsory_excess`, `hist_pol_flood_3y`; data-quality flag: `dq_exposure_fixed`, `dq_sum_insured_fixed`; governance: `discount_applied`, `written_premium`, `email_domain`, `marketing_consent`, `document_preference`, `previous_insurer`, `campaign_code`, `branch_id`, `marketing_segment`; leakage: `customer_service_contacts`, `renewal_decision`; near-duplicate: `wx_mean_temperature_c_mean`, `holds_buildings`; no value: `bicycle_cover`, `licence_points`, `years_at_address`, `soil_type`, `property_type`, `roof_type`, `basement`, `trees_within_5m`, `listed_building`, `bedrooms`, `energy_rating`, `solar_panels`, `winter_fraction`, `ends_before_year_end`, `property_age`, `wx_hail_days`, `wx_wind_exposure_index`, `hist_cust_claims_other_lines_3y`; not testable: `hist_years_available`; personal data: `surname`, `date_of_birth`; redundant: `new_business`, `customer_since_year`, `year_built`

## Targets in the training tables

* `n_claims`, `claim_cost`: claims whose loss date is in the row's period, cost = incurred - recoveries (net of the excess), repudiated claims excluded, valued at 31 December 2024; `n_open`: of which open.
* `n_<peril>`, `cost_<peril>`: the same by peril.
* `comp_count`, `comp_cost`: share of the row's claims (count, cost) already reported at 31 December 2024, from reporting-delay patterns corrected for truncation; `comp_<peril>` by peril. Use `log(exposure x comp_count)` or `log(exposure x comp_cost)` as the offset.
