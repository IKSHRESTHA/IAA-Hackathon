# Data dictionary - 2nd round (emerging market)

Files of the case study *IAA AI-Assisted Hackathon 2026 - Part 2*, as read by `01_data_reading.py`.

## `customers`

| Column | Type | Meaning | Missing % | Distinct | Values |
|---|---|---|---:|---:|---|
| `customer_id` | id | customer identifier (8 digits) | 0.00 | 135,856 | 38613727 (1); 88219131 (1); 25030155 (1); 95537081 (1); 70892415 (1); 14469298 (1) |
| `surname` | text | surname (personal data) | 0.00 | 600 | Surname503 (271); Surname361 (269); Surname042 (268); Surname238 (263); Surname272 (262);  |
| `gender` | category | f or m (a permitted rating factor in this market) | 0.00 | 2 | F (67,973); M (67,883) |
| `occupation` | category | occupation (free text; may start with 'self-employed') | 12.16 | 80 | Retired (9,218); Student (6,445); retired (4,431); student (3,236); Retired  (2,736); Care |
| `marital_status` | category | single, married, divorced, widowed | 12.03 | 4 | single (54,935); married (51,481); divorced (8,858); widowed (4,234) |
| `household_size` | integer | people in the household | 11.80 | 7 | 1 to 7 (median 2) |
| `prior_claims_5y` | integer | claims declared for the last 5 years (any insurer) | 0.00 | 6 | 0 to 5 (median 0) |
| `licence_points` | integer | penalty points on the driving licence | 0.00 | 4 | 0 to 9 (median 0) |
| `years_at_address` | integer | years lived at the current address | 11.94 | 46 | 0 to 53 (median 6) |
| `customer_since_year` | integer | first year with the insurer | 0.00 | 18 | 2008 to 2025 (median 2025) |
| `email_domain` | category | e-mail provider | 0.00 | 5 | other (27,324); inboxly (27,295); posthub (27,119); netbox (27,082); mailone (27,036) |
| `marketing_consent` | boolean | agreed to receive marketing | 0.00 | 2 | True (75,092); False (60,764) |
| `document_preference` | category | email, post or app | 0.00 | 3 | email (81,586); app (33,818); post (20,452) |
| `previous_insurer` | category | insurer before joining | 0.00 | 40 | INS31 (3,537); INS10 (3,533); INS05 (3,527); INS22 (3,486); INS18 (3,472); INS26 (3,464) |
| `campaign_code` | category | marketing campaign that acquired the customer (missing: none) | 20.19 | 15 | CMP10 (7,357); CMP02 (7,330); CMP13 (7,309); CMP07 (7,277); CMP08 (7,274); CMP14 (7,242) |
| `branch_id` | category | servicing branch | 0.00 | 25 | 9 (5,572); 4 (5,553); 19 (5,537); 6 (5,516); 2 (5,510); 10 (5,497) |
| `marketing_segment` | category | segment assigned by marketing | 0.00 | 12 | young_affluent (34,870); young_mainstream (16,106); mature_affluent (13,310); midlife_affl |

## `addresses`

| Column | Type | Meaning | Missing % | Distinct | Values |
|---|---|---|---:|---:|---|
| `address_id` | id | address identifier | 0.00 | 136,914 | A9736411 (1); A6546739 (1); A1065263 (1); A5881586 (1); A0550616 (1); A1190346 (1) |
| `region` | category | region E01-E06 | 0.00 | 6 | E03 (40,212); E06 (26,633); E02 (24,481); E05 (22,479); E01 (17,235); E04 (5,874) |
| `urban_density` | category | rural, suburban, urban, dense_urban | 0.00 | 4 | suburban (61,895); urban (33,735); rural (27,519); dense_urban (13,765) |
| `flood_zone` | ordinal | flood zone 1 (lowest risk) to 5 | 2.95 | 5 | 1 to 5 (median 1) |
| `distance_to_river_km` | numeric | distance to the nearest river | 11.99 | 1,756 | 0.01 to 30 (median 2.37) |
| `elevation_m` | numeric | elevation above sea level | 0.00 | 580 | 0 to 60 (median 7.5) |
| `soil_type` | category | clay, sand, loam, chalk, rock | 12.04 | 5 | clay (42,529); chalk (19,499); loam (19,491); rock (19,476); sand (19,439) |
| `year_built` | integer | year the property was built | 0.00 | 160 | 1862 to 2021 (median 1964) |
| `property_type` | category | detached, semi_detached, terraced, flat | 0.00 | 4 | semi_detached (40,812); detached (34,496); terraced (34,410); flat (27,196) |
| `roof_type` | category | pitched or flat | 11.93 | 2 | pitched (108,677); flat (11,900) |
| `basement` | boolean | has a basement | 0.00 | 2 | False (116,410); True (20,504) |
| `trees_within_5m` | boolean | trees within 5 metres of the building | 0.00 | 2 | False (120,551); True (16,363) |
| `listed_building` | boolean | protected (listed) building | 0.00 | 2 | False (132,860); True (4,054) |
| `bedrooms` | integer | number of bedrooms | 0.00 | 7 | 1 to 7 (median 3) |
| `energy_rating` | ordinal | energy rating A (best) to G | 12.06 | 7 | C (24,213); B (23,733); D (20,576); A (19,265); E (13,892); F (9,688) |
| `solar_panels` | boolean | has solar panels | 0.00 | 2 | False (123,182); True (13,732) |

## `weather`

| Column | Type | Meaning | Missing % | Distinct | Values |
|---|---|---|---:|---:|---|
| `address_id` | id | address identifier | 0.00 | 136,914 | A9736411 (3); A6546739 (3); A1065263 (3); A5881586 (3); A0550616 (3); A1190346 (3) |
| `year` | integer | calendar year (2023-2025) | 0.00 | 3 | 2023 to 2025 (median 2024) |
| `hail_days` | numeric | days with hail | 0.00 | 76 | 0 to 83 (median 12) |
| `wind_exposure_index` | numeric | wind exposure index 0-10 | 0.00 | 101 | 0 to 10 (median 5.2) |
| `freezing_days` | numeric | days below freezing | 0.00 | 74 | 0 to 74 (median 22) |
| `dry_summer_index` | numeric | summer dryness index 0-1 | 0.00 | 810 | 0.132 to 1 (median 0.669) |
| `wildfire_index` | numeric | wildfire index 0-1 | 0.00 | 862 | 0.104 to 1 (median 0.51) |
| `rainfall_mm` | numeric | annual rainfall | 0.00 | 1,770 | 622 to 2727 (median 1302) |
| `mean_temperature_c` | numeric | annual mean temperature | 0.00 | 98 | 9.6 to 19.3 (median 15.1) |

## `claims`

| Column | Type | Meaning | Missing % | Distinct | Values |
|---|---|---|---:|---:|---|
| `claim_id` | id | claim identifier | 0.00 | 9,258 | CL32268026 (1); CL28349650 (1); CL40256686 (1); CL02220435 (1); CL75733240 (1); CL30407354 |
| `policy_id` | id | policy that made the claim | 0.00 | 7,416 | M27385197 (7); M91131447 (6); M55315974 (6); M85798540 (6); M47767258 (6); M09323987 (6) |
| `line` | category | motor, buildings or contents | 0.00 | 3 | motor (7,183); buildings (1,059); contents (1,016) |
| `peril` | category | cause of loss | 0.00 | 11 | own_damage (2,712); third_party_prop (2,263); glass_hail (1,350); theft (826); storm (466) |
| `loss_date` | date | date of the loss | 0.00 | 760 | 2024-10-07 (26); 2023-12-11 (25); 2023-09-12 (24); 2023-12-28 (23); 2023-08-19 (23); 2024- |
| `report_date` | date | date the claim was reported | 0.00 | 729 | 2024-08-12 (24); 2023-12-15 (23); 2023-12-31 (23); 2023-02-28 (23); 2023-12-09 (23); 2024- |
| `status` | category | closed, open or repudiated (declined) | 0.00 | 3 | closed (8,082); open (1,056); repudiated (120) |
| `paid_to_date` | numeric | paid so far, net of excess | 0.00 | 9,021 | 0 to 361093 (median 1029.01) |
| `incurred_to_date` | numeric | paid + case reserve, net of excess | 0.00 | 9,043 | 0 to 361093 (median 1214.62) |
| `recoveries` | numeric | amounts recovered (salvage, third parties) | 0.00 | 753 | 0 to 10255.8 (median 0) |
| `repairer_id` | id | repairer (motor) | 46.26 | 987 | RP0417 (83); RP0901 (81); RP0733 (80); RP0757 (12); RP0104 (12); RP0492 (12) |
| `third_party_id` | id | third party (liability perils) | 70.70 | 2,713 | TP6024392 (1); TP6057218 (1); TP0834574 (1); TP8002966 (1); TP8905674 (1); TP9678573 (1) |

## `policies_motor`

| Column | Type | Meaning | Missing % | Distinct | Values |
|---|---|---|---:|---:|---|
| `row_id` | id | one policy row: a policy over one calendar-year period, or part of one | 0.00 | 47,283 | M00000000 (1); M00000001 (1); M00000002 (1); M00000003 (1); M00000004 (1); M00000005 (1) |
| `policy_id` | id | the policy; constant across its years and mid-term changes | 0.00 | 23,722 | M61105325 (4); M03367245 (4); M51980251 (4); M96948734 (4); M42168510 (4); M54779473 (4) |
| `customer_id` | id | the policyholder (customers table) | 0.00 | 23,722 | 23128226 (4); 36316135 (4); 49767091 (4); 38275564 (4); 86896118 (4); 54084415 (4) |
| `address_id` | id | the insured or garaging address (addresses table) | 0.00 | 23,799 | A8242647 (10); A9579498 (9); A1021018 (9); A2958263 (8); A3015082 (8); A9830968 (7) |
| `period_start` | date | first day on cover in this row | 0.00 | 728 | 2024-01-01 (20,084); 2023-01-01 (17,655); 2024-11-08 (28); 2024-01-02 (26); 2024-04-30 (26 |
| `period_end` | date | last day on cover in this row (inclusive) | 0.00 | 726 | 2024-12-30 (20,097); 2023-12-31 (20,084); 2024-10-22 (26); 2024-02-05 (25); 2024-10-25 (25 |
| `exposure` | numeric | years on cover: (period_end - period_start + 1) / 365 | 0.00 | 432 | 0.0027 to 365 (median 1) |
| `driver_age_band` | ordinal | age band of the main driver: 17-24, 25-34, 35-49, 50-64, 65+ | 0.00 | 5 | 65+ (13,311); 17-24 (9,419); 25-34 (9,040); 50-64 (7,908); 35-49 (7,605) |
| `years_licensed` | integer | years since the main driver passed the test | 0.00 | 74 | 0 to 73 (median 24) |
| `ncb_years` | integer | no-claims bonus years (0-15) | 0.00 | 16 | 0 to 15 (median 6) |
| `vehicle_value_group` | ordinal | vehicle value group 1-20 | 0.00 | 20 | 1 to 20 (median 11) |
| `vehicle_age` | integer | vehicle age in years | 11.85 | 24 | 3 to 26 (median 9) |
| `make_model` | category | vehicle make and model code (300 codes) | 0.00 | 300 | MM164 (207); MM056 (203); MM089 (202); MM275 (195); MM141 (194); MM051 (194) |
| `engine_size_cc` | numeric | engine size | 12.15 | 31 | 500 to 3600 (median 2100) |
| `body_type` | category | hatchback, saloon, estate, suv, coupe | 0.00 | 5 | hatchback (18,836); saloon (9,454); suv (9,282); estate (5,692); coupe (4,019) |
| `transmission` | category | manual or automatic | 0.00 | 2 | manual (25,968); automatic (21,315) |
| `seats` | integer | number of seats | 0.00 | 4 | 2 to 7 (median 5) |
| `vehicle_colour` | category | colour | 0.00 | 8 | black (6,047); blue (5,990); red (5,984); green (5,891); silver (5,879); other (5,873) |
| `fuel_type` | category | petrol, diesel, hybrid, electric | 0.00 | 4 | petrol (25,367); diesel (14,046); hybrid (6,900); electric (970) |
| `driver_assist` | boolean | has driver-assistance systems | 0.00 | 2 | False (33,174); True (14,109) |
| `cover_type` | category | comprehensive, third_party_fire_theft, third_party_only | 0.00 | 3 | comprehensive (37,219); third_party_fire_theft (5,143); third_party_only (4,921) |
| `overnight_parking` | category | garage, driveway, street | 12.05 | 3 | driveway (20,823); street (12,546); garage (8,217) |
| `vehicle_use` | category | social, commuting, business | 0.00 | 3 | social (21,294); commuting (21,224); business (4,765) |
| `named_young_driver` | boolean | a young driver is named on the policy | 0.00 | 2 | False (44,835); True (2,448) |
| `voluntary_excess` | integer | excess chosen by the customer: 0, 250, 500 or 1000 | 0.00 | 4 | 0 to 1000 (median 250) |
| `compulsory_excess` | integer | excess set by the insurer | 0.00 | 1 | 250 to 250 (median 250) |
| `channel` | category | sales channel: direct, comparison_site, broker | 0.00 | 3 | broker (33,161); direct (11,806); comparison_site (2,316) |
| `payment_method` | category | annual or monthly | 0.00 | 2 | annual (26,147); monthly (21,136) |
| `missed_payments_12m` | integer | premium payments missed in the last 12 months | 0.00 | 6 | 0 to 5 (median 0) |
| `new_business` | boolean | first year with the insurer | 0.00 | 2 | False (40,748); True (6,535) |
| `renewal_count` | integer | number of renewals so far | 0.00 | 17 | 0 to 16 (median 7) |
| `days_quote_to_inception` | integer | days between quote and start of cover | 0.00 | 31 | 0 to 30 (median 13) |
| `discount_applied` | numeric | discount on the premium (share) | 0.00 | 3 | 0 to 0.1 (median 0) |
| `written_premium` | numeric | annual premium charged by the current tariff | 0.00 | 33,413 | 192.54 to 2199.17 (median 502.06) |
| `customer_service_contacts` | integer | contacts with customer service during the period | 0.00 | 20 | 0 to 20 (median 0) |
| `renewal_decision` | outcome | what happened at the end of the period (training files only) | 0.00 | 5 | renewed (39,968); mid_term_change (3,477); lapsed_by_customer (2,369); cancelled (1,250);  |

## `policies_buildings`

| Column | Type | Meaning | Missing % | Distinct | Values |
|---|---|---|---:|---:|---|
| `row_id` | id | one policy row: a policy over one calendar-year period, or part of one | 0.00 | 27,447 | B00021680 (1); B00021681 (1); B00021682 (1); B00021683 (1); B00021684 (1); B00021685 (1) |
| `policy_id` | id | the policy; constant across its years and mid-term changes | 0.00 | 14,299 | B53263320 (4); B82610770 (4); B83685744 (4); B17670082 (4); B93998530 (4); B00996831 (4) |
| `customer_id` | id | the policyholder (customers table) | 0.00 | 14,299 | 67975739 (4); 31107093 (4); 89465785 (4); 83503998 (4); 30096071 (4); 24324764 (4) |
| `address_id` | id | the insured or garaging address (addresses table) | 0.00 | 15,340 | A6546739 (2); A1190346 (2); A6944531 (2); A7385385 (2); A1371358 (2); A4692371 (2) |
| `period_start` | date | first day on cover in this row | 0.00 | 727 | 2024-01-01 (12,107); 2023-01-01 (10,654); 2024-06-28 (15); 2024-02-05 (15); 2024-09-06 (15 |
| `period_end` | date | last day on cover in this row (inclusive) | 0.00 | 623 | 2023-12-31 (12,107); 2024-12-30 (12,103); 2024-11-02 (16); 2024-09-19 (16); 2024-09-09 (15 |
| `exposure` | numeric | years on cover: (period_end - period_start + 1) / 365 | 0.00 | 414 | 0.0027 to 365 (median 1) |
| `policyholder_age_band` | ordinal | age band of the policyholder: 17-24, 25-34, 35-49, 50-64, 65+ | 0.00 | 5 | 65+ (7,845); 17-24 (5,468); 25-34 (5,156); 50-64 (4,566); 35-49 (4,412) |
| `sum_insured` | numeric | rebuilding sum insured | 0.00 | 632 | 44000 to 4.7e+08 (median 229000) |
| `construction_type` | category | brick, stone, timber, non_standard | 0.00 | 4 | BRK (19,194); WOD (3,302); MIX (2,781); STN (2,170) |
| `occupancy` | category | owner_occupied or let | 0.00 | 2 | owner_occupied (24,676); let (2,771) |
| `days_unoccupied` | numeric | days the property is unoccupied in a year | 11.99 | 104 | 0 to 119 (median 7) |
| `smoke_alarms` | boolean | smoke alarms fitted | 11.90 | 2 | True (20,536); False (3,645) |
| `accidental_damage_cover` | boolean | accidental damage cover bought | 0.00 | 2 | False (16,475); True (10,972) |
| `mortgage` | boolean | property is mortgaged | 0.00 | 2 | True (15,145); False (12,302) |
| `voluntary_excess` | integer | excess chosen by the customer: 0, 250, 500 or 1000 | 0.00 | 4 | 0 to 1000 (median 250) |
| `compulsory_excess` | integer | excess set by the insurer | 0.00 | 1 | 350 to 350 (median 350) |
| `channel` | category | sales channel: direct, comparison_site, broker | 0.00 | 3 | broker (19,284); direct (6,753); comparison_site (1,410) |
| `payment_method` | category | annual or monthly | 0.00 | 2 | annual (15,199); monthly (12,248) |
| `missed_payments_12m` | integer | premium payments missed in the last 12 months | 0.00 | 6 | 0 to 5 (median 0) |
| `new_business` | boolean | first year with the insurer | 0.00 | 2 | False (23,661); True (3,786) |
| `renewal_count` | integer | number of renewals so far | 0.00 | 17 | 0 to 16 (median 7) |
| `days_quote_to_inception` | integer | days between quote and start of cover | 0.00 | 31 | 0 to 30 (median 13) |
| `discount_applied` | numeric | discount on the premium (share) | 0.00 | 3 | 0 to 0.1 (median 0) |
| `written_premium` | numeric | annual premium charged by the current tariff | 0.00 | 19,221 | 63.45 to 4629.86 (median 237.56) |
| `customer_service_contacts` | integer | contacts with customer service during the period | 0.00 | 12 | 0 to 11 (median 0) |
| `renewal_decision` | outcome | what happened at the end of the period (training files only) | 0.00 | 5 | renewed (24,102); lapsed_by_customer (1,460); mid_term_change (1,041); cancelled (731); de |

## `policies_contents`

| Column | Type | Meaning | Missing % | Distinct | Values |
|---|---|---|---:|---:|---|
| `row_id` | id | one policy row: a policy over one calendar-year period, or part of one | 0.00 | 34,566 | C00034260 (1); C00034261 (1); C00034262 (1); C00034263 (1); C00034264 (1); C00034265 (1) |
| `policy_id` | id | the policy; constant across its years and mid-term changes | 0.00 | 17,689 | C17689895 (4); C27295025 (4); C30249839 (4); C03576850 (4); C74814014 (4); C27611796 (4) |
| `customer_id` | id | the policyholder (customers table) | 0.00 | 17,689 | 72349334 (4); 67975739 (4); 66270075 (4); 89465785 (4); 83503998 (4); 30096071 (4) |
| `address_id` | id | the insured or garaging address (addresses table) | 0.00 | 18,935 | A4876398 (6); A5240222 (6); A8043799 (6); A1664253 (6); A4294324 (4); A4809555 (4) |
| `period_start` | date | first day on cover in this row | 0.00 | 726 | 2024-01-01 (14,982); 2023-01-01 (13,236); 2024-06-19 (21); 2024-06-28 (20); 2024-02-03 (20 |
| `period_end` | date | last day on cover in this row (inclusive) | 0.00 | 688 | 2023-12-31 (14,982); 2024-12-30 (14,922); 2024-09-09 (21); 2024-02-18 (20); 2024-01-01 (19 |
| `exposure` | numeric | years on cover: (period_end - period_start + 1) / 365 | 0.00 | 431 | 0.0027 to 365 (median 1) |
| `policyholder_age_band` | ordinal | age band of the policyholder: 17-24, 25-34, 35-49, 50-64, 65+ | 0.00 | 5 | 65+ (9,770); 17-24 (6,997); 25-34 (6,504); 50-64 (5,721); 35-49 (5,574) |
| `sum_insured` | numeric | contents sum insured | 0.00 | 1,316 | 5200 to 1.316e+08 (median 35800) |
| `tenure` | category | owner or renter | 0.00 | 2 | owner (19,685); renter (14,881) |
| `security_level` | ordinal | standard, locks, alarm_and_locks | 12.26 | 3 | standard (10,584); alarm_and_locks (10,545); locks (9,199) |
| `high_value_items` | integer | declared high-value items (0-5) | 11.77 | 6 | 0 to 5 (median 0) |
| `accidental_damage_cover` | boolean | accidental damage cover bought | 0.00 | 2 | False (19,153); True (15,413) |
| `works_from_home` | boolean | policyholder works from home | 0.00 | 2 | False (25,932); True (8,634) |
| `lodgers` | boolean | lodgers in the home | 0.00 | 2 | False (32,832); True (1,734) |
| `bicycle_cover` | boolean | bicycle cover bought | 0.00 | 2 | False (29,316); True (5,250) |
| `voluntary_excess` | integer | excess chosen by the customer: 0, 250, 500 or 1000 | 0.00 | 4 | 0 to 1000 (median 0) |
| `compulsory_excess` | integer | excess set by the insurer | 0.00 | 1 | 350 to 350 (median 350) |
| `channel` | category | sales channel: direct, comparison_site, broker | 0.00 | 3 | broker (24,305); direct (8,516); comparison_site (1,745) |
| `payment_method` | category | annual or monthly | 0.00 | 2 | annual (19,288); monthly (15,278) |
| `missed_payments_12m` | integer | premium payments missed in the last 12 months | 0.00 | 7 | 0 to 6 (median 0) |
| `new_business` | boolean | first year with the insurer | 0.00 | 2 | False (29,850); True (4,716) |
| `renewal_count` | integer | number of renewals so far | 0.00 | 17 | 0 to 16 (median 7) |
| `days_quote_to_inception` | integer | days between quote and start of cover | 0.00 | 31 | 0 to 30 (median 13) |
| `discount_applied` | numeric | discount on the premium (share) | 0.00 | 3 | 0 to 0.1 (median 0) |
| `written_premium` | numeric | annual premium charged by the current tariff | 0.00 | 15,791 | 26.62 to 573.35 (median 118.95) |
| `customer_service_contacts` | integer | contacts with customer service during the period | 0.00 | 12 | 0 to 11 (median 0) |
| `renewal_decision` | outcome | what happened at the end of the period (training files only) | 0.00 | 5 | renewed (29,877); mid_term_change (1,895); lapsed_by_customer (1,828); cancelled (936); de |

## `test_motor`

| Column | Type | Meaning | Missing % | Distinct | Values |
|---|---|---|---:|---:|---|
| `row_id` | id | one policy row: a policy over one calendar-year period, or part of one | 0.00 | 90,521 | M00109296 (1); M00109297 (1); M00109298 (1); M00109299 (1); M00109300 (1); M00109301 (1) |
| `policy_id` | id | the policy; constant across its years and mid-term changes | 0.00 | 83,907 | M09357681 (2); M62433702 (2); M67198758 (2); M69784298 (2); M87247390 (2); M44727105 (2) |
| `customer_id` | id | the policyholder (customers table) | 0.00 | 83,907 | 21008507 (2); 84843680 (2); 89338282 (2); 30983815 (2); 84764310 (2); 31597160 (2) |
| `address_id` | id | the insured or garaging address (addresses table) | 0.00 | 81,670 | A5422481 (8); A9579498 (7); A2995018 (7); A0143842 (7); A6517743 (7); A2324483 (7) |
| `period_start` | date | first day on cover in this row | 0.00 | 364 | 2025-01-01 (20,097); 2025-01-07 (4,422); 2025-01-14 (4,418); 2025-01-06 (4,408); 2025-01-0 |
| `period_end` | date | last day on cover in this row (inclusive) | 0.00 | 364 | 2025-12-31 (80,330); 2025-10-10 (44); 2025-01-18 (42); 2025-04-16 (42); 2025-08-30 (41); 2 |
| `exposure` | numeric | years on cover: (period_end - period_start + 1) / 365 | 0.00 | 365 | 0.0027 to 1 (median 0.9781) |
| `driver_age_band` | ordinal | age band of the main driver: 17-24, 25-34, 35-49, 50-64, 65+ | 0.00 | 5 | 25-34 (22,945); 17-24 (22,473); 65+ (21,353); 50-64 (11,942); 35-49 (11,808) |
| `years_licensed` | integer | years since the main driver passed the test | 0.00 | 75 | 0 to 74 (median 13) |
| `ncb_years` | integer | no-claims bonus years (0-15) | 0.00 | 16 | 0 to 15 (median 5) |
| `vehicle_value_group` | ordinal | vehicle value group 1-20 | 0.00 | 20 | 1 to 20 (median 11) |
| `vehicle_age` | integer | vehicle age in years | 12.23 | 25 | 3 to 27 (median 10) |
| `make_model` | category | vehicle make and model code (300 codes) | 0.00 | 300 | MM155 (351); MM279 (348); MM292 (346); MM199 (342); MM018 (340); MM056 (339) |
| `engine_size_cc` | numeric | engine size | 12.16 | 31 | 500 to 3600 (median 2100) |
| `body_type` | category | hatchback, saloon, estate, suv, coupe | 0.00 | 5 | hatchback (39,462); suv (16,570); saloon (16,404); estate (9,418); coupe (8,667) |
| `transmission` | category | manual or automatic | 0.00 | 2 | manual (49,833); automatic (40,688) |
| `seats` | integer | number of seats | 0.00 | 4 | 2 to 7 (median 5) |
| `vehicle_colour` | category | colour | 0.00 | 8 | silver (11,404); black (11,385); red (11,384); blue (11,362); grey (11,314); white (11,283 |
| `fuel_type` | category | petrol, diesel, hybrid, electric | 0.00 | 4 | petrol (48,948); diesel (26,484); hybrid (13,267); electric (1,822) |
| `driver_assist` | boolean | has driver-assistance systems | 0.00 | 2 | False (63,322); True (27,199) |
| `cover_type` | category | comprehensive, third_party_fire_theft, third_party_only | 0.00 | 3 | comprehensive (70,194); third_party_fire_theft (10,214); third_party_only (10,113) |
| `overnight_parking` | category | garage, driveway, street | 12.00 | 3 | driveway (39,965); street (23,768); garage (15,929) |
| `vehicle_use` | category | social, commuting, business | 0.00 | 3 | social (40,801); commuting (40,772); business (8,948) |
| `named_young_driver` | boolean | a young driver is named on the policy | 0.00 | 2 | False (86,323); True (4,198) |
| `voluntary_excess` | integer | excess chosen by the customer: 0, 250, 500 or 1000 | 0.00 | 4 | 0 to 1000 (median 250) |
| `compulsory_excess` | integer | excess set by the insurer | 0.00 | 1 | 250 to 250 (median 250) |
| `channel` | category | sales channel: direct, comparison_site, broker | 0.00 | 3 | broker (63,320); direct (22,671); comparison_site (4,530) |
| `payment_method` | category | annual or monthly | 0.00 | 2 | annual (49,770); monthly (40,751) |
| `missed_payments_12m` | integer | premium payments missed in the last 12 months | 0.00 | 8 | 0 to 7 (median 0) |
| `new_business` | boolean | first year with the insurer | 0.00 | 2 | True (68,823); False (21,698) |
| `renewal_count` | integer | number of renewals so far | 0.00 | 18 | 0 to 17 (median 0) |
| `days_quote_to_inception` | integer | days between quote and start of cover | 0.00 | 31 | 0 to 30 (median 13) |
| `discount_applied` | numeric | discount on the premium (share) | 0.00 | 3 | 0 to 0.1 (median 0) |
| `written_premium` | numeric | annual premium charged by the current tariff | 0.00 | 54,027 | 173.93 to 2415.21 (median 539.5) |
| `customer_service_contacts` | integer | contacts with customer service during the period | 0.00 | 6 | 0 to 5 (median 0) |

## `test_buildings`

| Column | Type | Meaning | Missing % | Distinct | Values |
|---|---|---|---:|---:|---|
| `row_id` | id | one policy row: a policy over one calendar-year period, or part of one | 0.00 | 52,295 | B00199817 (1); B00199818 (1); B00199819 (1); B00199820 (1); B00199821 (1); B00199822 (1) |
| `policy_id` | id | the policy; constant across its years and mid-term changes | 0.00 | 50,300 | B09555667 (2); B00545736 (2); B26539093 (2); B85701518 (2); B92203730 (2); B28036246 (2) |
| `customer_id` | id | the policyholder (customers table) | 0.00 | 50,300 | 08956650 (2); 92605381 (2); 51360512 (2); 35800486 (2); 20631727 (2); 88483618 (2) |
| `address_id` | id | the insured or garaging address (addresses table) | 0.00 | 52,295 | A6546739 (1); A6944531 (1); A7385385 (1); A1371358 (1); A4692371 (1); A1740905 (1) |
| `period_start` | date | first day on cover in this row | 0.00 | 364 | 2025-01-01 (12,103); 2025-01-07 (2,697); 2025-01-08 (2,647); 2025-01-14 (2,632); 2025-01-0 |
| `period_end` | date | last day on cover in this row (inclusive) | 0.00 | 364 | 2025-12-31 (48,146); 2025-11-20 (21); 2025-10-04 (20); 2025-03-16 (20); 2025-08-22 (20); 2 |
| `exposure` | numeric | years on cover: (period_end - period_start + 1) / 365 | 0.00 | 365 | 0.0027 to 1 (median 0.9808) |
| `policyholder_age_band` | ordinal | age band of the policyholder: 17-24, 25-34, 35-49, 50-64, 65+ | 0.00 | 5 | 25-34 (13,214); 17-24 (12,907); 65+ (12,343); 50-64 (6,961); 35-49 (6,870) |
| `sum_insured` | numeric | rebuilding sum insured | 0.00 | 685 | 45000 to 5.68e+08 (median 241000) |
| `construction_type` | category | brick, stone, timber, non_standard | 0.00 | 4 | BRK (36,709); WOD (6,255); MIX (5,214); STN (4,117) |
| `occupancy` | category | owner_occupied or let | 0.00 | 2 | owner_occupied (47,055); let (5,240) |
| `days_unoccupied` | numeric | days the property is unoccupied in a year | 11.93 | 104 | 0 to 119 (median 7) |
| `smoke_alarms` | boolean | smoke alarms fitted | 12.12 | 2 | True (39,155); False (6,799) |
| `accidental_damage_cover` | boolean | accidental damage cover bought | 0.00 | 2 | False (31,431); True (20,864) |
| `mortgage` | boolean | property is mortgaged | 0.00 | 2 | True (28,501); False (23,794) |
| `voluntary_excess` | integer | excess chosen by the customer: 0, 250, 500 or 1000 | 0.00 | 4 | 0 to 1000 (median 250) |
| `compulsory_excess` | integer | excess set by the insurer | 0.00 | 1 | 350 to 350 (median 350) |
| `channel` | category | sales channel: direct, comparison_site, broker | 0.00 | 3 | broker (36,543); direct (13,128); comparison_site (2,624) |
| `payment_method` | category | annual or monthly | 0.00 | 2 | annual (28,756); monthly (23,539) |
| `missed_payments_12m` | integer | premium payments missed in the last 12 months | 0.00 | 7 | 0 to 7 (median 0) |
| `new_business` | boolean | first year with the insurer | 0.00 | 2 | True (39,716); False (12,579) |
| `renewal_count` | integer | number of renewals so far | 0.00 | 18 | 0 to 17 (median 0) |
| `days_quote_to_inception` | integer | days between quote and start of cover | 0.00 | 31 | 0 to 30 (median 13) |
| `discount_applied` | numeric | discount on the premium (share) | 0.00 | 3 | 0 to 0.1 (median 0) |
| `written_premium` | numeric | annual premium charged by the current tariff | 0.00 | 30,197 | 76.04 to 4717.45 (median 256.77) |
| `customer_service_contacts` | integer | contacts with customer service during the period | 0.00 | 7 | 0 to 6 (median 0) |

## `test_contents`

| Column | Type | Meaning | Missing % | Distinct | Values |
|---|---|---|---:|---:|---|
| `row_id` | id | one policy row: a policy over one calendar-year period, or part of one | 0.00 | 66,247 | C00252112 (1); C00252113 (1); C00252114 (1); C00252115 (1); C00252116 (1); C00252117 (1) |
| `policy_id` | id | the policy; constant across its years and mid-term changes | 0.00 | 62,636 | C17588925 (2); C18897135 (2); C28377895 (2); C41386338 (2); C10252638 (2); C41741411 (2) |
| `customer_id` | id | the policyholder (customers table) | 0.00 | 62,636 | 55462563 (2); 92605381 (2); 18786089 (2); 21845187 (2); 51360512 (2); 33673917 (2) |
| `address_id` | id | the insured or garaging address (addresses table) | 0.00 | 64,690 | A1176467 (4); A8938958 (4); A2084291 (4); A2547095 (4); A2290780 (4); A0423247 (4) |
| `period_start` | date | first day on cover in this row | 0.00 | 364 | 2025-01-01 (14,922); 2025-01-07 (3,381); 2025-01-08 (3,365); 2025-01-13 (3,295); 2025-01-1 |
| `period_end` | date | last day on cover in this row (inclusive) | 0.00 | 364 | 2025-12-31 (59,938); 2025-02-18 (31); 2025-05-02 (30); 2025-07-16 (28); 2025-10-24 (28); 2 |
| `exposure` | numeric | years on cover: (period_end - period_start + 1) / 365 | 0.00 | 365 | 0.0027 to 1 (median 0.9781) |
| `policyholder_age_band` | ordinal | age band of the policyholder: 17-24, 25-34, 35-49, 50-64, 65+ | 0.00 | 5 | 25-34 (16,805); 17-24 (16,346); 65+ (15,679); 50-64 (8,774); 35-49 (8,643) |
| `sum_insured` | numeric | contents sum insured | 0.00 | 1,480 | 4900 to 9.81e+07 (median 36300) |
| `tenure` | category | owner or renter | 0.00 | 2 | owner (37,230); renter (29,017) |
| `security_level` | ordinal | standard, locks, alarm_and_locks | 12.15 | 3 | standard (20,428); alarm_and_locks (20,279); locks (17,490) |
| `high_value_items` | integer | declared high-value items (0-5) | 12.01 | 6 | 0 to 5 (median 0) |
| `accidental_damage_cover` | boolean | accidental damage cover bought | 0.00 | 2 | False (36,209); True (30,038) |
| `works_from_home` | boolean | policyholder works from home | 0.00 | 2 | False (49,691); True (16,556) |
| `lodgers` | boolean | lodgers in the home | 0.00 | 2 | False (63,041); True (3,206) |
| `bicycle_cover` | boolean | bicycle cover bought | 0.00 | 2 | False (56,152); True (10,095) |
| `voluntary_excess` | integer | excess chosen by the customer: 0, 250, 500 or 1000 | 0.00 | 4 | 0 to 1000 (median 0) |
| `compulsory_excess` | integer | excess set by the insurer | 0.00 | 1 | 350 to 350 (median 350) |
| `channel` | category | sales channel: direct, comparison_site, broker | 0.00 | 3 | broker (46,313); direct (16,630); comparison_site (3,304) |
| `payment_method` | category | annual or monthly | 0.00 | 2 | annual (36,427); monthly (29,820) |
| `missed_payments_12m` | integer | premium payments missed in the last 12 months | 0.00 | 6 | 0 to 5 (median 0) |
| `new_business` | boolean | first year with the insurer | 0.00 | 2 | True (50,441); False (15,806) |
| `renewal_count` | integer | number of renewals so far | 0.00 | 18 | 0 to 17 (median 0) |
| `days_quote_to_inception` | integer | days between quote and start of cover | 0.00 | 31 | 0 to 30 (median 13) |
| `discount_applied` | numeric | discount on the premium (share) | 0.00 | 3 | 0 to 0.1 (median 0) |
| `written_premium` | numeric | annual premium charged by the current tariff | 0.00 | 20,608 | 29.17 to 665.49 (median 127.03) |
| `customer_service_contacts` | integer | contacts with customer service during the period | 0.00 | 6 | 0 to 5 (median 0) |

## `sample_submission`

| Column | Type | Meaning | Missing % | Distinct | Values |
|---|---|---|---:|---:|---|
| `row_id` | id | row to predict (the 2025 test rows, in submission order) | 0.00 | 209,063 | M00109296 (1); M00109297 (1); M00109298 (1); M00109299 (1); M00109300 (1); M00109301 (1) |
| `prediction` | numeric | expected ultimate claim cost of the row | 0.00 | 1 | 0 to 0 (median 0) |

## `expenses`

| Column | Type | Meaning | Missing % | Distinct | Values |
|---|---|---|---:|---:|---|
| `item` | category | expense item | 0.00 | 8 | fixed_expense_per_policy (1); variable_expense (1); commission_direct (1); commission_comp |
| `developed` | numeric | value in the developed market | 0.00 | 7 | 0 to 45 (median 0.11) |
| `emerging` | numeric | value in this (emerging) market | 12.50 | 6 | 0 to 85 (median 0.12) |
| `unit` | category | CU per policy, or a share of premium | 0.00 | 3 | share of premium (6); CU (1); share of property risk premium (1) |

## `reference_policies`

| Column | Type | Meaning | Missing % | Distinct | Values |
|---|---|---|---:|---:|---|
| `row_id` | id | one policy row: a policy over one calendar-year period, or part of one | 0.00 | 20 | M00161473 (1); M00176777 (1); M00185311 (1); M00167496 (1); M00129413 (1); M00118569 (1) |
| `policy_id` | id | the policy; constant across its years and mid-term changes | 0.00 | 20 | M92909747 (1); M42023395 (1); M83715550 (1); M61457154 (1); M41089337 (1); M75811658 (1) |
| `customer_id` | id | the policyholder (customers table) | 0.00 | 20 | 88632978 (1); 30215970 (1); 37533851 (1); 48843509 (1); 05068935 (1); 81575987 (1) |
| `address_id` | id | the insured or garaging address (addresses table) | 0.00 | 20 | A3687131 (1); A8460609 (1); A1367449 (1); A6557449 (1); A0759646 (1); A6924348 (1) |
| `period_start` | date | first day on cover in this row | 0.00 | 12 | 2025-01-04 (3); 2025-01-11 (3); 2025-01-13 (3); 2025-01-14 (2); 2025-01-01 (2); 2025-01-07 |
| `period_end` | date | last day on cover in this row (inclusive) | 0.00 | 3 | 2025-12-31 (18); 2025-12-09 (1); 2025-02-06 (1) |
| `exposure` | numeric | years on cover: (period_end - period_start + 1) / 365 | 0.00 | 14 | 0.0685 to 1 (median 0.9726) |
| `driver_age_band` | ordinal | age band of the main driver: 17-24, 25-34, 35-49, 50-64, 65+ | 65.00 | 3 | 17-24 (3); 25-34 (3); 35-49 (1) |
| `years_licensed` | integer | years since the main driver passed the test | 65.00 | 6 | 0 to 29 (median 5) |
| `ncb_years` | integer | no-claims bonus years (0-15) | 65.00 | 5 | 0 to 10 (median 5) |
| `vehicle_value_group` | ordinal | vehicle value group 1-20 | 65.00 | 6 | 5 to 12 (median 8) |
| `vehicle_age` | integer | vehicle age in years | 75.00 | 4 | 4 to 15 (median 11) |
| `make_model` | category | vehicle make and model code (300 codes) | 65.00 | 7 | MM029 (1); MM181 (1); MM012 (1); MM273 (1); MM186 (1); MM278 (1) |
| `engine_size_cc` | numeric | engine size | 65.00 | 6 | 1300 to 2100 (median 1700) |
| `body_type` | category | hatchback, saloon, estate, suv, coupe | 65.00 | 5 | hatchback (2); suv (2); coupe (1); saloon (1); estate (1) |
| `transmission` | category | manual or automatic | 65.00 | 2 | manual (4); automatic (3) |
| `seats` | integer | number of seats | 65.00 | 2 | 4 to 5 (median 5) |
| `vehicle_colour` | category | colour | 65.00 | 4 | silver (4); red (1); white (1); green (1) |
| `fuel_type` | category | petrol, diesel, hybrid, electric | 65.00 | 2 | petrol (5); diesel (2) |
| `driver_assist` | boolean | has driver-assistance systems | 65.00 | 2 | True (4); False (3) |
| `cover_type` | category | comprehensive, third_party_fire_theft, third_party_only | 65.00 | 3 | comprehensive (4); third_party_fire_theft (2); third_party_only (1) |
| `overnight_parking` | category | garage, driveway, street | 70.00 | 2 | garage (4); driveway (2) |
| `vehicle_use` | category | social, commuting, business | 65.00 | 3 | social (3); commuting (2); business (2) |
| `named_young_driver` | boolean | a young driver is named on the policy | 65.00 | 1 | False (7) |
| `voluntary_excess` | integer | excess chosen by the customer: 0, 250, 500 or 1000 | 0.00 | 3 | 0 to 1000 (median 250) |
| `compulsory_excess` | integer | excess set by the insurer | 0.00 | 2 | 250 to 350 (median 350) |
| `channel` | category | sales channel: direct, comparison_site, broker | 0.00 | 2 | broker (17); direct (3) |
| `payment_method` | category | annual or monthly | 0.00 | 2 | monthly (10); annual (10) |
| `missed_payments_12m` | integer | premium payments missed in the last 12 months | 0.00 | 1 | 0 to 0 (median 0) |
| `new_business` | boolean | first year with the insurer | 0.00 | 2 | True (18); False (2) |
| `renewal_count` | integer | number of renewals so far | 0.00 | 3 | 0 to 14 (median 0) |
| `days_quote_to_inception` | integer | days between quote and start of cover | 0.00 | 19 | 0 to 29 (median 18.5) |
| `discount_applied` | numeric | discount on the premium (share) | 0.00 | 2 | 0 to 0.05 (median 0) |
| `written_premium` | numeric | annual premium charged by the current tariff | 0.00 | 20 | 83.18 to 1131.78 (median 270.835) |
| `customer_service_contacts` | integer | contacts with customer service during the period | 0.00 | 2 | 0 to 1 (median 0) |
| `line` | category | line of the reference policy | 0.00 | 3 | motor (7); buildings (7); contents (6) |
| `policyholder_age_band` | ordinal | age band of the policyholder: 17-24, 25-34, 35-49, 50-64, 65+ | 35.00 | 5 | 50-64 (4); 65+ (3); 35-49 (2); 17-24 (2); 25-34 (2) |
| `sum_insured` | numeric | contents sum insured | 35.00 | 12 | 22100 to 378000 (median 198000) |
| `construction_type` | category | brick, stone, timber, non_standard | 65.00 | 3 | BRK (4); MIX (2); WOD (1) |
| `occupancy` | category | owner_occupied or let | 65.00 | 2 | owner_occupied (6); let (1) |
| `days_unoccupied` | numeric | days the property is unoccupied in a year | 70.00 | 6 | 3 to 9 (median 5.5) |
| `smoke_alarms` | boolean | smoke alarms fitted | 70.00 | 2 | True (5); False (1) |
| `accidental_damage_cover` | boolean | accidental damage cover bought | 35.00 | 2 | False (10); True (3) |
| `mortgage` | boolean | property is mortgaged | 65.00 | 2 | False (4); True (3) |
| `tenure` | category | owner or renter | 70.00 | 2 | owner (4); renter (2) |
| `security_level` | ordinal | standard, locks, alarm_and_locks | 70.00 | 3 | alarm_and_locks (3); standard (2); locks (1) |
| `high_value_items` | integer | declared high-value items (0-5) | 75.00 | 4 | 0 to 5 (median 3) |
| `works_from_home` | boolean | policyholder works from home | 70.00 | 2 | False (4); True (2) |
| `lodgers` | boolean | lodgers in the home | 70.00 | 1 | False (6) |
| `bicycle_cover` | boolean | bicycle cover bought | 70.00 | 1 | False (6) |
