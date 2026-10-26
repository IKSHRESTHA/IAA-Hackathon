# Data dictionary - 1st round (developed market)

Files of the case study *IAA AI-Assisted Hackathon 2026 - Part 1*, as read by `01_data_reading.py`.

## `customers`

| Column | Type | Meaning | Missing % | Distinct | Values |
|---|---|---|---:|---:|---|
| `customer_id` | id | customer identifier (8 digits) | 0.00 | 545,240 | 88505748 (1); 15394897 (1); 23374905 (1); 74127862 (1); 67002274 (1); 05180324 (1) |
| `surname` | text | surname (personal data) | 0.00 | 600 | Surname143 (988); Surname476 (987); Surname093 (985); Surname035 (980); Surname257 (978);  |
| `date_of_birth` | date | date of birth | 0.00 | 27,380 | 2003-12-29 (82); 1995-03-22 (58); 1999-06-06 (58); 2000-05-04 (56); 2001-09-22 (55); 1999- |
| `occupation` | category | occupation | 2.47 | 20 | Retired (98,624); Student (40,063); Hairdresser (23,869); Care worker (23,383); Nurse (23, |
| `marital_status` | category | single, married, divorced, widowed | 2.52 | 4 | married (248,383); single (217,752); divorced (43,816); widowed (21,527) |
| `household_size` | integer | people in the household | 2.47 | 7 | 1 to 7 (median 2) |
| `credit_band` | ordinal | credit score band, A (best) to E | 0.00 | 5 | B (162,890); A (136,779); C (136,103); D (65,627); E (43,841) |
| `prior_claims_5y` | integer | claims declared for the last 5 years (any insurer) | 0.00 | 7 | 0 to 7 (median 0) |
| `licence_points` | integer | penalty points on the driving licence | 0.00 | 4 | 0 to 9 (median 0) |
| `years_at_address` | integer | years lived at the current address | 2.51 | 55 | 0 to 57 (median 6) |
| `customer_since_year` | integer | first year with the insurer | 0.00 | 20 | 2006 to 2025 (median 2020) |
| `email_domain` | category | e-mail provider | 0.00 | 5 | inboxly (109,445); netbox (109,181); posthub (109,048); other (108,787); mailone (108,779) |
| `marketing_consent` | boolean | agreed to receive marketing | 0.00 | 2 | True (300,209); False (245,031) |
| `document_preference` | category | email, post or app | 0.00 | 3 | email (328,154); app (135,799); post (81,287) |
| `previous_insurer` | category | insurer before joining | 0.00 | 40 | INS08 (13,866); INS28 (13,865); INS34 (13,831); INS11 (13,812); INS01 (13,775); INS13 (13, |
| `campaign_code` | category | marketing campaign that acquired the customer (missing: none) | 51.19 | 15 | CMP01 (18,027); CMP03 (17,844); CMP05 (17,816); CMP06 (17,810); CMP11 (17,761); CMP08 (17, |
| `branch_id` | category | servicing branch | 0.00 | 25 | 22 (22,181); 18 (22,137); 6 (21,984); 4 (21,952); 16 (21,942); 5 (21,927) |
| `marketing_segment` | category | segment assigned by marketing | 0.00 | 12 | young_affluent (83,747); midlife_affluent (73,109); senior_affluent (71,448); mature_afflu |

## `addresses`

| Column | Type | Meaning | Missing % | Distinct | Values |
|---|---|---|---:|---:|---|
| `address_id` | id | address identifier | 0.00 | 575,983 | A1980663 (1); A7251041 (1); A4921189 (1); A9620763 (1); A7357961 (1); A7322730 (1) |
| `region` | category | region R01-R12 | 0.00 | 12 | R01 (93,090); R12 (57,416); R03 (56,128); R10 (54,250); R11 (49,046); R05 (45,865) |
| `urban_density` | category | rural, suburban, urban, dense_urban | 0.00 | 4 | suburban (258,647); urban (144,467); rural (114,812); dense_urban (58,057) |
| `flood_zone` | ordinal | flood zone 1 (lowest risk) to 5 | 2.92 | 5 | 1 to 5 (median 1) |
| `distance_to_river_km` | numeric | distance to the nearest river | 2.50 | 2,093 | 0.01 to 30 (median 2.37) |
| `elevation_m` | numeric | elevation above sea level | 0.00 | 949 | 0 to 196.9 (median 8.1) |
| `soil_type` | category | clay, sand, loam, chalk, rock | 2.50 | 5 | clay (233,518); rock (82,487); sand (81,985); loam (81,873); chalk (81,728) |
| `year_built` | integer | year the property was built | 0.00 | 160 | 1862 to 2021 (median 1965) |
| `property_type` | category | detached, semi_detached, terraced, flat | 0.00 | 4 | semi_detached (172,771); terraced (144,140); detached (144,000); flat (115,072) |
| `roof_type` | category | pitched or flat | 2.50 | 2 | pitched (505,193); flat (56,389) |
| `basement` | boolean | has a basement | 0.00 | 2 | False (489,427); True (86,556) |
| `trees_within_5m` | boolean | trees within 5 metres of the building | 0.00 | 2 | False (506,138); True (69,845) |
| `listed_building` | boolean | protected (listed) building | 0.00 | 2 | False (558,676); True (17,307) |
| `bedrooms` | integer | number of bedrooms | 0.00 | 7 | 1 to 7 (median 3) |
| `energy_rating` | ordinal | energy rating A (best) to G | 2.51 | 7 | C (113,043); B (110,780); D (95,889); A (89,966); E (65,210); F (45,156) |
| `solar_panels` | boolean | has solar panels | 0.00 | 2 | False (518,668); True (57,315) |

## `weather`

| Column | Type | Meaning | Missing % | Distinct | Values |
|---|---|---|---:|---:|---|
| `address_id` | id | address identifier | 0.00 | 575,983 | A1980663 (5); A7251041 (5); A4921189 (5); A9620763 (5); A7357961 (5); A7322730 (5) |
| `year` | integer | calendar year (2021-2025) | 0.00 | 5 | 2021 to 2025 (median 2023) |
| `hail_days` | numeric | days with hail | 0.00 | 42 | 0 to 48 (median 6) |
| `wind_exposure_index` | numeric | wind exposure index 0-10 | 0.00 | 101 | 0 to 10 (median 4.7) |
| `freezing_days` | numeric | days below freezing | 0.00 | 109 | 0 to 110 (median 36) |
| `dry_summer_index` | numeric | summer dryness index 0-1 | 0.00 | 1,001 | 0 to 1 (median 0.542) |
| `wildfire_index` | numeric | wildfire index 0-1 | 0.00 | 641 | 0 to 0.669 (median 0.147) |
| `rainfall_mm` | numeric | annual rainfall | 0.00 | 1,967 | 400 to 2841 (median 1012) |
| `mean_temperature_c` | numeric | annual mean temperature | 0.00 | 162 | 3.2 to 19.3 (median 13.2) |

## `claims`

| Column | Type | Meaning | Missing % | Distinct | Values |
|---|---|---|---:|---:|---|
| `claim_id` | id | claim identifier | 0.00 | 163,150 | CL00485349 (1); CL88548306 (1); CL99833897 (1); CL96654761 (1); CL80491193 (1); CL27009613 |
| `policy_id` | id | policy that made the claim | 0.00 | 125,103 | M99747073 (21); M80065512 (15); M83947458 (15); M29469138 (13); M90444461 (13); M44905252  |
| `line` | category | motor, buildings or contents | 0.00 | 3 | motor (112,606); buildings (26,442); contents (24,102) |
| `peril` | category | cause of loss | 0.00 | 11 | own_damage (48,964); third_party_prop (33,826); glass_hail (15,732); escape_of_water (15,5 |
| `loss_date` | date | date of the loss | 0.00 | 1,473 | 2020-12-01 (252); 2022-12-24 (167); 2022-12-04 (154); 2023-12-30 (150); 2023-02-27 (150);  |
| `report_date` | date | date the claim was reported | 0.00 | 1,459 | 2022-12-29 (154); 2023-10-02 (153); 2022-12-05 (152); 2023-01-16 (152); 2022-11-10 (148);  |
| `status` | category | closed, open or repudiated (declined) | 0.00 | 3 | closed (151,292); open (10,147); repudiated (1,711) |
| `paid_to_date` | numeric | paid so far, net of excess | 0.00 | 139,258 | 0 to 1.11887e+06 (median 1702.4) |
| `incurred_to_date` | numeric | paid + case reserve, net of excess | 0.00 | 141,111 | 0 to 1.18943e+07 (median 1859.76) |
| `recoveries` | numeric | amounts recovered (salvage, third parties) | 0.00 | 14,464 | 0 to 22328.9 (median 0) |
| `repairer_id` | id | repairer (motor) | 49.26 | 1,000 | RP0733 (738); RP0901 (702); RP0417 (699); RP0404 (112); RP0664 (107); RP0650 (105) |
| `third_party_id` | id | third party (liability perils) | 74.62 | 41,163 | TP9802665 (3); TP0630869 (2); TP4280612 (2); TP1302154 (2); TP3893336 (2); TP0474696 (2) |

## `policies_motor`

| Column | Type | Meaning | Missing % | Distinct | Values |
|---|---|---|---:|---:|---|
| `row_id` | id | one policy row: a policy over one calendar-year period, or part of one | 0.00 | 979,968 | M00000000 (1); M00000001 (1); M00000002 (1); M00000003 (1); M00000004 (1); M00000005 (1) |
| `policy_id` | id | the policy; constant across its years and mid-term changes | 0.00 | 308,109 | M23308937 (8); M21713701 (8); M20580780 (8); M78923948 (8); M98967090 (8); M45195028 (7) |
| `customer_id` | id | the policyholder (customers table) | 0.00 | 308,109 | 50531880 (8); 812070 (8); 49442630 (8); 27719981 (8); 43852369 (8); 96193921 (7) |
| `address_id` | id | the insured or garaging address (addresses table) | 0.00 | 309,499 | A3513685 (24); A1373405 (23); A4266658 (23); A2901539 (22); A2831597 (21); A4954957 (21) |
| `period_start` | date | first day on cover in this row | 0.00 | 1,456 | 2022-01-01 (199,977); 2024-01-01 (199,752); 2023-01-01 (199,609); 2021-01-01 (176,035); 20 |
| `period_end` | date | last day on cover in this row (inclusive) | 0.00 | 1,456 | 2024-12-30 (200,063); 2021-12-31 (199,977); 2023-12-31 (199,752); 2022-12-31 (199,609); 20 |
| `exposure` | numeric | years on cover: (period_end - period_start + 1) / 365 | 0.00 | 956 | 0.0027 to 365 (median 1) |
| `driver_age` | integer | age of the main driver at the start of the row | 0.00 | 75 | 17 to 91 (median 50) |
| `years_licensed` | integer | years since the main driver passed the test | 0.00 | 88 | 0 to 87 (median 28) |
| `ncb_years` | integer | no-claims bonus years (0-15) | 0.00 | 16 | 0 to 15 (median 7) |
| `vehicle_value_group` | ordinal | vehicle value group 1-20 | 0.00 | 20 | 1 to 20 (median 11) |
| `vehicle_age` | integer | vehicle age in years | 2.49 | 29 | 0 to 28 (median 7) |
| `make_model` | category | vehicle make and model code (300 codes) | 0.00 | 300 | MM090 (3,629); MM255 (3,609); MM013 (3,526); MM031 (3,514); MM204 (3,474); MM005 (3,473) |
| `engine_size_cc` | numeric | engine size | 2.52 | 31 | 600 to 3600 (median 2100) |
| `body_type` | category | hatchback, saloon, estate, suv, coupe | 0.00 | 5 | hatchback (371,142); saloon (206,914); suv (206,569); estate (120,815); coupe (74,528) |
| `transmission` | category | manual or automatic | 0.00 | 2 | manual (537,961); automatic (442,007) |
| `seats` | integer | number of seats | 0.00 | 4 | 2 to 7 (median 5) |
| `vehicle_colour` | category | colour | 0.00 | 8 | grey (123,232); silver (123,215); black (122,510); red (122,501); green (122,375); blue (1 |
| `fuel_type` | category | petrol, diesel, hybrid, electric | 0.00 | 4 | petrol (495,571); diesel (270,454); hybrid (135,075); electric (78,868) |
| `driver_assist` | boolean | has driver-assistance systems | 0.00 | 2 | False (684,782); True (295,186) |
| `cover_type` | category | comprehensive, third_party_fire_theft, third_party_only | 0.00 | 3 | comprehensive (802,694); third_party_fire_theft (89,070); third_party_only (88,204) |
| `annual_mileage` | numeric | declared annual mileage | 2.52 | 497 | 0 to 86000 (median 10000) |
| `overnight_parking` | category | garage, driveway, street | 2.49 | 3 | driveway (478,840); street (286,359); garage (190,398) |
| `vehicle_use` | category | social, commuting, business | 0.00 | 3 | social (442,594); commuting (440,538); business (96,836) |
| `named_young_driver` | boolean | a young driver is named on the policy | 0.00 | 2 | False (927,622); True (52,346) |
| `telematics_opt_in` | boolean | telematics box fitted | 0.00 | 2 | False (858,223); True (121,745) |
| `telematics_score` | numeric | telematics driving score 1-100 (opted-in only) | 87.58 | 98 | 1 to 100 (median 60) |
| `voluntary_excess` | integer | excess chosen by the customer: 0, 250, 500 or 1000 | 0.00 | 4 | 0 to 1000 (median 250) |
| `compulsory_excess` | integer | excess set by the insurer | 0.00 | 1 | 250 to 250 (median 250) |
| `channel` | category | sales channel: direct, comparison_site, broker | 0.00 | 3 | direct (392,737); comparison_site (342,826); broker (244,405) |
| `payment_method` | category | annual or monthly | 0.00 | 2 | annual (537,917); monthly (442,051) |
| `missed_payments_12m` | integer | premium payments missed in the last 12 months | 0.00 | 8 | 0 to 7 (median 0) |
| `new_business` | boolean | first year with the insurer | 0.00 | 2 | False (837,446); True (142,522) |
| `renewal_count` | integer | number of renewals so far | 0.00 | 19 | 0 to 18 (median 6) |
| `days_quote_to_inception` | integer | days between quote and start of cover | 0.00 | 31 | 0 to 30 (median 13) |
| `discount_applied` | numeric | discount on the premium (share) | 0.00 | 3 | 0 to 0.1 (median 0) |
| `written_premium` | numeric | annual premium charged by the current tariff | 0.00 | 122,304 | 159.12 to 2406.52 (median 479.54) |
| `customer_service_contacts` | integer | contacts with customer service during the period | 0.00 | 31 | 0 to 36 (median 0) |
| `renewal_decision` | outcome | what happened at the end of the period (training files only) | 0.00 | 5 | renewed (793,414); mid_term_change (72,521); lapsed_by_customer (71,662); cancelled (35,97 |

## `policies_buildings`

| Column | Type | Meaning | Missing % | Distinct | Values |
|---|---|---|---:|---:|---|
| `row_id` | id | one policy row: a policy over one calendar-year period, or part of one | 0.00 | 707,726 | B00216008 (1); B00216009 (1); B00216010 (1); B00216011 (1); B00216012 (1); B00216013 (1) |
| `policy_id` | id | the policy; constant across its years and mid-term changes | 0.00 | 230,921 | B32188869 (7); B99627375 (7); B21986025 (7); B77740398 (7); B13883748 (7); B65151617 (7) |
| `customer_id` | id | the policyholder (customers table) | 0.00 | 230,921 | 11881596 (7); 66558535 (7); 13746721 (7); 95266532 (7); 52490109 (7); 07470822 (7) |
| `address_id` | id | the insured or garaging address (addresses table) | 0.00 | 258,068 | A9620763 (4); A2736290 (4); A2727550 (4); A1903092 (4); A1684027 (4); A0873922 (4) |
| `period_start` | date | first day on cover in this row | 0.00 | 1,456 | 2024-01-01 (149,990); 2023-01-01 (149,887); 2022-01-01 (149,781); 2021-01-01 (131,761); 20 |
| `period_end` | date | last day on cover in this row (inclusive) | 0.00 | 1,456 | 2024-12-30 (150,111); 2023-12-31 (149,990); 2022-12-31 (149,887); 2021-12-31 (149,781); 20 |
| `exposure` | numeric | years on cover: (period_end - period_start + 1) / 365 | 0.00 | 870 | 0.0027 to 365 (median 1) |
| `policyholder_age` | integer | age of the policyholder at the start of the row | 0.00 | 75 | 17 to 91 (median 49) |
| `sum_insured` | numeric | rebuilding sum insured | 0.00 | 1,135 | 36000 to 5.94e+08 (median 223000) |
| `construction_type` | category | brick, stone, timber, non_standard | 0.00 | 11 | Brick (456,072); non_standard (68,036); stone (54,703); W (42,319); T (38,937); brick (24, |
| `occupancy` | category | owner_occupied or let | 0.00 | 2 | owner_occupied (637,341); let (70,385) |
| `days_unoccupied` | numeric | days the property is unoccupied in a year | 2.48 | 104 | 0 to 119 (median 7) |
| `smoke_alarms` | boolean | smoke alarms fitted | 2.52 | 2 | True (586,178); False (103,681) |
| `accidental_damage_cover` | boolean | accidental damage cover bought | 0.00 | 2 | False (425,155); True (282,571) |
| `mortgage` | boolean | property is mortgaged | 0.00 | 2 | True (390,554); False (317,172) |
| `voluntary_excess` | integer | excess chosen by the customer: 0, 250, 500 or 1000 | 0.00 | 4 | 0 to 1000 (median 0) |
| `compulsory_excess` | integer | excess set by the insurer | 0.00 | 1 | 350 to 350 (median 350) |
| `channel` | category | sales channel: direct, comparison_site, broker | 0.00 | 3 | direct (282,617); comparison_site (247,706); broker (177,403) |
| `payment_method` | category | annual or monthly | 0.00 | 2 | annual (388,570); monthly (319,156) |
| `missed_payments_12m` | integer | premium payments missed in the last 12 months | 0.00 | 8 | 0 to 7 (median 0) |
| `new_business` | boolean | first year with the insurer | 0.00 | 2 | False (604,562); True (103,164) |
| `renewal_count` | integer | number of renewals so far | 0.00 | 19 | 0 to 18 (median 6) |
| `days_quote_to_inception` | integer | days between quote and start of cover | 0.00 | 31 | 0 to 30 (median 13) |
| `discount_applied` | numeric | discount on the premium (share) | 0.00 | 3 | 0 to 0.1 (median 0) |
| `written_premium` | numeric | annual premium charged by the current tariff | 0.00 | 104,995 | 45.43 to 5076.06 (median 216.66) |
| `customer_service_contacts` | integer | contacts with customer service during the period | 0.00 | 18 | 0 to 18 (median 0) |
| `renewal_decision` | outcome | what happened at the end of the period (training files only) | 0.00 | 5 | renewed (596,186); lapsed_by_customer (53,633); mid_term_change (27,147); cancelled (26,91 |

## `policies_contents`

| Column | Type | Meaning | Missing % | Distinct | Values |
|---|---|---|---:|---:|---|
| `row_id` | id | one policy row: a policy over one calendar-year period, or part of one | 0.00 | 721,617 | C00371737 (1); C00371738 (1); C00371739 (1); C00371740 (1); C00371741 (1); C00371742 (1) |
| `policy_id` | id | the policy; constant across its years and mid-term changes | 0.00 | 231,255 | C98095985 (7); C93949159 (7); C08361985 (7); C96996619 (7); C20581773 (7); C34960476 (7) |
| `customer_id` | id | the policyholder (customers table) | 0.00 | 231,255 | 62246096 (7); 12036901 (7); 66558535 (7); 48493564 (7); 11295347 (7); 44250876 (7) |
| `address_id` | id | the insured or garaging address (addresses table) | 0.00 | 257,529 | A4483201 (16); A2446797 (16); A1077535 (16); A5211036 (15); A7343631 (15); A4824100 (14) |
| `period_start` | date | first day on cover in this row | 0.00 | 1,456 | 2024-01-01 (149,997); 2022-01-01 (149,949); 2023-01-01 (149,935); 2021-01-01 (131,915); 20 |
| `period_end` | date | last day on cover in this row (inclusive) | 0.00 | 1,456 | 2024-12-30 (150,071); 2023-12-31 (149,997); 2021-12-31 (149,949); 2022-12-31 (149,935); 20 |
| `exposure` | numeric | years on cover: (period_end - period_start + 1) / 365 | 0.00 | 892 | 0.0027 to 365 (median 1) |
| `policyholder_age` | integer | age of the policyholder at the start of the row | 0.00 | 75 | 17 to 91 (median 49) |
| `sum_insured` | numeric | contents sum insured | 0.00 | 2,569 | 3800 to 1.526e+08 (median 39000) |
| `tenure` | category | owner or renter | 0.00 | 2 | owner (505,016); renter (216,601) |
| `security_level` | ordinal | standard, locks, alarm_and_locks | 2.51 | 3 | standard (246,824); alarm_and_locks (245,675); locks (211,031) |
| `high_value_items` | integer | declared high-value items (0-5) | 2.51 | 6 | 0 to 5 (median 1) |
| `accidental_damage_cover` | boolean | accidental damage cover bought | 0.00 | 2 | False (397,233); True (324,384) |
| `works_from_home` | boolean | policyholder works from home | 0.00 | 2 | False (540,692); True (180,925) |
| `lodgers` | boolean | lodgers in the home | 0.00 | 2 | False (685,681); True (35,936) |
| `bicycle_cover` | boolean | bicycle cover bought | 0.00 | 2 | False (613,803); True (107,814) |
| `voluntary_excess` | integer | excess chosen by the customer: 0, 250, 500 or 1000 | 0.00 | 4 | 0 to 1000 (median 250) |
| `compulsory_excess` | integer | excess set by the insurer | 0.00 | 1 | 350 to 350 (median 350) |
| `channel` | category | sales channel: direct, comparison_site, broker | 0.00 | 3 | direct (287,395); comparison_site (252,879); broker (181,343) |
| `payment_method` | category | annual or monthly | 0.00 | 2 | annual (396,210); monthly (325,407) |
| `missed_payments_12m` | integer | premium payments missed in the last 12 months | 0.00 | 8 | 0 to 7 (median 0) |
| `new_business` | boolean | first year with the insurer | 0.00 | 2 | False (616,407); True (105,210) |
| `renewal_count` | integer | number of renewals so far | 0.00 | 19 | 0 to 18 (median 6) |
| `days_quote_to_inception` | integer | days between quote and start of cover | 0.00 | 31 | 0 to 30 (median 13) |
| `discount_applied` | numeric | discount on the premium (share) | 0.00 | 3 | 0 to 0.1 (median 0) |
| `written_premium` | numeric | annual premium charged by the current tariff | 0.00 | 31,964 | 20.27 to 865.95 (median 127.48) |
| `customer_service_contacts` | integer | contacts with customer service during the period | 0.00 | 18 | 0 to 18 (median 0) |
| `renewal_decision` | outcome | what happened at the end of the period (training files only) | 0.00 | 5 | renewed (599,110); lapsed_by_customer (54,087); mid_term_change (40,481); cancelled (27,04 |

## `test_motor`

| Column | Type | Meaning | Missing % | Distinct | Values |
|---|---|---|---:|---:|---|
| `row_id` | id | one policy row: a policy over one calendar-year period, or part of one | 0.00 | 254,986 | M02409311 (1); M02409312 (1); M02409313 (1); M02409314 (1); M02409315 (1); M02409316 (1) |
| `policy_id` | id | the policy; constant across its years and mid-term changes | 0.00 | 236,008 | M83508481 (2); M93608866 (2); M95437242 (2); M76129970 (2); M81788884 (2); M91092398 (2) |
| `customer_id` | id | the policyholder (customers table) | 0.00 | 236,008 | 11502638 (2); 83544627 (2); 57479117 (2); 32071776 (2); 23099857 (2); 13409308 (2) |
| `address_id` | id | the insured or garaging address (addresses table) | 0.00 | 231,032 | A9356786 (8); A1373405 (7); A4266658 (7); A0884055 (7); A4954957 (7); A9024570 (7) |
| `period_start` | date | first day on cover in this row | 0.00 | 364 | 2025-01-01 (200,063); 2025-01-10 (202); 2025-12-30 (201); 2025-12-23 (198); 2025-01-07 (19 |
| `period_end` | date | last day on cover in this row (inclusive) | 0.00 | 364 | 2025-12-31 (199,842); 2025-12-29 (203); 2025-01-01 (197); 2025-12-26 (194); 2025-01-05 (19 |
| `exposure` | numeric | years on cover: (period_end - period_start + 1) / 365 | 0.00 | 365 | 0.0027 to 1 (median 1) |
| `driver_age` | integer | age of the main driver at the start of the row | 0.00 | 76 | 17 to 92 (median 48) |
| `years_licensed` | integer | years since the main driver passed the test | 0.00 | 88 | 0 to 87 (median 27) |
| `ncb_years` | integer | no-claims bonus years (0-15) | 0.00 | 16 | 0 to 15 (median 8) |
| `vehicle_value_group` | ordinal | vehicle value group 1-20 | 0.00 | 20 | 1 to 20 (median 11) |
| `vehicle_age` | integer | vehicle age in years | 2.48 | 30 | 0 to 29 (median 9) |
| `make_model` | category | vehicle make and model code (300 codes) | 0.00 | 300 | MM005 (934); MM052 (919); MM090 (916); MM255 (915); MM115 (915); MM077 (911) |
| `engine_size_cc` | numeric | engine size | 2.53 | 33 | 500 to 3700 (median 2100) |
| `body_type` | category | hatchback, saloon, estate, suv, coupe | 0.00 | 5 | hatchback (99,751); suv (52,121); saloon (51,942); estate (30,567); coupe (20,605) |
| `transmission` | category | manual or automatic | 0.00 | 2 | manual (140,161); automatic (114,825) |
| `seats` | integer | number of seats | 0.00 | 4 | 2 to 7 (median 5) |
| `vehicle_colour` | category | colour | 0.00 | 8 | grey (32,034); silver (31,987); green (31,948); red (31,914); white (31,847); black (31,82 |
| `fuel_type` | category | petrol, diesel, hybrid, electric | 0.00 | 4 | petrol (128,763); diesel (70,430); hybrid (35,285); electric (20,508) |
| `driver_assist` | boolean | has driver-assistance systems | 0.00 | 2 | False (178,155); True (76,831) |
| `cover_type` | category | comprehensive, third_party_fire_theft, third_party_only | 0.00 | 3 | comprehensive (207,529); third_party_fire_theft (23,729); third_party_only (23,728) |
| `annual_mileage` | numeric | declared annual mileage | 2.46 | 432 | 0 to 75000 (median 10000) |
| `overnight_parking` | category | garage, driveway, street | 2.51 | 3 | driveway (124,122); street (74,898); garage (49,568) |
| `vehicle_use` | category | social, commuting, business | 0.00 | 3 | social (115,174); commuting (114,475); business (25,337) |
| `named_young_driver` | boolean | a young driver is named on the policy | 0.00 | 2 | False (241,920); True (13,066) |
| `telematics_opt_in` | boolean | telematics box fitted | 0.00 | 2 | False (221,126); True (33,860) |
| `telematics_score` | numeric | telematics driving score 1-100 (opted-in only) | 86.72 | 98 | 1 to 100 (median 60) |
| `voluntary_excess` | integer | excess chosen by the customer: 0, 250, 500 or 1000 | 0.00 | 4 | 0 to 1000 (median 250) |
| `compulsory_excess` | integer | excess set by the insurer | 0.00 | 1 | 250 to 250 (median 250) |
| `channel` | category | sales channel: direct, comparison_site, broker | 0.00 | 3 | direct (102,258); comparison_site (88,961); broker (63,767) |
| `payment_method` | category | annual or monthly | 0.00 | 2 | annual (139,963); monthly (115,023) |
| `missed_payments_12m` | integer | premium payments missed in the last 12 months | 0.00 | 7 | 0 to 6 (median 0) |
| `new_business` | boolean | first year with the insurer | 0.00 | 2 | False (216,108); True (38,878) |
| `renewal_count` | integer | number of renewals so far | 0.00 | 20 | 0 to 19 (median 3) |
| `days_quote_to_inception` | integer | days between quote and start of cover | 0.00 | 31 | 0 to 30 (median 13) |
| `discount_applied` | numeric | discount on the premium (share) | 0.00 | 3 | 0 to 0.1 (median 0) |
| `written_premium` | numeric | annual premium charged by the current tariff | 0.00 | 78,173 | 182.9 to 2684.73 (median 516.92) |
| `customer_service_contacts` | integer | contacts with customer service during the period | 0.00 | 7 | 0 to 6 (median 0) |

## `test_buildings`

| Column | Type | Meaning | Missing % | Distinct | Values |
|---|---|---|---:|---:|---|
| `row_id` | id | one policy row: a policy over one calendar-year period, or part of one | 0.00 | 184,155 | B02664297 (1); B02664298 (1); B02664299 (1); B02664300 (1); B02664301 (1); B02664302 (1) |
| `policy_id` | id | the policy; constant across its years and mid-term changes | 0.00 | 176,946 | B55597937 (2); B66576487 (2); B87490157 (2); B10070797 (2); B29680848 (2); B73304424 (2) |
| `customer_id` | id | the policyholder (customers table) | 0.00 | 176,946 | 94909815 (2); 76131967 (2); 63876283 (2); 37270622 (2); 22157037 (2); 83054857 (2) |
| `address_id` | id | the insured or garaging address (addresses table) | 0.00 | 184,155 | A9620763 (1); A2736290 (1); A2727550 (1); A1903092 (1); A1684027 (1); A0873922 (1) |
| `period_start` | date | first day on cover in this row | 0.00 | 364 | 2025-01-01 (150,111); 2025-01-13 (133); 2025-02-11 (122); 2025-01-28 (121); 2025-12-28 (11 |
| `period_end` | date | last day on cover in this row (inclusive) | 0.00 | 364 | 2025-12-31 (150,113); 2025-12-27 (142); 2025-12-26 (131); 2025-03-25 (120); 2025-05-17 (11 |
| `exposure` | numeric | years on cover: (period_end - period_start + 1) / 365 | 0.00 | 365 | 0.0027 to 1 (median 1) |
| `policyholder_age` | integer | age of the policyholder at the start of the row | 0.00 | 76 | 17 to 92 (median 48) |
| `sum_insured` | numeric | rebuilding sum insured | 0.00 | 887 | 37000 to 5.77e+08 (median 240000) |
| `construction_type` | category | brick, stone, timber, non_standard | 0.00 | 9 | Brick (118,393); W (21,099); non_standard (17,824); stone (14,323); brick (6,460); BRICK ( |
| `occupancy` | category | owner_occupied or let | 0.00 | 2 | owner_occupied (165,913); let (18,242) |
| `days_unoccupied` | numeric | days the property is unoccupied in a year | 2.49 | 104 | 0 to 119 (median 7) |
| `smoke_alarms` | boolean | smoke alarms fitted | 2.50 | 2 | True (152,500); False (27,053) |
| `accidental_damage_cover` | boolean | accidental damage cover bought | 0.00 | 2 | False (110,559); True (73,596) |
| `mortgage` | boolean | property is mortgaged | 0.00 | 2 | True (101,383); False (82,772) |
| `voluntary_excess` | integer | excess chosen by the customer: 0, 250, 500 or 1000 | 0.00 | 4 | 0 to 1000 (median 250) |
| `compulsory_excess` | integer | excess set by the insurer | 0.00 | 1 | 350 to 350 (median 350) |
| `channel` | category | sales channel: direct, comparison_site, broker | 0.00 | 3 | direct (73,615); comparison_site (64,469); broker (46,071) |
| `payment_method` | category | annual or monthly | 0.00 | 2 | annual (100,813); monthly (83,342) |
| `missed_payments_12m` | integer | premium payments missed in the last 12 months | 0.00 | 7 | 0 to 6 (median 0) |
| `new_business` | boolean | first year with the insurer | 0.00 | 2 | False (156,264); True (27,891) |
| `renewal_count` | integer | number of renewals so far | 0.00 | 20 | 0 to 19 (median 3) |
| `days_quote_to_inception` | integer | days between quote and start of cover | 0.00 | 31 | 0 to 30 (median 13) |
| `discount_applied` | numeric | discount on the premium (share) | 0.00 | 3 | 0 to 0.1 (median 0) |
| `written_premium` | numeric | annual premium charged by the current tariff | 0.00 | 57,932 | 64.06 to 4954.83 (median 248.85) |
| `customer_service_contacts` | integer | contacts with customer service during the period | 0.00 | 7 | 0 to 6 (median 0) |

## `test_contents`

| Column | Type | Meaning | Missing % | Distinct | Values |
|---|---|---|---:|---:|---|
| `row_id` | id | one policy row: a policy over one calendar-year period, or part of one | 0.00 | 187,498 | C02848452 (1); C02848453 (1); C02848454 (1); C02848455 (1); C02848456 (1); C02848457 (1) |
| `policy_id` | id | the policy; constant across its years and mid-term changes | 0.00 | 176,957 | C97653379 (2); C98546986 (2); C56685883 (2); C99390852 (2); C61734767 (2); C84891522 (2) |
| `customer_id` | id | the policyholder (customers table) | 0.00 | 176,957 | 10593260 (2); 72346252 (2); 94909815 (2); 63876283 (2); 32399144 (2); 05760932 (2) |
| `address_id` | id | the insured or garaging address (addresses table) | 0.00 | 183,587 | A4483201 (4); A2446797 (4); A4586422 (4); A5084506 (4); A5462748 (4); A7343631 (4) |
| `period_start` | date | first day on cover in this row | 0.00 | 364 | 2025-01-01 (150,071); 2025-12-28 (142); 2025-01-28 (140); 2025-01-29 (129); 2025-01-07 (12 |
| `period_end` | date | last day on cover in this row (inclusive) | 0.00 | 364 | 2025-12-31 (150,010); 2025-12-27 (156); 2025-12-26 (143); 2025-01-13 (136); 2025-03-19 (13 |
| `exposure` | numeric | years on cover: (period_end - period_start + 1) / 365 | 0.00 | 365 | 0.0027 to 1 (median 1) |
| `policyholder_age` | integer | age of the policyholder at the start of the row | 0.00 | 76 | 17 to 92 (median 48) |
| `sum_insured` | numeric | contents sum insured | 0.00 | 1,958 | 4100 to 1.468e+08 (median 40900) |
| `tenure` | category | owner or renter | 0.00 | 2 | owner (131,162); renter (56,336) |
| `security_level` | ordinal | standard, locks, alarm_and_locks | 2.49 | 3 | standard (64,006); alarm_and_locks (63,808); locks (55,007) |
| `high_value_items` | integer | declared high-value items (0-5) | 2.51 | 6 | 0 to 5 (median 1) |
| `accidental_damage_cover` | boolean | accidental damage cover bought | 0.00 | 2 | False (103,124); True (84,374) |
| `works_from_home` | boolean | policyholder works from home | 0.00 | 2 | False (140,437); True (47,061) |
| `lodgers` | boolean | lodgers in the home | 0.00 | 2 | False (178,267); True (9,231) |
| `bicycle_cover` | boolean | bicycle cover bought | 0.00 | 2 | False (159,445); True (28,053) |
| `voluntary_excess` | integer | excess chosen by the customer: 0, 250, 500 or 1000 | 0.00 | 4 | 0 to 1000 (median 250) |
| `compulsory_excess` | integer | excess set by the insurer | 0.00 | 1 | 350 to 350 (median 350) |
| `channel` | category | sales channel: direct, comparison_site, broker | 0.00 | 3 | direct (74,896); comparison_site (65,565); broker (47,037) |
| `payment_method` | category | annual or monthly | 0.00 | 2 | annual (102,802); monthly (84,696) |
| `missed_payments_12m` | integer | premium payments missed in the last 12 months | 0.00 | 7 | 0 to 6 (median 0) |
| `new_business` | boolean | first year with the insurer | 0.00 | 2 | False (159,070); True (28,428) |
| `renewal_count` | integer | number of renewals so far | 0.00 | 20 | 0 to 19 (median 3) |
| `days_quote_to_inception` | integer | days between quote and start of cover | 0.00 | 31 | 0 to 30 (median 13) |
| `discount_applied` | numeric | discount on the premium (share) | 0.00 | 3 | 0 to 0.1 (median 0) |
| `written_premium` | numeric | annual premium charged by the current tariff | 0.00 | 28,226 | 27.25 to 837.77 (median 145.08) |
| `customer_service_contacts` | integer | contacts with customer service during the period | 0.00 | 6 | 0 to 5 (median 0) |

## `sample_submission`

| Column | Type | Meaning | Missing % | Distinct | Values |
|---|---|---|---:|---:|---|
| `row_id` | id | row to predict (the 2025 test rows, in submission order) | 0.00 | 626,639 | M02409311 (1); M02409312 (1); M02409313 (1); M02409314 (1); M02409315 (1); M02409316 (1) |
| `prediction` | numeric | expected ultimate claim cost of the row | 0.00 | 1 | 0 to 0 (median 0) |
