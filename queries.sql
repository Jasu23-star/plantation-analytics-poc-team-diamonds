-- Table: harvest_records (one row per block per harvest cycle; 3,200 rows)
-- Yield must be area-weighted: SUM(tonnes) / SUM(hectares). Never add up yields.

-- name: 1. Yield by estate (area-weighted)
SELECT estate_name,
       ROUND(SUM(ffb_tonnes), 0)                        AS total_tonnes,
       ROUND(SUM(ffb_tonnes) / SUM(productive_area_ha), 3) AS yield_t_ha_per_cycle
FROM harvest_records
GROUP BY estate_name
ORDER BY yield_t_ha_per_cycle;

-- name: 2. Yield by palm age band
SELECT palm_age_band,
       ROUND(AVG(palm_age_years), 1)  AS avg_age,
       ROUND(AVG(ffb_yield_t_ha), 3)  AS avg_yield,
       COUNT(*)                        AS records
FROM harvest_records
GROUP BY palm_age_band
ORDER BY avg_age;

-- name: 3. Water deficit vs yield (equal 14-day cycle)
SELECT CASE WHEN water_deficit_prior_90d_mm = 0  THEN 'a) None'
            WHEN water_deficit_prior_90d_mm <= 20 THEN 'b) 1-20 mm'
            WHEN water_deficit_prior_90d_mm <= 60 THEN 'c) 21-60 mm'
            ELSE 'd) >60 mm' END                          AS water_deficit_band,
       ROUND(AVG(ffb_yield_t_ha * 14.0 / harvest_interval_days), 3) AS yield_per_14_days,
       COUNT(*)                                            AS records
FROM harvest_records
GROUP BY water_deficit_band
ORDER BY water_deficit_band;

-- name: 4. Road access vs harvest losses
SELECT CASE WHEN road_accessibility_score_1_5 <= 2 THEN 'a) Poor (1-2)'
            WHEN road_accessibility_score_1_5 <= 3 THEN 'b) Fair (2-3)'
            ELSE 'c) Good (3-5)' END                       AS road_access,
       ROUND(AVG(harvest_interval_days), 1)    AS avg_interval_days,
       ROUND(AVG(overripe_bunch_pct), 2)       AS overripe_pct,
       ROUND(AVG(estimated_field_loss_pct), 2) AS field_loss_pct,
       COUNT(*)                                 AS records
FROM harvest_records
GROUP BY road_access
ORDER BY road_access;

-- name: 5. Estate yield vs age-expected yield
WITH age_expected AS (
    SELECT palm_age_years, AVG(ffb_yield_t_ha) AS expected_yield
    FROM harvest_records GROUP BY palm_age_years
)
SELECT h.estate_name,
       ROUND((SUM(h.ffb_tonnes) / SUM(h.productive_area_ha * a.expected_yield)) * 100 - 100, 2)
           AS pct_vs_age_expected,
       ROUND(AVG(h.water_deficit_prior_90d_mm), 1) AS avg_water_deficit_mm
FROM harvest_records h
JOIN age_expected a ON a.palm_age_years = h.palm_age_years
GROUP BY h.estate_name
ORDER BY pct_vs_age_expected;

-- name: 6. Priority blocks (tonnes lost vs age-expected)
WITH age_expected AS (
    SELECT palm_age_years, AVG(ffb_yield_t_ha) AS expected_yield
    FROM harvest_records GROUP BY palm_age_years
),
block_perf AS (
    SELECT h.block_id, h.estate_name, MAX(h.palm_age_years) AS age,
           MAX(h.productive_area_ha) AS area_ha,
           AVG(h.ffb_yield_t_ha) AS block_yield,
           AVG(a.expected_yield) AS expected_yield
    FROM harvest_records h
    JOIN age_expected a ON a.palm_age_years = h.palm_age_years
    GROUP BY h.block_id, h.estate_name
)
SELECT block_id, estate_name, age,
       ROUND((block_yield / expected_yield - 1) * 100, 1)      AS pct_below_expected,
       ROUND((expected_yield - block_yield) * area_ha, 1)     AS tonnes_lost_per_cycle
FROM block_perf
ORDER BY tonnes_lost_per_cycle DESC
LIMIT 10;
