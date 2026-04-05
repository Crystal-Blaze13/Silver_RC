WITH base AS (
  SELECT
    DATE(_PARTITIONTIME) AS event_date,
    SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64) AS tone,
    SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(1)] AS FLOAT64) AS positive,
    SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(2)] AS FLOAT64) AS negative,
    LOWER(IFNULL(V2Themes, '')) AS themes,
    LOWER(IFNULL(V2Organizations, '')) AS orgs,
    LOWER(IFNULL(V2Locations, '')) AS locs
  FROM `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE DATE(_PARTITIONTIME) BETWEEN DATE '2015-01-01' AND DATE '2026-03-31'
)

SELECT
  event_date AS date,
  metal,
  AVG(tone) AS avg_tone,
  AVG(positive) AS avg_positive,
  AVG(negative) AS avg_negative,
  COUNT(*) AS article_count
FROM (
  -- GOLD
  SELECT
    event_date,
    tone,
    positive,
    negative,
    'gold' AS metal
  FROM base
  WHERE CONCAT(themes, ' ', orgs, ' ', locs) LIKE '%gold%'
    AND REGEXP_CONTAINS(
      CONCAT(themes, ' ', orgs, ' ', locs),
      r'(india|indian|mcx|nse|rbi|rupee)'
    )

  UNION ALL

  -- SILVER
  SELECT
    event_date,
    tone,
    positive,
    negative,
    'silver' AS metal
  FROM base
  WHERE CONCAT(themes, ' ', orgs, ' ', locs) LIKE '%silver%'
    AND REGEXP_CONTAINS(
      CONCAT(themes, ' ', orgs, ' ', locs),
      r'(india|indian|mcx|nse|rbi|rupee)'
    )
)

GROUP BY date, metal
ORDER BY date, metal;
