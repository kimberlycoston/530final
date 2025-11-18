WITH crrt_patients AS (
  SELECT DISTINCT 
    icd.subject_id,
    icd.hadm_id,
    icd.stay_id
  FROM `physionet-data.mimiciv_3_1_derived.crrt` crrt
  INNER JOIN `physionet-data.mimiciv_3_1_derived.icustay_detail` icd
    ON crrt.stay_id = icd.stay_id
),

crrt_data AS (
  SELECT 
    cp.subject_id,
    cp.hadm_id,
    crrt.*
  FROM `physionet-data.mimiciv_3_1_derived.crrt` crrt
  INNER JOIN crrt_patients cp ON crrt.stay_id = cp.stay_id
),

admissions_filtered AS (
  SELECT a.subject_id, a.hadm_id, a.race
  FROM `physionet-data.mimiciv_3_1_hosp.admissions` a
  INNER JOIN crrt_patients cp 
    ON a.subject_id = cp.subject_id AND a.hadm_id = cp.hadm_id
),

dialysis_lines AS (
  SELECT il.stay_id, il.line_type, il.line_site
  FROM `physionet-data.mimiciv_3_1_derived.invasive_line` il
  INNER JOIN crrt_patients cp ON il.stay_id = cp.stay_id
  WHERE il.line_type = 'Dialysis'
),

cbc_matched AS (
  SELECT 
    cd.stay_id,
    cd.charttime AS crrt_charttime,
    cbc.hematocrit,
    cbc.hemoglobin,
    cbc.platelet,
    cbc.rbc,
    cbc.wbc,
    ROW_NUMBER() OVER (
      PARTITION BY cd.stay_id, cd.charttime 
      ORDER BY cbc.charttime DESC
    ) AS rn
  FROM crrt_data cd
  INNER JOIN `physionet-data.mimiciv_3_1_derived.complete_blood_count` cbc
    ON cd.subject_id = cbc.subject_id 
    AND cd.hadm_id = cbc.hadm_id
    AND cbc.charttime <= cd.charttime
    AND cbc.charttime >= TIMESTAMP_SUB(cd.charttime, INTERVAL 24 HOUR)
),

coag_matched AS (
  SELECT 
    cd.stay_id,
    cd.charttime AS crrt_charttime,
    coag.d_dimer,
    coag.fibrinogen,
    coag.thrombin,
    coag.inr,
    coag.pt,
    coag.ptt,
    ROW_NUMBER() OVER (
      PARTITION BY cd.stay_id, cd.charttime 
      ORDER BY coag.charttime DESC
    ) AS rn
  FROM crrt_data cd
  INNER JOIN `physionet-data.mimiciv_3_1_derived.coagulation` coag
    ON cd.subject_id = coag.subject_id 
    AND cd.hadm_id = coag.hadm_id
    AND coag.charttime <= cd.charttime
    AND coag.charttime >= TIMESTAMP_SUB(cd.charttime, INTERVAL 24 HOUR)
),

chem_matched AS (
  SELECT 
    cd.stay_id,
    cd.charttime AS crrt_charttime,
    chem.albumin,
    chem.globulin,
    chem.total_protein,
    chem.aniongap,
    chem.bicarbonate,
    chem.bun,
    chem.calcium,
    chem.chloride,
    chem.creatinine,
    chem.glucose,
    chem.sodium,
    chem.potassium,
    ROW_NUMBER() OVER (
      PARTITION BY cd.stay_id, cd.charttime 
      ORDER BY chem.charttime DESC
    ) AS rn
  FROM crrt_data cd
  INNER JOIN `physionet-data.mimiciv_3_1_derived.chemistry` chem
    ON cd.subject_id = chem.subject_id 
    AND cd.hadm_id = chem.hadm_id
    AND chem.charttime <= cd.charttime
    AND chem.charttime >= TIMESTAMP_SUB(cd.charttime, INTERVAL 24 HOUR)
),

-- Blood gas for lactate, pH, bicarbonate, pCO2
bg_matched AS (
  SELECT 
    cd.stay_id,
    cd.charttime AS crrt_charttime,
    bg.lactate,
    bg.pco2,
    bg.bicarbonate,
    bg.ph,
    ROW_NUMBER() OVER (
      PARTITION BY cd.stay_id, cd.charttime 
      ORDER BY bg.charttime DESC
    ) AS rn
  FROM crrt_data cd
  INNER JOIN `physionet-data.mimiciv_3_1_derived.bg` bg
    ON cd.subject_id = bg.subject_id 
    AND cd.hadm_id = bg.hadm_id
    AND bg.charttime <= cd.charttime
    AND bg.charttime >= TIMESTAMP_SUB(cd.charttime, INTERVAL 24 HOUR)
),

-- Triglyceride (itemid 51000) - 24 hour window
triglyceride_matched AS (
  SELECT 
    cd.stay_id,
    cd.charttime AS crrt_charttime,
    SAFE_CAST(lab.value AS FLOAT64) AS triglyceride,
    ROW_NUMBER() OVER (
      PARTITION BY cd.stay_id, cd.charttime 
      ORDER BY lab.charttime DESC
    ) AS rn
  FROM crrt_data cd
  INNER JOIN `physionet-data.mimiciv_3_1_hosp.labevents` lab
    ON cd.subject_id = lab.subject_id 
    AND cd.hadm_id = lab.hadm_id
    AND lab.itemid = 51000
    AND lab.value IS NOT NULL
    AND lab.charttime <= cd.charttime
    AND lab.charttime >= TIMESTAMP_SUB(cd.charttime, INTERVAL 24 HOUR)  -- Changed to 24 hours
),

-- Magnesium (itemid 50960)
magnesium_matched AS (
  SELECT 
    cd.stay_id,
    cd.charttime AS crrt_charttime,
    SAFE_CAST(lab.value AS FLOAT64) AS magnesium,
    ROW_NUMBER() OVER (
      PARTITION BY cd.stay_id, cd.charttime 
      ORDER BY lab.charttime DESC
    ) AS rn
  FROM crrt_data cd
  INNER JOIN `physionet-data.mimiciv_3_1_hosp.labevents` lab
    ON cd.subject_id = lab.subject_id 
    AND cd.hadm_id = lab.hadm_id
    AND lab.itemid = 50960
    AND lab.value IS NOT NULL
    AND lab.charttime <= cd.charttime
    AND lab.charttime >= TIMESTAMP_SUB(cd.charttime, INTERVAL 24 HOUR)
),

-- Phosphate (itemid 50970)
phosphate_matched AS (
  SELECT 
    cd.stay_id,
    cd.charttime AS crrt_charttime,
    SAFE_CAST(lab.value AS FLOAT64) AS phosphate,
    ROW_NUMBER() OVER (
      PARTITION BY cd.stay_id, cd.charttime 
      ORDER BY lab.charttime DESC
    ) AS rn
  FROM crrt_data cd
  INNER JOIN `physionet-data.mimiciv_3_1_hosp.labevents` lab
    ON cd.subject_id = lab.subject_id 
    AND cd.hadm_id = lab.hadm_id
    AND lab.itemid = 50970
    AND lab.value IS NOT NULL
    AND lab.charttime <= cd.charttime
    AND lab.charttime >= TIMESTAMP_SUB(cd.charttime, INTERVAL 24 HOUR)
),

-- Antithrombin III (itemid 52078) - 24 hour window
antithrombin_matched AS (
  SELECT 
    cd.stay_id,
    cd.charttime AS crrt_charttime,
    SAFE_CAST(lab.value AS FLOAT64) AS antithrombin,
    ROW_NUMBER() OVER (
      PARTITION BY cd.stay_id, cd.charttime 
      ORDER BY lab.charttime DESC
    ) AS rn
  FROM crrt_data cd
  INNER JOIN `physionet-data.mimiciv_3_1_hosp.labevents` lab
    ON cd.subject_id = lab.subject_id 
    AND cd.hadm_id = lab.hadm_id
    AND lab.itemid = 52078
    AND lab.value IS NOT NULL
    AND lab.charttime <= cd.charttime
    AND lab.charttime >= TIMESTAMP_SUB(cd.charttime, INTERVAL 24 HOUR)  -- Changed to 24 hours
),

-- Lactate Dehydrogenase / LD (itemid 50954) - 24 hour window
ldh_matched AS (
  SELECT 
    cd.stay_id,
    cd.charttime AS crrt_charttime,
    SAFE_CAST(lab.value AS FLOAT64) AS ldh,
    ROW_NUMBER() OVER (
      PARTITION BY cd.stay_id, cd.charttime 
      ORDER BY lab.charttime DESC
    ) AS rn
  FROM crrt_data cd
  INNER JOIN `physionet-data.mimiciv_3_1_hosp.labevents` lab
    ON cd.subject_id = lab.subject_id 
    AND cd.hadm_id = lab.hadm_id
    AND lab.itemid = 50954
    AND lab.value IS NOT NULL
    AND lab.charttime <= cd.charttime
    AND lab.charttime >= TIMESTAMP_SUB(cd.charttime, INTERVAL 24 HOUR)  -- Changed to 24 hours
),

-- HDL Cholesterol (itemid 50904) - 24 hour window
hdl_matched AS (
  SELECT 
    cd.stay_id,
    cd.charttime AS crrt_charttime,
    SAFE_CAST(lab.value AS FLOAT64) AS hdl_cholesterol,
    ROW_NUMBER() OVER (
      PARTITION BY cd.stay_id, cd.charttime 
      ORDER BY lab.charttime DESC
    ) AS rn
  FROM crrt_data cd
  INNER JOIN `physionet-data.mimiciv_3_1_hosp.labevents` lab
    ON cd.subject_id = lab.subject_id 
    AND cd.hadm_id = lab.hadm_id
    AND lab.itemid = 50904
    AND lab.value IS NOT NULL
    AND lab.charttime <= cd.charttime
    AND lab.charttime >= TIMESTAMP_SUB(cd.charttime, INTERVAL 24 HOUR)  -- Changed to 24 hours
),

-- LDL Cholesterol - prefer measured (50906), fall back to calculated (50905) - 24 hour window
ldl_matched AS (
  SELECT 
    cd.stay_id,
    cd.charttime AS crrt_charttime,
    SAFE_CAST(lab.value AS FLOAT64) AS ldl_cholesterol,
    lab.itemid AS ldl_source,  -- Track which one we used
    ROW_NUMBER() OVER (
      PARTITION BY cd.stay_id, cd.charttime 
      ORDER BY 
        CASE WHEN lab.itemid = 50906 THEN 1 ELSE 2 END,  -- Prefer measured
        lab.charttime DESC
    ) AS rn
  FROM crrt_data cd
  INNER JOIN `physionet-data.mimiciv_3_1_hosp.labevents` lab
    ON cd.subject_id = lab.subject_id 
    AND cd.hadm_id = lab.hadm_id
    AND lab.itemid IN (50905, 50906)  -- Both calculated and measured
    AND lab.value IS NOT NULL
    AND lab.charttime <= cd.charttime
    AND lab.charttime >= TIMESTAMP_SUB(cd.charttime, INTERVAL 24 HOUR)  -- Changed to 24 hours
)

-- Final SELECT
SELECT 
  cd.*,
  a.race,
  dl.line_type,
  dl.line_site,
  
  -- CBC
  cbc.hematocrit,
  cbc.hemoglobin,
  cbc.platelet,
  cbc.rbc,
  cbc.wbc,
  
  -- Coagulation
  coag.d_dimer,
  coag.fibrinogen,
  coag.thrombin,
  coag.inr,
  coag.pt,
  coag.ptt,
  
  -- Chemistry
  chem.albumin,
  chem.globulin,
  chem.total_protein,
  chem.aniongap,
  chem.bicarbonate,
  chem.bun,
  chem.calcium,
  chem.chloride,
  chem.creatinine,
  chem.glucose,
  chem.sodium,
  chem.potassium,
  
  -- Blood gas
  bg.lactate,
  bg.ph,
  bg.pco2,
  bg.bicarbonate, 
  
  -- Lipids (24 hour window)
  trig.triglyceride,
  hdl.hdl_cholesterol,
  ldl.ldl_cholesterol,
  ldl.ldl_source,  -- Shows if measured (50906) or calculated (50905)
  
  -- Other labs
  mag.magnesium,
  phos.phosphate,
  antith.antithrombin,  -- 24 hour window
  ldh.ldh  -- 24 hour window

FROM crrt_data cd

LEFT JOIN admissions_filtered a 
  ON cd.subject_id = a.subject_id AND cd.hadm_id = a.hadm_id

LEFT JOIN dialysis_lines dl 
  ON cd.stay_id = dl.stay_id

LEFT JOIN cbc_matched cbc
  ON cd.stay_id = cbc.stay_id 
  AND cd.charttime = cbc.crrt_charttime
  AND cbc.rn = 1

LEFT JOIN coag_matched coag
  ON cd.stay_id = coag.stay_id 
  AND cd.charttime = coag.crrt_charttime
  AND coag.rn = 1

LEFT JOIN chem_matched chem
  ON cd.stay_id = chem.stay_id 
  AND cd.charttime = chem.crrt_charttime
  AND chem.rn = 1

LEFT JOIN bg_matched bg
  ON cd.stay_id = bg.stay_id
  AND cd.charttime = bg.crrt_charttime
  AND bg.rn = 1

LEFT JOIN triglyceride_matched trig
  ON cd.stay_id = trig.stay_id
  AND cd.charttime = trig.crrt_charttime
  AND trig.rn = 1

LEFT JOIN magnesium_matched mag
  ON cd.stay_id = mag.stay_id
  AND cd.charttime = mag.crrt_charttime
  AND mag.rn = 1

LEFT JOIN phosphate_matched phos
  ON cd.stay_id = phos.stay_id
  AND cd.charttime = phos.crrt_charttime
  AND phos.rn = 1

LEFT JOIN antithrombin_matched antith
  ON cd.stay_id = antith.stay_id
  AND cd.charttime = antith.crrt_charttime
  AND antith.rn = 1

LEFT JOIN ldh_matched ldh
  ON cd.stay_id = ldh.stay_id
  AND cd.charttime = ldh.crrt_charttime
  AND ldh.rn = 1

LEFT JOIN hdl_matched hdl
  ON cd.stay_id = hdl.stay_id
  AND cd.charttime = hdl.crrt_charttime
  AND hdl.rn = 1

LEFT JOIN ldl_matched ldl
  ON cd.stay_id = ldl.stay_id
  AND cd.charttime = ldl.crrt_charttime
  AND ldl.rn = 1

WHERE cd.clots IS NOT NULL  -- Only rows with CRRT machine data documented