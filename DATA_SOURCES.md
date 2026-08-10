# Data Sources and Provenance

This file records the origin, licensing terms and processing lineage of every
dataset used in this project. It exists so that results can be reproduced and
so that upstream providers are correctly credited.

**Raw input data is not relicensed by this repository.** The MIT and CC BY 4.0
licenses cover this project's code and analysis only. Each source below
retains its own terms.

## ⚠️ Licensing disclaimer — read before reusing any source data

**This repository does not reproduce, mirror, summarise or warrant the license
terms of any upstream data provider.**

Where a field below reads **"Not reproduced here — consult the provider"**, it
means exactly that: the authoritative terms live on the provider's own site,
they change without notice, and we have deliberately not paraphrased them.
A paraphrase of a license is not a license.

**If you intend to redistribute, publish, or commercially use anything derived
from these sources, it is your responsibility to obtain and read the current
terms directly from the provider.** Every source has its URL recorded below;
start there. Do not treat this file as legal clearance — it is a provenance
record, not a rights assessment.

Two specific cautions:

- **Absence of a stated license is not permission.** Several Mexican public
  portals publish data without an attached machine-readable license. That is
  *unresolved*, not *public domain*.
- **A permissive license on our code says nothing about the data.** `LICENSE`
  (MIT) and `LICENSE-DOCS` (CC BY 4.0) cover this project's own code, prose and
  figures. They confer no rights whatsoever over the upstream observations.

**On `Accessed` dates:** these are the **local file mtime** of the downloaded
artefact. `wget` preserves the server's `Last-Modified` header by default, so
for files fetched that way the mtime may be the *publication* date rather than
the download date. Treat them as indicative, and confirm against your own
download records if the distinction matters.

---

## 1. Carpetas de Investigación (FGJ) — crime incidents

This is the primary target dataset: investigation files opened by the *Fiscalía
General de Justicia* of Mexico City. It supplies `y_cnt` (incident counts) and
`y_ind` (incident indicator).

| Field | Value |
|---|---|
| Provider | Portal de Datos Abiertos de la Ciudad de México ("Sistema Ajolote") |
| URL | `https://archivo.datos.cdmx.gob.mx/FGJ/carpetas/carpetasFGJ_acumulado_2025_01.csv` |
| Portal page | `https://www.datos.cdmx.gob.mx/` |
| Terms of use | **Not reproduced here — consult the provider** at the dataset page below. Not recorded in this repo. |
| License | **Not reproduced here — consult the provider**. No license is asserted by this project; the portal publishes no machine-readable license we have verified. Absence of a stated license is **not** permission. |
| Accessed | 2025-09-05 (mtime of `data/raw/carpetasFGJ.csv`) |
| Version / snapshot | `carpetasFGJ_acumulado_2025_01.csv`, 559,894,090 bytes; stored locally as `data/raw/carpetasFGJ.csv` |
| Temporal coverage | Filtered to **2016-01-01 onward** (`start_year: 2016`, `config/eda_fgr_conf.yml`); snapshot runs through **2025-01** |
| Spatial unit | Point geometry (`latitud`/`longitud`, EPSG:4326), plus `alcaldia_hecho`; snapped to street segments downstream |
| Redistributed here? | **No.** `data/raw/**` is git-ignored. Re-fetch with the `wget` command below. |

```bash
wget https://archivo.datos.cdmx.gob.mx/FGJ/carpetas/carpetasFGJ_acumulado_2025_01.csv \
  -O ./data/raw/carpetasFGJ.csv --no-check-certificate
```

`--no-check-certificate` is required because the official portal's TLS
certificate is intermittently expired (documented in the README FAQ).

**Required attribution string (verbatim, if the provider specifies one):**

> **Not reproduced here — consult the provider.** No credit line is recorded in
> this repository, and we do not paraphrase one. If the portal specifies a
> required attribution string, it is on the dataset page at
> `datos.cdmx.gob.mx`; obtain it there before redistributing anything derived
> from this source.

**Known limitations:** these are substantial and directly affect how the risk
estimates should be read.

- **Under-reporting dominates the signal.** INEGI's ENVIPE 2025 estimates the
  *cifra negra* — crimes neither reported nor investigated — at roughly **93%**.
  These records capture the visible tip of the distribution, not the phenomenon.
- **Geocoding gaps.** `latitud`/`longitud` are null in **4.82%** of rows,
  costing roughly **101k rows** from any spatial analysis.
- **Duplicates.** 4,735 exact duplicates (0.226%); 11,432 rows (0.545%) share an
  incident key (`fecha_hecho`, `hora_hecho`, `delito`, lat, lon, `colonia_hecho`,
  `alcaldia_hecho`). Deduplicate on the incident key for counting; keep all rows
  for caseload analysis.
- **Dead and low-information columns.** `alcaldia_catalogo` is 99.16% null (drop
  it); `municipio_hecho` is the constant `CDMX` across all ~2.1M rows;
  `competencia` is 50.70% null.
- **`hora_hecho` placeholder concentration** — reported times cluster on
  placeholder values, so hour-of-day analysis is biased.
- **Implausible `anio_hecho` values.** The incident year is self-reported and
  runs back to **1966** — the AGEB aggregation in `data/proc/fgr_dbs/` produced
  36 year-files spanning 1966–2025 (1966, 1969, 1972, 1976, 1981, … ). These are
  cases opened recently about events reported as decades old, plus data-entry
  errors. `anio_hecho` must be range-filtered before use; note that
  `config/eda_fgr_conf.yml` sets only a lower bound (`start_year: 2016`) and the
  modelling set is restricted to 2022–2024 separately.
- **Catalog vs free-text twins.** `colonia_catalogo`/`colonia_hecho` and
  `alcaldia_catalogo`/`alcaldia_hecho` are paired; the `_hecho` variants are far
  more complete and are treated as canonical.
- **Endpoint instability.** The README notes public crime endpoints change over
  time; the accumulated-file URL is versioned by month and older ones disappear.

---

## 2. Administrative boundaries — INEGI Marco Geoestadístico

| Field | Value |
|---|---|
| Provider | INEGI — *Marco Geoestadístico* |
| URL | `https://www.inegi.org.mx/app/biblioteca/ficha.html?upc=794551163061` |
| Terms of use | **Not reproduced here — consult the provider**. INEGI publishes a general *Términos de Libre Uso de la Información*; obtain the current text from INEGI directly rather than relying on this note. |
| License | **Not reproduced here — consult the provider**. Governed by INEGI's own terms. |
| Accessed | 2025-07-03 (mtime of `data/spatial/pd/*.shp`) |
| Vintage | **Unrecorded** — the Marco Geoestadístico edition/year was not captured at download time. This is a reproducibility gap, not a licensing one: boundary vintage determines whether *delegación* or *alcaldía* naming applies (the 2016 constitutional change) and affects joins on `CVEGEO`. Record it on the next refresh. |
| CRS | Source **EPSG:4326**; metric operations reprojected to **EPSG:6372** (Mexico ITRF2008 / LCC) |
| Redistributed here? | **No.** `data/spatial/**` is git-ignored. |

Layers actually used, all filtered to entidad `09` (CDMX):

| File | Geometry | Role |
|---|---|---|
| `data/spatial/pd/09mun.shp` | Polygon | Municipios / alcaldías — AoI clipping, `CVE_MUN` subclass |
| `data/spatial/pd/09m.shp` | Polygon | Manzanas (city blocks) — MZA-level aggregation |
| `data/spatial/pd/09a.shp` | Polygon | AGEBs — AGEB-level aggregation |
| `data/spatial/pd/09e.shp` | LineString | Street axes — split corner-to-corner into segments, the primary modelling unit |
| `09ent.shp` | Polygon | CDMX outer boundary |

**Known limitations:** street segments are produced by
`split_streets_at_intersections` (`src/utils/geom.py`), so segment identity is
derived, not official — `CVEGEO` for a segment is a constructed composite key
(`STREET_ID` built from `CVEGEO + CVE_ENT + CVE_MUN + CVE_LOC + CVEVIAL + CVESEG`)
and is not comparable across boundary vintages.

---

## 3. AlphaEarth Satellite Embedding V1 (Annual) — model features

| Field | Value |
|---|---|
| Provider | Google DeepMind / Google Earth Engine catalog |
| Collection id | `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` |
| URL | `https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL` |
| Method reference | `https://deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/` (Brown et al., 2025) |
| Terms of use | Google Earth Engine dataset terms, per catalog page |
| License | **CC BY 4.0** (as stated in the README and the GEE catalog entry) |
| Accessed | 2025-07-18 (mtime of `data/raw/alpha_earth_cdmx_*/`) |
| Version / snapshot | V1 Annual; exported per municipality as GeoTIFF — `mun_<CVE_MUN>_<year>.tif`, 16 municipality files per year |
| Temporal coverage | **2022, 2023, 2024, 2025** (`data/raw/alpha_earth_cdmx_{2022,2023,2024,2025}`) |
| Spatial unit | 10 m pixel; 64-dimensional vector per pixel per year, unit-norm by construction (values in [-1, 1], on \(S^{63}\)) |
| Redistributed here? | **No** for the rasters. One small derived sample *is* tracked: `data/sample/poi_embeddings_cdmx.csv`. |

Access requires a **Google Earth Engine account**; the Python API reads the
project id from `.env` (`EE_PROJECT_NAME`). Export parameters used: `scale: 10`
(native resolution), source CRS `EPSG:4326`, band prefix `A` → `A00`–`A63`
(`config/etl_embedding_alpha_earth.yml`).

**Required attribution string (verbatim, if the provider specifies one):**

> **Not reproduced here — consult the provider.** This source *is* CC BY 4.0,
> which makes attribution a binding condition rather than a courtesy. The exact
> required citation is published on the Earth Engine catalog page linked above;
> take it from there verbatim before redistributing derived products.

**Known limitations:**

- **Annual composite.** One vector per pixel per year; no sub-annual dynamics,
  so it cannot represent seasonal or diurnal variation in the environment.
- **Opportunistic / population confound.** The embeddings encode structure that
  correlates with human activity and therefore with population. High-risk areas
  correlate strongly with populated areas, and per-capita normalization is *not*
  a clean fix — the embeddings already carry population-correlated geography.
  This is flagged as unresolved future work in the project conclusions.
- **Training-domain limits.** What the encoder can represent is bounded by its
  pretraining data; similarity between two visually-alike streets may have
  nothing to do with crime.
- **Aggregation is a modelling choice.** Pixel→geometry assignment uses
  nearest/coverage rules (`src/etl/assign_point_to_geometry.py`); a different
  rule yields different features from identical source rasters.

---

## 4. Censo de Población y Vivienda 2020 — AGEB y manzana urbana

Source of the 28 socio-demographic covariates (`POBTOT`, `POBFEM`, `GRAPROES`,
`VPH_INTER`, …) that form the `other` feature set in the benchmarks.

| Field | Value |
|---|---|
| Provider | INEGI — Censo de Población y Vivienda 2020, "Principales resultados por AGEB y manzana urbana" (bulk open-data CSV, **not** the Indicadores API) |
| URL | `https://www.inegi.org.mx/contenidos/programas/ccpv/2020/datosabiertos/ageb_manzana/ageb_mza_urbana_09_cpv2020_csv.zip` (`{ent}` = `09` for CDMX) |
| Terms of use | **Not reproduced here — consult the provider**. Governed by INEGI's own terms. |
| License | **Not reproduced here — consult the provider**. |
| Accessed | **Not captured** — fetched at runtime by script; no cached artefact carries a date. |
| Version / snapshot | CPV 2020, entidad 09 |
| Temporal coverage | **2020 only — a single cross-section.** |
| Spatial unit | Manzana (city block) and AGEB, joined on `CVEGEO` |
| Redistributed here? | **No.** Downloaded on demand by `src/etl/get_socioecon_data.py`. |

**Known limitations:**

- **No panel.** The census is decennial, so there is **no 2022–2024 manzana
  panel**. A single 2020 cross-section is broadcast across all modelled years,
  which means the covariates cannot explain year-over-year change and grow
  staler the further a year sits from 2020.
- **Censored values.** The census marks suppressed cells with `*`, `N/D`, `-`
  or blank (`CENSUS_CENSOR_MARKS`); these are coerced to null, not zero.
  Suppression is applied for disclosure control on small blocks, so missingness
  is *not* random — it correlates with low population.

---

## 5. Macroeconomic indicators — INEGI BIE + Banxico SIE

| Field | Value |
|---|---|
| Providers | INEGI (Banco de Información Económica) and **Banco de México** (SIE) — interest rates are a Banxico product, not INEGI |
| URLs | `https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml` · `https://www.banxico.org.mx/SieAPIRest/service/v1` |
| Terms of use | **Not reproduced here — consult the provider** — separately for **each** API. INEGI and Banxico are distinct providers with distinct terms. |
| License | **Not reproduced here — consult the provider**. |
| Accessed | 2025-07-13 (mtime of `data/raw/cdmx_macro_2022_2024.csv`) |
| Version / snapshot | `data/raw/cdmx_macro_2022_2024.csv` (161 bytes) |
| Temporal coverage | 2022–2024, annual |
| Spatial unit | **National** — see limitations |
| Redistributed here? | **No.** |

Series used:

| Concept | Source | Id |
|---|---|---|
| Inflation (annual % change, INPC) | INEGI BIE | `628229` |
| Unemployment rate (ENOE, monthly) | INEGI BIE | `444612` |
| INPC index level | INEGI BIE | `628194`, `216064` |
| Interest rate (tasa objetivo) | Banxico SIE | `SF61745` |

Both APIs require free tokens, read from `.env` as `INEGI_TOKEN` and
`BANXICO_TOKEN`.

**Known limitations:**

- **Wrong geography for the research question.** The INPC and unemployment
  series above are **national**, not CDMX. CDMX-level unemployment exists under
  different state-ENOE indicator ids and would have to be swapped in.
- **Indicator ids are unstable.** INEGI restructured the BIE on **2025-12-01**,
  merging per-geography series into single keys and **retiring the classic ids**.
  `444612` in particular is a classic key and is likely retired. Verify current
  ids with the [Constructor de consultas](https://www.inegi.org.mx/app/querybuilder2/)
  or the script's `probe_inegi()` helper before re-running. This is the single
  most likely reason a fresh run of the macro ETL will fail.
- **Annual granularity, national scope** — these contribute little spatial
  discrimination and are not part of the street-level feature sets.

---

## 6. Sources referenced but *not* ingested

Listed so the distinction is explicit — none of these feed the models.

| Source | Use | Status |
|---|---|---|
| **C5 citizen reports** (`inViales_2022_2024.csv`, `https://archivo.datos.cdmx.gob.mx/C5/incidentes_viales/inViales_2022_2024.csv`) | Documented in Ch. 3 as an alternative target and suggested as a reader exercise | **Not used.** `config/eda_c5_conf.yml` is intentionally empty and no C5 file exists under `data/raw/`. |
| **ENVIPE 2025** (`https://www.inegi.org.mx/programas/envipe/2025/`) | Cited for victimization, *cifra negra* (~93%) and cost-of-crime context | Narrative citation only |
| **ENSU** (`https://www.inegi.org.mx/programas/ensu/`) | Cited for perception-of-insecurity statistics | Narrative citation only |

---

## Processing lineage

Stage names follow the script table in `docs/main.tex`. All modules run as
packages: `python -m src.<subpackage>.<module>`.

| Stage | Script | Input | Output |
|---|---|---|---|
| Validation (pre-flight) | `src/eda/app.py`, `src/eda/app_esp.py` | `data/raw/carpetasFGJ.csv` | Interactive EDA dashboard; figure exports |
| Validation (pre-flight) | `src/eda/dq.py` | `data/raw/carpetasFGJ.csv` | Null / duplicate / coverage profile |
| Extract — socio-economic | `src/etl/get_socioecon_data.py` | INEGI census zip, INEGI BIE API, Banxico SIE API | `data/raw/cdmx_macro_2022_2024.csv`, census frame |
| Transform — embeddings | `src/etl/process_embeddings.py` | `data/raw/alpha_earth_cdmx_<year>/*.tif` | `data/proc/embeddings/alpha_earth/cdmx/year=<year>/` (Hive-partitioned parquet) |
| Transform — crime cleaning | `src/eda/core.py` | `data/raw/carpetasFGJ.csv` | `data/proc/fgr_dbs/` |
| Transform — street splitting | `src/utils/geom.py` | `data/spatial/pd/09e.shp` | Corner-to-corner street segments |
| Transform — point assignment | `src/etl/assign_point_to_geometry.py` | Embedding pixels + geometries | Geometry-keyed embedding features |
| Load — training sets | `src/etl/get_training_sets.py` | `data/proc/fgr_dbs/asaltos_by_streets_year_<y>.geojson`, `data/proc/embeddings/.../year=<y>/` | `data/proc/training_sets/cdmx_asaltos.parquet` (534,111 rows × 99 cols) |
| Supervised — full benchmark | `src/supervised/model_single_run.py` | `data/proc/training_sets/cdmx_asaltos.parquet` | `models/classification/`, `models/regression/`, `models/tuning/`, `models/champion/` |
| Supervised — classification | `src/supervised/run_classification.py` | same | `models/classification/**`, `docs/book/resources/supervised/classification/` |
| Supervised — regression | `src/supervised/run_regression.py` | same | `models/regression/**` |
| Supervised — tuning | `src/supervised/run_tuning.py` | same | `models/tuning/` (Optuna study) |
| Supervised — inference | `src/supervised/predict.py` | `models/champion/` | Scored geometries |
| Unsupervised — dim. reduction | `src/unsupervised/dim_reduction.py` | `data/proc/training_sets/cdmx_asaltos.parquet` | `docs/resources/unsupervised/` |
| Unsupervised — manifold learning | `src/unsupervised/manifold_learning.py` | same | `docs/resources/unsupervised/` |
| Unsupervised — K-means | `src/unsupervised/clustering_kmeans.py` | same | `data/proc/training_sets/cdmx_asaltos_labeled.parquet`, `data/infer/kmeans_*.parquet`, `docs/resources/unsupervised/` |

**Nothing under `data/` is versioned except `data/sample/poi_embeddings_cdmx.csv`.**
`data/raw/**`, `data/proc/**`, `data/clean/**`, `data/spatial/**`, all `*.parquet`
and all `*.csv` are git-ignored, as is `models/**/*`.

---

## Reproducibility

- **Environment** pinned in `requirements.txt` (Python 3.12; matches the
  `ds_research` conda environment). There is no `environment.yml` or lock file.
- **Random seeds:**
  - `config/data.yaml` → `cv.seed: 69` (cross-validation)
  - `config/clf_nn.yaml`, `config/reg_nn.yaml` → `seed: 69` (torch heads)
  - `config/kmeans_conf.yml` → `rng_seed: 42`
- **K-means:** `n_init_sweep: 5`, `max_iter_sweep: 200`, `batch_size: 10000`,
  spherical variant `n_iter: 50`, `tol: 1e-4`, sweep over `k_range: 2..13`
  (`config/kmeans_conf.yml`).
- **Train/validation/holdout split** — defined declaratively in
  `config/data.yaml` and applied at `src/supervised/cv.py:43` (`split_holdout`):

  ```yaml
  holdout:
    mode: union          # union | intersection of the column masks
    year: [2024]
    CVE_MUN: ["002", "017", "004"]
  ```

  The split is **neither purely temporal nor stratified random**: it is the
  *union* of a year mask and a municipality mask, so the holdout is every row
  from 2024 **plus** every row from municipalities 002/017/004 in any year. It
  is a deliberate spatio-temporal extrapolation test — a model is asked to
  generalize to an unseen year *and* to unseen municipalities at once.
  Actual sizes on the current training set: **dev 319,936 / holdout 214,175**.

- **Cross-validation** — `GroupKFold`, `n_splits: 3`, grouped on the joint key
  `[year, CVE_MUN]` (`src/supervised/cv.py:62`), so no subclass cell straddles
  folds.
- **Never-overwrite outputs** — figure and metric writers version their
  filenames (`name_v2.png`, `name_v3.png`) rather than overwriting, so a re-run
  never silently replaces a previous result.

**Known reproducibility hazards:**

1. The CDMX crime endpoint is versioned by month; the exact
   `carpetasFGJ_acumulado_2025_01.csv` snapshot may no longer be served.
2. INEGI retired the classic BIE indicator ids on 2025-12-01 (see §5).
3. AlphaEarth extraction needs a Google Earth Engine account and a one-off
   manual export; it is not fully scripted end to end.
4. Marco Geoestadístico vintage is unrecorded (see §2), so boundary joins may
   not reproduce exactly.

---

## Citing the sources

If you reuse the derived data in `data/proc/` or `data/infer/`, cite **both**
this repository **and** the original providers listed above. Citing only this
repository misattributes the underlying observations.

Machine-readable citation metadata is in **`CITATION.cff`**. Use it for this
repository; use each provider's own required citation for the observations.

**This project's own licensing** (see `NOTICE` for the summary):

| Component | License | File |
|---|---|---|
| Source code (`src/`) | MIT | `LICENSE` |
| Documentation & figures (`docs/`, `**/resources/`) | CC BY 4.0 | `LICENSE-DOCS` |
| Derived data (`data/proc/`, `data/infer/`) | CC BY 4.0 | `LICENSE-DOCS` |
| Raw input data (`data/raw/`, `data/spatial/`) | **Provider's own terms** | this file |

**Repository gaps to close before redistribution:**

- `data/sample/showcase_streets_2025.parquet` is **not tracked** (caught by the
  `*.parquet` ignore rule), although `src/showcase/app.py` and `app_esp.py` both
  read it. A fresh clone cannot run the showcase app until the file is
  regenerated with `python -m src.showcase.prepare_sample`, which itself needs
  the full upstream pipeline. Consider force-adding it (`git add -f`) since it is
  a deliberately small sample.
