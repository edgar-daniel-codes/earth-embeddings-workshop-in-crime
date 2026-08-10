### Summer Internship - Earth Embeddings
### ETL - Get Socio-economical Data
### By Edgar Daniel


"""
get_socioecon_data.py
=====================================================================
Extract, for Mexico City (CDMX, entidad = "09"):

  (A) Manzana (MZA / city-block) level sociodemographic data
      -> Censo de Poblacion y Vivienda 2020, "Principales resultados
         por AGEB y manzana urbana" (bulk CSV, NOT the Indicadores API).
         This is a single 2020 cross-section: the census is decennial,
         so there is NO 2022-2024 manzana panel.

  (B) Yearly macroeconomic variables, 2022-2024:
        - inflation      : INEGI Indicadores API, INPC id 628194
                           (annual inflation derived Dec/Dec and avg/avg)
        - unemployment   : INEGI Indicadores API, id 444612
                           (national "tasa de desocupacion")
        - interest rate  : Banxico SIE API, series SF61745 (tasa objetivo)
                           -- interest rates are a Banxico product, not INEGI

Design notes
------------
* INEGI Indicadores API v2.0 endpoint:
    .../jsonxml/INDICATOR/{id}/es/{area}/{recent}/{source}/2.0/{token}?type=json
  area:   "00" national, "07000009" style geo keys for lower levels
  source: "BIE" (economic) or "BISE" (sociodemographic)
* INPC / unemployment BIE series above are national. CDMX-level
  unemployment exists under *different* (state ENOE) indicator ids;
  swap them into INEGI_INDICATORS if you need the entidad breakdown.
* Indicator ids are re-based periodically -> verify in the
  "Constructor de consultas": https://www.inegi.org.mx/app/querybuilder2/

Requirements:  pip install requests pandas
               (optional) pip install python-dotenv   # nicer .env parsing
Tokens (both free) are read from a .env file, e.g.:

    # .env
    INEGI_TOKEN=your-inegi-token
    BANXICO_TOKEN=your-banxico-token
    ENTIDAD=09          # optional; defaults below
    YEAR_START=2022
    YEAR_END=2024

    cfg = Config.from_env(".env")
    data = build_cdmx_dataset(cfg)

Real process environment variables override the .env file. If python-dotenv
is not installed, a small built-in parser is used instead.

Get tokens (both free):
  INEGI  -> https://www.inegi.org.mx/app/api/indicadores/interna_v1_1/tokenVerify.aspx
  Banxico-> https://www.banxico.org.mx/SieAPIRest/service/v1/token
=====================================================================
"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------


from __future__ import annotations

import io
import os
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import geopandas as gpd 
import requests


### -------------------------------------------------------------------------------
### Functions and Classes ---------------------------------------------------------


# Typed errors (let callers distinguish "wrong id" from "bad token")


class INEGIError(RuntimeError):
    """Any non-2xx / unusable response from the INEGI Indicadores API."""


class INEGITokenError(INEGIError):
    """Token invalid / not activated -- fatal, retrying other ids won't help."""


class INEGINoResults(INEGIError):
    """ErrorCode 100: no series for this id/area/source -- try another id."""


# Configuration

INEGI_BASE = "https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml"
BANXICO_BASE = "https://www.banxico.org.mx/SieAPIRest/service/v1"

# Census "AGEB y manzana urbana" open-data zip, one per entidad.
CENSO_MZA_URL = (
    "https://www.inegi.org.mx/contenidos/programas/ccpv/2020/"
    "datosabiertos/ageb_manzana/ageb_mza_urbana_{ent}_cpv2020_csv.zip"
)

# INEGI indicator ids. NOTE: INEGI restructured the BIE on 2025-12-01, merging
# per-geography series into single keys and RETIRING the old classic ids.
# Each concept lists candidate ids tried in order; run `probe_inegi()` (below)
# with your token to discover / confirm the current key for a concept.

INEGI_INDICATORS = {
    # Inflación General (annual % change of the INPC) -- CONFIRMED post-migration.
    "inflation":    {"ids": ["628229"], "source": "BIE", "kind": "rate"},
    # National unemployment rate ("tasa de desocupacion", ENOE mensual).
    # 444612 is the *classic* key (likely retired). Confirm the new one 
    "unemployment": {"ids": ["444612"], "source": "BIE", "kind": "rate"},
    # INPC *index level* (only if you'd rather derive inflation yourself).
    "inpc_index":   {"ids": ["628194", "216064"], "source": "BIE", "kind": "index"},
}

# Geographic-area codes to try (the single-key migration changed area behavior;
# national is usually "00", some price/coyuntura series answer to "0700").
INEGI_AREAS = ("00", "0700")

# Banxico series ids.
BANXICO_SERIES = {
    "interest_rate": "SF61745",   # tasa objetivo (overnight target rate)
}

# A compact, analysis-ready subset of the ~230 census columns.
CENSUS_KEEP_COLS = [
    "ENTIDAD", "NOM_ENT", "MUN", "NOM_MUN", "LOC", "NOM_LOC", "AGEB", "MZA",
    "POBTOT", "POBFEM", "POBMAS",          # total / female / male population
    "P_0A2", "P_3YMAS", "P_15YMAS", "P_60YMAS",  # age structure
    "REL_H_M",                             # men-per-100-women
    "POB0_14", "POB15_64", "POB65_MAS",    # dependency-ratio inputs
    "PEA", "PE_INAC", "POCUPADA", "PDESOCUP",    # economic activity
    "GRAPROES",                            # avg schooling years
    "PSINDER", "PDER_SS",                  # health-service (non)affiliation
    "TVIVHAB", "TVIVPARHAB", "PROM_OCUP", "OCUPVIVPAR",  # dwellings/occupancy
    "VPH_S_ELEC", "VPH_AGUAFV", "VPH_NODREN",  # dwellings lacking services
    "VPH_INTER", "VPH_PC", "VPH_CEL",      # ICT access
]


# ----------------------------------------------------------------------
# Census numeric rules (INEGI data dictionary)
# ----------------------------------------------------------------------
# From the "diccionario_de_datos" CSV shipped inside the same census zip
# (also published at inegi.org.mx/programas/ccpv/2020, AGEB-manzana urbana):
#   * every indicator is numeric and non-negative;
#   * counts (POBTOT, P_*, POB*, PEA, ..., TVIV*, VPH_*) are integers;
#   * REL_H_M, GRAPROES, PROM_OCUP are 2-decimal averages/ratios whose
#     printed formats (###.## / ##.## / ##.##) bound their valid ranges;
#   * censored cells: '*' = suppressed by confidentiality (manzanas with
#     1-2 inhabited dwellings), 'N/D' = not available. Both -- plus any
#     out-of-range value -- become NaN and are median-imputed.
CENSUS_CENSOR_MARKS = {"*", "N/D", "-", ""}

CENSUS_DECIMAL_RULES = {           # col -> (min, max), kept as floats
    "REL_H_M": (0.0, 999.99),      # men per 100 women
    "GRAPROES": (0.0, 99.99),      # average schooling years
    "PROM_OCUP": (0.0, 99.99),     # average occupants per inhabited dwelling
}

CENSUS_ID_COLS = {"CVEGEO", "ENTIDAD", "NOM_ENT", "MUN", "NOM_MUN",
                  "LOC", "NOM_LOC", "AGEB", "MZA"}


def apply_census_numeric_rules(
    df: pd.DataFrame,
    impute_group: str = "MUN",
) -> pd.DataFrame:
    """
    Enforce the data-dictionary numeric rules on every non-identifier column
    and median-impute censored cells.

    Per column: censor marks ('*', 'N/D', '-') and values outside the
    dictionary range become NaN; NaNs are filled with the median of the
    column within `impute_group` (default: municipio), falling back to the
    entidad-wide median for groups that are fully censored. Count columns
    are then rounded to nullable integers; REL_H_M / GRAPROES / PROM_OCUP
    stay as floats.
    """
    df = df.copy()
    for c in (c for c in df.columns if c not in CENSUS_ID_COLS):
        raw = df[c].astype(str).str.strip()
        vals = pd.to_numeric(raw.mask(raw.isin(CENSUS_CENSOR_MARKS)),
                             errors="coerce")
        lo, hi = CENSUS_DECIMAL_RULES.get(c, (0.0, None))
        vals = vals.where(vals >= lo)
        if hi is not None:
            vals = vals.where(vals <= hi)
        n_censored = int(vals.isna().sum())
        if n_censored:
            if impute_group in df.columns:
                vals = vals.fillna(
                    vals.groupby(df[impute_group]).transform("median"))
            vals = vals.fillna(vals.median())
            print(f"[census] {c}: {n_censored:,} censored/out-of-range cells "
                  f"median-imputed", flush=True)
        if c not in CENSUS_DECIMAL_RULES:
            vals = vals.round().astype("Int64")
        df[c] = vals
    return df


@dataclass
class Config:
    inegi_token: str = ""
    banxico_token: str = ""
    entidad: str = "09"          # CDMX
    year_start: int = 2022
    year_end: int = 2024
    timeout: int = 60

    @classmethod
    def from_env(cls, path: str | Path = ".env") -> "Config":
        """
        Build a Config from a .env file. Recognized keys (real process
        environment variables take precedence over the file):

            INEGI_TOKEN=your-inegi-token
            BANXICO_TOKEN=your-banxico-token
            ENTIDAD=09          # optional
            YEAR_START=2022     # optional
            YEAR_END=2024       # optional
            TIMEOUT=60          # optional

        Uses python-dotenv if available, else a minimal built-in parser.
        """
        values = _read_dotenv(Path(path).expanduser())

        def get(key: str, default: str = "") -> str:
            # process env wins over the .env file (12-factor convention)
            return os.environ.get(key, values.get(key, default))

        cfg = cls(
            inegi_token=get("INEGI_TOKEN").strip(),
            banxico_token=get("BANXICO_TOKEN").strip(),
            entidad=str(get("ENTIDAD", cls.entidad)).strip(),
            year_start=int(get("YEAR_START", str(cls.year_start))),
            year_end=int(get("YEAR_END", str(cls.year_end))),
            timeout=int(get("TIMEOUT", str(cls.timeout))),
        )
        cfg.require_tokens()
        return cfg

    def require_tokens(self) -> None:
        """Fail fast with a clear message if either token is missing."""
        missing = [name for name, val in
                   (("INEGI_TOKEN", self.inegi_token),
                    ("BANXICO_TOKEN", self.banxico_token)) if not val]
        if missing:
            raise ValueError(
                f"Missing {missing} in the .env file (or process environment)."
            )


def _read_dotenv(path: Path) -> dict[str, str]:
    """
    Return the key/value pairs from a .env file as a dict, without mutating
    os.environ. Prefers python-dotenv; falls back to a small parser that
    handles `KEY=value`, `export KEY=value`, comments and quoted values.
    """
    if not path.is_file():
        raise FileNotFoundError(f".env file not found: {path}")

    try:
        from dotenv import dotenv_values  # python-dotenv, if installed
        return {k: v for k, v in dotenv_values(path).items() if v is not None}
    except ImportError:
        pass

    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.strip().strip('"').strip("'")
        out[key.strip()] = val
    return out

### -------------------------------------------------------------------------------
### Main --------------------------------------------------------------------------


# (A) Manzana-level sociodemographic data (Census 2020)

def fetch_census_manzana(
    entidad: str = "09",
    keep_cols: Iterable[str] | None = CENSUS_KEEP_COLS,
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Download and parse the Census-2020 AGEB/manzana-urbana file for one
    entidad, returning ONLY true urban manzanas (AGEB != '0000', MZA != '000').

    Returns a tidy DataFrame with a 16-digit `CVEGEO` manzana key and a
    sociodemographic subset typed per the INEGI data dictionary (integer
    counts, bounded 2-decimal averages). Censored cells ('*' confidentiality
    suppression, 'N/D' not available) are median-imputed within the
    municipio -- see `apply_census_numeric_rules`.
    """
    url = CENSO_MZA_URL.format(ent=entidad)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        # The data CSV lives under .../conjunto_de_datos/conjunto_de_datos_*.csv
        csv_name = next(
            n for n in zf.namelist()
            if n.endswith(".csv") and "conjunto_de_datos" in n and "/diccionario" not in n
        )
        with zf.open(csv_name) as fh:
            df = pd.read_csv(fh, dtype=str, encoding="utf-8", low_memory=False)

    df.columns = [c.strip().upper() for c in df.columns]

    # Keep only urban manzanas; drop entidad/municipio/AGEB aggregate rows.
    df = df[(df["AGEB"] != "0000") & (df["MZA"] != "000")].copy()

    # Zero-padded canonical geo key: ENT(2)+MUN(3)+LOC(4)+AGEB(4)+MZA(3) = 16.
    df["CVEGEO"] = (
        df["ENTIDAD"].str.zfill(2) + df["MUN"].str.zfill(3) + df["LOC"].str.zfill(4)
        + df["AGEB"].str.zfill(4) + df["MZA"].str.zfill(3)
    )

    if keep_cols:
        cols = ["CVEGEO"] + [c for c in keep_cols if c in df.columns]
        df = df[cols]

    # Dictionary-based numeric rules + median imputation of censored cells.
    df = apply_census_numeric_rules(df)

    return df.reset_index(drop=True)


# (B1) INEGI Indicadores API


def inegi_series(
    indicator_id: str,
    token: str,
    area: str = "00",
    source: str = "BIE",
    recent: bool = False,
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Fetch one INEGI indicator series (v2.0 API) as a DataFrame with a
    parsed `date` (period start) and numeric `value`. Handles annual,
    quarterly, monthly and bi-weekly TIME_PERIOD formats.
    """
    if not token:
        raise ValueError("Missing INEGI token (set INEGI_TOKEN or pass explicitly).")

    recent_flag = "true" if recent else "false"
    url = (f"{INEGI_BASE}/INDICATOR/{indicator_id}/es/{area}/"
           f"{recent_flag}/{source}/2.0/{token}?type=json")

    resp = requests.get(url, timeout=timeout)
    if not resp.ok:
        # INEGI returns HTTP 400 for BOTH an invalid/unactivated token AND an
        # indicator id absent from the requested bank. 
        body = resp.text or ""
        safe_url = url.replace(token, f"***{token[-4:]}")
        low = body.lower()
        msg = (f"INEGI {resp.status_code} for indicator {indicator_id} "
               f"(area={area}, source={source}). Body: {body[:300]!r}")
        if "token" in low:
            raise INEGITokenError(msg + " -- token looks invalid/unactivated.")
        if "errorcode:100" in low.replace(" ", "") or "no se encontraron" in low:
            raise INEGINoResults(msg + f"\nURL: {safe_url}")
        raise INEGIError(msg + f"\nURL: {safe_url}")

    payload = resp.json()

    series = payload.get("Series") or payload.get("SERIES")
    if not series:
        raise INEGINoResults(
            f"No 'Series' for indicator {indicator_id}/{area}/{source}: {payload}")

    obs = series[0].get("OBSERVATIONS", [])
    rows = [{"period": o["TIME_PERIOD"], "value": o["OBS_VALUE"]} for o in obs]
    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = _parse_inegi_period(df["period"])
    df["indicator_id"] = indicator_id
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def inegi_series_any(
    ids: Sequence[str],
    token: str,
    areas: Sequence[str] = ("00",),
    source: str = "BIE",
    recent: bool = False,
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Resilient fetch: try each (id, area) combination until one returns a
    non-empty series. A bad token aborts immediately (retrying is pointless);
    "no results" just advances to the next candidate. On total failure, emit
    a warning and return an empty frame instead of raising -- so an upstream
    pipeline can still assemble whatever else succeeded.
    """
    last: Exception | None = None
    for ind in ids:
        for area in areas:
            try:
                df = inegi_series(ind, token, area=area, source=source,
                                  recent=recent, timeout=timeout)
                if not df.empty:
                    return df
                last = INEGINoResults(f"{ind}/{area}/{source} returned 0 rows")
            except INEGITokenError:
                raise                      # fatal for every candidate
            except INEGIError as e:
                last = e                   # try the next candidate
            except requests.RequestException as e:
                last = e                   # network/transport hiccup
    warnings.warn(f"INEGI: no data for ids={list(ids)} areas={list(areas)} "
                  f"source={source}. Last error: {last}")
    return pd.DataFrame(columns=["date", "value", "indicator_id"])


def probe_inegi(
    candidate_ids: Sequence[str],
    token: str,
    areas: Sequence[str] = INEGI_AREAS,
    sources: Sequence[str] = ("BIE", "BISE"),
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Discovery helper for the post-2025 BIE re-keying. Probes every
    (id, area, source) combination and reports which ones resolve, so you can
    pin the current key for a concept without guessing. Returns a DataFrame:
        indicator_id | area | source | status | n_obs | last_date | last_value
    Usage:
        probe_inegi(["444612", "628230", "628231", "628232"], token)
    then read off the row with status == "ok" and copy its id into
    INEGI_INDICATORS. Candidate ids come from the Constructor de consultas.
    """
    rows = []
    for ind in candidate_ids:
        for source in sources:
            for area in areas:
                rec = {"indicator_id": ind, "area": area, "source": source,
                       "status": "", "n_obs": 0,
                       "last_date": pd.NaT, "last_value": pd.NA}
                try:
                    df = inegi_series(ind, token, area=area, source=source,
                                      recent=False, timeout=timeout)
                    if df.empty:
                        rec["status"] = "empty"
                    else:
                        rec["status"] = "ok"
                        rec["n_obs"] = len(df)
                        rec["last_date"] = df["date"].iloc[-1]
                        rec["last_value"] = df["value"].iloc[-1]
                except INEGITokenError as e:
                    rec["status"] = f"token_error: {e}"
                except INEGINoResults:
                    rec["status"] = "no_results"
                except Exception as e:                       # noqa: BLE001
                    rec["status"] = f"error: {type(e).__name__}"
                rows.append(rec)
    out = pd.DataFrame(rows)
    hits = out[out["status"] == "ok"]
    if hits.empty:
        warnings.warn("probe_inegi: no candidate resolved. Pull fresh ids from "
                      "inegi.org.mx/app/querybuilder2/ (new BIE keys).")
    return out.sort_values("status").reset_index(drop=True)


def _parse_inegi_period(periods: pd.Series) -> pd.Series:
    """Map INEGI TIME_PERIOD strings to period-start Timestamps."""
    def to_ts(p: str):
        p = str(p).strip()
        if "/" in p:                       # "2022/01" (month), "2022/03" (quarter-as-month)
            y, m = p.split("/")[:2]
            return pd.Timestamp(int(y), int(m), 1)
        if len(p) == 4 and p.isdigit():    # "2022" annual
            return pd.Timestamp(int(p), 1, 1)
        return pd.NaT
    return periods.map(to_ts)


# (B2) Banxico SIE API


def banxico_series(
    serie_id: str,
    token: str,
    start: str,
    end: str,
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Fetch one Banxico SIE series over [start, end] (YYYY-MM-DD) as a
    DataFrame with parsed `date` and numeric `value`.
    """
    if not token:
        raise ValueError("Missing Banxico token (set BANXICO_TOKEN or pass explicitly).")

    url = f"{BANXICO_BASE}/series/{serie_id}/datos/{start}/{end}"
    resp = requests.get(url, headers={"Bmx-Token": token,
                                      "Accept": "application/json"}, timeout=timeout)
    resp.raise_for_status()
    datos = resp.json()["bmx"]["series"][0].get("datos", [])
    df = pd.DataFrame(datos)
    if df.empty:
        return df.assign(date=pd.NaT, value=pd.NA)
    df["date"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y")
    df["value"] = pd.to_numeric(df["dato"].str.replace(",", ""), errors="coerce")
    df["serie_id"] = serie_id
    return df[["date", "value", "serie_id"]].sort_values("date").reset_index(drop=True)


# Yearly collapse of the macro block

def annual_from_rate(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Collapse a monthly *rate* series (e.g. annual inflation = 910406, already
    a percentage) to yearly: December value (headline / Dec-Dec) and the
    within-year mean. Returns an empty frame unchanged.
    """
    if df.empty:
        return pd.DataFrame(columns=["year", f"{prefix}_dec_pct", f"{prefix}_avg_pct"])
    g = df.dropna(subset=["date"]).copy()
    g["year"] = g["date"].dt.year
    dec = g[g["date"].dt.month == 12].set_index("year")["value"].rename(f"{prefix}_dec_pct")
    avg = g.groupby("year")["value"].mean().rename(f"{prefix}_avg_pct")
    return pd.concat([dec, avg], axis=1).reset_index()


def annual_inflation_from_inpc(inpc: pd.DataFrame) -> pd.DataFrame:
    """
    Alternative inflation path: from a monthly INPC *index* series, compute
    Dec/Dec and average/average annual inflation. Use only if you fetch an
    index id (e.g. 216064) instead of the direct inflation series 910406.
    """
    if inpc.empty:
        return pd.DataFrame(columns=["year", "inflation_dec_pct", "inflation_avg_pct"])
    s = inpc.set_index("date")["value"].sort_index()
    dec = s[s.index.month == 12]
    dec_yoy = (dec.pct_change() * 100).rename("inflation_dec_pct")
    avg = s.groupby(s.index.year).mean()
    avg_yoy = (avg.pct_change() * 100)
    out = dec_yoy.reset_index()
    out["year"] = pd.to_datetime(out["date"]).dt.year
    out["inflation_avg_pct"] = avg_yoy.reindex(out["year"]).values
    return out[["year", "inflation_dec_pct", "inflation_avg_pct"]]


def to_annual_mean(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Collapse a sub-annual (monthly/quarterly/daily) series to a yearly mean."""
    g = df.dropna(subset=["date"]).copy()
    g["year"] = g["date"].dt.year
    out = g.groupby("year")["value"].mean().rename(value_name).reset_index()
    return out


# Orchestrator

def build_macro_yearly(cfg: Config) -> pd.DataFrame:
    """
    Assemble the yearly macro table (inflation, unemployment, interest rate).
    Every source is fetched independently; a failure in one leaves its
    column(s) as NaN and emits a warning instead of aborting the whole run.
    The output schema is stable regardless of which sources succeeded.
    """
    y0, y1 = cfg.year_start, cfg.year_end
    macro = pd.DataFrame({"year": list(range(y0, y1 + 1))})

    def _safe(label: str, fn):
        try:
            part = fn()
            return part if part is not None else pd.DataFrame()
        except INEGITokenError as e:
            warnings.warn(f"[{label}] token error (all INEGI columns will be NaN): {e}")
        except Exception as e:                       # noqa: BLE001 - deliberate catch-all
            warnings.warn(f"[{label}] failed, column left as NaN: {e}")
        return pd.DataFrame()

    def _inflation():
        c = INEGI_INDICATORS["inflation"]
        df = inegi_series_any(c["ids"], cfg.inegi_token, areas=INEGI_AREAS,
                              source=c["source"], timeout=cfg.timeout)
        return annual_from_rate(df, "inflation")

    def _unemployment():
        c = INEGI_INDICATORS["unemployment"]
        df = inegi_series_any(c["ids"], cfg.inegi_token, areas=INEGI_AREAS,
                              source=c["source"], timeout=cfg.timeout)
        return to_annual_mean(df, "unemployment_rate_pct")

    def _interest_rate():
        df = banxico_series(BANXICO_SERIES["interest_rate"], cfg.banxico_token,
                            start=f"{y0}-01-01", end=f"{y1}-12-31", timeout=cfg.timeout)
        return to_annual_mean(df, "interest_rate_pct")

    for part in (_safe("inflation", _inflation),
                 _safe("unemployment", _unemployment),
                 _safe("interest_rate", _interest_rate)):
        if not part.empty:
            macro = macro.merge(part, on="year", how="left")

    # Guarantee a stable schema even when a source failed.
    expected = ["inflation_dec_pct", "inflation_avg_pct",
                "unemployment_rate_pct", "interest_rate_pct"]
    for col in expected:
        if col not in macro.columns:
            macro[col] = pd.NA
    return macro[["year"] + expected].sort_values("year").reset_index(drop=True)


def build_cdmx_dataset(cfg: Config | None = None) -> dict[str, pd.DataFrame]:
    """
    Full extraction for CDMX. Returns:
      {"manzana": <2020 sociodemographic cross-section>,
       "macro":   <yearly inflation/unemployment/rate>}
    Both blocks fail soft: an empty DataFrame (plus a warning) is returned for
    whichever source is unavailable, so the other still comes back usable.
    """
    cfg = cfg or Config()
    try:
        manzana = fetch_census_manzana(cfg.entidad, timeout=cfg.timeout)
    except Exception as e:                           # noqa: BLE001
        warnings.warn(f"[manzana] census download/parse failed: {e}")
        manzana = pd.DataFrame()
    macro = build_macro_yearly(cfg)                  # already resilient
    return {"manzana": manzana, "macro": macro}



# Showcase CDMX database for benchmark on models 

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract CDMX manzana sociodemographics + yearly macro variables."
    )
    parser.add_argument("--env", default=".env",
                        help="Path to the .env file with API tokens (default: .env).")
    args = parser.parse_args()

    cfg = Config.from_env(args.env)   # raises if tokens are missing

    # read geo shape
    gpd_mza = gpd.read_file("./data/spatial/pd/09m.shp")

    # Resilient extraction: each block returns data or an empty frame + warning.
    data = build_cdmx_dataset(cfg)
    mza, macro = data["manzana"], data["macro"]

    if not mza.empty:
        # ``CVEGEO`` is already built (zero-padded) by fetch_census_manzana
        mza = gpd.GeoDataFrame(
            mza.merge(
                gpd_mza[["CVEGEO", "geometry"]],
                how="left",
                on="CVEGEO",
            ),
            geometry="geometry",
            crs=gpd_mza.crs,
        )
        mza["geometry"] = mza.geometry.centroid

    if not mza.empty:
        print(f"[MZA] entidad {cfg.entidad}: {len(mza):,} manzanas, {mza.shape[1]} cols")
        print(mza.head(3).to_string(index=False))
        mza.to_parquet(f"./data/raw/cdmx_manzana_{cfg.entidad}_2020.parquet", index=False)
    else:
        print("[MZA] no manzana data (see warnings).")
