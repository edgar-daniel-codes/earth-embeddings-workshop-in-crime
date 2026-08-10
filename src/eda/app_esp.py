### Summer Internship - Earth Embeddings
### EDA - Tablero de Streamlit (versión en español)
### By Edgar Daniel


"""

Tablero de Streamlit para el EDA de delitos de la CDMX, con la interfaz y
las figuras en español.

Ejecutar:  streamlit run src/eda/app_esp.py [-- <archivo_config.yml>]
El impacto está en las gráficas; la interfaz es intencionalmente mínima.

"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow both `streamlit run src/eda/app_esp.py` and `python -m src.eda.app_esp`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import yaml
from streamlit.components.v1 import html as st_html

from src.eda import core_esp as ec
from src.eda import dq_esp as dq

# Config file reading

parser = argparse.ArgumentParser(description="Ejecuta el flujo EDA/FGR.")
parser.add_argument(
    "config_file",
    nargs="?",
    default="eda_fgr_conf_esp.yml",
    help="Ruta al archivo YAML de configuración (por defecto: eda_fgr_conf_esp.yml).",
)

args, _ = parser.parse_known_args()

CFG_FILE = args.config_file

REPO_ROOT = Path(__file__).resolve().parents[2]

with open(REPO_ROOT / "config" / CFG_FILE, "r") as f:
    _cfg = yaml.safe_load(f)

INPUT_FILE = REPO_ROOT / _cfg["paths"]["input_file"]
PD_SHP_PATH = REPO_ROOT / _cfg["paths"]["pd_shp_path"]
STREETS_SHP_PATH = REPO_ROOT / _cfg["paths"]["streets_shp_path"]

SELECTED_COLUMNS_DICT = _cfg["selected_columns"]
GEO_INDEX_COL = _cfg["geo_index_col"]

START_YEAR = _cfg["start_year"]
BBOX = _cfg["bbox"]

EPSG_PROJ = _cfg["dbscan"]["epsg_proj"]
EPSILON_M = _cfg["dbscan"]["epsilon_m"]
MIN_SAMPLES = _cfg["dbscan"]["min_samples"]
ALPHA_HULL = _cfg["dbscan"]["alpha_hull"]

PAGE_TITLE = _cfg["labels"]["title"]
PAGE_CAPTION = _cfg["labels"]["caption"]


st.set_page_config(page_title="EDA de Delitos", layout="wide",
                   initial_sidebar_state="expanded")
ec.apply_theme()
st.session_state.setdefault("exports", {})   # nombre -> bytes/str, para exportar todo


### -------------------------------------------------------------------------------
### Cached data access ------------------------------------------------------------


@st.cache_data(show_spinner="Cargando y preparando los datos...")
def get_data():
    return ec.prepare(ec.load_raw(INPUT_FILE, SELECTED_COLUMNS_DICT),
                      SELECTED_COLUMNS_DICT)


@st.cache_data(show_spinner="Cargando las geometrías de las alcaldías...")
def get_districts():
    return ec.load_districts(PD_SHP_PATH)


@st.cache_data(show_spinner="Agregando los conteos por alcaldía...")
def get_cvegeo(year_range, categories, crimes, districts):
    flt = ec.filter_data(get_data(), year_range, categories, crimes, districts)
    return ec.cvegeo_aggregate(flt, get_districts())


@st.cache_data(show_spinner="Dividiendo calles en las intersecciones (una vez, lento)...")
def get_streets_split():
    return ec.load_streets(STREETS_SHP_PATH)


@st.cache_data(show_spinner="Agregando los conteos por segmento de calle...")
def get_streets(year_range, categories, crimes, districts):
    flt = ec.filter_data(get_data(), year_range, categories, crimes, districts)
    return ec.street_aggregate(flt, get_streets_split())


### -------------------------------------------------------------------------------
### Render helpers (Including artefacts for the export-all) -----------------------


def show(fig, name: str, dpi: int = 150) -> None:
    png = ec.fig_to_png_bytes(fig, dpi)
    st.session_state["exports"][f"{name}.png"] = png
    st.pyplot(fig, use_container_width=True)
    st.download_button("Descargar PNG", png, file_name=f"{name}.png",
                       mime="image/png", key=f"dl_{name}")


def show_gif(builder, data, name: str, label: str) -> None:
    if st.button(label, key=f"btn_{name}"):
        with st.spinner("Construyendo la animación..."):
            st.session_state[name] = builder(data)
    if name in st.session_state:
        st.session_state["exports"][f"{name}.gif"] = st.session_state[name]
        st.image(st.session_state[name])
        st.download_button("Descargar GIF", st.session_state[name],
                           file_name=f"{name}.gif", mime="image/gif",
                           key=f"dl_{name}")


### -------------------------------------------------------------------------------
### Sidebar filters ---------------------------------------------------------------

df = get_data()

st.sidebar.title("Filtros")
years = sorted(int(y) for y in df["year"].dropna().unique() if y >= START_YEAR)
if not years:
    st.warning(f"No hay incidentes a partir de start_year={START_YEAR}.")
    st.stop()
year_range = st.sidebar.slider("Rango de años", years[0], years[-1],
                               (max(START_YEAR, years[0]), years[-1]))

cats = sorted(df["crime_cat"].dropna().unique())
sel_cats = st.sidebar.multiselect("Categoría de delito", cats)

delito_pool = df[df["crime_cat"].isin(sel_cats)] if sel_cats else df
crimes = sorted(delito_pool["crime"].dropna().unique())
sel_crimes = st.sidebar.multiselect("Tipo de delito", crimes)

with st.sidebar.expander("Más filtros"):
    districts = sorted(df["district"].dropna().unique())
    sel_districts = st.multiselect("Alcaldía", districts)

flt = ec.filter_data(df, year_range, sel_cats, sel_crimes, sel_districts)
geo, geo_out = ec.geo_split(flt)

st.sidebar.caption(
    f"{len(flt):,} incidentes seleccionados  ·  {len(geo):,} geolocalizados")

# Export all cached figures to a local folder
st.sidebar.divider()
st.sidebar.subheader("Exportar")
out_dir = st.sidebar.text_input(
    "Carpeta de salida",
    value=str(ec.REPO_ROOT / "docs" / "resources" / "crime_eda_esp"))
st.sidebar.caption(
    f"{len(st.session_state['exports'])} figuras actualmente en caché")
if st.sidebar.button("Exportar todas las figuras en caché"):
    written = ec.export_all(st.session_state["exports"], out_dir)
    st.sidebar.success(f"Se escribieron {len(written)} archivos en {out_dir}")


### -------------------------------------------------------------------------------
### Layout ------------------------------------------------------------------------

st.title(PAGE_TITLE)
st.caption(PAGE_CAPTION)

if flt.empty:
    st.warning("Ningún incidente coincide con los filtros actuales.")
    st.stop()

(tab_over, tab_time, tab_space, tab_dyn,
 tab_dist, tab_street, tab_quality) = st.tabs(
    ["Panorama", "Temporal", "Espacial",
     "Dinámicas", "Alcaldías", "Calles", "Calidad de datos"])

with tab_over:
    show(ec.top_crime_types(flt), "01_top_crime_types")
    show(ec.incidents_by_category(flt), "02_incidents_by_category")
    show(ec.annual_volume(flt), "04_annual_volume")

with tab_time:
    show(ec.monthly_time_series(flt), "03_monthly_time_series")
    show(ec.time_series_by_category(flt), "05_time_series_by_category")
    show(ec.seasonality_month_dow(flt), "06_seasonality_month_dow")
    show(ec.hourly_rhythm(flt), "07_hourly_rhythm")
    show(ec.incidents_by_hour(flt), "08_incidents_by_hour")
    show(ec.intraday_by_category(flt), "09_intraday_by_category")

with tab_space:
    show(ec.spatial_outliers(geo, geo_out), "13_spatial_outliers")
    show(ec.kde_overall(geo), "14_kde_overall")
    show(ec.kde_top3_categories(geo), "15_kde_top3_categories")
    st.subheader("Mapa de calor de densidad H3")
    if st.button("Construir mapa de calor H3", key="btn_h3"):
        with st.spinner("Agrupando en celdas H3..."):
            st.session_state["h3"] = ec.h3_heatmap(geo)._repr_html_()
    if "h3" in st.session_state:
        st.session_state["exports"]["16_h3_heatmap.html"] = st.session_state["h3"]
        st_html(st.session_state["h3"], height=600)

with tab_dyn:
    st.caption("Las animaciones se calculan bajo demanda y se guardan en la "
               "memoria de la sesión.")
    c1, c2, c3 = st.columns(3)
    with c1:
        show_gif(ec.gif_scatter_by_year, geo, "crime_by_year",
                 "Dispersión por año")
    with c2:
        show_gif(ec.gif_kde_by_year, geo, "crime_kde_yearly", "KDE por año")
    with c3:
        show_gif(ec.gif_dbscan_by_year, geo, "crime_dbscan", "DBSCAN por año")

with tab_dist:
    st.caption("Agregación a nivel alcaldía — la unión espacial está en caché.")
    if st.button("Calcular la agregación por alcaldía", key="btn_cvegeo"):
        st.session_state["cvegeo_ready"] = True
    if st.session_state.get("cvegeo_ready"):
        crime_gdf = get_cvegeo(year_range, tuple(sel_cats), tuple(sel_crimes),
                               tuple(sel_districts))
        show(ec.cvegeo_distribution(crime_gdf), "17_per_cvegeo_distribution")
        show_gif(ec.gif_cvegeo_by_year, crime_gdf, "crime_by_year_cvegeo",
                 "Polígonos por año")

with tab_street:
    st.caption("Nivel segmento de calle: las calles se dividen en las "
               "intersecciones (una sola vez, lento) y los incidentes se "
               "asignan al segmento más cercano.")
    if st.button("Calcular la agregación por calle", key="btn_street"):
        st.session_state["street_ready"] = True
    if st.session_state.get("street_ready"):
        street_gdf = get_streets(year_range, tuple(sel_cats), tuple(sel_crimes),
                                 tuple(sel_districts))
        show(ec.street_distribution(street_gdf), "18_per_street_distribution")
        show_gif(ec.gif_street_by_year, street_gdf, "crime_by_year_street",
                 "Segmentos por año")

with tab_quality:
    st.caption("Perfil agnóstico del esquema de los datos — las fechas y las "
               "coordenadas se infieren (EN/ES); todo lo demás es numérico o "
               "categórico.")
    schema = dq.infer_schema(df)

    st.subheader("Perfil de columnas")
    st.dataframe(dq.profile_df(df), use_container_width=True)

    st.subheader("Valores faltantes")
    show(dq.missingness_bar(df), "27_missingness_bar")
    show(dq.null_heatmap(df), "28_null_heatmap")
    if schema["datetime"]:
        dcol = st.selectbox("Columna de fecha para los nulos en el tiempo",
                            schema["datetime"])
        show(dq.nulls_over_time(df, dcol), "29_nulls_over_time")

    if schema["numeric"]:
        st.subheader("Columnas numéricas")
        st.dataframe(dq.numeric_summary(df), use_container_width=True)
        ncol = st.selectbox("Columna numérica", schema["numeric"])
        show(dq.numeric_distribution(df, ncol), f"30_dist_{ncol}")
        show(dq.numeric_cdf(df, ncol), f"31_cdf_{ncol}")
        show(dq.numeric_boxplot(df, ncol), f"32_box_{ncol}")

    if schema["categorical"]:
        st.subheader("Columnas categóricas")
        ccol = st.selectbox("Columna categórica", schema["categorical"])
        show(dq.categorical_frequency(df, ccol), f"33_freq_{ccol}")

    if schema["lat_col"] and schema["lon_col"]:
        st.subheader("Verificación geoespacial")
        show(dq.geo_scatter(df, schema["lat_col"], schema["lon_col"]),
             "34_geo_scatter")
