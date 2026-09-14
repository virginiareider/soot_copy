from pathlib import Path
import re

import pandas as pd
import streamlit as st

from soot_tool.graphing import build_figure

st.set_page_config(page_title="City / Season Dataset Explorer", layout="wide")

DATASET_DIR = Path("datasets")
SEASON_ALIASES = {
    "spring": "spring",
    "summer": "summer",
    "autumn": "autumn",
    "fall": "fall",
    "winter": "winter",
}


def normalize_text(value: str) -> str:
    return re.sub(r"[_\-\s]+", " ", str(value).strip()).strip().lower()


def infer_dataset_metadata(dataset_path: Path) -> dict[str, str]:
    stem = dataset_path.stem
    stem_tokens = [token for token in re.split(r"[_\-\s]+", stem) if token]

    season = "unknown"
    for token in reversed(stem_tokens):
        token_norm = normalize_text(token)
        if token_norm in SEASON_ALIASES:
            season = SEASON_ALIASES[token_norm]
            stem_tokens = stem_tokens[: stem_tokens.index(token)]
            break

    if stem_tokens:
        city = " ".join(stem_tokens).strip()
    else:
        city = dataset_path.parent.name

    city = city.title() if city else dataset_path.parent.name.title()
    season = season.title() if season != "unknown" else "Unknown"

    return {
        "path": dataset_path,
        "city": city,
        "season": season,
        "dataset_name": dataset_path.name,
    }


@st.cache_data(show_spinner=False)
def load_dataset(dataset_path: Path) -> pd.DataFrame:
    return pd.read_csv(dataset_path)


@st.cache_data(show_spinner=False)
def list_datasets() -> list[dict[str, str]]:
    if not DATASET_DIR.exists():
        DATASET_DIR.mkdir(parents=True, exist_ok=True)

    dataset_files = sorted(DATASET_DIR.rglob("*.csv"))
    return [infer_dataset_metadata(path) for path in dataset_files]


@st.cache_data(show_spinner=False)
def resolve_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lowered:
            return lowered[key]
    return None


st.title("City / Season Dataset Explorer")
st.write(
    "Place your CSV datasets in the datasets folder. The app will auto-discover them and let you choose the city and season."
)

all_datasets = list_datasets()

if not all_datasets:
    st.warning(
        "No CSV datasets were found in the datasets folder yet. Add CSV files named like 'boston_spring.csv' or 'boston-spring.csv'."
    )
    st.stop()

cities = sorted({item["city"] for item in all_datasets})
city = st.selectbox("City", cities)

city_datasets = [item for item in all_datasets if item["city"] == city]
seasons = sorted({item["season"] for item in city_datasets})
season = st.selectbox("Season", seasons)

season_datasets = [item for item in city_datasets if item["season"] == season]

if len(season_datasets) == 1:
    selected_dataset = season_datasets[0]
else:
    dataset_labels = [item["dataset_name"] for item in season_datasets]
    selected_dataset_name = st.selectbox("Dataset", dataset_labels)
    selected_dataset = next(item for item in season_datasets if item["dataset_name"] == selected_dataset_name)

st.caption(f"Loading: {selected_dataset['dataset_name']}")

with st.spinner("Loading dataset..."):
    dataset_df = load_dataset(selected_dataset["path"])

st.dataframe(dataset_df.head(200), use_container_width=True)

altitude_col = resolve_col(
    dataset_df,
    ["Altitude_m_MSL", "Altitude_m", "Altitude", "altitude", "altitude_msl"],
)
ozone_col = resolve_col(
    dataset_df,
    ["Ozone_ppbv", "O3_ppbv", "Ozone", "ozone", "ozone_ppbv"],
)
temp_col = resolve_col(
    dataset_df,
    ["Temperature_C", "Temp_C", "Temperature", "Temp", "temp", "air_temperature"],
)

if altitude_col is None:
    st.error("This dataset does not contain an altitude column. Expected one of: Altitude_m_MSL, Altitude_m, Altitude.")
    st.stop()

if ozone_col is None:
    st.error("This dataset does not contain an ozone column. Expected one of: Ozone_ppbv, O3_ppbv, Ozone.")
    st.stop()

if temp_col is None:
    st.warning("No temperature column was detected. Plotting ozone vs altitude only.")

st.sidebar.header("Graph Controls")
poly_order = st.sidebar.slider("Polynomial Order", min_value=1, max_value=6, value=3, step=1)
window = st.sidebar.slider("Rolling Window (Number of Points)", min_value=3, max_value=400, value=100, step=1)
show_raw = st.sidebar.checkbox("Show Raw Scatter", value=True)
show_smoothed = st.sidebar.checkbox("Show Smoothed Graph", value=True)
smooth_vertical = st.sidebar.checkbox("Smooth Vertically", value=False)
reverse_vertical = st.sidebar.checkbox("Reverse Y-Axis", value=False)

st.subheader("Ozone vs Altitude")
fig_ozone = build_figure(
    dataset_df,
    y_col=altitude_col,
    x_col=ozone_col,
    poly_order=poly_order,
    window=window,
    show_raw=show_raw,
    show_smoothed=show_smoothed,
    smooth_vertical=smooth_vertical,
    reverse_vertical=reverse_vertical,
    title=f"Ozone vs Altitude — {city} ({season})",
)
st.pyplot(fig_ozone)

if temp_col is not None:
    st.subheader("Temperature vs Altitude")
    fig_temp = build_figure(
        dataset_df,
        y_col=altitude_col,
        x_col=temp_col,
        poly_order=poly_order,
        window=window,
        show_raw=show_raw,
        show_smoothed=show_smoothed,
        smooth_vertical=smooth_vertical,
        reverse_vertical=reverse_vertical,
        title=f"Temperature vs Altitude — {city} ({season})",
    )
    st.pyplot(fig_temp)
