from pathlib import Path

import pandas as pd
import streamlit as st

from visualizations.common import smooth_by_altitude
from visualizations.profile_plot import create_profile_figure
from visualizations.scatter_plot import (
    create_ozone_scatter_figure,
    create_temperature_scatter_figure,
)
from visualizations.violin_plot import create_violin_figure

st.set_page_config(page_title="NASA Ozone Profiles", layout="wide")

PROJECT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_DIR / "datasets" / "processed"
CAMPAIGNS = [
    {
        "location": "Chesapeake Bay region",
        "season": "Summer",
        "label": "OWLETS 2017 · Sonde",
        "stem": "OWLETS_2017_SONDE_MCGEE",
    },
    {
        "location": "Louisiana Gulf Coast",
        "season": "Summer",
        "label": "SCOAPE 2024 · Sonde",
        "stem": "SCOAPE_2024_SONDE_STAUFFER",
    },
    {
        "location": "North Atlantic",
        "season": "Summer",
        "label": "AJAX 2018 · Alpha",
        "stem": "AJAX_2018_ALPHA_IRACI",
    },
    {
        "location": "Lake Michigan",
        "season": "Summer",
        "label": "LMOS 2017 · SCI-AVIATION",
        "stem": "LMOS_2017_SCI-AVIATION_CONLEY",
    },
    {
        "location": "Maryland",
        "season": "Summer",
        "label": "Discover-AQ 2011 · Maryland",
        "stem": "DISCOVERAQ_2011-MD_MERGE_MRG",
    },
    {
        "location": "Multiple U.S. locations",
        "season": "Spring-Fall 2023",
        "label": "STAQS 2023 · Walter sonde",
        "stem": "STAQS_2023_SONDE_WALTER",
    },
    {
        "location": "Lake Michigan region",
        "season": "Summer 2023",
        "label": "STAQS 2023 · McKinney sonde",
        "stem": "STAQS_2023_SONDE_MCKINNEY",
    },
    {
        "location": "STAQS study region",
        "season": "Summer 2023",
        "label": "STAQS 2023 · Newchurch drone",
        "stem": "STAQS_2023_DRONE_NEWCHURCH",
    },
    {
        "location": "California and Nevada",
        "season": "Spring-Summer 2019",
        "label": "AJAX 2019 · SCI-AVIATION",
        "stem": "AJAX_2019_SCI-AVIATION_IRACI",
    },
]


@st.cache_data(show_spinner=False)
def load_dataset(dataset_path: Path) -> pd.DataFrame:
    return pd.read_csv(dataset_path)


st.title("NASA Ozone and Temperature Profiles")

available_campaigns = [
    campaign
    for campaign in CAMPAIGNS
    if all(
        (PROCESSED_DIR / folder / f"{campaign['stem']}_avg_{bin_size}m.csv").exists()
        for folder, bin_size in (
            ("averaged_10m", 10),
            ("averaged_50m", 50),
            ("averaged_100m", 100),
            ("averaged_500m", 500),
        )
    )
]

if not available_campaigns:
    st.warning("No processed campaign profiles were found. Run `python process_datasets.py` first.")
    st.stop()

locations = sorted({campaign["location"] for campaign in available_campaigns})
location_filter = st.selectbox(
    "Location",
    ["All locations", *locations],
    key="location_filter_v2",
)
location_campaigns = [
    campaign
    for campaign in available_campaigns
    if location_filter == "All locations" or campaign["location"] == location_filter
]
seasons = sorted({campaign["season"] for campaign in location_campaigns})
season_filter = st.selectbox(
    "Season",
    ["All seasons", *seasons],
    key="season_filter_v2",
)
season_campaigns = [
    campaign
    for campaign in location_campaigns
    if season_filter == "All seasons" or campaign["season"] == season_filter
]
selected_label = st.selectbox(
    "Dataset",
    [campaign["label"] for campaign in season_campaigns],
    key="dataset_selector_v2",
)
selected_campaign = next(
    campaign for campaign in season_campaigns if campaign["label"] == selected_label
)
location = selected_campaign["location"]
season = selected_campaign["season"]
resolution = st.selectbox("Altitude interval", ["10 m", "50 m", "100 m", "500 m"], index=1)
bin_size_m = int(resolution.split()[0])
average_folder = f"averaged_{bin_size_m}m"
dataset_path = (
    PROCESSED_DIR
    / average_folder
    / f"{selected_campaign['stem']}_avg_{bin_size_m}m.csv"
)

with st.spinner("Loading averaged profile..."):
    profile = load_dataset(dataset_path)

required_columns = ["Altitude_km", "Ozone_ppmv"]
has_temperature = "Air_Temperature_C" in profile.columns
if not set(required_columns).issubset(profile.columns):
    st.error(f"The selected profile is missing required columns: {required_columns}.")
    st.stop()

st.caption(f"{selected_campaign['label']} · {location} · {season} · {resolution} bins")
st.caption(
    "Dashed tropopause boundary separates the troposphere below from the stratosphere above. "
    "Its height is adjustable because the real tropopause varies."
)

maximum_smoothing_m = max(100, min(2000, int(profile["Altitude_km"].max() * 1000)))
maximum_smoothing_m = max(100, (maximum_smoothing_m // 100) * 100)
with st.sidebar:
    st.subheader("Profile style")
    show_bin_means = st.checkbox("Show altitude-bin means", value=True)
    show_smooth_curve = st.checkbox("Show smoothed curve", value=True)
    smoothing_span_m = st.slider(
        "Smoothing span (m)",
        min_value=100,
        max_value=maximum_smoothing_m,
        value=min(500, maximum_smoothing_m),
        step=100,
    )
    color_palette = [
        "Teal", "Blue", "Green", "Orange", "Red", "Purple", "Brown", "Black",
    ]
    ozone_color = st.selectbox("Ozone point color", color_palette, index=0)
    temperature_color = st.selectbox("Temperature point color", color_palette, index=3)
    line_color = st.selectbox("Smooth line color", color_palette, index=1)
    marker_shape = st.selectbox(
        "Data marker shape",
        ["circle", "square", "diamond", "triangle", "cross", "x", "star"],
    )
    marker_size = st.slider("Data marker size", min_value=3, max_value=14, value=6)
    curve_style = st.selectbox("Smooth curve style", ["solid", "dash", "dot", "dashdot"])
    reverse_altitude = st.checkbox("Reverse altitude axis", value=False)
    show_tropopause = st.checkbox("Show tropopause boundary", value=True)
    tropopause_km = st.number_input(
        "Tropopause altitude (km)",
        min_value=5.0,
        max_value=20.0,
        value=12.0,
        step=0.5,
        disabled=not show_tropopause,
    )

altitude_min = float(profile["Altitude_km"].min())
altitude_max = float(profile["Altitude_km"].max())
ozone_min = float(profile["Ozone_ppmv"].min())
ozone_max = float(profile["Ozone_ppmv"].max())

temperature_limits = None
if has_temperature:
    temperature_min = float(profile["Air_Temperature_C"].min())
    temperature_max = float(profile["Air_Temperature_C"].max())

with st.expander("Axis ranges", expanded=False):
    altitude_range, ozone_range = st.columns(2)
    altitude_limits = altitude_range.slider(
        "Altitude (km)",
        min_value=altitude_min,
        max_value=altitude_max,
        value=(altitude_min, altitude_max),
        step=max(bin_size_m / 1000, 0.001),
        format="%.3f",
    )
    ozone_limits = ozone_range.slider(
        "Ozone (ppmv)",
        min_value=ozone_min,
        max_value=ozone_max,
        value=(ozone_min, ozone_max),
        step=max((ozone_max - ozone_min) / 500, 0.00001),
        format="%.4f",
    )
    if has_temperature:
        temperature_range = st.columns(1)[0]
        temperature_limits = temperature_range.slider(
            "Air temperature (°C)",
            min_value=temperature_min,
            max_value=temperature_max,
            value=(temperature_min, temperature_max),
            step=max((temperature_max - temperature_min) / 500, 0.01),
            format="%.1f",
        )

visible_profile = profile.loc[
    profile["Altitude_km"].between(*altitude_limits)
].sort_values("Altitude_km")
if visible_profile.empty:
    st.warning("No altitude-bin averages fall within the selected altitude range.")
    st.stop()

tab_labels = [
    "Profiles",
    "Overall violins",
    "Ozone vs altitude",
]
if has_temperature:
    tab_labels.append("Temperature vs altitude")
tabs = st.tabs(tab_labels)

with tabs[0]:
    figure = create_profile_figure(
        visible_profile,
        ozone_limits,
        temperature_limits,
        altitude_limits,
        ozone_color,
        temperature_color,
        line_color,
        marker_shape,
        marker_size,
        curve_style,
        smoothing_span_m,
        reverse_altitude,
        show_bin_means,
        show_smooth_curve,
        show_tropopause,
        tropopause_km,
        selected_campaign["label"],
        resolution,
        has_temperature,
    )
    st.plotly_chart(figure, width="stretch", config={"scrollZoom": True, "displaylogo": False})

with tabs[1]:
    st.caption("Each violin shows the distribution of altitude-bin means for that variable.")
    figure = create_violin_figure(
        visible_profile,
        altitude_limits,
        ozone_limits,
        temperature_limits,
        ozone_color,
        temperature_color,
        show_tropopause,
        tropopause_km,
        has_temperature,
    )
    st.plotly_chart(figure, width="stretch", config={"scrollZoom": True, "displaylogo": False})

with tabs[2]:
    figure = create_ozone_scatter_figure(
        visible_profile,
        altitude_limits,
        ozone_limits,
        temperature_limits,
        marker_shape,
        marker_size,
        reverse_altitude,
        "Linear",
        show_tropopause,
        tropopause_km,
        has_temperature,
    )
    st.plotly_chart(figure, width="stretch", config={"scrollZoom": True, "displaylogo": False})

if has_temperature:
    with tabs[3]:
        figure = create_temperature_scatter_figure(
            visible_profile,
            altitude_limits,
            ozone_limits,
            temperature_limits,
            marker_shape,
            marker_size,
            reverse_altitude,
            show_tropopause,
            tropopause_km,
        )
        st.plotly_chart(figure, width="stretch", config={"scrollZoom": True, "displaylogo": False})

with st.expander("Averaged profile data"):
    st.dataframe(visible_profile, width="stretch", hide_index=True)
