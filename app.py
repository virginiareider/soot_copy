from pathlib import Path

import pandas as pd
import streamlit as st

#edit pls 
from visualizations.all_data_plot import (
    create_all_data_figure,
    filter_data_by_date,
)
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
SOURCE_DIR = PROJECT_DIR / "datasets" / "Data_Nasa_Soot"
CAMPAIGNS = [
    {
        "location": "Chesapeake Bay",
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
        "location": "California and Nevada",
        "season": "Winter-Spring",
        "label": "AJAX 2018 · Alpha",
        "stem": "AJAX_2018_ALPHA_IRACI",
    },
    {
        "location": "Texas and Connecticut",
        "season": "Spring-Summer",
        "label": "STAQS 2023 · Walter sonde",
        "stem": "STAQS_2023_SONDE_WALTER",
    },
    {
        "location": "Lake Michigan",
        "season": "Spring-Summer",
        "label": "STAQS 2023 · McKinney sonde",
        "stem": "STAQS_2023_SONDE_MCKINNEY",
    },
    {
        "location": "California and Nevada",
        "season": "Spring-Summer",
        "label": "AJAX 2019 · SCI-AVIATION",
        "stem": "AJAX_2019_SCI-AVIATION_IRACI",
    },
    {
        "location": "Texas",
        "season": "Summer",
        "label": "SEAC4RS 2013 · Merge WAS",
        "stem": "SEAC4RS_2013_MERGE_WAS",
    },
    {
        "location": "Texas",
        "season": "Summer",
        "label": "SEAC4RS 2013 · Merge SAGAAERO",
        "stem": "SEAC4RS_2013_MERGE_SAGAAERO",
    },
    {
        "location": "Texas",
        "season": "Summer",
        "label": "SEAC4RS 2013 · Merge SAGAMC",
        "stem": "SEAC4RS_2013_MERGE_SAGAMC",
    },
    {
        "location": "NYC",
        "season": "Spring",
        "label": "LISTOS 2017 · UMD-AIRCRAFT",
        "stem": "LISTOS_2017_UMD-AIRCRAFT_DICKERSON",
    },
        {
        "location": "Lake Michigan",
        "season": "Summer",
        "label": "LMOS 2017 · UWEC-AUTO",
        "stem": "LMOS_2017_UWEC-AUTO_CLEARY",
    },
        {
        "location": "Colorado",
        "season": "Summer",
        "label": "Discover-AQ 2014 · Merge",
        "stem": "DISCOVERAQ_2014-CO_MERGE_MRG",
    },
     {
        "location": "Colorado",
        "season": "Summer",
        "label": "Discover-AQ 2014 · Sonde ",
        "stem": "DISCOVERAQ_2014-CO_SONDES_THOMPSON",
    },
]


@st.cache_data(show_spinner=False)
def load_dataset(dataset_path: Path) -> pd.DataFrame:
    return pd.read_csv(dataset_path)


@st.cache_data(show_spinner="Preparing original source CSV...")
def load_source_bytes(source_path: Path) -> bytes:
    return source_path.read_bytes()


def slider_bounds(values: pd.Series, minimum_span: float) -> tuple[float, float]:
    lower = float(values.min())
    upper = float(values.max())
    if lower >= upper:
        padding = max(abs(lower) * 0.01, minimum_span / 2)
        return lower - padding, upper + padding
    return lower, upper


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
source_matches = list(SOURCE_DIR.rglob(f"{selected_campaign['stem']}.csv"))
if source_matches:
    original_source_path = source_matches[0]
    st.download_button(
        "Download original source CSV",
        data=load_source_bytes(original_source_path),
        file_name=original_source_path.name,
        mime="text/csv",
        help="Download the original, unprocessed campaign data file.",
    )
else:
    st.warning(f"Original source CSV not found for {selected_campaign['label']}.")
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
source_data_path = PROCESSED_DIR / "cleaned" / f"{selected_campaign['stem']}_clean.csv"

with st.spinner("Loading individual measurements..."):
    source_data = load_dataset(source_data_path)

required_columns = ["Altitude_km", "Ozone_ppmv"]
has_temperature = "Air_Temperature_C" in profile.columns
if not set(required_columns).issubset(profile.columns):
    st.error(f"The selected profile is missing required columns: {required_columns}.")
    st.stop()

st.caption(f"{selected_campaign['label']} · {location} · {season} · {resolution} bins")
st.caption(
    "Dashed tropopause boundary separates the troposphere below from the stratosphere above. "
    "Its height is adjustable because the real tropopause varies. add more description here"
)

maximum_smoothing_m = max(10, min(2000, int(profile["Altitude_km"].max() * 1000)))
maximum_smoothing_m = max(10, (maximum_smoothing_m // 100) * 100)
with st.sidebar:
    st.subheader("Chart style")
    show_bin_means = st.checkbox("Show datapoints", value=True)
    show_smooth_curve = st.checkbox("Show smoothed curve", value=True)
    smoothing_span_m = st.slider(
        "Smooth Curve Level (m)",
        min_value=10,
        max_value=maximum_smoothing_m,
        value=min(10, maximum_smoothing_m),
        step=10,
    )
    color_palette = [
        "Teal", "Blue", "Green", "Orange", "Red", "Purple", "Brown", "Black",
    ]
    ozone_color = st.selectbox("Ozone point color", color_palette, index=0)
    temperature_color = st.selectbox("Temperature point color", color_palette, index=3)
    line_color = st.selectbox("Line color", color_palette, index=1)
    marker_shape = st.selectbox(
        "Data point shape",
        ["circle", "square", "diamond", "triangle-up", "cross", "x", "star"],
    )
    marker_size = st.slider("Data point size", min_value=3, max_value=14, value=6)
    curve_style = st.selectbox("Line style", ["solid", "dash", "dot", "dashdot"])
    reverse_altitude = st.checkbox("Reverse axis", value=False)
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

source_dates = pd.to_datetime(source_data["Datetime"], errors="coerce").dropna()
date_filtered_data = source_data.iloc[0:0].copy()
if not source_dates.empty:
    earliest_date = source_dates.min().date()
    latest_date = source_dates.max().date()
    with st.expander("Individual measurement date range", expanded=False):
        selected_dates = st.date_input(
            "Dates used by All measurements",
            value=(earliest_date, latest_date),
            min_value=earliest_date,
            max_value=latest_date,
            key=f"measurement_dates_{selected_campaign['stem']}",
        )
    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = end_date = selected_dates
    date_filtered_data = filter_data_by_date(source_data, start_date, end_date)

location_filtered_data = date_filtered_data.copy()

axis_data = (
    location_filtered_data
    if not location_filtered_data.empty
    else source_data
)
all_altitude_min, all_altitude_max = slider_bounds(axis_data["Altitude_km"], 0.001)
all_ozone_min, all_ozone_max = slider_bounds(axis_data["Ozone_ppmv"], 0.00001)
all_temperature_limits = None
if has_temperature:
    all_temperature_min, all_temperature_max = slider_bounds(
        axis_data["Air_Temperature_C"], 0.01
    )

with st.expander("Axis ranges", expanded=False):
    st.subheader("Averaged profile")
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
    st.subheader("All measurements")
    all_axis_columns = st.columns(3 if has_temperature else 2)
    all_altitude_limits = all_axis_columns[0].slider(
        "Altitude range (km)",
        min_value=all_altitude_min,
        max_value=all_altitude_max,
        value=(all_altitude_min, all_altitude_max),
        step=max((all_altitude_max - all_altitude_min) / 500, 0.001),
        format="%.3f",
        key=f"all_altitude_range_{selected_campaign['stem']}",
    )
    all_ozone_limits = all_axis_columns[1].slider(
        "Ozone range (ppmv)",
        min_value=all_ozone_min,
        max_value=all_ozone_max,
        value=(all_ozone_min, all_ozone_max),
        step=max((all_ozone_max - all_ozone_min) / 500, 0.00001),
        format="%.4f",
        key=f"all_ozone_range_{selected_campaign['stem']}",
    )
    if has_temperature:
        all_temperature_limits = all_axis_columns[2].slider(
            "Temperature range (°C)",
            min_value=all_temperature_min,
            max_value=all_temperature_max,
            value=(all_temperature_min, all_temperature_max),
            step=max((all_temperature_max - all_temperature_min) / 500, 0.01),
            format="%.1f",
            key=f"all_temperature_range_{selected_campaign['stem']}",
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
tab_labels.append("All measurements")
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
    st.caption("Each violin shows the distribution of for that variable.")
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

all_measurements_tab = 4 if has_temperature else 3
with tabs[all_measurements_tab]:



    st.caption(
        "Dots are individual cleaned measurements from the selected date range. "
        "Date and axis ranges filter both the measurements and summary statistics. "
        "The curve is calculated from the filtered measurements."
    )
    if location_filtered_data.empty:
        st.warning("This dataset has no usable dates or no measurements in the selected date range.")
    else:
        filtered_measurements = location_filtered_data.loc[
            location_filtered_data["Altitude_km"].between(*all_altitude_limits)
            & location_filtered_data["Ozone_ppmv"].between(*all_ozone_limits)
        ].copy()
        if has_temperature:
            filtered_measurements = filtered_measurements.loc[
                filtered_measurements["Air_Temperature_C"].between(*all_temperature_limits)
            ]

        if filtered_measurements.empty:
            st.warning("No measurements fall within the selected date and axis ranges.")
        else:
            st.caption(f"{len(filtered_measurements):,} measurements within selected filters")
            if (
                not filtered_measurements.empty
                and {"Latitude", "Longitude"}.issubset(filtered_measurements.columns)
            ):

                st.subheader("Measurement Locations")

                map_points = (
                    filtered_measurements[
                        ["Latitude", "Longitude"]
                    ]
                    .dropna()
                )

                if not map_points.empty:

                    st.caption(
                        f"{len(map_points):,} measurements within selected filters"
                    )

                    st.map(
                        map_points.rename(
                            columns={
                                "Latitude": "lat",
                                "Longitude": "lon",
                            }
                        ),
                        use_container_width=True,
                    )
            ozone_stats = filtered_measurements["Ozone_ppmv"]
            ozone_peak_index = ozone_stats.idxmax()
            ozone_summary = st.columns(6)
            ozone_summary[0].metric("Measurements", f"{len(filtered_measurements):,}")
            ozone_summary[1].metric("Mean ozone", f"{ozone_stats.mean():.4g} ppmv")
            ozone_summary[2].metric("Median ozone", f"{ozone_stats.median():.4g} ppmv")
            ozone_summary[3].metric("Minimum ozone", f"{ozone_stats.min():.4g} ppmv")
            ozone_summary[4].metric("Maximum ozone", f"{ozone_stats.max():.4g} ppmv")
            ozone_summary[5].metric(
                "Altitude at max",
                f"{filtered_measurements.loc[ozone_peak_index, 'Altitude_km']:.3f} km",
            )
            if has_temperature:
                temperature_stats = filtered_measurements["Air_Temperature_C"]
                minimum_temperature_index = temperature_stats.idxmin()
                temperature_summary = st.columns(5)
                temperature_summary[0].metric("Mean temperature", f"{temperature_stats.mean():.1f} °C")
                temperature_summary[1].metric("Median temperature", f"{temperature_stats.median():.1f} °C")
                temperature_summary[2].metric("Minimum temperature", f"{temperature_stats.min():.1f} °C")
                temperature_summary[3].metric("Maximum temperature", f"{temperature_stats.max():.1f} °C")
                temperature_summary[4].metric(
                    "Altitude at min temperature",
                    f"{filtered_measurements.loc[minimum_temperature_index, 'Altitude_km']:.3f} km",
                )

            figure = create_all_data_figure(
                filtered_measurements,
                smoothing_span_m,
                ozone_color,
                temperature_color,
                line_color,
                marker_shape,
                marker_size,
                curve_style,
                show_smooth_curve,
                reverse_altitude,
                show_tropopause,
                tropopause_km,
                all_ozone_limits,
                all_altitude_limits,
                all_temperature_limits,
            )
            st.plotly_chart(
                figure,
                width="stretch",
                config={"scrollZoom": True, "displaylogo": False},
            )

with st.expander("Averaged profile data"):
    st.dataframe(visible_profile, width="stretch", hide_index=True)
