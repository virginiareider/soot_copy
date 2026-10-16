from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .common import smooth_by_altitude


def filter_data_by_date(
    data: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    timestamps = pd.to_datetime(data["Datetime"], errors="coerce")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date + timedelta(days=1))
    return data.loc[timestamps.ge(start) & timestamps.lt(end)].copy()


def create_all_data_figure(
    data: pd.DataFrame,
    smoothing_span_m: int,
    ozone_color: str,
    temperature_color: str,
    line_color: str,
    marker_shape: str,
    marker_size: int,
    curve_style: str,
    show_smooth_curve: bool,
    reverse_altitude: bool,
    show_tropopause: bool,
    tropopause_km: float,
    ozone_limits: tuple[float, float],
    altitude_limits: tuple[float, float],
    temperature_limits: tuple[float, float] | None,
) -> go.Figure:
    has_temperature = "Air_Temperature_C" in data.columns
    column_count = 2 if has_temperature else 1
    subplot_titles = (
        ("Ozone vs altitude", "Air temperature vs altitude")
        if has_temperature
        else ("Ozone vs altitude",)
    )
    figure = make_subplots(rows=1, cols=column_count, subplot_titles=subplot_titles)

    altitude = data["Altitude_km"].to_numpy(dtype=float)
    ozone = data["Ozone_ppmv"].to_numpy(dtype=float)
    observation_times = pd.to_datetime(data["Datetime"], errors="coerce").dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    series = [(1, ozone, "Ozone (ppmv)", ozone_color)]
    if has_temperature:
        temperature = data["Air_Temperature_C"].to_numpy(dtype=float)
        series.append((2, temperature, "Air temperature (°C)", temperature_color))

    altitude_bin_width_km = max(smoothing_span_m / 1000 / 10, 0.005)
    altitude_bins = np.floor(altitude / altitude_bin_width_km).astype("int64")
    binned = data.assign(_Altitude_Bin=altitude_bins).groupby("_Altitude_Bin").agg(
        Altitude_km=("Altitude_km", "mean"),
        Ozone_ppmv=("Ozone_ppmv", "mean"),
        **(
            {"Air_Temperature_C": ("Air_Temperature_C", "mean")}
            if has_temperature
            else {}
        ),
    ).sort_values("Altitude_km")
    binned_altitude = binned["Altitude_km"].to_numpy(dtype=float)
    smoothing_sigma_km = smoothing_span_m / 1000

    for column_index, values, axis_title, color in series:
        figure.add_trace(
            go.Scattergl(
                x=values,
                y=altitude,
                mode="markers",
                name=f"Individual {axis_title.lower()}",
                customdata=observation_times,
                marker={
                    "color": color,
                    "size": marker_size,
                    "symbol": marker_shape,
                    "opacity": 0.35,
                },
                hovertemplate=(
                    f"Date: %{{customdata}}<br>{axis_title}: %{{x:.4g}}"
                    "<br>Altitude: %{y:.3f} km"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=column_index,
        )

        column = "Ozone_ppmv" if column_index == 1 else "Air_Temperature_C"
        binned_values = binned[column].to_numpy(dtype=float)
        if show_smooth_curve and len(binned_values) > 1:
            smoothed = smooth_by_altitude(
                binned_altitude,
                binned_values,
                smoothing_sigma_km,
            )
            figure.add_trace(
                go.Scattergl(
                    x=smoothed,
                    y=binned_altitude,
                    mode="lines",
                    name=f"Smoothed {axis_title.lower()}",
                    line={"color": line_color, "width": 5, "dash": curve_style},
                    hovertemplate=(
                        f"Smoothed {axis_title.lower()}: %{{x:.4g}}"
                        "<br>Altitude: %{y:.3f} km<extra></extra>"
                    ),
                ),
                row=1,
                col=column_index,
            )
        axis_limits = ozone_limits if column_index == 1 else temperature_limits
        figure.update_xaxes(
            title_text=axis_title,
            range=list(axis_limits),
            showgrid=True,
            row=1,
            col=column_index,
        )

    altitude_range = (
        [altitude_limits[1], altitude_limits[0]]
        if reverse_altitude
        else list(altitude_limits)
    )
    figure.update_yaxes(
        title_text="Altitude (km)",
        showgrid=True,
        range=altitude_range,
        row=1,
        col=1,
    )
    if has_temperature:
        figure.update_yaxes(range=altitude_range, row=1, col=2)
    if show_tropopause and altitude_limits[0] <= tropopause_km <= altitude_limits[1]:
        for column_index in range(1, column_count + 1):
            figure.add_hline(
                y=tropopause_km,
                line_color="#bf3f45",
                line_dash="dash",
                line_width=1.5,
                annotation_text=f"Tropopause · {tropopause_km:g} km",
                annotation_position="top left",
                row=1,
                col=column_index,
            )
    figure.update_layout(
        title="Individual measurements with smoothed profile",
        height=650,
        legend={"orientation": "h", "y": -0.2},
        margin={"t": 80, "b": 90},
    )
    return figure


def create_temperature_ozone_figure(
    data: pd.DataFrame,
    marker_shape: str,
    marker_size: int,
) -> go.Figure:
    figure = go.Figure(
        go.Scattergl(
            x=data["Ozone_ppmv"],
            y=data["Air_Temperature_C"],
            mode="markers",
            marker={
                "color": data["Altitude_km"],
                "colorscale": "Turbo",
                "colorbar": {"title": "Altitude (km)"},
                "size": marker_size,
                "symbol": marker_shape,
                "opacity": 0.65,
            },
            customdata=data[["Altitude_km"]],
            hovertemplate=(
                "Ozone: %{x:.4g} ppmv<br>Air temperature: %{y:.2f} °C"
                "<br>Altitude: %{customdata[0]:.3f} km<extra></extra>"
            ),
        )
    )
    figure.update_xaxes(title_text="Ozone (ppmv)", showgrid=True)
    figure.update_yaxes(title_text="Air temperature (°C)", showgrid=True)
    figure.update_layout(
        title="Air temperature vs ozone, colored by altitude",
        height=650,
        margin={"t": 80, "b": 60},
    )
    return figure


def create_ozone_time_altitude_heatmap(
    data: pd.DataFrame,
    time_bin: str,
    altitude_bin_m: int,
) -> go.Figure:
    timestamps = pd.to_datetime(data["Datetime"], errors="coerce")
    altitude_bins = np.floor(data["Altitude_km"] * 1000 / altitude_bin_m).astype("int64")
    binned = data.assign(
        _Time_Bin=timestamps.dt.floor(time_bin),
        _Altitude_Bin=altitude_bins,
    )
    grouped = binned.groupby(["_Altitude_Bin", "_Time_Bin"], observed=True)["Ozone_ppmv"].mean()
    grid = grouped.unstack("_Time_Bin").sort_index().sort_index(axis=1)
    altitude_centers_km = (grid.index.to_numpy(dtype=float) + 0.5) * altitude_bin_m / 1000
    figure = go.Figure(
        go.Heatmap(
            x=grid.columns,
            y=altitude_centers_km,
            z=grid.to_numpy(dtype=float),
            colorscale="Turbo",
            colorbar={"title": "Ozone (ppmv)"},
            hovertemplate=(
                "Time: %{x}<br>Altitude: %{y:.3f} km"
                "<br>Mean ozone: %{z:.4g} ppmv<extra></extra>"
            ),
            connectgaps=False,
        )
    )
    figure.update_xaxes(title_text="Measurement time", showgrid=False)
    figure.update_yaxes(title_text="Altitude (km)", showgrid=True)
    figure.update_layout(
        title=f"Ozone over time and altitude · {time_bin} × {altitude_bin_m} m bins",
        height=650,
        margin={"t": 80, "b": 90},
    )
    return figure
