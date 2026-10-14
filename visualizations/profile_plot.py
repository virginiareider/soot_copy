from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .common import smooth_by_altitude


def create_profile_figure(
    profile,
    ozone_limits: tuple[float, float],
    temperature_limits: tuple[float, float] | None,
    altitude_limits: tuple[float, float],
    ozone_color: str,
    temperature_color: str,
    line_color: str,
    marker_shape: str,
    marker_size: int,
    curve_style: str,
    smoothing_span_m: int,
    reverse_altitude: bool,
    show_bin_means: bool,
    show_smooth_curve: bool,
    show_tropopause: bool,
    tropopause_km: float,
    selected_label: str,
    resolution: str,
    has_temperature: bool = True,
) -> go.Figure:
    column_count = 2 if has_temperature else 1
    subplot_titles = (
        ("Ozone profile", "Air temperature profile")
        if has_temperature
        else ("Ozone profile",)
    )
    figure = make_subplots(
        rows=1,
        cols=column_count,
        shared_yaxes=True,
        subplot_titles=subplot_titles,
    )
    altitude_values = profile["Altitude_km"].to_numpy()
    smoothing_sigma_km = smoothing_span_m / 1000

    columns = [(1, "Ozone_ppmv", ozone_color, "Ozone profile", ozone_limits)]
    if has_temperature:
        columns.append(
            (2, "Air_Temperature_C", temperature_color, "Air temperature profile", temperature_limits)
        )
    for column_index, column, color, title, limits in columns:
        values = profile[column].to_numpy()
        if show_bin_means:
            figure.add_trace(
                go.Scatter(
                    x=values,
                    y=altitude_values,
                    mode="markers",
                    name=f"{title} bin means",
                    marker={"color": color, "size": marker_size, "symbol": marker_shape},
                    customdata=profile["Sample_Count"],
                    hovertemplate=(
                        "Altitude: %{y:.3f} km<br>Value: %{x:.4g}"
                        "<br>Source samples: %{customdata}<extra></extra>"
                    ),
                ),
                row=1,
                col=column_index,
            )
        if show_smooth_curve and len(values) > 1:
            smoothed = smooth_by_altitude(altitude_values, values, smoothing_sigma_km)
            figure.add_trace(
                go.Scatter(
                    x=smoothed,
                    y=altitude_values,
                    mode="lines",
                    name=f"{title} smooth",
                    line={"color": line_color, "width": 3, "dash": curve_style},
                    hovertemplate="Altitude: %{y:.3f} km<br>Smoothed: %{x:.4g}<extra></extra>",
                ),
                row=1,
                col=column_index,
            )
        figure.update_xaxes(
            range=list(limits),
            showgrid=True,
            title_text=("Ozone (ppmv)" if column_index == 1 else "Air temperature (°C)"),
            row=1,
            col=column_index,
        )
    figure.update_yaxes(
        range=[altitude_limits[1], altitude_limits[0]] if reverse_altitude else list(altitude_limits),
        title_text="Altitude (km)",
        showgrid=True,
        row=1,
        col=1,
    )
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
        title=f"{selected_label} · {resolution} altitude averages",
        height=650,
        legend={"orientation": "h", "y": -0.2},
        margin={"t": 80, "b": 100},
    )
    return figure
