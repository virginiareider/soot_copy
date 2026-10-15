from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def create_ozone_scatter_figure(
    profile,
    altitude_limits: tuple[float, float],
    ozone_limits: tuple[float, float],
    temperature_limits: tuple[float, float] | None,
    marker_shape: str,
    marker_size: int,
    reverse_altitude: bool,
    scatter_scale: str,
    show_tropopause: bool,
    tropopause_km: float,
    has_temperature: bool = True,
) -> go.Figure:
    marker = {
        "color": profile["Ozone_ppmv"],
        "colorscale": "Turbo",
        "cmin": ozone_limits[0],
        "cmax": ozone_limits[1],
        "colorbar": {"title": "Ozone (ppmv)"},
        "size": max(3, marker_size - 2),
        "symbol": marker_shape,
        "opacity": 0.45,
        "line": {"color": "white", "width": 0.35},
    }
    customdata = profile[["Sample_Count"]]
    hovertemplate = (
        "Ozone: %{x:.4g} ppmv<br>Altitude: %{y:.3f} km"
        "<br>Source samples: %{customdata[0]}<extra></extra>"
    )
    if has_temperature:
        marker = {
            "color": profile["Air_Temperature_C"],
            "colorscale": "Turbo",
            "cmin": temperature_limits[0],
            "cmax": temperature_limits[1],
            "colorbar": {"title": "Air temp (°C)"},
            "size": max(3, marker_size - 2),
            "symbol": marker_shape,
            "opacity": 0.45,
            "line": {"color": "white", "width": 0.35},
        }
        customdata = profile[["Air_Temperature_C", "Sample_Count"]]
        hovertemplate = (
            "Ozone: %{x:.4g} ppmv<br>Altitude: %{y:.3f} km"
            "<br>Air temperature: %{customdata[0]:.2f} °C"
            "<br>Source samples: %{customdata[1]}<extra></extra>"
        )
    figure = go.Figure(
        go.Scattergl(
            x=profile["Ozone_ppmv"],
            y=profile["Altitude_km"],
            mode="markers",
            marker=marker,
            customdata=customdata,
            hovertemplate=hovertemplate,
        )
    )
    figure.update_xaxes(
        title_text="Ozone (ppmv)",
        type="log" if scatter_scale == "Logarithmic" else "linear",
        range=(np.log10(ozone_limits) if scatter_scale == "Logarithmic" else ozone_limits),
    )
    figure.update_yaxes(
        title_text="Altitude (km)",
        range=[altitude_limits[1], altitude_limits[0]] if reverse_altitude else list(altitude_limits),
    )
    figure.update_layout(
        title="Ozone by altitude, colored by air temperature"
        if has_temperature
        else "Ozone by altitude",
        height=650,
        margin={"t": 70, "b": 50},
    )
    if show_tropopause and altitude_limits[0] <= tropopause_km <= altitude_limits[1]:
        figure.add_hline(
            y=tropopause_km,
            line_color="#bf3f45",
            line_dash="dash",
            line_width=1.5,
            annotation_text=f"Tropopause · {tropopause_km:g} km",
            annotation_position="top left",
        )
    return figure


def create_temperature_scatter_figure(
    profile,
    altitude_limits: tuple[float, float],
    ozone_limits: tuple[float, float],
    temperature_limits: tuple[float, float],
    marker_shape: str,
    marker_size: int,
    reverse_altitude: bool,
    show_tropopause: bool,
    tropopause_km: float,
) -> go.Figure:
    figure = go.Figure(
        go.Scattergl(
            x=profile["Air_Temperature_C"],
            y=profile["Altitude_km"],
            mode="markers",
            marker={
                "color": profile["Ozone_ppmv"],
                "colorscale": "Turbo",
                "cmin": ozone_limits[0],
                "cmax": ozone_limits[1],
                "colorbar": {"title": "Ozone (ppmv)"},
                "size": max(3, marker_size - 2),
                "symbol": marker_shape,
                "opacity": 0.45,
                "line": {"color": "white", "width": 0.35},
            },
            customdata=profile[["Ozone_ppmv", "Sample_Count"]],
            hovertemplate=(
                "Air temperature: %{x:.2f} °C<br>Altitude: %{y:.3f} km"
                "<br>Ozone: %{customdata[0]:.4g} ppmv"
                "<br>Source samples: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    figure.update_xaxes(title_text="Air temperature (°C)", range=list(temperature_limits))
    figure.update_yaxes(
        title_text="Altitude (km)",
        range=[altitude_limits[1], altitude_limits[0]] if reverse_altitude else list(altitude_limits),
    )
    figure.update_layout(
        title="Air temperature by altitude, colored by ozone",
        height=650,
        margin={"t": 70, "b": 50},
    )
    if show_tropopause and altitude_limits[0] <= tropopause_km <= altitude_limits[1]:
        figure.add_hline(
            y=tropopause_km,
            line_color="#bf3f45",
            line_dash="dash",
            line_width=1.5,
            annotation_text=f"Tropopause · {tropopause_km:g} km",
            annotation_position="top left",
        )
    return figure
