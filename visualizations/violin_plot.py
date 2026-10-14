from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_violin_figure(
    profile,
    altitude_limits: tuple[float, float],
    ozone_limits: tuple[float, float],
    temperature_limits: tuple[float, float] | None,
    ozone_color: str,
    temperature_color: str,
    show_tropopause: bool,
    tropopause_km: float,
    has_temperature: bool = True,
) -> go.Figure:
    columns = 3 if has_temperature else 2
    subplot_titles = ("Altitude", "Ozone") if not has_temperature else ("Altitude", "Ozone", "Air temperature")
    figure = make_subplots(rows=1, cols=columns, subplot_titles=subplot_titles)
    for column_index, label, values, units, color, limits in (
        (1, "Altitude", profile["Altitude_km"], "km", "#3572a5", altitude_limits),
        (2, "Ozone", profile["Ozone_ppmv"], "ppmv", ozone_color, ozone_limits),
    ):
        figure.add_trace(
            go.Violin(
                x=[label] * len(values),
                y=values,
                name=label,
                box_visible=True,
                meanline_visible=True,
                points=False,
                fillcolor=color,
                opacity=0.55,
                line_color=color,
                hovertemplate=f"{label}: %{{y:.4g}} {units}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=column_index,
        )
        figure.update_yaxes(title_text=units, range=list(limits), row=1, col=column_index)
    if has_temperature:
        figure.add_trace(
            go.Violin(
                x=["Air temperature"] * len(profile["Air_Temperature_C"]),
                y=profile["Air_Temperature_C"],
                name="Air temperature",
                box_visible=True,
                meanline_visible=True,
                points=False,
                fillcolor=temperature_color,
                opacity=0.55,
                line_color=temperature_color,
                hovertemplate="Air temperature: %{y:.4g} °C<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=3,
        )
        figure.update_yaxes(title_text="°C", range=list(temperature_limits), row=1, col=3)
        figure.add_trace(
            go.Violin(
                x=[label] * len(values),
                y=values,
                name=label,
                box_visible=True,
                meanline_visible=True,
                points=False,
                fillcolor=color,
                opacity=0.55,
                line_color=color,
                hovertemplate=f"{label}: %{{y:.4g}} {units}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=column_index,
        )
        figure.update_yaxes(title_text=units, range=list(limits), row=1, col=column_index)
    if show_tropopause and altitude_limits[0] <= tropopause_km <= altitude_limits[1]:
        figure.add_hline(
            y=tropopause_km,
            line_color="#bf3f45",
            line_dash="dash",
            line_width=1.5,
            annotation_text=f"Tropopause · {tropopause_km:g} km",
            annotation_position="top left",
            row=1,
            col=1,
        )
    figure.update_layout(height=550, showlegend=False, margin={"t": 80, "b": 40})
    return figure
