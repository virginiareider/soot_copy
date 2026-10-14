from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .common import mark_tropopause


def create_distribution_figure(
    profile,
    band_positions: np.ndarray,
    ozone_groups: list[np.ndarray],
    temperature_groups: list[np.ndarray],
    ozone_limits: tuple[float, float],
    temperature_limits: tuple[float, float] | None,
    altitude_limits: tuple[float, float],
    ozone_color: str,
    temperature_color: str,
    band_width_km: float,
    reverse_altitude: bool,
    show_tropopause: bool,
    tropopause_km: float,
    has_temperature: bool = True,
) -> plt.Figure:
    axes = [plt.subplots(ncols=2, figsize=(14, max(6, len(band_positions) * 0.24)), sharey=True)[1]]
    if has_temperature:
        figure, (ozone_axis, temperature_axis) = plt.subplots(
            ncols=2,
            figsize=(14, max(6, len(band_positions) * 0.24)),
            sharey=True,
        )
        axis_pairs = [
            (ozone_axis, ozone_groups, ozone_color, "Ozone by altitude band", "Ozone (ppmv)", ozone_limits),
            (
                temperature_axis,
                temperature_groups,
                temperature_color,
                "Air temperature by altitude band",
                "Air temperature (°C)",
                temperature_limits,
            ),
        ]
    else:
        figure, ozone_axis = plt.subplots(
            ncols=1,
            figsize=(7, max(6, len(band_positions) * 0.24)),
            sharey=True,
        )
        axis_pairs = [
            (ozone_axis, ozone_groups, ozone_color, "Ozone by altitude band", "Ozone (ppmv)", ozone_limits),
        ]
    for axis, groups, color, title, xlabel, limits in axis_pairs:
        violin_groups = [values for values in groups if len(values) > 1]
        violin_positions = [
            position for position, values in zip(band_positions, groups) if len(values) > 1
        ]
        if violin_groups:
            violins = axis.violinplot(
                violin_groups,
                positions=violin_positions,
                vert=False,
                widths=band_width_km * 0.78,
                showextrema=False,
            )
            for body in violins["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.32)
        boxes = axis.boxplot(
            groups,
            positions=band_positions,
            vert=False,
            widths=band_width_km * 0.24,
            patch_artist=True,
            showmeans=True,
            manage_ticks=False,
        )
        for box in boxes["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.8)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_xlim(*limits)
        axis.set_yticks(band_positions)
        axis.set_ylabel("Altitude band center (km)")
        axis.grid(axis="x", alpha=0.2)

    if show_tropopause and altitude_limits[0] <= tropopause_km <= altitude_limits[1]:
        mark_tropopause(ozone_axis, tropopause_km)
        if has_temperature:
            mark_tropopause(temperature_axis, tropopause_km)
    if reverse_altitude:
        ozone_axis.set_ylim(altitude_limits[1], altitude_limits[0])
    figure.tight_layout()
    return figure
