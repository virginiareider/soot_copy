from __future__ import annotations

import matplotlib.pyplot as plt

from .common import mark_tropopause, weighted_altitude_bands


def create_band_figure(
    profile,
    band_width_km: float,
    ozone_limits: tuple[float, float],
    temperature_limits: tuple[float, float] | None,
    altitude_limits: tuple[float, float],
    ozone_color: str,
    temperature_color: str,
    reverse_altitude: bool,
    show_tropopause: bool,
    tropopause_km: float,
    has_temperature: bool = True,
) -> plt.Figure:
    bands = weighted_altitude_bands(profile, band_width_km)
    if has_temperature:
        figure, (ozone_axis, temperature_axis) = plt.subplots(
            ncols=2,
            figsize=(13, 6),
            sharey=True,
        )
    else:
        figure, ozone_axis = plt.subplots(ncols=1, figsize=(7, 6), sharey=True)
    ozone_axis.barh(
        bands["Altitude_km"],
        bands["Ozone_ppmv"],
        height=band_width_km * 0.78,
        color=ozone_color,
        alpha=0.85,
    )
    ozone_axis.set_title("Weighted mean ozone by altitude band")
    ozone_axis.set_xlabel("Ozone (ppmv)")
    ozone_axis.set_ylabel("Altitude band center (km)")
    ozone_axis.set_xlim(*ozone_limits)
    ozone_axis.grid(axis="x", alpha=0.25)

    if has_temperature:
        temperature_axis.barh(
            bands["Altitude_km"],
            bands["Air_Temperature_C"],
            height=band_width_km * 0.78,
            color=temperature_color,
            alpha=0.85,
        )
        temperature_axis.set_title("Weighted mean temperature by altitude band")
        temperature_axis.set_xlabel("Air temperature (°C)")
        temperature_axis.set_xlim(*temperature_limits)
        temperature_axis.grid(axis="x", alpha=0.25)
    if show_tropopause and altitude_limits[0] <= tropopause_km <= altitude_limits[1]:
        mark_tropopause(ozone_axis, tropopause_km)
        if has_temperature:
            mark_tropopause(temperature_axis, tropopause_km)
    if reverse_altitude:
        ozone_axis.set_ylim(altitude_limits[1], altitude_limits[0])
    else:
        ozone_axis.set_ylim(*altitude_limits)
    figure.tight_layout()
    return figure
