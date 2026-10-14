from __future__ import annotations

import numpy as np
import pandas as pd


def smooth_by_altitude(
    altitude_km: np.ndarray,
    values: np.ndarray,
    sigma_km: float,
) -> np.ndarray:
    """Smooth values using Gaussian weights over a local altitude window."""
    smoothed = np.empty_like(values, dtype=float)
    for index, altitude in enumerate(altitude_km):
        start = np.searchsorted(altitude_km, altitude - 3 * sigma_km)
        stop = np.searchsorted(altitude_km, altitude + 3 * sigma_km, side="right")
        local_altitude = altitude_km[start:stop]
        weights = np.exp(-0.5 * ((local_altitude - altitude) / sigma_km) ** 2)
        smoothed[index] = np.average(values[start:stop], weights=weights)
    return smoothed


def weighted_altitude_bands(profile: pd.DataFrame, band_width_km: float) -> pd.DataFrame:
    """Return weighted mean ozone and temperature for each altitude band."""
    banded = profile.assign(
        _band=np.floor(profile["Altitude_km"] / band_width_km).astype(int),
        _weighted_ozone=profile["Ozone_ppmv"] * profile["Sample_Count"],
        _weighted_altitude=profile["Altitude_km"] * profile["Sample_Count"],
    )
    grouped = banded.groupby("_band").agg(
        Sample_Count=("Sample_Count", "sum"),
        Ozone_Total=("_weighted_ozone", "sum"),
        Altitude_Total=("_weighted_altitude", "sum"),
    )
    grouped["Altitude_km"] = grouped.index * band_width_km + band_width_km / 2
    grouped["Ozone_ppmv"] = grouped["Ozone_Total"] / grouped["Sample_Count"]
    if "Air_Temperature_C" in profile.columns:
        banded["_weighted_temperature"] = profile["Air_Temperature_C"] * profile["Sample_Count"]
        grouped["Temperature_Total"] = banded.groupby("_band")["_weighted_temperature"].sum()
        grouped["Air_Temperature_C"] = grouped["Temperature_Total"] / grouped["Sample_Count"]
    return grouped.reset_index(drop=True)


def mark_tropopause(axis, altitude_km: float) -> None:
    """Draw a dashed tropopause marker and text label on a Matplotlib axis."""
    axis.axhline(altitude_km, color="#bf3f45", linestyle="--", linewidth=1.5, alpha=0.9)
    axis.text(
        0.99,
        altitude_km,
        f"Tropopause · {altitude_km:g} km",
        transform=axis.get_yaxis_transform(),
        color="#8c2630",
        fontsize=8,
        ha="right",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1},
    )
