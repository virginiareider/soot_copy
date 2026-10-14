from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_DIR / "datasets" / "Data_Nasa_Soot"
OUTPUT_DIR = PROJECT_DIR / "datasets" / "processed"


@dataclass(frozen=True)
class DatasetSpec:
    altitude_column: str
    ozone_column: str
    temperature_column: str | None
    altitude_to_km: float
    ozone_to_ppmv: float
    temperature_to_c: float = 0.0
    datetime_column: str | None = None
    minimum_altitude_km: float | None = None
    unit_notes: str = "Source columns already use the output units."


DATASET_SPECS = {
    "AJAX_2018_ALPHA_IRACI.csv": DatasetSpec(
        altitude_column="G_Alt",
        ozone_column="O3",
        temperature_column=None,
        altitude_to_km=1 / 1000,
        ozone_to_ppmv=1,
        datetime_column="Datetime",
        unit_notes="G_Alt is in meters, converted to km; ozone is in ppmv; temperature is unavailable.",
    ),
    "OWLETS_2017_SONDE_MCGEE.csv": DatasetSpec(
        altitude_column="Altitude_km",
        ozone_column="Ozone_ppmv",
        temperature_column="Temp_degC",
        altitude_to_km=1,
        ozone_to_ppmv=1,
        datetime_column="Seconds_Datetime",
    ),
    "SCOAPE_2024_SONDE_STAUFFER.csv": DatasetSpec(
        altitude_column="Altitude_km",
        ozone_column="Ozone_ppmv",
        temperature_column="Temp_degC",
        altitude_to_km=1,
        ozone_to_ppmv=1,
        datetime_column="Seconds_Datetime",
    ),
    "LMOS_2017_SCI-AVIATION_CONLEY.csv": DatasetSpec(
        altitude_column="Altitude",
        ozone_column="O3",
        temperature_column="Temperature",
        altitude_to_km=1 / 1000,
        ozone_to_ppmv=1 / 1000,
        datetime_column="Datetime",
    ),
    "DISCOVERAQ_2011-MD_MERGE_MRG.csv": DatasetSpec(
        altitude_column="ALTP",
        ozone_column="O3",
        temperature_column="TEMPERATURE",
        altitude_to_km=1,
        ozone_to_ppmv=1 / 1000,
        temperature_to_c=273.15,
        datetime_column="Datetime",
    ),
}

ALTITUDE_CANDIDATES = ("Altitude_km", "Altitude", "G_Alt", "GPSAltitude_km")
OZONE_CANDIDATES = (
    "Ozone_ppmv",
    "O3_ppbv",
    "O3_ppb",
    "O3",
    "2b_o3",
    "M2_O3",
)
TEMPERATURE_CANDIDATES = ("Temp_degC", "Temperature", "Temp", "M2_T")


def _first_present(columns: list[str], candidates: tuple[str, ...]) -> str:
    return next((name for name in candidates if name in columns), "")


def _source_audit() -> pd.DataFrame:
    audit_rows = []
    for path in sorted(SOURCE_DIR.glob("*.csv")):
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        spec = DATASET_SPECS.get(path.name)
        altitude_column = spec.altitude_column if spec else _first_present(columns, ALTITUDE_CANDIDATES)
        ozone_column = spec.ozone_column if spec else _first_present(columns, OZONE_CANDIDATES)
        temperature_column = spec.temperature_column if spec else _first_present(columns, TEMPERATURE_CANDIDATES)
        supported = spec is not None
        if supported:
            notes = spec.unit_notes
        else:
            missing = [
                variable
                for variable, column in (
                    ("altitude", altitude_column),
                    ("ozone", ozone_column),
                    ("air temperature", temperature_column),
                )
                if not column
            ]
            notes = "Not processed; missing " + ", ".join(missing) + ". No values were inferred."

        audit_rows.append(
            {
                "Source_File": path.name,
                "Status": "processed" if supported else "excluded",
                "Altitude_Source_Column": altitude_column,
                "Ozone_Source_Column": ozone_column,
                "Temperature_Source_Column": temperature_column,
                "Available_Columns": " | ".join(columns),
                "Notes": notes,
            }
        )
    return pd.DataFrame(audit_rows)


def _load_clean_dataset(path: Path, spec: DatasetSpec) -> pd.DataFrame:
    columns = [spec.altitude_column, spec.ozone_column]
    if spec.temperature_column:
        columns.append(spec.temperature_column)
    if spec.datetime_column:
        columns.append(spec.datetime_column)
    frame = pd.read_csv(path, usecols=columns, low_memory=False)

    cleaned = pd.DataFrame(
        {
            "Altitude_km": pd.to_numeric(frame[spec.altitude_column], errors="coerce")
            * spec.altitude_to_km,
            "Ozone_ppmv": pd.to_numeric(frame[spec.ozone_column], errors="coerce")
            * spec.ozone_to_ppmv,
        }
    )
    if spec.temperature_column:
        cleaned["Air_Temperature_C"] = pd.to_numeric(
            frame[spec.temperature_column], errors="coerce"
        ) - spec.temperature_to_c
    if spec.datetime_column:
        cleaned["Datetime"] = frame[spec.datetime_column]

    required = ["Altitude_km", "Ozone_ppmv"]
    if spec.temperature_column:
        required.append("Air_Temperature_C")
    numeric = cleaned[required].to_numpy(dtype=float)
    valid = np.isfinite(numeric).all(axis=1) & (cleaned["Ozone_ppmv"] > 0)
    if spec.minimum_altitude_km is not None:
        valid &= cleaned["Altitude_km"] >= spec.minimum_altitude_km

    cleaned = cleaned.loc[valid].copy()
    if cleaned.empty:
        raise ValueError(f"No valid rows remain in {path.name}.")
    return cleaned


def _average_by_altitude(cleaned: pd.DataFrame, bin_size_m: int) -> pd.DataFrame:
    altitude_bin = np.floor(cleaned["Altitude_km"] * 1000 / bin_size_m).astype("int64")
    grouped = cleaned.assign(_Altitude_Bin=altitude_bin).groupby("_Altitude_Bin")
    value_columns = ["Altitude_km", "Ozone_ppmv"]
    if "Air_Temperature_C" in cleaned.columns:
        value_columns.append("Air_Temperature_C")
    averaged = grouped[value_columns].mean()
    averaged["Bin_Lower_Altitude_km"] = averaged.index * bin_size_m / 1000
    averaged["Bin_Upper_Altitude_km"] = (averaged.index + 1) * bin_size_m / 1000
    averaged["Sample_Count"] = grouped.size()
    columns = [
        "Altitude_km",
        "Ozone_ppmv",
        "Bin_Lower_Altitude_km",
        "Bin_Upper_Altitude_km",
        "Sample_Count",
    ]
    if "Air_Temperature_C" in averaged.columns:
        columns.insert(2, "Air_Temperature_C")
    return averaged.reset_index(drop=True)[columns]


def main() -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Dataset source folder not found: {SOURCE_DIR}")

    clean_dir = OUTPUT_DIR / "cleaned"
    average_dirs = {
        bin_size_m: OUTPUT_DIR / f"averaged_{bin_size_m}m"
        for bin_size_m in (10, 50, 100, 500)
    }
    clean_dir.mkdir(parents=True, exist_ok=True)
    for directory in average_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    audit = _source_audit()
    audit.to_csv(OUTPUT_DIR / "source_audit.csv", index=False)

    for filename, spec in DATASET_SPECS.items():
        source_path = SOURCE_DIR / filename
        if not source_path.exists():
            raise FileNotFoundError(f"Expected source dataset not found: {source_path}")

        cleaned = _load_clean_dataset(source_path, spec)
        stem = source_path.stem
        cleaned.to_csv(clean_dir / f"{stem}_clean.csv", index=False)
        for bin_size_m, directory in average_dirs.items():
            averaged = _average_by_altitude(cleaned, bin_size_m)
            averaged.to_csv(directory / f"{stem}_avg_{bin_size_m}m.csv", index=False)
            print(
                f"{filename}: {len(cleaned):,} clean rows -> "
                f"{len(averaged):,} altitude bins at {bin_size_m} m"
            )

    print(f"Source column audit: {OUTPUT_DIR / 'source_audit.csv'}")
    print(f"Cleaned files: {clean_dir}")
    print("Averaged files: " + ", ".join(str(directory) for directory in average_dirs.values()))


if __name__ == "__main__":
    main()