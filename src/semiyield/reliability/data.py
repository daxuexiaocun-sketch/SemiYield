"""Contracts and validation for lifetime, temperature, and censoring data."""

import pandas as pd

RELIABILITY_COLUMNS = ("unit_id", "time_to_event", "event_observed")


def validate_lifetime_data(
    frame: pd.DataFrame, *, require_temperature: bool = False
) -> pd.DataFrame:
    required = list(RELIABILITY_COLUMNS)
    if require_temperature:
        required.append("temperature_c")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing reliability columns: {', '.join(missing)}")
    validated = frame.copy()
    validated["time_to_event"] = pd.to_numeric(validated["time_to_event"], errors="raise")
    if (validated["time_to_event"] <= 0).any():
        raise ValueError("time_to_event must be strictly positive")
    events = validated["event_observed"]
    if not events.isin([0, 1, False, True]).all():
        raise ValueError("event_observed must contain only 0/1 or boolean values")
    validated["event_observed"] = events.astype(bool)
    if not validated["event_observed"].any():
        raise ValueError("At least one observed failure is required")
    if validated["unit_id"].isna().any() or validated["unit_id"].astype(str).duplicated().any():
        raise ValueError("unit_id must be non-null and unique")
    if require_temperature:
        validated["temperature_c"] = pd.to_numeric(validated["temperature_c"], errors="raise")
        if validated["temperature_c"].nunique() < 2:
            raise ValueError("Arrhenius fitting requires at least two temperatures")
        if (validated["temperature_c"] <= -273.15).any():
            raise ValueError("temperature_c must be above absolute zero")
    return validated
