"""Public contracts for raw, linked three-stage simulation data."""

VERSION = "semiyield-three-stage-v2"
SENSORS = [f"sensor_{i:03d}" for i in range(24)]
PACKAGING_FEATURES = [f"X{i}" for i in range(1, 17)]
TABLES = ("manufacturing.csv", "packaging_candidates.csv")
