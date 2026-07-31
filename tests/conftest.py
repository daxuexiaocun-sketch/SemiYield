import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    rows = 120
    frame = pd.DataFrame(
        {
            "sensor_000": rng.normal(size=rows),
            "sensor_001": rng.normal(size=rows),
            "sensor_002": np.ones(rows),
            "sensor_003": rng.normal(size=rows),
        }
    )
    frame.loc[::4, "sensor_001"] = np.nan
    frame.loc[:80, "sensor_003"] = np.nan
    target = pd.Series((frame["sensor_000"] > 1).astype(int), name="failed")
    target.iloc[[0, 1, 2, 3, 4, 5]] = 1
    return frame, target
