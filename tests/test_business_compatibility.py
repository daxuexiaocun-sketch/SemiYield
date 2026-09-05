import importlib
import pickle
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from semiyield.cli import app
from semiyield.modeling import ModelArtifact, train_model
from semiyield.preprocessing import ColumnCleaner


def test_old_imports_share_identity():
    for old, new in [
        ("data", "yield_risk.data"),
        ("benchmark", "yield_risk.benchmark"),
        ("nasa", "reliability.nasa"),
        ("modeling", "common.modeling"),
        ("preprocessing", "common.preprocessing"),
    ]:
        assert importlib.import_module(f"semiyield.{old}") is importlib.import_module(
            f"semiyield.{new}"
        )


def test_legacy_pickle_class_paths(monkeypatch, tmp_path):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    features = pd.DataFrame(rng.normal(size=(30, 3)), columns=["a", "b", "c"])
    artifact = train_model(features, pd.Series([0, 1] * 15), calibrate=False)
    with monkeypatch.context() as legacy:
        legacy.setattr(ModelArtifact, "__module__", "semiyield.modeling")
        legacy.setattr(ColumnCleaner, "__module__", "semiyield.preprocessing")
        path = tmp_path / "legacy.joblib"
        artifact.save(path)
        # Old serialized globals resolve without re-training.
        payload = pickle.dumps(artifact)
    loaded = ModelArtifact.load(path)
    np.testing.assert_allclose(
        loaded.estimator.predict_proba(features),
        pickle.loads(payload).estimator.predict_proba(features),
    )


def test_new_and_legacy_command_help():
    for args in [
        ["yield", "train"],
        ["train"],
        ["nasa", "prepare"],
        ["reliability", "report"],
        ["packaging", "train"],
        ["demo", "run"],
    ]:
        result = CliRunner().invoke(app, [*args, "--help"])
        assert result.exit_code == 0, result.output


def test_star_script_in_installed_isolated_interpreter():
    script = Path(__file__).parents[1] / "scripts/update_star_history.py"
    result = subprocess.run(
        [sys.executable, "-I", str(script), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "--repository" in result.stdout
