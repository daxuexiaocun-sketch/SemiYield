# Three-stage synthetic demonstration / 三环节模拟演示

This offline demonstration follows the same devices through manufacturing, packaging and lifetime.
**All records are synthetic. They are not SECOM, NASA derivatives, or measured packaging outcomes.**

## Run

```bash
uv sync --locked --extra demo --extra dev
uv run semiyield demo quickstart
```

Open `reports/demo/three_stage/README.md` for the complete report, linked SVG charts and CSV metrics.
After installing dependencies the workflow does not download data, models, or access external APIs.
For separate generation and execution:

```bash
uv run semiyield demo generate --output-dir data/demo/three_stage --seed 42
uv run semiyield demo run --data-dir data/demo/three_stage --output-dir reports/demo/three_stage
```

These are alternatives to quickstart, not extra steps after it. Outputs already containing files
are rejected unless `--force` is supplied. Source and report directories must be separate and
non-nested. Use dedicated directories; the workflow never needs `reports/reference` or
`reports/verified`. Default generated data and reports are ignored by Git; CI uploads demo reports.

For a smaller demonstration:

```bash
uv run semiyield demo quickstart --data-dir data/demo/small --output-dir reports/demo/small --batches 20 --units-per-batch 50 --seed 7
```

## Data contract and mechanism

Defaults: seed 42, 100 batches, 100 devices per batch. At least two batches and one device per batch
are required. Batch selection assigns 20% of batches (rounded up, minimum one) to test before
creating observations. IDs and split assignment are identical across all subsequent stage tables.

| File | Population | Outcomes and units |
|---|---|---|
| `manufacturing.csv` | All devices | 24 anonymous sensors, observed binary `failed`; sensors in arbitrary units |
| `packaging.csv` | Manufacturing passes only | Mixed `X1`–`X16`, `Y` in synthetic units/hour, `proxy_failed` |
| `lifetime.csv` | Both earlier stages passed | `temperature_c`, positive `time_to_event` in hours, `event_observed` |
| `trace.csv` | All devices | Stage entry flags, nullable outcomes, final stage disposition |
| `manifest.json` | Dataset description | Generator version, seed, dimensions, split, threshold, field meanings, file hashes |

A shared latent batch/device quality variable affects early measurements and failure propensity.
Packaging adds independent process noise; throughput decreases with poor quality/process conditions.
Lifetime uses Weibull draws with temperature acceleration and upstream quality/process damage.
Follow-up limits create right censoring: time is `min(failure_time, followup)`, and event is 1 only
when failure occurs before follow-up ends. Latent variables and unobserved future failure times are
not exported or used as features. These mechanisms illustrate correlation, not validated causal
relationships for real manufacturing.

Only manufacturing-passed **training** devices determine the packaging 10% threshold. It then labels
both training and test devices. Manufacturing rejects never enter packaging; low-throughput proxy
rejects never enter lifetime. Missing downstream outcomes remain blank. Model predictions evaluate
screening capability; observed generated outcomes determine routing.

## Evaluation and interpretation

Manufacturing and packaging run Dummy and Logistic Regression on shared training batches and report
metrics on test batches. IDs, split, role flags and labels are excluded from input features.
The generator manifest supplies source hashes and the packaging threshold is checked when running.

Weibull fits training survivors only and describes their pooled stress-temperature cohort; it is
not a holdout accuracy metric or a use-temperature curve. Arrhenius fitting runs only with two
training stress groups having at least three observed failures each, with an explicit extrapolation
note for the 55 °C use condition. Survival forests are not required or run by this demo.

The report includes stage entries and failures, conditional process failure rates, overall process
pass fraction, holdout comparison, throughput histogram/threshold, survival curve and example device
traces. Lifetime observed-event fraction is follow-up dependent; right-censored devices are not
called passes. There is no single combined process-and-lifetime failure probability.

If a small cohort lacks train/test devices, both training classes or enough lifetime failures,
that fit is marked skipped with a reason while stage statistics remain available. Undefined metrics
are recorded as null with reasons. Fixed seed generation is byte reproducible in the locked
environment; fitted model serialization is not promised to match byte-for-byte across platforms.
