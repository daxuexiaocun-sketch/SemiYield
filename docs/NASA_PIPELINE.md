# NASA MOSFET Data Pipeline

## Purpose

Prepare the external NASA Power MOSFET thermal-overstress archive for device-level reliability analysis.

Install the local smoke table with `semiyield reliability example install` when needed.

## Inspect and convert

```bash
uv run semiyield reliability nasa verify /path/to/nasa-mosfet.zip
uv run semiyield reliability nasa notices /path/to/nasa-mosfet.zip
uv run semiyield reliability nasa inspect /path/to/nasa-mosfet.zip
uv run semiyield reliability nasa convert-official /path/to/nasa-mosfet.zip
uv run semiyield reliability nasa prepare --source data/interim/nasa_mosfet_normalized
```

For archive variants, inspect a representative MATLAB file and provide explicit dotted-path mapping:

```bash
uv run semiyield reliability nasa inspect-mat /path/to/Test_1_run_1.mat
uv run semiyield reliability nasa convert-matlab /path/to/extracted nasa_mapping.json
```

## Inputs and outputs

The normalized input schema requires `device_id`, `time_s`, `temperature_c`, `vds_v`, and `id_a`; `event_observed` is optional. Preparation calculates RDS(on), aggregates sequential windows, creates device-level splits, and writes features, provenance, and lifetime-sensitivity artifacts.

## Data handling

The converter records accepted and excluded inputs in a manifest. The source archive and row-level
derived tables remain outside Git until dataset-specific redistribution rights are confirmed. The
verified reliability report may publish protected-test device labels and SVG-only RSF model-predicted
survival curves; it does not publish raw rows, baseline features, or observed device outcomes.
