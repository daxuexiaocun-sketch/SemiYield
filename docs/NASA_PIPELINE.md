# NASA MOSFET Data Pipeline

## Purpose

Prepare the external NASA Power MOSFET thermal-overstress archive for device-level reliability analysis.

The local `semiyield quickstart` workflow does not use this archive. It uses UCI SECOM and a synthetic reliability example only.

## Inspect and convert

```bash
uv run semiyield nasa verify /path/to/nasa-mosfet.zip
uv run semiyield nasa notices /path/to/nasa-mosfet.zip
uv run semiyield nasa inspect /path/to/nasa-mosfet.zip
uv run semiyield nasa convert-official /path/to/nasa-mosfet.zip
uv run semiyield nasa prepare --source data/interim/nasa_mosfet_normalized
```

For archive variants, inspect a representative MATLAB file and provide explicit dotted-path mapping:

```bash
uv run semiyield nasa inspect-mat /path/to/Test_1_run_1.mat
uv run semiyield nasa convert-matlab /path/to/extracted nasa_mapping.json
```

## Inputs and outputs

The normalized input schema requires `device_id`, `time_s`, `temperature_c`, `vds_v`, and `id_a`; `event_observed` is optional. Preparation calculates RDS(on), aggregates sequential windows, creates device-level splits, and writes features, provenance, and lifetime-sensitivity artifacts.

## Data handling

The converter records accepted and excluded inputs in a manifest. The source archive and row-level derived tables remain outside Git until dataset-specific redistribution rights are confirmed.
