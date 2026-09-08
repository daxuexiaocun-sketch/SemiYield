# Reliability Data Card

## Purpose

Support local validation and MOSFET thermal-overstress reliability analysis.

## Included example data

`src/semiyield/assets/mosfet_lifetime_smoke.csv` is a deterministic synthetic dataset for command, censoring, fitting, and plotting checks. It is not NASA data and must not be used as an experimental result.

Its reliability-report RSF contract uses the documented compatibility aliases `temperature_c`,
`initial_rds_on_ohm`, and `rds_on_slope`. Inputs with no declared device split use a deterministic
device-level 80/20 hold-out. NASA reports instead map
`baseline_temperature_c_mean`, `baseline_rds_on_ohm_mean`, and
`baseline_rds_on_ohm_slope` and preserve their documented 24/8/9 device split.

Every reliability report attempts RSF. When the survival dependency or any required data condition
is absent, its report JSON and CLI output contain `required_but_unavailable`; Weibull and
Arrhenius–Weibull outputs remain available. Completed NASA reports may publish the explicitly
authorized protected-test device labels and their SVG-only individual RSF model predictions. They do
not publish device-level feature values, observed outcomes, raw rows, or downloadable predictions.

## NASA Power MOSFET data

The NASA Ames Prognostics Center of Excellence archive contains thermal-overstress run-to-failure experiments on discrete power MOSFETs. The upstream archive is approximately 7.85 GB and remains external to this repository.

The project does not redistribute NASA row-level derivatives because reuse terms have not been
confirmed. The sole public exception is the protected-test device label plus SVG-only RSF prediction
identified above. See the [NASA data pipeline](NASA_PIPELINE.md) and archive notices before
publishing other derivative records.
