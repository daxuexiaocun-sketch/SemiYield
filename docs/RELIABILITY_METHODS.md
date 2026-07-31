# Reliability Methods

## Purpose

Estimate degradation and lifetime distributions from MOSFET thermal-overstress measurements.

## Degradation endpoint

SemiYield uses change in on-state resistance, ΔRDS(on), relative to each device's initial median. The primary endpoint is ΔRDS(on) ≥ 0.05 Ω; 0.045 Ω is also reported as a sensitivity analysis. The endpoint requires three consecutive filtered windows above threshold.

RDS(on) is calculated as VDS/ID for selected on-state samples and adjusted to an initial device temperature reference. This is an analysis correction, not an electro-thermal model.

## Models and outputs

- Weibull maximum likelihood with right censoring: shape β, characteristic life η, B10, likelihood, and parameter intervals.
- Arrhenius–Weibull accelerated-life model: activation energy and use-condition lifetime when temperature groups are sufficient.
- Optional random survival forest: device-level concordance index.

## Limitations and references

Failure-mode consistency is required to interpret Weibull shape or extrapolate accelerated-life results. Statistical survival models do not replace qualification.

- [NASA technical report 20140010628](https://ntrs.nasa.gov/api/citations/20140010628/downloads/20140010628.pdf)
- [NASA technical report 20140010629](https://ntrs.nasa.gov/api/citations/20140010629/downloads/20140010629.pdf)
