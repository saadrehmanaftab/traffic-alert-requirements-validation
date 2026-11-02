# Requirements Based Validation of a Traffic Alerting Function

Personal project. A simple tau based traffic alerting function, written as three system
requirements and validated by Monte Carlo simulation in Python (NumPy, pandas, Matplotlib).
All inputs are **assumed**; this is not a model of any certified collision avoidance system.

## System requirements (assumed)
- **R1 Alert lead time:** for threats (true miss distance below 150 m), at least 99% of encounters
  alert 25 s or more before closest approach.
- **R2 Nuisance alerts:** at most 5% of non threatening encounters (miss distance above 600 m) alert.
- **R3 Crew margin (human factors):** at the 5th percentile, the time left after an assumed 5 s crew
  recognition and response time is at least 15 s.

## Function under test
- Surveillance gives the intruder's relative position every 1 s with 30 m noise per axis.
- A tracker estimates relative velocity by a least squares fit over the last N updates.
- Alert when tau (time to closest approach) is below a threshold and the predicted miss distance
  is below a horizontal miss distance (HMD) threshold.

## Validation
4,000 random encounters per setting (closing speed 80 to 250 m/s, miss distance 0 to 1,500 m), swept
over tracker window (5 and 10 updates), tau threshold (20 to 45 s) and HMD threshold (150 to 300 m).

## Results (from the run, `results.csv`)
| tracker | tau | HMD | threats alerted in time | nuisance alerts | margin after crew | verdict |
|---|---|---|---|---|---|---|
| 5 updates | 30 s | 300 m | 99.5% | 14.0% | 22 s | fails R2 |
| 10 updates | 30 s | 225 m | 100% | 0% | 22 s | passes R1, R2, R3 |

Lengthening the tracker window from 5 to 10 updates removes the nuisance alerts caused by noisy
velocity estimates, so a 30 s threshold meets all three requirements.

## Assumptions and limits
Two dimensional, constant velocity encounters; no vertical dimension, no manoeuvres, no coordination;
Gaussian position noise; assumed requirement values and crew response time. A learning exercise in
requirements based validation, not a certification argument.

## Run
```
python alert_validation.py
```
