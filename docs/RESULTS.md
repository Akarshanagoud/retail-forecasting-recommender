# Verified results

Output captured by actually running this repository. Nothing here is hand-written.

| | |
|---|---|
| Date | 2026-10-15 |
| Commit | `39e8c86` |
| Python | Python 3.11.9 |
| OS | Windows 11 |

Timing figures (ms, rps) vary by machine and run. Counts and accuracy figures are deterministic.

## Reproduce

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]" pytest
pytest -q
python examples\quickstart.py
```

## Test suite

```text
........................................................................ [ 61%]
.............................................                            [100%]
117 passed in 3.41s
```

## `examples/quickstart.py`

```text

==============================================================================
1. Synthetic data with the structure that makes retail hard
==============================================================================
  14,600 rows across 20 series
  1,917 zero-demand days (intermittent SKUs)
  2,669 promotion days with price elasticity
  6,997 customer interactions

  weekly seasonality of the first series:
    Mon   212.8  #####################################################
    Tue   236.7  ###########################################################
    Wed   232.9  ##########################################################
    Thu   254.8  ###############################################################
    Fri   314.5  ##############################################################################
    Sat   322.0  ################################################################################
    Sun   248.9  ##############################################################

==============================================================================
2. Features that cannot leak, proved rather than promised
==============================================================================
  702 rows x 32 features (28 warmup rows dropped)
  perturbing a future target changed no earlier feature row

==============================================================================
3. Rolling-origin backtest against the baseline that matters
==============================================================================
MODEL                        SERIES     WAPE    STDEV    MASE  BEATS SN   FIT ms
--------------------------------------------------------------------------------
ridge                            20   0.1516   0.0580   0.459       110    37.28
ensemble                         20   0.2056   0.0774   0.590       102    39.45
seasonal_moving_average          20   0.2722   0.2353   0.789        82     0.00
seasonal_naive                   20   0.3135   0.2790   0.877        74     0.00
holt_winters                     20   0.3568   0.2642   1.038        54     0.22
naive                            20   0.4174   0.4744   1.138        57     0.00

  MASE below 1.0 means better than 'same day last week'.
  ridge improves WAPE by 51.6% over the baseline.
  STDEV is the spread across origins: a low mean with a high spread
  is a coin flip, not a deployable model.

==============================================================================
4. Recommender, scored against showing everyone the bestsellers
==============================================================================
  catalogue of 40 items, k=10

MODEL                   PREC@10   RECALL     MAP    NDCG     HIT  COVERAGE  NOVELTY
-----------------------------------------------------------------------------------
item-item CF             0.1069   0.7079  0.3088  0.4229  0.7964    1.0000    5.325
popularity               0.0442   0.2762  0.0996  0.1524  0.3573    0.6000    5.208

  explainable recommendations for CUST-0000:
    SKU-014    because you bought SKU-019, SKU-034
    SKU-039    because you bought SKU-029, SKU-034
    SKU-009    because you bought SKU-029, SKU-019

==============================================================================
5. Drift detection: what moved, and does it matter
==============================================================================
overall verdict: drifted
performance    : drifted (+55.0% vs baseline error)
FEATURE                           PSI       KS  VERDICT
rolling_mean_28                1.1898   0.4587  drifted
rolling_mean_7                 0.8820   0.3675  drifted
rolling_std_7                  0.5264   0.1709  drifted
discount_depth                 0.3915   0.1254  drifted
lag_2                          0.3745   0.2507  drifted
lag_1                          0.3586   0.2536  drifted
lag_7                          0.3350   0.2365  drifted
seasonal_lag                   0.3350   0.2365  drifted
lag_14                         0.3172   0.2308  drifted
lag_28                         0.2896   0.2137  drifted
annual_sin_1                   0.0392   0.0427  stable
annual_cos_2                   0.0167   0.0427  stable
annual_sin_2                   0.0159   0.0427  stable
annual_cos_1                   0.0081   0.0427  stable
day_of_month                   0.0067   0.0370  stable

==============================================================================
6. Retraining requires evidence, and promotion requires improvement
==============================================================================
  trigger : ['performance_drift', 'feature_drift']
  marginal candidate: promoted=False - improvement 0.3% is below the 2% promotion gate
  material candidate: promoted=True - improves error by 20.0%

==============================================================================
7. Every run is tracked, every promotion is gated
==============================================================================
  leaderboard:
    ridge                        WAPE 0.1516
    ensemble                     WAPE 0.2056
    seasonal_moving_average      WAPE 0.2722

  promote v1: True - promoted v1: no incumbent
  promote v2: False - not promoted: v2 improves wape by -35.6%, below the 2% gate (incumbent v1)

Done. Every number above was computed on this machine, in seconds.
```
