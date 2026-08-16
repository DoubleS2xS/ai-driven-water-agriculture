# Forecasting Irrigation Events from Soil-Moisture History: A Reproducible Benchmark with Explainable Models

This repository contains the full, deterministic pipeline behind the study.
It predicts whether a drip-irrigation electrovalve will be **open during
hour *t*** using only information observable at **hour *t* − 1 or earlier**,
and benchmarks gradient-boosted trees against four reference models.

Every number quoted in the manuscript is produced by `python -m
src.evaluate_pipeline` and written to `data/outputs/`. None is transcribed
by hand.

---

# 1. Data and site

| | |
|:--|:--|
| Sensor data | Mendeley Data, [doi:10.17632/cjb4vy4mzj.3](https://data.mendeley.com/datasets/cjb4vy4mzj/3) — soil moisture, electrovalve relay, flow meter |
| Site | Areguá, Central Department, **Paraguay** (−25.31, −57.39), strawberry field |
| Period | 2022-07-12 → 2022-09-16 local time (UTC−04:00) |
| Weather | NASA POWER hourly MERRA-2 reanalysis for the **same** coordinates |
| Design matrix | 1 313 supervised hours × 40 causal features, 23.6 % positive |

Mendeley timestamps are local Paraguayan wall-clock time and are converted
to UTC by adding four hours before the merge. Paraguayan DST does not
overlap the observation window, so a single constant offset is exact.

The reconciliation is verified automatically after every merge: the mean
diurnal cycle of `solar_radiation` must peak at 16:00 UTC ± 2 h, or the
loader raises. See `src/data_loader.py::validate_diurnal_alignment`.

---

# 2. Problem statement

**Target** — `irrigation_event(t)` ∈ {0, 1}.

**Features** — strictly causal. Every column on row *t* is a function of
observations at *t* − 1 or earlier: soil-moisture lags, drying-rate
differences, causal rolling statistics, weather lags and daily
aggregates, calendar encodings, and autoregressive lags of the target.

Two groups of columns are excluded **by construction** and the exclusion
is enforced by `src/features.py::assert_no_forbidden_features`, not merely
intended:

* `soil_moisture(t)` — moisture rises *because* the valve opened, so
  conditioning on it turns forecasting into after-the-fact detection.
* `flow_l`, `flow_l_cumulative` — the metered volume the valve delivered.
  Forbidden at every lag.

The no-look-ahead contract is machine-checked: `tests/test_features.py`
blanks the tail of the input and asserts that no earlier feature moves,
and separately blanks a single row and asserts that row's own features do
not move. Together these pin the contract to *t* − 1 and earlier exactly.

---

# 3. Validation protocol

* **Rolling-origin cross-validation**, 5 expanding-window folds — primary.
* **Chronological 80/20 holdout**, single ordered split — secondary, for
  comparability with the wider literature.
* Nothing is shuffled. `src/validation.py::assert_splits_are_ordered`
  raises if any fold's test block is not strictly after its training data.
* All preprocessing is fitted **inside the training fold** via
  `sklearn.Pipeline`; there is no code path that fits a transformer on the
  test block.

The irrigation regime is strongly non-stationary — the test-block positive
rate ranges from 3.2 % (late July) to 79.8 % (early September). Folds are
therefore **not exchangeable**, and per-fold results are reported
alongside every aggregate. See `data/outputs/folds.csv`.

---

# 4. Repository structure

```text
.
├── configs/
│   └── default.yaml              site coordinates, timezone, feature windows
│
├── data/
│   ├── raw/                      cached NASA POWER JSON (coords in filename)
│   ├── processed/                merged_hourly.csv
│   └── outputs/                  all result artefacts and figures
│
├── src/
│   ├── config.py                 every tunable constant
│   ├── data_loader.py            merge, UTC reconciliation, diurnal check
│   ├── features.py               causal feature construction, leakage guard
│   ├── validation.py             rolling-origin CV, ordering invariants
│   ├── preprocessing.py          per-fold preprocessing policy
│   ├── metrics.py                imbalanced-classification metric set
│   ├── statistics.py             confidence intervals, paired bootstrap
│   ├── baselines.py              the four reference models
│   ├── export.py                 result artefacts and run provenance
│   ├── evaluate_pipeline.py      main experiment
│   ├── robustness_experiment.py  corruption/healing study (separate)
│   ├── data_corruption.py        drift and missingness injection
│   ├── data_healing.py           MICE imputation, drift compensation
│   └── models/
│       ├── irrigation_ml.py      XGBoost / LightGBM wrapper
│       └── explanation.py        SHAP figures and instance selection
│
├── tests/                        298 tests
│   ├── test_data_loader.py       32   merge, timezone, diurnal alignment
│   ├── test_features.py          41   causality contract, leakage guard
│   ├── test_validation.py        31   fold ordering, preprocessing leakage
│   ├── test_baselines.py         35   baselines and metrics
│   ├── test_statistics.py        37   intervals, bootstrap
│   ├── test_ablation.py          18   feature sets, shared-rows invariant
│   ├── test_explanation.py       34   SHAP instance selection
│   ├── test_export.py            29   artefacts and provenance
│   ├── test_models.py            12   predictor API
│   ├── test_data_corruption.py   16   drift/missingness injection
│   └── test_data_healing.py      13   imputation and compensation
│
├── requirements.txt
└── data_provenance.yaml
```

---

# 5. Installation

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

# 6. Running

```bash
python -m src.data_loader --config configs/default.yaml   # rebuild dataset
python -m src.evaluate_pipeline                           # main experiment
python -m src.robustness_experiment                       # robustness study
pytest -q                                                 # 298 tests
```

Useful flags for `evaluate_pipeline`:

```
--n-seeds N          repeated runs (default 10)
--bootstrap N        bootstrap iterations (default 10000)
--compare-metric     roc_auc | pr_auc
--shap-fold          best | last
```

---

# 7. Metrics

Positive class ≈ 23.6 %, so the metric set is chosen for imbalance:

* **PR-AUC (average precision)** — the metric of record. Compare it
  against the **positive rate (0.2361)**, never against 0.5.
* ROC-AUC, F1, precision, recall, Brier score, confusion matrix.

No metric is reported without an interval. Intervals are Student-*t* over
the five folds, clipped to [0, 1].

**All six models are deterministic** — ten seeds produce byte-identical
results, because the tree ensembles run without row or column subsampling
and the remaining models have no random component. The seed-level interval
therefore has zero width and is *not* used as the confidence interval; it
is preserved in the `seed_*` columns of `results_summary.csv` and flagged
by `deterministic_across_seeds`.

---

# 8. Results

Numbers live in `data/outputs/`, regenerated on every run:

| File | Contents |
|:--|:--|
| `results_summary.csv` | model × metric × mean × SD × 95 % CI |
| `baselines.csv` | baselines with no-skill reference and gap to the best main model |
| `ablation.csv` | feature set × model × metric |
| `feature_importance.csv` | mean \|SHAP\| per feature, ranked, tagged by block |
| `folds.csv` | fold periods, sizes, class balance |
| `model_comparison.json` | paired bootstrap: difference, CI, p-value |
| `run_metadata.json` | library versions, git commit, dataset shape, site |

Headline findings, all reproducible from those files:

1. Under rolling-origin CV the gradient-boosted models **do not beat** a
   soil-moisture threshold rule or logistic regression. The single 80/20
   split reverses this ranking, because its test block lies entirely in
   the easy late-season regime — which is why the holdout is reported as
   a secondary result only.
2. Weather **degrades** performance: adding the 16 meteorological features
   to the soil-moisture lags lowers PR-AUC (ablation A → B).
3. The leakage control (set E, weather only) sits at chance —
   ROC-AUC ≈ 0.51–0.54 against a no-skill 0.50.

---

# 9. Feature ablation

| Set | Features | Purpose |
|:--|:--|:--|
| A | soil-moisture lags only | how far past moisture alone gets |
| B | + weather | does regional reanalysis add anything |
| C | + calendar | does time-of-day/season add anything |
| D | + irrigation autoregressive lags | full set |
| E | weather only | **leakage control** — must be near chance |

Every set is a **column subset of one matrix**, so all five are scored on
the identical 1 090 held-out rows. Rebuilding per set would drop a
different warm-up period and confound features with sample size.

---

# 10. Explainability outputs

Generated in `data/outputs/`:

* `shap_summary_beeswarm.png` — global importance and effect direction.
* `shap_waterfall_confident_true_positive.png` — the most confident
  correct alarm.
* `shap_waterfall_confident_false_positive.png` — the most confident
  *wrong* alarm, the most informative single plot in the set.
* `shap_waterfall_borderline.png` — the instance nearest the 0.5 boundary.
* `shap_dependence_{1,2,3}_<feature>.png` — the top three features.
* `shap_instances.json` — the row index, timestamp and probability of
  every instance plotted, so the manuscript can name the hours it shows.

Instances are selected by what each demonstrates, not by position. Reading
the waterfall plots requires the training fold's moisture distribution
alongside: an absolute value that looks high in the record as a whole can
be near the bottom of a given fold's training range.

---

# 11. Robustness study

`src/robustness_experiment.py` — sensor-drift and packet-loss injection
with MICE imputation and drift compensation. Reported as a robustness
study, deliberately **outside** the main pipeline, because its injected
drift is periodic and the models exploit that periodicity. See that
module's docstring for the measured outcome and its caveats.

---

# 12. Reproducibility

* Fixed seeds; every model deterministic.
* NASA POWER responses cached under `data/raw/`, with the coordinates in
  the filename so a change of site cannot be masked by a stale cache.
* `run_metadata.json` records the git commit **and whether the working
  tree was dirty**. Results produced from uncommitted code are not
  reproducible from that commit, and the file says so.
* `data_provenance.yaml` documents sources, licences, the timezone
  conversion, and the assumptions A1–A3.

---

# 13. Citation

```bibtex
@mastersthesis{shaya_water_management_2026,
  author       = {Your Name and Shaya, Ibrahim},
  title        = {Forecasting Irrigation Events from Soil-Moisture History:
                  A Reproducible Benchmark with Explainable Models},
  school       = {Department of Technologies and Information Systems},
  year         = {2026},
  address      = {Karaganda, Kazakhstan},
  month        = {June}
}
```

---

# 14. License

See `LICENCE`. The Mendeley dataset is CC BY 4.0; NASA POWER data are
public domain.

---

# 15. Acknowledgments

Developed as part of a Master's thesis under the supervision of
**Professor Ibrahim Shaya**.
