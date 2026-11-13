# Energy Expenditure (METs) and Activity Intensity: Scoping Document

**Date:** November 2026
**Scope:** Wrist-worn triaxial accelerometer only. Output one MET value per 1-minute epoch and one intensity class per epoch.
**Status:** Scoping. No code yet. `physiodsp/activity/energy_expenditure.py` is currently a `NotImplementedError` stub.

---

## 1. Problem Definition

### 1.1 Outputs

| Output | Resolution | Definition |
|---|---|---|
| `met` | 1 min | Metabolic equivalent: VO2 (ml/kg/min) / 3.5. Floor at the resting value (default 1.0). |
| `intensity` | 1 min | `sedentary` (< 1.5 MET), `light` (1.5 to < 3), `moderate` (3 to < 6), `vigorous` (>= 6). |
| `kcal` (optional) | 1 min | `met * body_mass_kg / 60`. Only when the user supplies body mass. |
| `ambulatory` (diagnostic) | 1 min | Fraction of the minute spent in a step bout (from `StepCount`). |

Intensity is computed by two methods, both implemented and benchmarked (decision 3):

- **`met`:** thresholds on the MET estimate. The two outputs can never disagree.
- **`enmo_cutpoints`:** non-dominant wrist ENMO thresholds (Hildebrand-style, re-tuned on our data). Comparable with GGIR-style literature and independent of the MET model.

The default is chosen in Phase 3 on CAPTURE-24 validation kappa. Both stay selectable.

**Target placement:** non-dominant wrist (decision 1).

### 1.2 Out of scope (v1)

- Sleep staging. Sleep is reported as `sedentary`. A `sleep` class can come later from the `sleep/` module.
- Non-wear detection. Epochs flagged as non-wear by the caller are returned as `NaN`.
- Heart-rate fusion. Accelerometer only, as requested.
- Activity type recognition. The `ActivityRecognition` stub stays untouched, but the CAPTURE-24 pipeline built here can feed it later.

### 1.3 Known physical limits of a wrist accelerometer

These bound the achievable accuracy and must be documented in the README:

- **Cycling and stationary bike:** arms are nearly still, so EE is underestimated.
- **Load carrying, pushing, uphill walking:** extra cost is not visible at the wrist.
- **Arm-only tasks** (dishwashing, gesturing, typing): wrist motion is high, cost is low, so EE is overestimated.
- **Posture:** sitting vs. standing (both ~1.3 to 1.8 MET) is only weakly observable from wrist angle.

The design uses step cadence to address the second and third points for walking. Cycling remains a documented blind spot.

---

## 2. Algorithm Options

Five tiers, ordered by complexity. All tiers share the same pre-processing and 1-minute output contract.

| Tier | Method | Training needed | Inference deps | Expected accuracy | Role |
|---|---|---|---|---|---|
| T0 | ENMO cut-points (Hildebrand 2014/2017) + linear ENMO-to-MET | No | numpy | Low for MET, moderate for intensity | Baseline |
| T1 | Staudenmayer 2015 wrist linear model (`sdvm`, `mangle`) | No | numpy | RMSE ~1.55 MET (published, lab) | Baseline |
| **T2** | **Cadence-aware hybrid regression** (this project) | Yes, offline | numpy | Target RMSE <= 1.2 MET | **v1 default** |
| T3 | Gradient-boosted trees on ~40 features + HMM smoothing | Yes, offline | numpy (JSON tree evaluator) | Likely best non-DL | Stretch, ship only if it beats T2 clearly |
| T4 | Self-supervised ResNet (ssl-wearables) fine-tuned on CAPTURE-24 | Yes | torch | Best on CAPTURE-24 | Out of scope for core; possible `physiodsp[ml]` extra |

### 2.1 Recommendation

Implement **T0 and T1 as baselines** and **T2 as the default model**. Evaluate T3 in the research harness. Promote T3 only if it improves held-out RMSE by more than 0.15 MET and macro-F1 by more than 0.03, since it costs interpretability and a custom tree evaluator.

Reasons:

1. Core dependencies are numpy, pandas, scipy, pydantic. T2 ships as a JSON file of coefficients and needs no new runtime dependency.
2. T2 reuses existing, tested building blocks: `ENMO` and `StepCount`.
3. Cadence is the strongest single wrist predictor of walking cost and is independent of arm swing amplitude. CADENCE-Adults reports 100 steps/min ~ 3 MET and 130 steps/min ~ 6 MET (ages 21 to 60).
4. Linear models trained on small calorimetry datasets (WEEE, 17 subjects) overfit less than tree ensembles.

### 2.2 T2 design: cadence-aware hybrid regression

```
Raw accel (x, y, z), any fs
        |
        v
[1] Pre-processing        resample to 30 Hz, clip to +-8 g, optional autocalibration,
                          left/right wrist harmonisation (mirror x axis)
        |
        v
[2] 10 s sub-window features
    - ENMO (mean), MAD of VM, sdvm
    - mean / sd of wrist angle (z vs. horizontal)
    - dominant frequency and spectral power 0.5 to 3.5 Hz
        |
        +------> [3] StepCount (existing) -> cadence per 10 s, ambulatory flag
        |
        v
[4] Branch per 10 s sub-window
    ambulatory:      MET = a0 + a1*cadence + a2*ENMO + a3*sdvm
    non-ambulatory:  MET = b0 + b1*log1p(ENMO) + b2*sdvm + b3*angle_sd + b4*MAD
        |
        v
[5] 1-min aggregation     mean of sub-window METs, floor at resting_met
        |
        v
[6] Temporal smoothing    centred 3-min median (optional, on by default)
        |
        v
[7] Intensity             MET thresholds 1.5 / 3 / 6
```

Notes:

- 10 s sub-windows keep a mixed minute (e.g. 30 s walking + 30 s sitting) from being averaged into a meaningless feature vector. Averaging METs is valid because EE is additive in time.
- Coefficients are fitted with non-negative constraints on intensity features so that MET is monotonic in movement. This is a cheap safety property and testable.
- If T3 is promoted, steps [4] to [6] are swapped for the tree model + HMM. The rest of the pipeline is unchanged.

### 2.3 Baseline details (to verify against the original papers before coding)

| Baseline | Formula / thresholds | Window | Source device | Verification needed |
|---|---|---|---|---|
| Staudenmayer 2015 LM | `MET = 1.89378 + 5.50821*sdvm - 0.02705*mangle` | 15 s | ActiGraph GT3X+, 80 Hz, wrist | Coefficients recalled from memory. Confirm in J Appl Physiol 119(4):396 (PMC4538283). |
| Hildebrand 2014/2017 ENMO | Sedentary / light / moderate / vigorous thresholds in mg, device-specific | 5 s | GENEActiv and ActiGraph, non-dominant wrist | Literature tables are inconsistent (values reported range from 35 to 52 mg for sedentary, 93 to 200 mg for moderate). Read both original papers. |

---

## 3. Datasets

### 3.1 Ground truth is the central issue

There are two kinds of MET labels and they are not interchangeable:

- **Measured:** indirect calorimetry (VO2). True continuous EE, but lab or semi-lab, small cohorts.
- **Assigned:** Compendium of Physical Activities value for an annotated activity. Large free-living cohorts, but every minute of "walking the dog" gets the same MET regardless of pace. No within-activity variance.

Consequence: **MET regression is fitted and tested on measured data. Intensity classification is fitted and tested on assigned data.** The datasets play different roles.

GOTOV is not accessible (decision 2). **WEEE is therefore the only measured-EE wrist dataset**, so there is no external MET test set. Section 3.6 covers the mitigation.

### 3.2 Candidate datasets

| Dataset | n | Wrist device | fs / range | Setting | Ground truth | License | Role |
|---|---|---|---|---|---|---|---|
| **CAPTURE-24** (Chan et al. 2024) | 151 | Axivity AX3, dominant wrist | 100 Hz, +-8 g | Free-living, ~24 h | Compendium METs from wearable-camera annotation; sleep diary | Check ORA terms | Intensity train / val / test |
| **WEEE** (Gashi et al. 2022) | 17 | Empatica E4 | 32 Hz, +-2 g (E4 spec, confirm in data) | Lab protocol | Indirect calorimetry (VO2 Master) + Compendium METs | Check Zenodo record | MET regression fit (LOSO) |
| PAMAP2 (UCI) | 9 | Colibri IMU, dominant wrist | 100 Hz | Lab protocol | Activity labels (map to Compendium) | Check UCI | Secondary intensity test (lab activities) |
| **NHANES 2011-2014** | > 10k | ActiGraph GT3X+, **non-dominant** wrist | 80 Hz | Free-living, 7 days | None | Public domain | Non-dominant plausibility check (daily MVPA and sedentary distributions vs. published NHANES figures) |

Rejected: GOTOV (not accessible). CATSE3, WSE3, OpenMetabolics (calorimetry, but thigh only). HARTH, HAR70+, MHEALTH, OxWalk (no EE or intensity labels, or no wrist).

### 3.3 Dataset-specific risks

| Risk | Dataset | Mitigation |
|---|---|---|
| +-2 g saturation during running | WEEE (E4) | Flag saturated sub-windows; exclude from fit or cap features. Report running separately. |
| 32 Hz limits spectral features above 16 Hz | WEEE | Common resampling to 30 Hz for all datasets. Use only features below 15 Hz. |
| Dominant-wrist training data, non-dominant target | CAPTURE-24, PAMAP2 | See section 3.7. |
| VO2 kinetics lag (1 to 3 min to steady state) | WEEE | Fit only on steady-state minutes (drop first 2 min of each activity). Report transitions separately. |
| Measured MET depends on body mass and resting rate | WEEE | Use VO2/kg/3.5 (standard MET). Report "corrected MET" (VO2 / measured resting VO2) as secondary. |
| Small n, single measured-EE dataset | WEEE | Leave-one-subject-out only. No random minute-level splits. See section 3.6. |
| Wrist side not documented | WEEE | Check the WEEE protocol. If mixed or dominant, treat like CAPTURE-24 (section 3.7). |
| Assigned METs are discrete | CAPTURE-24 | Do not report MET RMSE on CAPTURE-24 as a headline metric. Use it for intensity metrics and daily time-in-class bias. |

### 3.4 Splits

All splits are **by subject**, never by minute.

| Dataset | Train | Validation | Test |
|---|---|---|---|
| CAPTURE-24 | 70% subjects | 15% subjects | 15% subjects (frozen, seeded; reuse the split from Chan et al. 2024 if published) |
| WEEE | LOSO folds | Nested inside LOSO | LOSO aggregate |
| PAMAP2 | No | No | 100% (intensity only) |
| NHANES | No | No | Distribution check only |

Final T2 coefficients are fitted on all of WEEE (MET branch) + CAPTURE-24 train/val (intensity thresholds, ENMO cut-points and smoothing), then reported on CAPTURE-24 test, PAMAP2 and NHANES.

### 3.5 Cross-dataset evaluation

Report a train-on-A, test-on-B matrix. Device shift (E4 vs. AX3 vs. ActiGraph) is the most likely failure mode in real use, and in-dataset numbers hide it.

### 3.6 No external MET test set

With GOTOV out, every measured-MET number comes from 17 WEEE subjects. Mitigations:

1. **Nested LOSO** on WEEE: hyper-parameters chosen inside each fold, so the LOSO score is not optimistic.
2. **Activity-level agreement on CAPTURE-24 test:** compare the median predicted MET per Compendium activity code against the assigned MET. This is coarse, but it is free-living, other device, other people.
3. **Compendium sanity table:** predicted MET for sitting, standing, slow/normal/brisk walking, running must fall inside the Compendium ranges.
4. **Report confidence intervals** (bootstrap over subjects), not point estimates.
5. **Keep searching** for a second wrist + calorimetry dataset (open or by request). Candidates to contact: authors of the Staudenmayer 2015 and Montoye wrist calibration studies.

### 3.7 Dominant-wrist data, non-dominant target

CAPTURE-24 and PAMAP2 are dominant wrist. The dominant wrist moves more during daily tasks, so models and cut-points tuned on it will **underestimate intensity on the non-dominant wrist**. Mitigations:

1. Use only **rotation-invariant features** (VM, ENMO, MAD, sdvm, angle magnitude) plus x-axis mirroring, so left/right geometry does not matter.
2. **ENMO cut-points:** start from published non-dominant values (Hildebrand), re-tune on CAPTURE-24 only as a comparison, and ship the non-dominant values.
3. **Distribution check on NHANES** (non-dominant): predicted daily sedentary / light / MVPA minutes must sit near published NHANES wrist estimates. A systematic shift there quantifies the dominant/non-dominant bias.
4. Expose a `wrist` setting (`"non_dominant"` default, `"dominant"`) that applies a scale factor on movement features. The factor is estimated from item 3 and documented as approximate.

---

## 4. Evaluation Protocol

### 4.1 Metrics

**MET (measured datasets, steady-state minutes):**
- RMSE, MAE, mean bias
- Bland-Altman limits of agreement
- Lin's CCC
- Per-activity breakdown (sitting, standing, walking, running, cycling, ADLs)

**Intensity (CAPTURE-24 test, PAMAP2; WEEE via thresholds on measured MET). Both methods (`met` and `enmo_cutpoints`) are reported side by side:**
- Macro-F1, Cohen's kappa, per-class recall, confusion matrix
- Daily minutes per class: bias and LoA per participant (this is what end users read)

### 4.2 Proposed acceptance targets

| Metric | Dataset | Target | Reference point |
|---|---|---|---|
| MET RMSE | WEEE LOSO | <= 1.2 MET | Staudenmayer 2015 RF wrist: 1.21; LM: ~1.55 |
| Median MET per activity | CAPTURE-24 test | Within Compendium range for > 80% of codes | |
| Intensity kappa (4 class) | CAPTURE-24 test | >= 0.70 (best of the two methods) | Walmsley 2021 (sleep/sed/light/MVPA, RF+HMM): 0.80 |
| Daily MVPA / sedentary minutes | NHANES | Within published NHANES wrist ranges | |
| MVPA minutes/day bias | CAPTURE-24 test | within +-10 min/day | |
| T2 beats T0 and T1 | All | Yes, on every dataset | Otherwise ship T1 as default |

These targets are proposals. Re-set them after the Phase 1 baseline run.

---

## 5. Repository Layout

```
physiodsp/activity/
    energy_expenditure.py        # EnergyExpenditure (fills the existing stub)
    activity_intensity.py        # ActivityIntensity (new)
    _ee_features.py              # 10 s sub-window features, shared
    models/ee_t2_v1.json         # fitted coefficients + metadata (datasets, commit, metrics)

research/energy_expenditure/     # NOT packaged, own requirements file
    requirements-research.txt    # scikit-learn, (lightgbm), matplotlib
    datasets/                    # loaders: capture24.py, weee.py, pamap2.py, nhanes.py
    harmonise.py                 # resample, units, calibration, wrist side, GT alignment
    fit_t2.py                    # constrained regression, writes models/ee_t2_v1.json
    benchmark.py                 # T0/T1/T2(/T3) on all datasets, writes report
    README.md                    # download steps, DATA_DIR env var, checksums

tests/
    test_activity_energy_expenditure.py
    test_activity_intensity.py

docs/algorithms/
    energy_expenditure.md
    activity_intensity.md
```

Required change: add `models/*.json` to `[tool.setuptools.package-data]` in `pyproject.toml`.

Data never enters git. Loaders read from `DATA_DIR`. CI runs only unit tests on synthetic signals and a small committed fixture.

### 5.1 API sketch

```python
from physiodsp.activity import EnergyExpenditure, EnergyExpenditureSettings, ActivityIntensity

ee = EnergyExpenditure(settings=EnergyExpenditureSettings(
    model="physiodsp_t2_v1",        # or "staudenmayer2015", "enmo_linear"
    body_mass_kg=72.0,              # optional, enables kcal
    wrist="non_dominant",
))
ee.run(accelerometer)
ee.biomarker        # timestamps | met | kcal | ambulatory

intensity = ActivityIntensity().run(accelerometer)          # runs EE internally
intensity = ActivityIntensity.from_energy_expenditure(ee)   # reuse, no recompute
intensity.biomarker  # timestamps | intensity | met
```

Settings (Pydantic): `epoch_len=60`, `sub_window_len=10`, `resting_met=1.0`, `smoothing="median3"`, `wrist="non_dominant"`.

`ActivityIntensitySettings`: `method="met" | "enmo_cutpoints"` (default set after Phase 3), `met_thresholds=(1.5, 3.0, 6.0)`, `enmo_cutpoints_mg=(sed, mod, vig)` with shipped non-dominant defaults. `from_energy_expenditure` is valid only with `method="met"`.

### 5.2 Unit tests (CI, no external data)

- Static signal at 1 g -> MET == `resting_met`, intensity `sedentary`.
- Synthetic walking (2 Hz, 0.5 g) -> MET in [3, 6), intensity `moderate`.
- MET is monotonic non-decreasing in amplitude and cadence.
- Output length == number of complete minutes. Partial trailing minute dropped (match `ENMO`).
- `kcal` present only when `body_mass_kg` set.
- Input at 25, 32, 50, 100 Hz gives consistent MET (resampling).
- Saturated input (+-2 g clipping) does not crash and is flagged.
- Frozen regression test on `tests/accelerometer.csv` (catch silent coefficient changes).
- `from_energy_expenditure` and `run` give identical output (`method="met"`).
- `method="enmo_cutpoints"`: ENMO just below / above each cut-point maps to the expected class.
- Both methods agree on the extremes (static -> sedentary, running -> vigorous).

---

## 6. Phased Plan

| Phase | Deliverable | Exit criterion |
|---|---|---|
| **P0 Data** | Loaders + harmonisation for CAPTURE-24, WEEE, PAMAP2, NHANES. Per-minute tables: features, measured MET, assigned MET, label. Confirm WEEE wrist side. | Row counts and MET distributions match the papers. Read each dataset's README and licence. |
| **P1 Baselines** | T0 and T1 in the research harness. First benchmark report. | Numbers on all datasets; acceptance targets re-set. |
| **P2 T2 model** | Constrained regression, nested LOSO on WEEE, activity-level check on CAPTURE-24. | T2 beats T0/T1 everywhere; meets or is near targets. |
| **P3 Intensity** | Both methods (`met`, `enmo_cutpoints`) tuned on CAPTURE-24 val, tested on CAPTURE-24 test, PAMAP2, NHANES. Pick the default. | Kappa and MVPA bias targets for at least one method. |
| **P4 Package** | `EnergyExpenditure`, `ActivityIntensity`, JSON model, tests, docs, README section. | `pytest` and `flake8` green. Benchmark report linked from docs. |
| P5 (optional) | T3 trees + HMM; T4 feasibility note. | Promotion rule in section 2.1. |

---

## 7. Decisions

### 7.1 Confirmed (2026-11-03)

| # | Question | Decision |
|---|---|---|
| 1 | Target wrist | **Non-dominant.** Dominant-wrist training data handled per section 3.7. |
| 2 | GOTOV | **Not accessible.** Dropped. WEEE is the only measured-EE set (section 3.6). |
| 3 | Intensity method | **Both** (`met` and `enmo_cutpoints`) implemented and benchmarked. Default picked in Phase 3. |

### 7.2 Open (defaults chosen, confirm or override)

| # | Question | Default |
|---|---|---|
| 4 | Target device class (fs, range)? | Any fs >= 25 Hz, range >= +-4 g. Below that, outputs flagged. |
| 5 | Separate `sleep` class? | No in v1 (`sedentary`). |
| 6 | Age-specific cadence thresholds (61 to 85: 100/110/120 spm for 3/4/5 MET)? | Not in v1. Optional `age` setting later. |
| 7 | Measured vs. standard resting MET (3.5 ml/kg/min)? | Standard, for comparability with Compendium. |

---

## 8. References

**Datasets**
- CAPTURE-24: Chan S. et al., Sci Data 11:1135 (2024). [arXiv 2402.19229](https://arxiv.org/abs/2402.19229). Data: [Oxford ORA](https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001)
- WEEE: Gashi S. et al., Sci Data 9:537 (2022), [doi:10.1038/s41597-022-01643-5](https://www.nature.com/articles/s41597-022-01643-5). Data: [Zenodo 6420886](https://zenodo.org/record/6420886). Site: [wearableenergyexpenditure.github.io](https://wearableenergyexpenditure.github.io)
- GOTOV (not accessible, kept for reference): Paraschiakos S. et al., [4TU.ResearchData](https://data.4tu.nl/datasets/f9bae0cd-ec4e-4cfb-aaa5-41bd1c5554ce)
- Open dataset index: [wearable-landscape.info](https://wearable-landscape.info/topics/open-accelerometer-datasets)

**Methods**
- Staudenmayer J. et al., J Appl Physiol 119(4):396-403 (2015), [PMC4538283](https://pmc.ncbi.nlm.nih.gov/articles/PMC4538283); model summary at [Accelerometer Repository](https://sites.google.com/view/accelerometerrepository/available-models/provided-in-paper/staudenmayer)
- Hildebrand M. et al., Med Sci Sports Exerc (2014) and Scand J Med Sci Sports (2017). ENMO cut-points.
- Walmsley R. et al., Br J Sports Med (2021), RF + HMM on CAPTURE-24, [medRxiv](https://www.medrxiv.org/content/10.1101/2020.11.10.20227769v3)
- Tudor-Locke C. et al., CADENCE-Adults, IJBNPA 16:8 (2019), [doi:10.1186/s12966-019-0769-6](https://ijbnpa.biomedcentral.com/track/pdf/10.1186/s12966-019-0769-6); ages 41-60 (2020) and 61-85 follow-ups
- Yuan H. et al., self-supervised wearables (ssl-wearables), npj Digit Med (2024)
