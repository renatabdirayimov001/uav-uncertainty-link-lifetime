# Uncertainty-Aware Link Lifetime Prediction and Proactive Routing for UAV Swarms

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23612462.svg)](https://doi.org/10.5281/zenodo.23612462)

Simulation code, raw results and figure sources for the manuscript

> R. Abdirayimov, *"Uncertainty-Aware Link Lifetime Prediction and Proactive Routing for Reliable UAV Swarm Communications"*, submitted to MDPI *Drones*, 2026.

The code implements **risk-calibrated lifetime-aware routing (RCLR)**: a lower-quantile model of the residual UAV-to-UAV link lifetime, one-sided conformal calibration (CQR), adaptive conformal inference (ACI) with idealized and observable feedback, and hop-constrained widest-path routing with proactive (make-before-break) switching. It also contains all baselines used in the paper (kinematic LET, ML point predictor, bootstrap ensemble, raw quantile regression, split conformal prediction on the point model, oracle, shortest hop).

## Requirements

Python ≥ 3.10 and the packages in `requirements.txt` (tested with Python 3.13, NumPy 2.5, SciPy 1.18, scikit-learn 1.9, Matplotlib 3.11):

```bash
pip install -r requirements.txt
```

## Files

| File | Purpose |
|---|---|
| `sim.py` | Heterogeneous 3-D Gauss–Markov swarm mobility, residual link-lifetime labels, GPS noise, HELLO losses and age, link features (incl. LET) |
| `exp2.py` | Scenario builder (`mk`), dataset generation (`ds`), training of the point model and the bootstrap ensemble (training trace seed 1, calibration trace seed 2) |
| `route.py` | Hop-constrained widest-path search (`hop_widest`) and connectivity helper |
| `route3.py` | Slot-level routing simulator with proactive switching and all routing metrics (PDR, breaks, reactive/proactive discoveries, coverage on used links and routes, kept-route failures by hop count) |
| `exp5.py` | All routing experiments of the paper (`main`, `shift`, `noretrain`, `N30`, `N80`) → `results5/<mode>_s<seed>.json` |
| `pred2.py` | Prediction-level coverage with trace-level confidence intervals (8 test traces) → `results5/pred_ci.json` |
| `pred3.py` | Ablation of the conformal step → `results5/ablation_cqr.json` |
| `analyze5.py` | Validation-based setting selection, paired confidence intervals, summary (`results5/summary.json`) and the result figures (`figures/`) |
| `gen_tables.py` | LaTeX tables of the paper from the JSON results (`tables/`) |
| `pubbase.py` | Re-implementations of the published schemes: LB-OPAR (Gharib et al., 2022), OLSR+ (Rahmani et al., 2022), LPAR (Li & Sun, 2025) and LPMD-GPSR (Dong & Tang, 2026) |
| `nplstm.py` | Two-layer LSTM regressor in NumPy (forward, BPTT, Adam) used for the LPMD-GPSR link-duration predictor |
| `exp6.py`, `exp6b.py` | Comparison with the published schemes on the same traces (`main`, `N80`, `noretrain`, `shift`) -> `results6/` |
| `pred4.py` | Prediction-level quality of the published lifetime estimators -> `results6/pred_pub.json` |
| `analyze6.py` | Validation-based parameter selection for the published schemes, comparison tables and figure |
| `pred5.py`, `pred5b.py` | Coverage with continuous-time (sub-step) lifetime labels; single vs pooled calibration -> `results5/pred_cont*.json` |
| `final_routing.py` | First-round experiment with models retrained in the degraded-information scenario (`SCEN=stale`) → `results/route_<scen>_s<seed>.json` |
| `results5/`, `results6/`, `results/` | The raw JSON results used in the paper |
| `figures_src/` | Scripts of the illustrative figures: `make_concept_chart.py` (Fig. 1, matplotlib), `make_fw_drawio.py` + `fig_framework_academic.drawio` (Fig. 3, editable draw.io), `make_arch5.py` + `render.py` (Fig. 2, draw.io). Fig. 2 uses isometric artwork by Freepik (free license with attribution), which is **not** redistributed here; download the assets from freepik.com to rebuild it. The device icons of the onboard-module callout are also cut from Freepik artwork and are not included. |
| `figures/`, `tables/` | Created by `analyze5.py`, `analyze6.py` and `gen_tables.py`; they reproduce all result figures and tables of the paper from the JSON files |

## Reproduce

```bash
# prediction-level results (about 10 min)
python pred2.py
python pred3.py

# routing, base scenario: validation traces 7-10, test traces 11-22 (about 15 min per trace)
python exp5.py main 7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22

# robustness without retraining and swarm-size sensitivity
python exp5.py shift 31,32,33,34
python exp5.py noretrain 51,52,53,54
python exp5.py N30 41,42,43
python exp5.py N80 41,42,43    # swarm-size runs use 600-s traces
python exp5.py weak 11,12,13,14,15,16,17,18,19,20,21,22   # quantile model trained on 2% of the data, raw vs CQR
python exp5.py shiftb 31,32,33,34                         # fixed beta = 0.02 under the speed shift
python pred5.py

# comparison with re-implemented published schemes
for m in main N80 noretrain shift; do python exp6.py $m <seeds>; python exp6b.py $m <seeds>; done   # seeds as above
python pred4.py
python analyze6.py

# degraded-information scenario with retrained models (first round)
SCEN=stale python final_routing.py 7,8,9,10

# summaries, result figures and all tables of the paper (no simulation needed, reads the JSON results)
python analyze5.py
python analyze6.py
python gen_tables.py
```

Every trace is generated from its own seed: training trace 1, calibration trace 2, prediction test traces 3–6 and 23–26, routing validation traces 7–10, routing test traces 11–22, speed-shift traces 31–34, swarm-size traces 41–43, degraded-information traces 51–54. The environment variable `TT` shortens the routing traces for quick tests (e.g. `TT=200 python exp5.py main 7`).

**Note on exact reproducibility.** scikit-learn's `HistGradientBoostingRegressor` uses early stopping on a random internal validation split, so re-runs can differ in the last digits. The "gradient-boosted point predictor" is scikit-learn's HistGradientBoostingRegressor, not the XGBoost library. The conclusions of the paper do not depend on these differences.

## License

MIT, see `LICENSE`. If you use this code, please cite the paper and the archived software release (see `CITATION.cff`).

## Quick check

```bash
TT=120 python exp5.py main 7     # about 30 s; run it in a copy of the repository, because it overwrites results5/main_s7.json
```
