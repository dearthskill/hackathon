# PathSnap — Map-Matching & Route Reconstruction

Noisy GPS trajectories are matched to the road network by a learned **candidate classifier**
(tuned LightGBM). An **HMM / Viterbi** decoder then makes the matched sequence follow the road
topology. The system is served as a **FastAPI** backend (Laptop 1) and a browser **UI** (Laptop 2).

```
.track/.nodes/.arcs ──► clean + project ──► candidate arcs (≤50 m) ──► 24 features
      ──► LightGBM P(arc is correct) ──► HMM/Viterbi over network distances ──► connected route
```

## Repository layout

| Folder | Contents |
|---|---|
| `pathsnap/` | Library: data loading & cleaning (`data_io.py`), projection (`geo.py`), road network & shortest paths (`network.py`), per-fix label alignment (`labeling.py`), candidate generation & feature engineering (`features.py`), dataset building (`dataset.py`), model zoo (`models.py`), metrics (`metrics.py`), HMM/Viterbi (`hmm.py`), inference pipeline (`pipeline.py`), config (`config.py`) |
| `scripts/` | `eda.py` (exploration + Folium), `train_models.py` (8-model comparison), `bayes_opt.py` (Optuna TPE), `hmm_eval.py` (decoder tuning + HMM vs classifier), `make_figures.py`, `make_maps.py` (Folium maps of predictions) |
| `api/` | FastAPI backend (`main.py`) |
| `ui/` | Evaluator UI (`index.html`, `app.js`, `style.css`; Leaflet) |
| `artifacts/` | Final model (`model.joblib`), `model_meta.json` (features, tuned params), `hmm_config.json` |
| `results/` | All experiment outputs: CSVs, Optuna study/trials, training histories, figures, Folium maps, logs |
| `report/` | Report generator and the final PDF (`PathSnap_Report.pdf`) |

## Setup

Python 3.11. On macOS, LightGBM/XGBoost need OpenMP: `brew install libomp`.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The raw data is not committed. By default it is expected at `../Student folder/{train,student_test}`.
Override this with `PATHSNAP_DATA_DIR=/path/to/data`, where the folder must contain `train/` and `student_test/`.

## Reproduce all results

```bash
./run_pipeline.sh                          # EDA → 8 models → Optuna → HMM → figures → maps (~40 min)
.venv/bin/python report/build_report.py    # rebuilds report/PathSnap_Report.pdf from results/
```

Every script fixes `SEED = 42`. Trip-grouped 5-fold CV is used throughout. The evaluator-testing partition is never read by any training or tuning script.

## Deployment (two laptops)

**Laptop 1: API.** Use the same repo and venv, with the evaluator trips in a folder of trip sub-folders:

```bash
PATHSNAP_EVAL_DIR="/path/to/evaluator_test" .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- Find the IP with `ipconfig getifaddr en0` (macOS) or `ipconfig` (Windows).
- Allow port 8000 through the firewall.
- The console prints one log line per request: trip, accuracy, route F1 and latency. Show this to the evaluator.

**Laptop 2: UI.** Only the `ui/` folder and a browser are needed. Internet access is required for the Leaflet library and map tiles.

```bash
cd ui && python3 -m http.server 5500
# open http://localhost:5500/?api=http://<LAPTOP1_IP>:8000
```

Then pick any of the ten trips. The UI sends only the `trip_id` to `POST /predict` and draws the route and metrics that come back.

### API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status and loaded model |
| GET | `/trips` | The (up to) ten evaluation trip ids in `PATHSNAP_EVAL_DIR` |
| POST | `/predict` | Body `{"trip_id": "..."}`. Runs the full pipeline and returns GPS, predicted route (HMM and classifier only), nearby network, aggregate metrics and timings |

### Leakage safeguards
- `pipeline.PathSnapMatcher.predict_trip` loads trips with `with_route=False`, so the route file never reaches feature building, the classifier or the HMM.
- Accuracy is computed afterwards in `pipeline.evaluate_trip`, on the server.
- The response contains only aggregate metrics. It never includes reference arcs, reference geometry or per-point correctness.
- No predictions are hardcoded. Each request runs the whole pipeline; see the API logs and `timings` in the response.
