# First evaluation: 5 models and the best one

**Goal of this evaluation:** show 5 trained models and justify picking the best one from experimental evidence.
We trained 8 models in total. For this evaluation we present the **5 below, one from each algorithm family**. The other 3 (Gaussian NB, Decision Tree, XGBoost) are backup, in case evaluators ask "what else did you try?".

All numbers come from `results/model_comparison.csv`, produced by `scripts/train_models.py`.

## The 5 models

| # | Model | Family | Key configuration | CV Macro F1 (mean ± sd) | Student-test Top-1 | Student-test Macro F1 | Log loss | Route continuity | Inference (ms / 1k fixes) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Logistic Regression | Linear | L2, C=1.0, standardised | 0.9315 ± 0.0139 | 0.8894 | 0.9315 | 0.0838 | 0.939 | 0.2 |
| 2 | KNN | Instance-based | k=25, distance-weighted, 60k reference rows | 0.9269 ± 0.0144 | 0.8805 | 0.9269 | 0.1032 | 0.914 | 143.2 |
| 3 | Random Forest | Ensemble (bagging) | 150 trees, min_samples_leaf=5 | **0.9375** ± 0.0133 | 0.8930 | 0.9360 | 0.0705 | **0.945** | 9.4 |
| 4 | **LightGBM** ✅ | Boosting | 400 trees, lr=0.05, 63 leaves | 0.9367 ± 0.0143 | **0.9033** | **0.9404** | **0.0666** | 0.944 | 16.5 |
| 5 | MLP | Neural network | 64-32 ReLU, Adam, early stopping | 0.9320 ± 0.0135 | 0.8931 | 0.9312 | 0.0761 | 0.944 | 0.8 |

- **Protocol:** 5-fold GroupKFold over the 80 training trips (a trip never appears in two folds). Each model is then refit on all 80 trips and tested on the 10 student-test trips.
- **Task:** for every (GPS fix, candidate road arc) pair, predict P(this arc is the true one). Top-1 picks the highest-probability arc per fix.
- **Primary selection metric:** Top-1 accuracy, which is the actual route-matching objective, together with log loss. Log loss matters because the probabilities become the HMM's emission scores.

## Why LightGBM is the best (the answer to "why not Random Forest?")

1. **Best student-test Top-1** (0.9033) and **best Macro F1** (0.9404) of all 5.
2. **Best log loss** (0.0666), so it gives the best-calibrated probabilities to feed into the HMM/Viterbi decoder.
3. **CV is a tie between the ensembles.** Random Forest's CV F1 is 0.0008 higher, but the fold standard deviation is ~0.014, about 17× larger, so that difference is noise.
4. **Practical:** it trains in seconds, has a much smaller artifact than Random Forest, and offers more hyperparameters worth tuning. That makes it the right candidate for Bayesian optimisation, the next stage.

Why the other models lost:
- **Logistic Regression** is linear, so it cannot model interactions such as "heading only matters when the vehicle is moving". Its train F1 equals its test F1 (0.93), which means it underfits.
- **KNN** memorises: train F1 = 1.000 vs test 0.927 (overfitting). It is also about 10× slower at inference than every other model (143 ms per 1k fixes).
- **MLP** is decent, but on tabular data with thresholds (ranks, degrees) it trails boosting. Early stopping halted training at 37 epochs.

Figures to show: `results/figures/cmp_cv_macro_f1.png`, `cmp_test_top1_f1.png`, `cmp_logloss_inference.png`, `cmp_train_vs_test.png`, `cmp_training_histories.png`, `cmp_confusion_matrices.png`.
(These figures also include the 3 backup models; say so if asked.)

---

## Who presents what

| Member | Name / SRN | Presents | Time |
|---|---|---|---|
| **M1** | ____________ | Problem framing, data, labels and features (the setup all 5 models share) | ~2 min |
| **M2** | ____________ | Model 1 Logistic Regression and Model 2 KNN | ~2 min |
| **M3** | ____________ | Model 3 Random Forest and Model 5 MLP | ~2 min |
| **M4** | ____________ | Model 4 LightGBM, the comparison table and the final selection | ~3 min |

### M1: Setup (shared by all models)
- Data: 80 train and 10 student-test trips; files `.track`, `.nodes`, `.arcs`, `.route`.
- Per-fix labels come from a monotone alignment of the GPS fixes to the route (`pathsnap/labeling.py`). Fixes more than 100 m from the route are marked unreliable.
- Candidates: arcs within 50 m, at most 12 per fix. Candidate miss rate is 1.4% on train and 0.12% on student-test.
- 24 features: distance, heading, geometry, ±3-fix context and topology. Key insight: the nearest arc is right only about 55% of the time because two-way roads have two arcs at the same distance, so heading and context features are essential.
- Evaluation protocol: GroupKFold by trip, stride-3 training rows, fixed seed 42.

### M2: Logistic Regression and KNN
- **LR:** why it's a good baseline, its L2 log-loss objective, and that it underfits (train ≈ test ≈ 0.93).
- **KNN:** why instance-based learning was tried, why scaling is needed, train F1 = 1.0 (memorisation), and the very slow inference.

### M3: Random Forest and MLP
- **RF:** how bagging reduces a single tree's variance; train 0.972 vs test 0.936 (mild overfit); it is tied with LightGBM on CV but has worse log loss.
- **MLP:** the 64-32 architecture, Adam, early stopping; show the epoch loss curve in `cmp_training_histories.png`.

### M4: LightGBM and the selection
- Boosting fits residuals sequentially; the leaf-wise trees use the log-loss objective.
- Overfitting evidence: in the boosting history, held-out log loss is lowest around round 130 and then rises slightly. That motivates the Bayesian tuning that comes next.
- Walk through the table above and the 4 reasons LightGBM was picked.

### Every member should be able to answer
- What Macro F1, Top-1 and log loss mean here, and why Top-1 plus log loss is the primary selection metric.
- Why GroupKFold by trip and not a random split: fixes from the same trip would leak between folds.
- That evaluator-test data was never touched.

## Quick commands

```bash
# rerun the 5-model (actually all 8) comparison
.venv/bin/python -W ignore scripts/train_models.py
# or only the 5 presented models
.venv/bin/python -W ignore scripts/train_models.py LogisticRegression KNN RandomForest LightGBM MLP
# regenerate comparison figures
.venv/bin/python -W ignore scripts/make_figures.py
```
