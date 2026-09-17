# NGBoost for LOS prediction — briefing

## Goal
Predict length of stay as a *probability distribution per patient*, not a single number. Same target as ordinary regression (bed-days), but each prediction carries its own calibrated uncertainty — so we can report an expected LOS, a range, and P(bed free by day *t*) for every patient individually.

## What it is
Gradient boosting (like XGBoost) but with a probabilistic output. Instead of minimising squared error toward the mean, it boosts trees to fit the *parameters of a chosen distribution*, using the natural gradient of a proper scoring rule. Engine = trees (nonlinear, learns interactions); output = a distribution.

## Output — the key point for integration
For each patient the model returns the parameters of one distribution (e.g. for a log-normal: two numbers). That parameter pair *is* the prediction. Everything operational is a deterministic readout from it, computed on demand:

- point estimate → `mean` (or median)
- uncertainty → `sd`, or quantiles (p10/p50/p90 for a range)
- `P(LOS ≤ t)` → the CDF at day *t*, for any *t*

So downstream you store ~2 numbers per patient and derive the rest — no re-running the model to answer a new "what's the chance they're out by day 10" question.

## Distribution choice
LOS is positive and right-skewed, so a positive-skewed family, not Normal. **LogNormal** is built in and a sensible default; Gamma/Weibull are alternatives (check availability in the installed version, may need a custom dist). This choice is a modelling decision the group should validate against the data, especially the long tail.

## How to use (sklearn-compatible)

```python
from ngboost import NGBRegressor
from ngboost.distns import LogNormal
from ngboost.scores import LogScore

model = NGBRegressor(Dist=LogNormal, Score=LogScore,
                     n_estimators=500, learning_rate=0.01)
model.fit(X_train, y_train, X_val=X_val, Y_val=y_val)  # early stopping on val

dist   = model.pred_dist(X_test)   # distribution per row
mean   = dist.mean()               # point prediction
p_le_t = dist.dist.cdf(t)          # P(LOS <= t)
q10, q50, q90 = dist.dist.ppf([.1, .5, .9])
params = dist.params               # the stored per-patient parameters
```

It slots into normal sklearn workflows: pipelines, cross-validation, `Score=CRPS` as an alternative objective. Base learner (tree depth etc.) is configurable. *Exact attribute names (params keys, scipy handle) vary by distribution/version — verify against the installed package.*

## Development / integration notes

- **Preprocessing:** the default tree base learner does *not* handle NaNs or raw categoricals — impute and encode upstream. Build it into a pipeline so serving and training match.
- **Validation:** score with NLL or CRPS (these judge the whole distribution), *plus a calibration check* (PIT histogram / reliability curve). A model can have a good mean and still be badly calibrated on spread — that's the failure mode that matters here, since spread is the reason we chose this.
- **Point accuracy** (MAE/RMSE on the median) is a secondary check, comparable to a plain XGBoost baseline — worth running that baseline to prove the distributional version earns its complexity.
- **Serving:** lightweight — persist the fitted model, output two params per patient, compute readouts in the app layer.

## Caveats to set expectations

- Slower to train than XGBoost; usually fine at this data size (~7k rows).
- Won't extrapolate beyond the training LOS range; the long tail (max ~89d) is data-sparse, so tail probabilities and extreme-patient certainty need scrutiny.
- Distribution-family and calibration are the two decisions that make or break it — not the boosting hyperparameters.

## One-line framing for the group
*Same input features and same bed-days target as a standard regressor; the difference is the model returns a per-patient LOS distribution instead of a point, which is what lets us quote ranges and discharge-by-day probabilities.*
