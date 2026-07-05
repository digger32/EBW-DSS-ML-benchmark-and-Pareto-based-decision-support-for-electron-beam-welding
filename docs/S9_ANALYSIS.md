# S-9 — Thorough analysis of the merged campaign (`s7_merged_Dp`)

Snapshot: 5652 aggregated configurations. This document separates results
that are **scientifically locked** (clean, 3-fold, unaffected by the D′ run)
from the **data-integrity problems introduced by the D′ run**, then states
what does and does not need a re-run.

---

## 0. Data-integrity audit (read this first)

### 0.1 What the merge contains

| Block | Rows | Fold setting | Status |
|-------|-----:|--------------|--------|
| `real` ablation, all 41 models × 12 opt | 492 | 3-fold | **clean, original** |
| `synth`/`real_plus_synth`/`tstr`, 34 "fast" models | ~3500 | 3-fold (synth/r+s), 1 (tstr) | **clean, original** |
| `synth`/`r+s`/`tstr`, 5 D′ models (ngb, mdn, vote, tabnet, ftt) | 540 | **5-fold** (synth/r+s), 1 (tstr) | **see 0.2** |
| 7 remaining slow models (gpr, anfis, node, saint, cnn1d, bnn, stack) on tvae/copula/physics | — | — | **never run** (out of D′ scope) |

### 0.2 Two defects in the D′ run

1. **Fold mismatch (launcher bug).** `bucket_D_prime.sh` did not pass
   `--n-folds 3`, so it inherited the `S7Config` default of **5**. The whole
   original campaign — and the S-8 limitations text — use **3-fold** for the
   non-tstr ablations. The 540 D′ synth/r+s rows are therefore 5-fold and are
   **not directly comparable** with the rest of the grid.

2. **Timeouts.** Of the 540 D′ configs: **311 ok, 217 timeout, 12 served from
   the old 3-fold bucket-D**. The 5-fold setting is the main cause: 5 outer
   folds × (10 trials × 5 inner CV + refit) ≈ 255 single-core fits/config vs
   153 at 3-fold. The deep NN models could not finish in the 7200 s cap:

   | model | timeouts | folds reached (of 5) before cap |
   |-------|---------:|----------------------------------|
   | ftt    | 78 | 0–1 (essentially unusable on synth/r+s) |
   | tabnet | 72 | 1–3 (partial) |
   | mdn    | 43 | mostly 3–4 (nearly complete) |
   | vote   | 24 | mixed 1–4 |
   | ngb    | 0  | complete |

   195 of the 217 timeout rows carry *partial* metrics (1–4 folds); 22 (all
   ftt) are empty.

### 0.3 Consequence

* **No headline result depends on the D′ data.** D′ only touched
  synth/r+s/tstr cells for 5 slow models. The `real` ablation — the basis for
  the model ranking, the optimiser ranking and the inverse-design Pareto set —
  is the original clean 3-fold data and was never touched.
* The D′ data is **not usable as-is** for a clean merged synthetic-ablation
  ranking (fold mismatch + truncation). It should be re-run at 3-fold or the
  slow-model synth cells declared partial in Limitations (§6).

---

## 1. Model ranking on the real ablation (clean)

Best macro R² across the 12 optimisers, per model. Friedman test across the 41
models (12 optimiser blocks): **χ² = 439.0, p = 1.3 × 10⁻⁶⁸** — reproduces the
S-8 value exactly, confirming the real data is intact.

| Rank | Model | Family | Best opt | R² | RMSE Depth | RMSE Width | Elapsed [s] |
|----:|--------|--------|----------|----:|-----------:|-----------:|------------:|
| 1 | ngb | Boosting | nsga2 | 0.9364 | 0.0723 | 0.0477 | 624 |
| 2 | mdn | NN/DL | random | 0.9280 | 0.0719 | 0.0532 | 299 |
| 3 | vote | Ensemble | de | 0.9270 | 0.0726 | 0.0531 | 8.5 |
| 4 | rfr | Trees | hyperopt_tpe | 0.9251 | 0.0730 | 0.0526 | 38 |
| 5 | bag | Trees | random | 0.9248 | 0.0731 | 0.0524 | 58 |
| 6 | grnn | NN/DL | grid | 0.9241 | 0.0741 | 0.0521 | 0.11 |
| 7 | kridge | Kernel | ga_deap | 0.9238 | 0.0737 | 0.0547 | 0.25 |
| 8 | anfis | NN/DL | grid | 0.9236 | 0.0735 | 0.0522 | 204 |
| 9 | cat | Boosting | random | 0.9232 | 0.0738 | 0.0533 | 19 |
| 10 | tabnet | NN/DL | hyperband | 0.9228 | 0.0750 | 0.0508 | 535 |

The top 10 fall within 1.4 percentage points of R² (0.9228–0.9364). For
decision support the choice should be driven by interpretability, runtime and
uncertainty support rather than raw accuracy.

---

## 2. Pareto front on the real ablation (correction to S-8)

Minimising (RMSE_Depth, RMSE_Width) over the full real grid, the strictly
non-dominated set is **{ngb, mdn} only**:

| Model | Optimiser | RMSE Depth | RMSE Width |
|-------|-----------|-----------:|-----------:|
| mdn | random | 0.0719 | 0.0532 |
| ngb | nsga2 | 0.0723 | 0.0477 |
| ngb | de | 0.0763 | 0.0474 |

**This corrects the interim S-8 statement of three Pareto models (ngb, vote,
mdn).** `vote` is dominated by ngb on both objectives (ngb has lower Depth and
much lower Width RMSE). `vote` should still be carried into the inverse-design
candidate set as a fast practical baseline (R² = 0.927 in 8.5 s), but it is not
Pareto-optimal and the manuscript should not claim it is.

**Inverse-design candidate set for S-9 (NSGA-II): {ngb, mdn} (+ vote as a fast
reference).** All trained on real data — ready now, independent of the D′ issue.

---

## 3. Speed/quality trade-off (practical recommendation)

Among models at R² ≥ 0.923 on real:

| Model | R² | Elapsed [s] |
|-------|----:|-----------:|
| grnn | 0.9241 | 0.11 |
| kridge | 0.9238 | 0.25 |
| vote | 0.9270 | 8.5 |
| cat | 0.9232 | 19 |
| rfr | 0.9251 | 38 |
| bag | 0.9248 | 58 |
| anfis | 0.9236 | 204 |
| mdn | 0.9280 | 299 |
| ngb | 0.9364 | 624 |

`vote` reaches 0.927 in 8.5 s; `ngb` needs 624 s for the extra 0.009 R². `grnn`
(closed-form, Specht 1991) reaches 0.924 in 0.11 s. Where uncertainty
quantification is not required, the voting ensemble dominates on speed/quality.

---

## 4. Optimiser ranking on real (clean)

Median macro R² per optimiser; mean shown for reference (means are contaminated
by the known MLP convergence failure, R² ≈ −52):

| Optimiser | Median R² | Mean R² |
|-----------|----------:|--------:|
| bo_gp | 0.9141 | −0.413* |
| cmaes | 0.9136 | 0.861 |
| tpe | 0.9136 | 0.872 |
| ga_deap | 0.9132 | 0.845 |
| hyperopt_tpe | 0.9130 | 0.872 |
| de | 0.9126 | 0.852 |
| nsga2 | 0.9126 | 0.852 |
| random | 0.9113 | 0.869 |
| ga_sk | 0.9102 | 0.851 |
| hyperband | 0.9094 | 0.877 |
| pso | 0.9086 | 0.864 |
| grid | 0.8947 | 0.836 |

Best–worst median spread = 0.019 (~2 pp). Consistent with Bergstra & Bengio
(2012): on small models with few hyperparameters, sophisticated optimisers are
only marginally better than random search; grid is worst. Practitioner
recommendation: TPE or random by default; avoid grid.

---

## 5. Synthetic generators (clean 3-fold, fast-model subset)

### 5.1 Generator ranking on `synth`

| Generator | n | Mean R² | Median R² | Min | Max | Usable |
|-----------|--:|--------:|----------:|----:|----:|--------|
| TVAE | 372 | 0.805 | 0.850 | −0.558 | 0.883 | Yes |
| Physics | 348 | 0.798 | 0.805 | 0.526 | 0.861 | Yes |
| Copula | 348 | 0.758 | 0.736 | 0.603 | 0.838 | Yes |
| CTGAN | 479 | −0.147 | −0.119 | −1.218 | −0.007 | **No** |

**CTGAN central finding holds and strengthens: 97.4 % of CTGAN synth configs
have R² < 0** (copula/physics 0 %, TVAE 1.3 %). At N = 72 real points and
default SDV hyperparameters, the conditional GAN does not preserve the X→Y
dependency of the EBW process. TVAE / physics / copula are statistically close
(medians 0.74–0.85); TVAE marginally ahead in median, physics ahead in
worst-case (min 0.53 vs TVAE −0.56).

### 5.2 Best model per (ablation, generator) — clean 3-fold

| Ablation | Generator | Best model / opt | R² |
|----------|-----------|------------------|----:|
| synth | tvae | nusvr / ga_deap | 0.883 |
| synth | physics | elm / grid | 0.861 |
| synth | copula | theilsen / grid | 0.838 |
| real+synth | tvae | knn / ga_deap | 0.878 |
| real+synth | physics | elm / nsga2 | 0.862 |
| real+synth | copula | theilsen / grid | 0.843 |
| tstr | tvae | lgbm / cmaes | 0.867 |
| tstr | physics | dtr / pso | 0.892 |
| tstr | copula | quantile / random | 0.838 |

Best model is generator-dependent (no single winner across generators).

### 5.3 Does mixing real + synth beat synth-only? No.

| Generator | best synth | best real+synth | Δ |
|-----------|-----------:|----------------:|---:|
| tvae | 0.883 | 0.878 | −0.005 |
| physics | 0.861 | 0.862 | +0.002 |
| copula | 0.838 | 0.843 | +0.005 |

Adding the 72 real points to ~10 000 synthetic points does not systematically
improve accuracy — within noise either way. Contradicts the common assumption
that mixing must help.

---

## 6. What still needs doing / Limitations

* **D′ re-run at 3-fold (recommended, optional).** Re-run the 5 slow models'
  synth/r+s/tstr cells with `--n-folds 3` (fixed launcher provided), discard
  the 5-fold rows, re-merge. At 3-fold the light models (ngb, mdn, vote) finish
  cleanly and fast; the deep NN (ftt, tabnet) remain heavy on CPU and need
  intra-op threads or a higher cap. **No headline depends on this** — it only
  completes the generator×model grid and enables a defensible synth CD diagram.
* **7 slow models uncovered on good generators** (gpr, anfis, node, saint,
  cnn1d, bnn, stack): out of D′ scope. Their *real* ablation is in the ranking;
  their synth cells are absent. Declare in Limitations, as S-8 already noted.
* **MLP failure** (R² ≈ −52 on real) persists; reported as a sklearn LBFGS
  small-data artefact.
* **Nemenyi CD** still under-powered for 41 models × 12 raters; use pairwise
  Wilcoxon + Holm on the top-10 (Benavoli 2017).

---

## 7. Ready for S-9 inverse design

The inverse-design step (NSGA-II to find process parameters hitting target
Depth/Width) uses models trained on the **real** ablation, which is clean.
Candidate set: **ngb** (best accuracy, native uncertainty), **mdn** (Pareto,
predictive distribution), with **vote** as a fast reference. This can proceed
now regardless of the D′ re-run decision.
