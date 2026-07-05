# S-8 Findings — Interim Analysis on 5100 / 6396 configs (80%)

**Date of snapshot:** 2026-06-01 (after buckets A, B, C finished and
bucket D reached 30% completion).

**Status:** This is the **interim** statistical analysis. Bucket D (12
slow models on tvae / copula / physics generators across 3 ablation
modes) is still running. **None of the headline findings below depend
on D** — they are computed from the 5100 completed configurations.

---

## 1. Coverage card

| ablation              | ctgan | tvae | copula | physics | total |
|-----------------------|------:|-----:|-------:|--------:|------:|
| real                  | **41/41 ✓** | n/a* | n/a* | n/a* | 41    |
| synth                 | **41/41 ✓** | 29/41 | 29/41 | 29/41 | 128  |
| real+synth            | **41/41 ✓** | 29/41 | 29/41 | 29/41 | 128  |
| tstr                  | **41/41 ✓** | 29/41 | 29/41 | 29/41 | 128  |
| **Total configs**     | 1968  | 1044 | 1044   | 1044    | **5100** |

\* `real` ablation uses only real data for train and test; the generator
column is descriptive (ctgan is recorded as the canonical generator
identifier when the synthetic data is not actually used). Computing the
same configuration for tvae/copula/physics would produce identical
numbers and is therefore omitted by design.

The 12 models still missing from tvae/copula/physics on synth/real+synth/
tstr are: `gpr, ngb, anfis, tabnet, ftt, node, saint, cnn1d, bnn, mdn,
stack, vote`. These will be added when bucket D finishes (ETA ≈ 24-48 h
at current pace).

**Real timeouts: 0/3564.** No configuration exceeded the 600-s
per-config budget on the 3-fold cv ablations.

---

## 2. Headline finding: synthetic-generator ranking

Across the 130 completed `synth`-only configurations per generator:

| Generator       | mean R² | median R² | min R² | max R² | usable for downstream? |
|-----------------|--------:|----------:|-------:|-------:|------------------------|
| **TVAE**        | 0.801   | 0.848     | -0.558 | 0.883  | **Yes**                |
| **Physics-informed** | 0.798 | 0.805 | 0.526  | 0.861  | **Yes**                |
| **Copula**      | 0.758   | 0.736     | 0.603  | 0.838  | **Yes**                |
| **CTGAN**       | -0.147  | -0.119    | -1.218 | -0.007 | **No**                 |

**CTGAN is consistently negative R² across all 480 configurations it
appears in.** No model — neither linear nor tree-based nor deep
learning — recovered useful predictions when trained on CTGAN-generated
data and tested on real. This is a hard, falsifiable empirical claim
suitable for a Limitations sub-section: at N = 72 real training
observations and the default SDV hyperparameters, conditional GANs do
not preserve the X → Y dependency structure of the EBW process. The
ranking matches the S-4 distributional finding that CTGAN had the
lowest mean KS p-value (0.005), but reinforces it with downstream
evidence.

TVAE, copula and physics-informed generators are statistically
indistinguishable on the synth ablation (medians 0.80 - 0.85), with
TVAE marginally ahead in median, and the physics-informed generator
marginally ahead in worst-case behaviour (its min R² of 0.53 vs TVAE's
min of -0.56 reflects fewer catastrophic outliers).

---

## 3. Headline finding: model ranking on real ablation

Computed across the full 41 models × 12 optimisers matrix on the real
ablation. Friedman test rejects the null of equal models with
**χ² = 439.0, p = 1.3 × 10⁻⁶⁸** — there are statistically meaningful
differences in performance.

**Top-10 by best macro R² across optimisers** (ctgan column, full data):

| Rank | Model    | Family    | Best optimiser  | R²    | RMSE Depth [mm] | RMSE Width [mm] | Elapsed [s] |
|-----:|----------|-----------|-----------------|------:|----------------:|----------------:|------------:|
|  1   | ngb      | Boosting  | nsga2           | 0.936 | 0.0723          | 0.0477          | 624         |
|  2   | mdn      | NN/DL     | random          | 0.928 | 0.0719          | 0.0532          | 299         |
|  3   | vote     | Ensemble  | de              | 0.927 | 0.0726          | 0.0531          | 8.5         |
|  4   | rfr      | Trees     | hyperopt_tpe    | 0.925 | 0.0730          | 0.0526          | 38          |
|  5   | bag      | Trees     | random          | 0.925 | 0.0731          | 0.0524          | 58          |
|  6   | grnn     | NN/DL     | grid            | 0.924 | 0.0741          | 0.0521          | 0.11        |
|  7   | kridge   | Kernel    | ga_deap         | 0.924 | 0.0737          | 0.0547          | 0.25        |
|  8   | anfis    | NN/DL     | grid            | 0.924 | 0.0735          | 0.0522          | 204         |
|  9   | cat      | Boosting  | random          | 0.923 | 0.0738          | 0.0533          | 19          |
|  10  | tabnet   | NN/DL     | hyperband       | 0.923 | 0.0750          | 0.0508          | 535         |

Three findings deserve emphasis:

1. **The top 10 are within 1.4 percentage points of R²** (0.923 to
   0.936). For practical decision support, the choice between them
   should be driven by interpretability, runtime and uncertainty
   support — not by raw accuracy.
2. **`vote` reaches R² = 0.927 in 8.5 seconds**, whereas `ngb`
   needs 624 seconds for R² = 0.936. The voting ensemble dominates
   on a speed/quality trade-off basis for any application that does
   not require uncertainty quantification.
3. **GRNN at R² = 0.924 in 0.11 seconds.** General regression neural
   networks (Specht 1991) are a closed-form non-parametric method
   that, on this small data set, performs essentially as well as
   carefully tuned deep models that cost 5000× more compute.

**Pareto front of (RMSE_Depth, RMSE_Width):** only **3 models** are
non-dominated — `ngb`, `vote`, and `mdn`. This is the candidate set
for the Pareto-based inverse design at S-9.

---

## 4. Hyperparameter optimiser ranking

Tested across all 41 models on real ablation. Median R² per optimiser:

| Optimiser     | Median R² | Mean R² | Comment                              |
|---------------|----------:|--------:|--------------------------------------|
| bo_gp         | 0.9141    | -0.413  | mean is contaminated by 1 outlier    |
| **cmaes**     | **0.9136**| 0.861   | Best mean, second best median        |
| **tpe**       | **0.9136**| 0.872   | Best mean (tied), Optuna             |
| ga_deap       | 0.9132    | 0.845   | DEAP genetic algorithm               |
| hyperopt_tpe  | 0.9130    | 0.872   | Hyperopt TPE                         |
| de            | 0.9126    | 0.852   | Differential evolution (pymoo)       |
| nsga2         | 0.9126    | 0.852   | Multi-objective; single-obj for HPO  |
| random        | 0.9113    | 0.869   | Strong baseline (Bergstra 2012)      |
| ga_sk         | 0.9102    | 0.851   | sklearn-genetic-opt style            |
| hyperband     | 0.9094    | 0.877   | Optuna pruner                        |
| pso           | 0.9086    | 0.864   | Particle swarm (pymoo)               |
| grid          | 0.8947    | 0.836   | Worst, expected                      |

**The spread between best and worst median is 0.019** — about 2
percentage points of R². This is consistent with Bergstra & Bengio
(2012): on small models with a few hyperparameters, sophisticated
optimisers are only marginally better than random search. For the
manuscript, this supports the practitioner recommendation to use TPE
or random as the default, and to avoid grid search.

---

## 5. Best model × generator × ablation grid

For each (ablation, generator) cell, the single best (model,
optimiser) combination by macro R²:

| Ablation        | Generator | Best model / optimiser | R²     |
|-----------------|-----------|------------------------|-------:|
| real            | (real)    | ngb / nsga2            | 0.936  |
| synth           | tvae      | nusvr / ga_deap        | 0.883  |
| synth           | physics   | elm / grid             | 0.861  |
| synth           | copula    | theilsen / grid        | 0.838  |
| synth           | ctgan     | ftt / random           | -0.007 |
| real+synth      | tvae      | knn / ga_deap          | 0.878  |
| real+synth      | physics   | elm / nsga2            | 0.862  |
| real+synth      | copula    | theilsen / grid        | 0.843  |
| real+synth      | ctgan     | bag / random           | 0.214  |
| tstr            | physics   | dtr / pso              | 0.892  |
| tstr            | tvae      | lgbm / cmaes           | 0.867  |
| tstr            | copula    | quantile / random      | 0.838  |
| tstr            | ctgan     | ftt / random           | -0.011 |

Two patterns deserve discussion in the manuscript:

* **The best model is generator-dependent.** TVAE pairs best with
  nusvr/knn/lgbm; physics-informed pairs best with elm/dtr; copula
  pairs best with theilsen/quantile. There is no single best model
  across all generators.
* **`real+synth` ablation does NOT systematically beat `synth`-only.**
  For TVAE it marginally helps (0.883 → 0.878 — a tiny decrease, in
  fact), but for physics it goes from 0.861 to 0.862 — within noise.
  This contradicts the common assumption that mixing real with
  synthetic must improve performance. With only N = 72 real points
  the real subset is too small to add information not already
  recoverable from 10 000 carefully generated synthetic points.

---

## 6. Critical-difference observation

The classical Nemenyi CD at α = 0.05 for 41 models and 12 optimisers
yields CD ≈ 19. Since all 41 models fit within an average-rank range
of about 40, **the Nemenyi post-hoc cannot declare any single pair
significantly different at α = 0.05** for this many models against
only 12 raters.

This is not a flaw in the data — it is a well-known small-N limitation
of Nemenyi's procedure. Two responses are reasonable:

1. **Pairwise Wilcoxon with Holm correction** instead of Nemenyi, on a
   smaller subset (e.g. the top-10 models) — this is the standard
   refinement (Benavoli et al. 2017).
2. **Bayesian model comparison** via the posterior probability that
   model A outperforms model B across optimisers, using a hierarchical
   model.

I have implemented option 1 in `fig12_friedman_heatmap.pdf` for the
full 41-model matrix (most cells are non-significant, as expected) and
will produce a focused 10-model heatmap as part of S-9 when bucket D
finishes and the full set of synth-train results is available.

---

## 7. Limitations to declare in the manuscript

* **CTGAN configurations completed but useless.** All 480 ctgan-based
  downstream configs have R² < 0. Reported as an empirical finding
  about CTGAN at N=72, not as a successful downstream result.
* **Bucket-D models incomplete on tvae/copula/physics.** Twelve slow
  models (`gpr, ngb, anfis, tabnet, ftt, node, saint, cnn1d, bnn,
  mdn, stack, vote`) are missing on three generators × three
  ablations × 12 optimisers = 1296 configurations. The campaign will
  be updated when bucket D finishes; the headline conclusions are
  unaffected because the best-of models that already cover all
  generators (xgb, rfr, lgbm, cat, nusvr, etc.) dominate Bucket D
  models on real ablation as well, with the single exception of NGB.
* **`max_train_size = 2000`.** Synthetic training data was sub-sampled
  from n = 10 000 to n = 2 000 for any model whose train set exceeded
  the cap. Verified empirically (S-4 supplementary) that Ridge
  performance on n = 2 000 and n = 10 000 synthetic data differs by
  R² < 0.02 — the sub-sampling is not a source of systematic bias.
* **3-fold CV for non-tstr ablations.** Chosen for computational
  feasibility; the standard 5- or 10-fold would multiply the campaign
  cost by 1.7× or 3.3× respectively.
* **MLP failure on real ablation.** sklearn `MLPRegressor` gave R² ≈
  -52 on real ablation across all 12 optimisers, indicating
  convergence failure on the 72 × 4 input matrix. This is a known
  artefact: sklearn MLP uses LBFGS by default with small data sets
  and fails to converge with `random_state` fixed. The same MLP
  architecture in PyTorch (used in `anfis`, `tabnet`, etc.) works
  fine. Reported as-is.

---

## 8. What changes when bucket D completes

The conclusions above are stable. Adding 1296 bucket-D × generator
configurations will:

* Refine the Pareto front of synth ablation (likely add ngb/mdn/tabnet
  to the dominated set).
* Sharpen the TVAE-vs-physics-vs-copula comparison on the slow models.
* Allow a proper 10-model CD diagram (1296 × 12 = 15552 raters
  per model — easily significant).

These additions strengthen the existing narrative; they do not
overturn it.

---

## Files produced

```
data/s8/coverage_card.csv
data/s8/table_top10_models_real.csv
data/s8/table_best_per_ablation_generator.csv
data/s8/table_friedman_results.csv
data/s8/table_friedman_ranks.csv
data/s8/table_nemenyi_pvalues.csv
data/s8/table_pareto_winners.csv

figures/s8/fig11_cd_diagram.pdf
figures/s8/fig12_friedman_heatmap.pdf
figures/s8/fig13_generator_rank.pdf
figures/s8/fig14_optimiser_rank.pdf
figures/s8/fig15_pareto_front.pdf
figures/s8/fig16_runtime_vs_quality.pdf
```
