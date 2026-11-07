# clusterBMA port: Step 0 notes

**Source read:** `of2/clusterBMA`, commit `4fba1361cb362a9843e192eaf3a5fbc7ef341ce1` (2024-02-05), package version 0.2.0, cloned to `reference/clusterBMA`. File references below are relative to that folder.

**Python/TensorFlow:** the repo has no Python files. All TensorFlow code is R calling `tf$...` through reticulate, in `R/SimplexClust.R`.

**How each claim was checked**

- The clone was only read. None of its code was run, so everything marked "inferred" is from reading the code and has not been tested.
- `clusterCrit` 1.3.0 and `max.col` were run directly, on a few lines of random numbers and a 3-row matrix.
- `Hmisc` is not installed here, so `wtd.mean` was not read.
- The paper's claims were first read through a summarising web fetch, then checked against the PDF (PLOS ONE, 21 Aug 2023) on 2026-11-03: pp. 4-5 and 8-9 for the equations and SSMF section, pp. 14 and 17-18 for the case study, discussion and computing passages. Page references below are to that PDF.

## a. Weights

**Default index:** Calinski-Harabasz, direction "max" (`R/clusterBMA.R:55`). The direction is a user argument. It is not looked up from the index, so `wt_crit_name = "S_Dbw"` with the default direction is silently weighted the wrong way.

**Hard projection:** `apply(soft_alloc_matrix, 1, which.max)`, so the first column wins a tie (`R/prob_to_hard_fn.R:9`, called from `R/clusterBMA.R:97`). Every model goes through this, hard or soft. `max.col(A, ties.method = "first")` gave the same result on a 3-row check that included an exact tie and a near tie.

**Index call:** `clusterCrit::intCriteria(traj = as.matrix(input_data), part = as.integer(labels), crit = name)`, taking the first list element (`R/new_weight_fn.R:27,30,32`). The data are used as supplied, with no scaling.

**Normalisation** (`R/new_weight_fn.R:39-48`):

- max: `W_m = crit_m / sum(crit)`
- min: `W_m = (1 / crit_m) / sum(1 / crit)`

**Priors** (`R/consensus_matrix_fn.R:25-27`): `combined = W_m * prior_m`, then `combined / sum(combined)`. With equal priors, `combined` equals `W_m`. The consensus uses `combined` (`:29`).

**Consensus:** the cell-wise `Hmisc::wtd.mean` of the similarity matrices (`R/consensus_matrix_fn.R:29`). The commented-out older version is the plain weighted sum (`:17`), and the weights already sum to 1, so I expect the two to be equal (`wtd.mean` itself not read). Similarity is `A %*% t(A)` with the diagonal set to 1 (`R/sim_mat_fn.R:12,14`).

**Not used:** `R/ch_xb_weight_fn.R` is dead code; its call is commented out at `R/clusterBMA.R:109`.

**Edge cases the source does not guard**

- There is no check for zero, negative or non-finite index values.
- A model with a single non-empty class gives `NaN` for CH (run on `clusterCrit`), which makes every weight `NaN`.
- Non-contiguous labels are fine: `intCriteria` canonifies them (run: labels {1,3} gave the same CH as {1,2}).
- An invalid direction only prints a message and leaves the weights `NA` (`R/new_weight_fn.R:49`).

## b. Factorisation

All from `R/SimplexClust.R`. Signature (`:25`): `SimplexClust(S, G, lambda = 1, eps = 1E-2, steps = 5000, choice_loss = "KL")`. `clusterBMA()` passes only `S` and `G` (`R/consensus_matrix_fn.R:36`), so users cannot change the other settings through it.

**Objective.** With `A = P %*% t(P)` (`:47`) and the default `choice_loss = "KL"` (`:52-57,64,72`):

```
total_loss = sum over i != j of [ -S_ij log(A_ij) - (1 - S_ij) log(1 - A_ij) ]
             + lambda * sum over k of sqrt( sum over i of P_ik^2 )
```

- The data term is binary cross-entropy. The code calls it KL; the two differ only by a constant in `S`, so the gradients match.
- It sums over ordered pairs, so each pair counts twice, with the diagonal masked out (`:43-45,64`).
- The alternative `choice_loss = "L2"` is `(S - A)^2` (`:59-62`). `clusterBMA()` never selects it.

**Constraints.** There is no constraint or projection. `P` comes from an unconstrained `N x G` variable `V`: `P0 = softplus(V)`, `P = P0 / rowSums(P0)` (`:39,40,46`). Rows therefore sum to 1 and entries are strictly positive by construction.

**Regularisation.** A group norm: the sum of column L2 norms, known as L2,1 (`:66,68`). Default strength `lambda = 1` (`:25`), not scaled by N. A commented-out relaxation term is inactive (`:70`).

**Optimiser.** `tf$compat$v1$train$AdamOptimizer(eps)` (`:74`). `eps` is the learning rate, 0.01, not Adam's epsilon. The other Adam settings are not set in the code, so TF defaults apply (beta1 .9, beta2 .999, epsilon 1e-8, from TF documentation; confirm in Step 1). Gradients are full batch.

**Initialisation.** `V ~ Uniform(0, 1)` in float32 (`:39`), so `softplus(V)` lies in about [0.69, 1.31] and rows start close to uniform. No seed is set or accepted anywhere, so `set.seed()` in R does not control it (inferred).

**Iterations.** A fixed 5000 steps (`:25,82`). The comment says it mostly converges after about 2000, the "original default" (`:5,21`).

**Convergence rule.** None: no tolerance and no early stop (`:82-86`). The data loss, without the penalty, is printed every 20 steps and not returned (`:84-85`).

**Empty clusters.** Not detected. `P` keeps all `G` columns and its entries can never reach exactly 0. The only trace is indirect: `table(alloc_vector)` counts only columns that are some point's argmax (`R/prob_to_hard_fn.R:17`), and `alloc_ordered` renumbers those by size (`:21-25`). So "empty" in practice means "never the argmax".

**Numerics.** Everything runs in float32 (`:39,45,49`). `log(A)` and `log(1 - A)` have no clipping (`:54-55`).

## c. Outputs

**Allocation uncertainty:** `1 - max(P[i, ])` for each person (`R/prob_to_hard_fn.R:15`). `alloc_prob` is `max(P[i, ])` (`:13`). It is computed from the final `P` only, with no separate within-model or between-model calculation. The paper defines uncertainty only in words, as `p(g_i != ghat_i)`, with no numbered equation or formula (p. 9), so `1 - max` is the code's implementation of that definition.

**Per-cluster model contribution:** none in the code. Weights exist only per model, across all clusters. The paper mentions assessing "the contribution from each algorithm" (p. 17), which in the outputs means the per-model weights.

**Return list** (`R/consensus_matrix_fn.R:49`, documented at `README.md:116-122`):

| Index | Content |
|---|---|
| 1 | consensus matrix |
| 2 | `P` (N x G, raw column order) |
| 3 | data frame: `alloc_vector`, `alloc_prob`, `alloc_uncertainty`, `alloc_ordered` |
| 4 | size-sorted table of hard allocations |
| 5 | `W_m` before priors |
| 6 | weights after priors |
| 7 | pheatmap object |

## d. Disagreements

### Code vs paper

1. **Algorithm.** The paper describes 5000 EM iterations, with gradient descents in each (p. 18). The code has no E or M step: it is one objective minimised by Adam. The count of 5000 matches.
2. **Penalty.** The paper (p. 9) and the docs (`R/clusterBMA.R:47`, `R/SimplexClust.R:4,20`) say L2. The code uses L2,1 on columns (`:66`).
3. **Empty clusters.** The paper says redundant clusters are emptied (p. 9). In the code, no entry reaches 0, so this only happens in the never-the-argmax sense.
4. **Adjustable iterations.** The paper says users can adjust the iteration count (p. 18). `clusterBMA()` has no argument for it, so this requires calling `SimplexClust()` directly (`R/consensus_matrix_fn.R:36`).
5. **Paper's own numbering.** Section 4.2 says the weight was calculated "using Eqs 11 and 12" (p. 14), but those are S_Dbw equations (p. 7). The weight equation is Eq 13, as the same page's results section says, so this looks like a numbering slip.

**Confirmed against the PDF, no disagreement:** Eq 13 for both max and min (p. 8) matches `R/new_weight_fn.R:39-48` and the spec. Eq 14 reduces to the normalised weight under equal priors. Eq 15 is the weighted average (p. 8). Eq 4 sets the diagonal to 1 (p. 5). Crisp projection of GMM results for the CIVI is stated for the case study (p. 14). The paper gives no factorisation objective, optimiser, learning rate or strength in text, as the spec says.

### Spec vs code

The spec says to follow the source where they disagree. Items marked **decision** are ones where that may not be what you want.

6. **Optimiser (Step 0 and `cbma_ssmf`).** The spec mentions EM iterations; there are none. The source is not projected gradient either, so the spec's condition for `cbma_project_simplex()` is not met and neither the helper nor its test is needed.
7. **`tol` and "iterations used".** The source has no convergence rule and always runs every step. **Decision:** to match it, `tol` would default to off.
8. **"Which clusters ended empty".** The source has no such output, so the port needs its own definition. Never-the-argmax is the one consistent with the source. The Step 3 test (K = 6, at most 3 non-empty) depends on this definition.
9. **`seed`.** The source has no seed. **This affects Step 1:** the fixtures' `seed` field cannot make the factorisation reproducible unless `make_reference.R` also sets a TF seed itself.
10. **Weight errors.** The spec stops on non-finite or non-positive values; the source does not check. **Decision:** I would keep the spec's stricter behaviour and record it as a deliberate difference, since the source's behaviour there gives `NaN` or weights outside [0, 1].
11. **Direction from `bestCriterion`.** The source takes direction from the user. `bestCriterion` also returns the position of the best value, not a direction. Seven indices (Ball_Hall, Ksq_DetW, Trace_W, Trace_WiB, Det_Ratio, Log_Det_Ratio, Log_SS_Ratio) use a second-difference elbow rule, so they have no max or min direction and do not fit Eq 13. **Decision:** the port needs its own lookup table that rejects those seven.
12. **Priors.** The spec has equal priors only. The source has a `prior_weights` argument, but it looks broken: `if (prior_weights == "equal")` (`R/clusterBMA.R:59`) gets a condition of length M from a numeric vector, which is an error in R 4.2 and later (inferred).
13. **`K_BMA` default.** The spec defaults to the maximum K across inputs; in the source `n_final_clust` is required.
14. **Consensus memory.** The source holds all M matrices in one array (`R/consensus_matrix_fn.R:11`); the spec accumulates in place. This should change memory use only, not results. The source's pheatmap call (`:33`) is left out, which also removes its cost at large N.
15. **Precision.** The source optimises in float32, while R uses doubles. Allocations may differ slightly for this reason alone, on top of the random initialisation.
16. **Versions.** `README.md:46` says Python 3.7.9 and TF 1.15.5, `README.md:10` says 3.10.9 and 2.11, and the installer pins only Python 3.10, with unpinned `tensorflow` (`R/clusterBMA_initial_python_setup.R:46-54,59`). So the fixture metadata must record what actually gets installed.
17. **No upstream tests.** `tests/testthat/test-clusterBMA.R:1-3` is a placeholder, so the fixtures will be the only reference.
18. **Commits.** The spec requires tests in the same commit as each function, but this project has no git repository (README).

### Risks to test (inferred, not run)

19. **Penalty strength at ABCD scale.** The data term sums about N^2 pairs, while the penalty is at most `G * sqrt(N)` with `lambda = 1`. The penalty therefore shrinks relative to the data term as N grows, so clusters that empty at N = 300 may not empty at N of 8,000-12,000. The Step 1 scenarios go up to N = 500 only, so they would not reveal this.
20. **`NaN` loss.** If a row of `P` gets close to one-hot, `A_ii` can round to 1 in float32. On the diagonal, `S_ii = 1` makes the term `0 * log(0)`, which is `NaN`, and the mask does not remove it because `NaN * 0` is `NaN` (`R/SimplexClust.R:55-56,64`). Off-diagonal pairs with `S_ij = 1` can do the same. Doubles push this threshold much further out, so the port and the reference could differ here.

## Decisions (2026-11-03)

Jamie accepted these after reading the items above. The spec now reflects them.

| # | Decision | Notes item |
|---|---|---|
| 1 | Port lives in its own folder and git repository, `clusterBMA_port` | 18 |
| 2 | `tol` off by default | 7 |
| 3 | `cbma_weights` stops on non-finite or non-positive values: deliberate difference | 10 |
| 4 | Direction from a port lookup table; seven elbow-rule indices rejected | 11 |
| 5 | Empty cluster = no observation's top class | 8 |
| 6 | Doubles, off-diagonal terms only, error if the objective is not finite: deliberate difference | 15, 20 |
| 7 | Equal priors only | 12 |
| 8 | Fixtures pinned to Python 3.10.9 and TF 2.11, TF seed set in `make_reference.R`, simulated data only | 9, 16 |
| 9 | `lambda = 1` stays; Step 6 also reports empty clusters at each N | 19 |
| 10 | Step 6 runs after the GMM optima experiment finishes | |

## Step 1 results (2026-11-03)

Fixtures are in `tests/testthat/fixtures/`, from `tools/make_reference.R`, simulated data only. Package commit `4fba136`, Python 3.10.9, TF 2.11.0 (`tensorflow-macos`, the arm64 build), numpy 1.23.5. The environment is in `.fixture_env/` (2.1 GB, gitignored); deleting that folder removes it.

| Scenario | N x dims | Inputs | K_BMA | Weights | Non-empty | ARI, truth | Other seed: ARI | Other seed: mean abs diff | Other seed: uncertainty r |
|---|---|---|---|---|---|---|---|---|---|
| A | 500 x 2 | kmeans 5, GMM 5 | 5 | .501, .499 | 5 | .985 | 1.000 | .0001 | 1.000 |
| B | 300 x 2 | kmeans 3, GMM 3, Ward 4 | 4 | .363, .363, .273 | 3 | 1.000 | 1.000 | .0000 | 1.000 |
| C | 500 x 20 | kmeans 3, 4, 5, GMM 4 | 5 | .292, .253, .227, .229 | 4 | .286 | 1.000 | .0003 | 1.000 |
| D | 300 x 2 | kmeans 2, GMM 3, Ward 6 | 6 | .081, .565, .354 | 3 | 1.000 | 1.000 | .0271 | .875 |

**Choices the spec left open:** `K_BMA` for B and C is the largest input K; D uses 3 true clusters and one each of k-means, GMM and Ward.

**Reproducibility:** with the TF seed set, the same seed gives an identical factorisation (max abs diff 0), and rerunning the whole script reproduced A-C exactly.

**Scenario D's tolerances (changed from the spec's defaults).** The other-seed columns above compare the reference with itself, on one seed pair. D missed two Step 4 tolerances that way, so the floor was measured against the D fixture across 20 seeds (`meta$seed_floor` in `D.rds`):

| Measure | Spec tolerance | Near solution (10 seeds) | Second solution (10 seeds) | D tolerance |
|---|---|---|---|---|
| Hard labels, ARI | >= .95 | 1.000 | 1.000 | >= .95 (unchanged) |
| Mean abs diff, non-empty columns | < .02 | .0000-.0013 | .0270-.0280 | < .035 |
| Uncertainty correlation | >= .95 | .9997-1.0000 | .8656-.8757 | >= .83 |

The D tolerances are the worst of the 20 seeds with a 25% margin (Jamie's decision to set them from the floor; the margin is my choice). They are looser, and they cannot tell the two solutions apart: a port on either one passes. A-C keep the spec's tolerances.

**Empty clusters confirm item 3:** in D, columns that are no observation's top class still reach probabilities of .26 and .22, so "empty" does not mean near zero. B and C also lose one column each, at N = 300 and 500.

**Precision:** reference rows sum to 1 within about 1.5e-7 (float32), so the 1e-8 row-sum test can apply to the port but not to the fixtures.

## Steps 2-3 (2026-11-03)

`R/cbma.R` and `tests/testthat/test-cbma.R`. Run: `Rscript --vanilla -e 'testthat::test_dir("tests/testthat")'`.

**How decision 6 was implemented.** A term with a zero coefficient is left out instead of being evaluated: `(1 - C_ij) log(1 - A_ij)` when `C_ij = 1`, and `C_ij log(A_ij)` when `C_ij = 0`. This is the cross-entropy limit, not clipping; without it, hard inputs would reach `0 * log(0)` and stop. Any other non-finite objective stops with an error.

**Additions beyond Step 2's list:** `cbma_ssmf` refuses a `C` with values outside [0, 1]; `cbma_check_inputs` carries out Step 5's two input checks, because `cbma_fit` needs them; `cbma_ssmf`'s `seed` defaults to 1 so no call depends on global RNG state. `cbma_project_simplex()` was not written (Step 0).

**Informal comparison with the fixtures (not the Step 4 tests), port seed 1, K_BMA as in the fixture:**

| Scenario | Max diff: CIVI, weights | Max diff: consensus | ARI | Mean abs diff | Uncertainty r | Empty: reference, port | Port time |
|---|---|---|---|---|---|---|---|
| A | 0, 0 | 0 | 1.000 | .0001 | 1.000 | 0, 0 | 82 s |
| B | 0, 0 | 3e-16 | 1.000 | .0001 | 1.000 | 1, 1 | 29 s |
| C | 0, 0 | 0 | 1.000 | .0002 | 1.000 | 1, 1 | 81 s |
| D | 0, 0 | 1e-16 | 1.000 | .0012 | 1.000 | 3, 3 | 29 s |

Times were measured with the GMM experiment using 8 of 10 cores, against 7-12 s for the reference. Step 6 will measure scale properly.

## Steps 4-5 (2026-11-03)

`tests/testthat/test-parity.R` and `test-determinism.R`. Full suite on the final code: 48 tests, 152 expectations, 0 failures, 0 errors, 0 skips, in about 5 minutes.

| Scenario, port seed | Max diff: CIVI, weights | Max diff: consensus | ARI | Mean abs diff | Uncertainty r | Empty: port, reference |
|---|---|---|---|---|---|---|
| A, 1 | 0, 0 | 0 | 1.0000 | .0001 | 1.0000 | 0, 0 |
| B, 1 | 0, 0 | 3e-16 | 1.0000 | .0001 | 1.0000 | 1, 1 |
| C, 1 | 0, 0 | 0 | 1.0000 | .0002 | 1.0000 | 1, 1 |
| D, 1 | 0, 0 | 1e-16 | 1.0000 | .0012 | .9998 | 3, 3 |
| D, 2 | | | 1.0000 | .0001 | 1.0000 | |
| D, 3 | | | 1.0000 | .0279 | .8664 | |
| D, 4 | | | 1.0000 | .0001 | 1.0000 | |
| D, 5 | | | 1.0000 | .0278 | .8672 | |

**Scenario D:** port seeds 3 and 5 land on the same second solution the reference reaches (reference floor: .0270-.0280, r .8656-.8757). The port reproduces both solutions, not just one. Five seeds are run because one could not exercise the loosened tolerance.

**What these tests do not cover:**
- Probabilities are compared on the reference's non-empty columns only, the same as the floor. Empty columns have no label overlap to align them by, so their leftover probability is not compared.
- Only the Calinski-Harabasz path is compared with the reference. `S_Dbw` and the "min" formula are unit-tested only, because no fixture uses them.
- Fixtures stop at N = 500. Nothing here tests behaviour at ABCD's size; that is Step 6.

**Bug the tests found:** the range check added to `cbma_ssmf` rejected `C` values above 1 + 1e-8, but rows accepted within 1e-6 of 1 can produce values just above 1, so valid input could stop. The check now allows 1 + 1e-5.

## Step 6 (2026-11-06)

`tools/benchmark.R`, one process per N under `/usr/bin/time -l`, with nothing else running. Raw rows are in `tools/benchmark_results.csv`. Simulated data: 4 unequal clusters in 4 dimensions, 6 inputs (k-means K = 2, 3, 4, 6; `Mclust` G = 3, 4), K_BMA = 6. R's default reference BLAS (`libRblas`), on the M5 Mac with 16 GB.

| N | Consensus time | Consensus peak | SSMF time, 5000 steps | SSMF peak (R heap) | Process peak footprint | Empty clusters | ARI, truth |
|---|---|---|---|---|---|---|---|
| 2,000 | 0.1 s | 186 MB | 8.7 min | 399 MB | 0.6 GB | 2 | .973 |
| 5,000 | 0.3 s | 990 MB | 51.4 min | 2.3 GB | 3.3 GB | 2 | .960 |
| 12,000 | 2.2 s | 5.5 GB | not run; 12.9 h projected | 10.5 GB in 20 steps | 15.0 GB | not measured | not measured |

**N = 12,000 is over both Step 6 limits (30 min, 8 GB), so it was stopped after a 20-step pilot.** Nothing was changed in response; the options are for Jamie to choose.

**The 12.9 h projection is probably pessimistic.** The pilot projection was accurate at 2,000 and 5,000 (9.4 vs 8.7 min; 51.8 vs 51.4 min), but at 12,000 the process hit memory pressure on a 16 GB machine: 65 s of system time in 199 s real, against 2-6 s of system time in the two smaller full runs. Scaling the N = 5,000 time by N squared gives about 5 h without that pressure. That is an extrapolation, not a measurement.

**Consensus and index times are not the problem.** Time and memory are in the factorisation.

**Sizes that matter for ABCD**, extrapolated by N squared from N = 5,000, without memory pressure: the sample is 10,088 (8,331 with one child per family), not 12,000. About 3.5 h and 9.5 GB R heap at 10,088; about 2.4 h and 6.5 GB at 8,331.

**Empty clusters (decision 9):** 2 of 6 at N = 2,000 and N = 5,000, the same as at N = 300 in the smoke test, with 4 true clusters. So `lambda = 1` does not weaken across this range on this data. Untested at 12,000, and these clusters are cleaner than CBCL scores.

**Options, none applied:**
- Plain R: reuse the masks and index vectors in `cbma_ssmf_step` rather than rebuilding them each step. No new dependency. Lowers memory; the time gain is unknown.
- Accelerated BLAS (Apple vecLib, shipped with CRAN R). No code change, but it replaces a library in the R installation. Could speed the matrix products; untested, and results may differ slightly in the last digits.
- RcppArmadillo (spec's option). Most control over allocation and speed; adds a compile step.
- Float storage (spec's option). Halves matrix memory; the source's float32 shows rows summing to 1 only within about 1.5e-7. Little speed gain expected.
- R `torch` (spec's option). Could use the GPU, but brings back a large dependency, and the port exists to avoid that kind of stack.

## Step 6, after tightening (2026-11-06)

Jamie chose to tighten the R code and test vecLib. `cbma_ssmf_step` now uses whole-matrix arithmetic and zeroes the left-out terms afterwards, with what never changes held in `cbma_ssmf_parts`. Rows are in `tools/benchmark_results_tightened.csv`.

| N | BLAS | SSMF time, 5000 steps | R heap peak | Process peak | Empty clusters | ARI, truth |
|---|---|---|---|---|---|---|
| 2,000 | vecLib | 4.1 min | 264 MB | 0.4 GB | 2 | .973 |
| 5,000 | vecLib | 23.2 min | 1.4 GB | 1.9 GB | 2 | .960 |
| 12,000 | reference | not run; 2.8 h projected | 7.8 GB | 8.3 GB | not measured | not measured |
| 12,000 | vecLib | not run; 2.3 h projected | 7.8 GB | 8.3 GB | not measured | not measured |
| 12,000 | vecLib, full run (2026-11-07) | 2.4 h (145.5 min) | 7.7 GB | 8.3 GB | 2 | .960 |

**N = 12,000 is still over the 30-minute limit, and at the 8 GB limit (under by R heap, just over by process footprint).** It stopped after the 20-step pilot. The pilot projections were again accurate at the sizes run in full (4.4 vs 4.1 min; 23.1 vs 23.2 min), and there is no sign of memory pressure now (under 2 s of system time).

**Most of the gain comes from tightening.** At N = 12,000: 12.9 h to 2.8 h from tightening, part of which is the end of memory pressure, then 2.8 h to 2.3 h from vecLib (about 18%). vecLib makes one 3000 x 3000 `tcrossprod` about 57 times faster, so that small overall gain suggests element-by-element arithmetic now takes most of each step. That part is an inference; no profile was run.

**Same results.**
- Old and tightened code: `identical()` allocations and objective traces on fixtures A-D and on C with exact 0s and 1s, with values just above 1, and from two hard solutions (500 steps each, reference BLAS).
- The full suite passes on both libraries: 48 tests, 152 expectations, 0 failures.
- Reference vs vecLib: at N = 2,000, K = 6, 300 steps, max allocation difference 9e-16, same labels. On fixture C, 5000 steps, the difference was exactly 0; `tcrossprod` with few columns was identical in both libraries at N = 2,000, while `C %*% P` differed by 2e-13.

**Extrapolated by N squared from the N = 12,000 vecLib pilot:** about 95 min at N = 10,088 and 65 min at N = 8,331. R heap about 5.5 GB and 3.7 GB.

**How vecLib was loaded:** one process at a time, with a symlink to R's `libRblas.vecLib.dylib` in a scratch folder and `DYLD_LIBRARY_PATH` set when calling `bin/exec/R` directly. The R installation was not changed and still uses the reference BLAS by default.

**Full run at N = 12,000 (2026-11-07, Jamie's go-ahead to exceed the time limit once):** 145.5 min against 140.1 projected, peak memory as in the pilot (8.3 GB process), no memory pressure (19 s of system time in 2.4 h). 2 of 6 clusters empty, ARI with truth .960, smallest class 8.9%: the same as at N = 2,000 and 5,000, so on this data `lambda = 1` does not weaken up to N = 12,000. These simulated clusters are cleaner than CBCL scores. The run used `tools/benchmark.R 12000 <file> force`.
