# SPEC: Native R port of clusterBMA (Bayesian model averaging for clustering)

## Purpose

Port the clusterBMA method (Forbes et al., 2023, PLOS ONE, doi:10.1371/journal.pone.0288000) to plain R with no Python or TensorFlow dependency. It will combine clustering solutions from the ABCD clustering multiverse (k-means, k-medoids, GMM, trajectory models) into one probabilistic allocation with model weights and per-person allocation uncertainty.

Reference implementation: https://github.com/of2/clusterBMA (R wrapper, factorization step in TensorFlow via reticulate).

Scope: the method only. No multiverse grid, no ABCD data loading. Those come later and call this code.

## Decisions recorded 2026-11-03

Accepted by Jamie after Step 0; reasons are in `NOTES.md`. Where these differ from the source, the difference is deliberate.

1. The port lives in its own folder and git repository (`clusterBMA_port`), separate from the ABCD project.
2. `tol` is off by default: all steps run, as in the source.
3. `cbma_weights` stops on non-finite or non-positive values. The source does not check.
4. Direction comes from a lookup table in the port, not `bestCriterion`. The seven elbow-rule indices are rejected.
5. An empty cluster is one that is no observation's top class.
6. `cbma_ssmf` uses doubles and off-diagonal terms only, and stops with an error if the objective is not finite. No silent clipping.
7. Equal priors only.
8. Fixtures: Python 3.10.9 and TF 2.11, TF seed set in `make_reference.R`, simulated data only.
9. `lambda = 1` stays. The Step 6 benchmark also reports empty clusters at each N; if clusters do not empty at large N, report before changing anything.
10. Step 6 runs only after the GMM optima experiment in the ABCD project has finished.

## Ground rules

- Base R plus `clusterCrit`, `clue` (label alignment), `mclust` (ARI), and `testthat`. Add RcppArmadillo only if the benchmark in Step 6 shows it is needed.
- Every function ships with its tests in the same commit.
- The original package is used for one thing: generating reference fixtures (Step 1). It is never a runtime dependency.
- Where this spec and the original source disagree, follow the source and record the difference in `NOTES.md`.

## Step 0. Read the source before writing code

Clone `of2/clusterBMA`. Read and summarize in `NOTES.md`:

1. `clusterBMA()`: how weights are computed, which CIVI is the default, how hard projections are made for CIVI computation.
2. `SimplexClust()` (and any Python/TF code it calls): the exact objective, constraints, regularization term and its default strength, optimizer, learning rate, initialization, number of EM iterations, convergence rule.
3. How allocation uncertainty and any per-cluster model contribution outputs are computed.

Do not start Step 2 until these are written down. The paper does not give the factorization objective in text, so the source is the only authority for it.

## Step 1. Reference fixtures (run once, in a throwaway conda env)

Script: `tools/make_reference.R`. Not part of the test suite and not run in CI. Environment: `tools/setup_reference_env.sh`, pinned to Python 3.10.9 and TF 2.11. The source sets no seed, so the script sets the TF seed itself.

Scenarios:

- A. The package vignette example: simulated 5 clusters, k-means + GMM, K = 5.
- B. Well-separated, 3 clusters, 2 dimensions, N = 300; inputs: k-means K=3, GMM K=3, hierarchical (Ward) K=4.
- C. Poorly separated, 4 clusters, 20 dimensions, N = 500; inputs: k-means K=3,4,5, GMM K=4.
- D. Inputs with different K (2, 3, 6) on the same data, K_BMA = 6, to exercise empty-cluster behavior.

For each scenario save to `tests/testthat/fixtures/<scenario>.rds`:
input data, list of input allocation matrices, seed, CIVI values, normalized weights, consensus matrix, final allocation matrix, final hard labels, allocation uncertainty.

Record the package commit hash, Python and TensorFlow versions in the fixture metadata.

## Step 2. Functions

All in `R/cbma.R`.

### `cbma_hard_to_prob(labels, K = NULL)`
Integer label vector to N x K one-hot matrix. Labels need not be contiguous; map to 1..K in sorted order.

### `cbma_civi(X, A, index = c("Calinski_Harabasz", "S_Dbw", ...))`
- Hard-project A to labels with `max.col(A, ties.method = "first")`.
- Compute the index with `clusterCrit::intCriteria`.
- Return the value and its direction (maximize or minimize) from a lookup table in the port. Stop with a clear error for the seven elbow-rule indices (Ball_Hall, Ksq_DetW, Trace_W, Trace_WiB, Det_Ratio, Log_Det_Ratio, Log_SS_Ratio).

### `cbma_weights(civi_values, direction)`
Paper Eq. 13, equal priors (Eq. 14):

- maximize: W_m = CIVI_m / sum(CIVI)
- minimize: W_m = (1 / CIVI_m) / sum(1 / CIVI)

Error on non-finite or non-positive CIVI values (deliberate difference: the source does not check).

### `cbma_similarity(A)`
Paper Eq. 4: S = A %*% t(A), then set diag(S) = 1. Use `tcrossprod(A)`.

### `cbma_consensus(A_list, weights)`
C = sum_m W_m * S_m. Accumulate in place; never hold all M similarity matrices at once.

### `cbma_ssmf(C, K, ...)`
Symmetric simplex matrix factorization: C (N x N) to A_BMA (N x K), rows non-negative and summing to 1. Implement exactly what Step 0 found in `SimplexClust()`. Expose: `K`, regularization strength, `n_iter` (default as in source, paper states 5000), learning rate, `tol`, `seed`. `tol` is off by default. Return A_BMA, objective trace, iterations used, and which clusters ended empty, where empty means no observation's top class.

Step 0 found Adam on a softplus parametrisation, not projected gradient, so no simplex projection is needed. Use doubles and off-diagonal terms only; stop with an error if the objective is not finite.

### `cbma_uncertainty(A_BMA)`
Per-person probability that the assigned cluster is wrong, p(g_i != ghat_i). Expected form: 1 - max_k A_BMA[i, k]. Confirm against the source in Step 0.

### `cbma_fit(X, A_list, K_BMA = NULL, index = "Calinski_Harabasz", ...)`
Wrapper. Default K_BMA = max K across inputs (the paper's heuristic). Returns a list:
`weights`, `civi`, `consensus` (optional, off by default for large N), `alloc` (A_BMA), `labels`, `uncertainty`, `empty_clusters`, `settings`, `objective_trace`.

## Step 3. Unit tests

- `cbma_hard_to_prob`: rows sum to 1, non-contiguous labels handled.
- `cbma_weights`: sum to 1; maximize vs minimize branches; error on zero or negative CIVI.
- `cbma_similarity`: symmetric, diagonal 1, values in [0, 1]; hard input gives only 0 and 1.
- `cbma_consensus`: equal to manual weighted sum on a small case; weights of (1, 0) return the first S exactly.
- `cbma_ssmf`:
  - block-diagonal C with 3 true blocks, K = 3: ARI = 1 with truth.
  - same C, K = 6: at most 3 non-empty clusters after regularization.
  - rows of output non-negative, sum to 1 within 1e-8.
  - same seed gives identical output.

## Step 4. Parity tests against fixtures

For each fixture scenario:

- CIVI values and weights: match within 1e-6.
- Consensus matrix: max abs diff below 1e-8.
- Final allocations: align clusters to the reference with `clue::solve_LSAP` on the cross-tabulation, then require ARI of hard labels >= 0.95 and mean abs diff of aligned allocation probabilities < 0.02.
- Uncertainty: correlation with reference >= 0.95.

The factorization is stochastic and TF numerics differ, so allocations are tested on agreement, not exact equality. If a scenario fails, report the numbers. Do not loosen a tolerance without saying so in `NOTES.md`.

## Step 5. Determinism and failure behavior

- Every random step takes a `seed` argument; no global RNG side effects.
- Inputs with mismatched N across A_list: stop with a clear error.
- Any input matrix whose rows do not sum to 1 within 1e-6: stop with a clear error.

## Step 6. Scale benchmark (script, not a test)

`tools/benchmark.R`: simulated data at N = 2,000, 5,000, 12,000; 6 input models; K_BMA = 6. Report wall time and peak memory for consensus and for SSMF separately, and the number of empty clusters at each N. Run only after the GMM optima experiment has finished. Full ABCD is roughly 12,000, so the dense consensus matrix is about 1.1 GB in double precision. If SSMF at N = 12,000 takes more than 30 minutes or memory exceeds 8 GB, stop and report; options then are RcppArmadillo, float storage, or the R `torch` package. Do not pick one without reporting first.

## Done when

- Step 0 notes written.
- All unit and parity tests pass.
- Benchmark table in `NOTES.md`.
- `README.md` with a 20-line usage example on scenario B.

## Known method caveats (document in README, do not fix)

- Weights come from internal validation indices, not likelihoods. CH favors compact spherical clusters; S_Dbw is the paper's alternative for other shapes.
- Linear normalization of CH tends to give fairly flat weights (the paper's case study: 0.35, 0.30, 0.35). Report weights alongside results so readers can see how much the averaging is doing.
- CIVIs for soft models are computed on hard projections.
- Allocation uncertainty is a point estimate, not a posterior distribution.
