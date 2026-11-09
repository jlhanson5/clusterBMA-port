# clusterBMA port

A plain R port of clusterBMA (Forbes et al., 2023, PLOS ONE, [doi:10.1371/journal.pone.0288000](https://doi.org/10.1371/journal.pone.0288000)), with no Python or TensorFlow. It combines clustering solutions from different methods into one probabilistic allocation, with model weights and per-person allocation uncertainty.

This repository holds the method only. The ABCD clustering multiverse uses it (16 fits at N = 8,331; see `NOTES.md`, "Use at scale"); no ABCD data are used or stored here, and the test fixtures are simulated.

**Status:** matches the original package on all four test scenarios, up to N = 500. Full fits are measured up to N = 12,000 on simulated data (about 2.4 hours) and run at N = 8,331 on real data (73-87 min each); see [Limits](#limits).

## Requirements

R with `clusterCrit`. The tests also need `testthat`, `clue` and `mclust`. Built on R 4.6.1 with clusterCrit 1.3.0, clue 0.3-68 and mclust 6.1.3. All functions are in `R/cbma.R`; there is no package to install.

## Usage

Scenario B from the test fixtures, run from the repository root:

```r
source("R/cbma.R")                      # needs clusterCrit
library(mclust)                         # only for the GMM input; Mclust() fails when called as mclust::Mclust()
set.seed(20261102)
# Scenario B: 3 well-separated clusters, 2 dimensions, N = 300
centres <- rbind(c(0, 0), c(8, 0), c(4, 7))
truth <- rep(1:3, each = 100)
X <- centres[truth, ] + matrix(rnorm(600), 300)
# Inputs: one N x K matrix per solution. Hard labels become one-hot; soft models give probabilities
inputs <- list(
  kmeans_3 = cbma_hard_to_prob(kmeans(X, 3, nstart = 20)$cluster),
  gmm_3    = Mclust(X, G = 3, verbose = FALSE)$z,
  ward_4   = cbma_hard_to_prob(cutree(hclust(dist(X), "ward.D2"), 4))
)
fit <- cbma_fit(X, inputs, seed = 1)    # K_BMA defaults to the largest input K, here 4

round(setNames(fit$weights, names(inputs)), 3)
fit$empty_clusters                      # clusters that are nobody's top class
table(truth, fit$labels)
summary(fit$uncertainty)                # per-person probability the assigned cluster is wrong
```

Output:

```
kmeans_3    gmm_3   ward_4
   0.363    0.363    0.273

[1] 1

     
truth   2   3   4
    1   0 100   0
    2   0   0 100
    3 100   0   0

     Min.   1st Qu.    Median      Mean   3rd Qu.      Max.
0.0002310 0.0003153 0.0003781 0.0257696 0.0007272 0.1657205
```

The fit recovers the three true clusters and leaves the fourth column empty. Cluster numbers are arbitrary. The data are identical to the fixture's `X`. The GMM input comes from `mclust` here and `ClusterR` in the fixture; the weights and the labels still agree with the original package's result (ARI = 1). The example takes under a minute.

All inputs must describe the same N people in the same order. `cbma_fit` stops if the row counts differ or if any row does not sum to 1 within 1e-6.

### Weights without a validity index

By default the weights come from `index`, as in the original. You can pass your own instead; these options are additions, not part of the original package:

```r
k <- c(3, 3, 4)                                       # the K of each input above
fit_eq   <- cbma_fit(X, inputs, weights = cbma_weights_equal(length(inputs)))
fit_kbal <- cbma_fit(X, inputs, weights = cbma_weights_grouped(k))   # each k gets the same total
fit_avg  <- cbma_fit(X, inputs, weights = cbma_weights_average(fit$weights, cbma_weights_equal(3)))
```

`cbma_weights_grouped` is for inputs that are uneven across some grouping, such as k when some methods have no solution at large k: with equal weights per input, the k values with more inputs would count for more. Supplied weights must be non-negative and sum to 1; `civi` is then `NA`.

## What `cbma_fit` returns

| Element | Contents |
|---|---|
| `weights`, `civi` | Model weights and the validity index value behind each one |
| `alloc` | N x K_BMA allocation probabilities; rows sum to 1 |
| `labels` | Each person's top cluster |
| `uncertainty` | 1 minus the top probability |
| `empty_clusters` | Clusters that are no one's top class |
| `consensus` | N x N consensus matrix, only with `keep_consensus = TRUE` |
| `settings`, `objective_trace` | Settings used; the objective at each step |

An empty cluster is not necessarily near zero. In scenario D, columns that are no one's top class still reach probabilities of .26 and .22, so check `alloc` as well as `labels`.

## Method caveats

- Weights come from internal validation indices, not likelihoods. Calinski-Harabasz (the default) favours compact spherical clusters; S_Dbw is the paper's alternative for other shapes.
- Linear normalisation of Calinski-Harabasz tends to give fairly flat weights (the paper's case study: .35, .30, .35). Report weights alongside results so readers can see how much the averaging is doing.
- Validity indices for soft models are computed on hard projections.
- Allocation uncertainty is a point estimate, not a posterior distribution. With hard (one-hot) inputs it measures disagreement between inputs, not uncertainty within any one model.

## Differences from the original

These are deliberate; reasons are in `NOTES.md`.

- `cbma_weights` stops on non-finite or non-positive index values. The original does not check.
- Index direction comes from a table in the port. The seven elbow-rule indices have no direction, so they are rejected.
- The factorisation uses doubles rather than float32 and stops if the objective is not finite. It does not clip silently.
- Equal priors only.
- `cbma_fit` can take supplied weights (`weights =`, with `cbma_weights_equal`, `cbma_weights_grouped` and `cbma_weights_average`). The original only weights by index.

## Tests

```bash
Rscript --vanilla -e 'testthat::test_dir("tests/testthat")'
```

The suite takes about 5 minutes. Per-scenario results are in `NOTES.md`. The fixtures came from the original package at commit `4fba136` (Python 3.10.9, TensorFlow 2.11) through `tools/make_reference.R`. Regenerating them is not needed to run the tests.

## Limits

- **Large N is slow.** Benchmark on simulated data, 6 inputs, K_BMA = 6 (`tools/benchmark.R`, tables in `NOTES.md`). With Apple's vecLib BLAS: 4.1 min at N = 2,000, 23 min at N = 5,000 and 2.4 hours at N = 12,000 (about 2.8 hours projected with R's default BLAS), peaking at 8.3 GB. That is over the spec's 30-minute limit. Results do not depend on the BLAS beyond rounding (about 1e-15).
- **Real data at N = 8,331** (ABCD CBCL, 32 hard inputs, K_BMA = 6, vecLib): 73-87 min per fit and about 4.0-4.5 GB per process, so at most two at once on a 16 GB machine. 3-4 of 6 clusters were used. A 10,000-step rerun agreed with 5,000 steps at ARI .96. Details in `NOTES.md`.
- **Empty clusters hold up with N.** The penalty stays at the original's `lambda = 1`. With 4 true clusters and K_BMA = 6, 2 clusters empty at N = 2,000, 5,000 and 12,000. These simulated clusters are cleaner than CBCL scores.
- **Only Calinski-Harabasz is compared with the original.** `S_Dbw` and the minimise branch have unit tests but no reference fixture.
- **Scenario D has looser tolerances.** Its factorisation has two solutions, and the original package reaches both. The port does too. See `NOTES.md`.
- **Calinski-Harabasz values do not compare across data scales**, which matters if inputs are fitted on different transformations of the data.
