# clusterBMA port

A plain R port of clusterBMA (Forbes et al., 2023, PLOS ONE, [doi:10.1371/journal.pone.0288000](https://doi.org/10.1371/journal.pone.0288000)), with no Python or TensorFlow. It combines clustering solutions from different methods into one probabilistic allocation, with model weights and per-person allocation uncertainty.

This repository holds the method only. The ABCD multiverse will call it later; no ABCD data are used or stored here, and the test fixtures are simulated.

**Status:** matches the original package on all four test scenarios, up to N = 500. Scale is not yet measured (see [Limits](#limits)).

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
- Allocation uncertainty is a point estimate, not a posterior distribution.

## Differences from the original

These are deliberate; reasons are in `NOTES.md`.

- `cbma_weights` stops on non-finite or non-positive index values. The original does not check.
- Index direction comes from a table in the port. The seven elbow-rule indices have no direction, so they are rejected.
- The factorisation uses doubles rather than float32 and stops if the objective is not finite. It does not clip silently.
- Equal priors only.

## Tests

```bash
Rscript --vanilla -e 'testthat::test_dir("tests/testthat")'
```

The suite takes about 5 minutes. Per-scenario results are in `NOTES.md`. The fixtures came from the original package at commit `4fba136` (Python 3.10.9, TensorFlow 2.11) through `tools/make_reference.R`. Regenerating them is not needed to run the tests.

## Limits

- **Scale is not measured.** The Step 6 benchmark (`tools/benchmark.R`) has not been written or run. Time grows with at least N squared: scenario B took 29 s at N = 300 on a busy machine, which extrapolates to roughly 13 hours at N = 12,000, ABCD's approximate size. That is a rough figure, not a measurement. The consensus matrix alone is about 1.1 GB at that size.
- **Empty clusters at large N are untested.** The penalty stays at the original's `lambda = 1`, and it may be too weak at large N.
- **Only Calinski-Harabasz is compared with the original.** `S_Dbw` and the minimise branch have unit tests but no reference fixture.
- **Scenario D has looser tolerances.** Its factorisation has two solutions, and the original package reaches both. The port does too. See `NOTES.md`.
- **Calinski-Harabasz values do not compare across data scales**, which matters if inputs are fitted on different transformations of the data.
