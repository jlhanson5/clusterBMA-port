# Unit tests for R/cbma.R (SPEC Step 3). Run from the repository root:
#   Rscript --vanilla -e 'testthat::test_dir("tests/testthat")'
source(testthat::test_path("..", "..", "R", "cbma.R"))

blocks <- function(sizes) {
  truth <- rep(seq_along(sizes), sizes)
  list(C = outer(truth, truth, `==`) * 1, truth = truth)
}

test_that("cbma_hard_to_prob: rows sum to 1, non-contiguous labels map in sorted order", {
  A <- cbma_hard_to_prob(c(7, 2, 2, 11))
  expect_equal(rowSums(A), rep(1, 4))
  expect_equal(ncol(A), 3L)
  expect_equal(max.col(A), c(2L, 1L, 1L, 3L))
})

test_that("cbma_hard_to_prob: K adds empty columns, and refuses too few or NA", {
  expect_equal(colSums(cbma_hard_to_prob(c(1, 2), K = 4)), c(1, 1, 0, 0))
  expect_error(cbma_hard_to_prob(c(1, 2, 3), K = 2), "below")
  expect_error(cbma_hard_to_prob(c(1, NA)), "NA")
})

test_that("cbma_civi: matches clusterCrit on the first-tie hard projection", {
  set.seed(11)
  X <- rbind(matrix(rnorm(40), 20), matrix(rnorm(40, 6), 20))
  A <- cbind(rep(c(.5, .9), each = 20), rep(c(.5, .1), each = 20))
  A[21:40, ] <- A[21:40, 2:1]                 # first 20 rows tie, so they go to column 1
  labels <- rep(1:2, each = 20)
  r <- cbma_civi(X, A)
  expect_equal(r$value, clusterCrit::intCriteria(X, labels, "Calinski_Harabasz")[[1]])
  expect_equal(r$direction, "maximize")
  expect_equal(cbma_civi(X, A, "S_Dbw")$direction, "minimize")
})

test_that("cbma_civi: direction table covers clusterCrit and agrees with bestCriterion", {
  expect_setequal(c(CBMA_MAXIMIZE, CBMA_MINIMIZE, CBMA_ELBOW), clusterCrit::getCriteriaNames(TRUE))
  for (i in CBMA_MAXIMIZE) expect_equal(clusterCrit::bestCriterion(c(1, 2, 3), i), 3L, info = i)
  for (i in CBMA_MINIMIZE) expect_equal(clusterCrit::bestCriterion(c(1, 2, 3), i), 1L, info = i)
})

test_that("cbma_civi: refuses elbow-rule and unknown indices", {
  X <- matrix(rnorm(20), 10); A <- cbma_hard_to_prob(rep(1:2, 5))
  for (i in CBMA_ELBOW) expect_error(cbma_civi(X, A, i), "elbow", info = i)
  expect_error(cbma_civi(X, A, "Nope"), "unknown")
})

test_that("cbma_weights: sum to 1, both directions", {
  expect_equal(cbma_weights(c(1, 3), "maximize"), c(.25, .75))
  expect_equal(cbma_weights(c(1, 3), "minimize"), c(.75, .25))
  expect_equal(sum(cbma_weights(c(2.5, 7, 11), "minimize")), 1)
})

test_that("cbma_weights: errors on zero, negative and non-finite values", {
  for (bad in list(c(1, 0), c(1, -1), c(1, NaN), c(1, Inf), c(1, NA)))
    expect_error(cbma_weights(bad, "maximize"), "finite and positive")
  expect_error(cbma_weights(c(1, 2), "sideways"))
})

test_that("cbma_similarity: symmetric, unit diagonal, values in [0, 1]", {
  set.seed(12)
  A <- matrix(runif(60), 20); A <- A / rowSums(A)
  S <- cbma_similarity(A)
  expect_equal(S, t(S))
  expect_equal(diag(S), rep(1, 20))
  expect_true(all(S >= 0 & S <= 1 + 1e-12))
})

test_that("cbma_similarity: hard input gives only 0 and 1", {
  expect_setequal(cbma_similarity(cbma_hard_to_prob(c(1, 1, 2, 3))), c(0, 1))
})

test_that("cbma_consensus: equals the manual weighted sum; weights (1, 0) give the first S", {
  A1 <- cbma_hard_to_prob(c(1, 1, 2, 2)); A2 <- cbma_hard_to_prob(c(1, 2, 2, 3))
  S1 <- cbma_similarity(A1); S2 <- cbma_similarity(A2)
  expect_equal(cbma_consensus(list(A1, A2), c(.3, .7)), .3 * S1 + .7 * S2)
  expect_identical(cbma_consensus(list(A1, A2), c(1, 0)), S1)
  expect_error(cbma_consensus(list(A1, A2), 1), "different lengths")
})

test_that("cbma_ssmf gradient agrees with numerical differences", {
  set.seed(13)
  A <- matrix(runif(36), 12); A <- A / rowSums(A)
  C <- cbma_similarity(A)
  V <- matrix(runif(36, -1, 1), 12)
  g <- cbma_ssmf_step(V, C, lambda = 1)$grad
  num <- V
  for (i in seq_along(V)) {
    h <- 1e-6; up <- V; dn <- V; up[i] <- V[i] + h; dn[i] <- V[i] - h
    num[i] <- (cbma_ssmf_step(up, C, 1, FALSE)$value - cbma_ssmf_step(dn, C, 1, FALSE)$value) / (2 * h)
  }
  expect_equal(g, num, tolerance = 1e-6)
})

test_that("cbma_ssmf: block-diagonal C with 3 blocks, K = 3 recovers the blocks", {
  b <- blocks(c(12, 10, 8))
  f <- cbma_ssmf(b$C, 3)
  expect_equal(mclust::adjustedRandIndex(max.col(f$alloc), b$truth), 1)
  expect_length(f$empty_clusters, 0)
})

test_that("cbma_ssmf: same C with K = 6 leaves at most 3 non-empty clusters", {
  b <- blocks(c(12, 10, 8))
  f <- cbma_ssmf(b$C, 6)
  expect_lte(6 - length(f$empty_clusters), 3)
  expect_equal(mclust::adjustedRandIndex(max.col(f$alloc), b$truth), 1)
})

test_that("cbma_ssmf: rows are non-negative and sum to 1 within 1e-8", {
  f <- cbma_ssmf(blocks(c(6, 6))$C, 4, n_iter = 200)
  expect_true(all(f$alloc >= 0))
  expect_lt(max(abs(rowSums(f$alloc) - 1)), 1e-8)
})

test_that("cbma_ssmf: same seed gives identical output; other seed differs", {
  C <- blocks(c(6, 6))$C
  expect_identical(cbma_ssmf(C, 3, n_iter = 100, seed = 5), cbma_ssmf(C, 3, n_iter = 100, seed = 5))
  expect_false(identical(cbma_ssmf(C, 3, n_iter = 100, seed = 5)$alloc,
                         cbma_ssmf(C, 3, n_iter = 100, seed = 6)$alloc))
})

test_that("cbma_ssmf: leaves the global RNG state alone", {
  C <- blocks(c(4, 4))$C
  set.seed(99); before <- .Random.seed
  cbma_ssmf(C, 2, n_iter = 5, seed = 1)
  expect_identical(.Random.seed, before)
})

test_that("cbma_ssmf: runs every step by default; tol stops early", {
  C <- blocks(c(6, 6))$C
  full <- cbma_ssmf(C, 2, n_iter = 300)
  expect_equal(full$iterations, 300)
  expect_length(full$objective_trace, 300)
  early <- cbma_ssmf(C, 2, n_iter = 5000, tol = 1e-3)
  expect_lt(early$iterations, 5000)
  expect_length(early$objective_trace, early$iterations)
})

test_that("cbma_ssmf: stops on a non-finite objective rather than clipping", {
  expect_error(cbma_ssmf(blocks(c(4, 4))$C, 2, lambda = Inf, n_iter = 5), "not finite at step 1")
})

test_that("cbma_ssmf: a hard C with A at 1 is not treated as non-finite", {
  b <- blocks(c(4, 4))
  V <- cbind(rep(c(60, -60), each = 4), rep(c(-60, 60), each = 4))
  expect_true(is.finite(cbma_ssmf_step(V, b$C, 1, gradient = FALSE)$value))
})

test_that("cbma_ssmf: refuses a C that is not a similarity matrix", {
  C <- blocks(c(4, 4))$C
  expect_error(cbma_ssmf(matrix(1, 2, 3), 2), "square")
  bad <- C; bad[1, 2] <- NaN
  expect_error(cbma_ssmf(bad, 2), "\\[0, 1\\]")
  bad <- C; bad[1, 2] <- 1.5
  expect_error(cbma_ssmf(bad, 2), "\\[0, 1\\]")
})

test_that("cbma_uncertainty: one minus the largest probability per row", {
  expect_equal(cbma_uncertainty(rbind(c(.7, .3), c(.5, .5), c(0, 1))), c(.3, .5, 0))
})

test_that("cbma_fit: end to end on well-separated data", {
  set.seed(14)
  truth <- rep(1:3, each = 15)
  X <- cbind(c(0, 8, 4)[truth], c(0, 0, 7)[truth]) + matrix(rnorm(90), 45)
  A_list <- list(cbma_hard_to_prob(kmeans(X, 3, nstart = 10)$cluster),
                 cbma_hard_to_prob(cutree(hclust(dist(X), "ward.D2"), 4)))
  f <- cbma_fit(X, A_list, seed = 2)
  expect_equal(f$settings$K_BMA, 4L)
  expect_equal(sum(f$weights), 1)
  expect_null(f$consensus)
  expect_equal(mclust::adjustedRandIndex(f$labels, truth), 1)
  expect_equal(f$uncertainty, cbma_uncertainty(f$alloc))
  expect_equal(dim(cbma_fit(X, A_list, keep_consensus = TRUE, n_iter = 5)$consensus), c(45L, 45L))
})

test_that("cbma_weights_equal / grouped / average: sums, group shares, means", {
  expect_equal(cbma_weights_equal(4), rep(.25, 4))
  expect_error(cbma_weights_equal(0), "positive whole")
  k <- rep(2:6, c(7, 7, 6, 6, 6))                  # the uneven-k case from the ABCD inputs
  w <- cbma_weights_grouped(k)
  expect_equal(sum(w), 1)
  expect_equal(as.vector(tapply(w, k, sum)), rep(.2, 5))
  expect_equal(w[k == 4], rep(1 / 30, 6))
  expect_equal(cbma_weights_grouped(factor(c("a", "b", "b"))), c(.5, .25, .25))
  expect_error(cbma_weights_grouped(c(1, NA)), "NA")
  a <- c(.5, .3, .2); b <- c(.2, .2, .6)
  expect_equal(cbma_weights_average(a, b, cbma_weights_equal(3)), (a + b + 1 / 3) / 3)
  expect_error(cbma_weights_average(a, c(.5, .5)), "different lengths")
  expect_error(cbma_weights_average(a, c(.5, .5, .5)), "sum to 1")
})

test_that("cbma_fit: supplied weights give the index path's result when they match it", {
  set.seed(15)
  truth <- rep(1:3, each = 15)
  X <- cbind(c(0, 8, 4)[truth], c(0, 0, 7)[truth]) + matrix(rnorm(90), 45)
  A_list <- list(cbma_hard_to_prob(kmeans(X, 3, nstart = 10)$cluster),
                 cbma_hard_to_prob(cutree(hclust(dist(X), "ward.D2"), 4)),
                 cbma_hard_to_prob(cutree(hclust(dist(X), "average"), 2)))
  f_idx <- cbma_fit(X, A_list, n_iter = 200, seed = 3)
  f_w <- cbma_fit(X, A_list, n_iter = 200, seed = 3, weights = f_idx$weights)
  expect_identical(f_w$alloc, f_idx$alloc)
  expect_true(all(is.na(f_w$civi)))
  expect_true(is.na(f_w$settings$index))
  f_eq <- cbma_fit(X, A_list, n_iter = 200, seed = 3, weights = cbma_weights_equal(3))
  expect_equal(f_eq$weights, rep(1 / 3, 3))
  expect_error(cbma_fit(X, A_list, weights = c(.5, .5)), "length 3")
  expect_error(cbma_fit(X, A_list, weights = c(.5, .6, -.1)), "non-negative")
  expect_error(cbma_fit(X, A_list, weights = c(.5, .3, .3)), "sum to 1")
})

test_that("cbma_fit: refuses mismatched N and rows that do not sum to 1", {
  X <- matrix(rnorm(20), 10); good <- cbma_hard_to_prob(rep(1:2, 5))
  expect_error(cbma_fit(X, list(good, good[-1, ])), "nrow")
  bad <- good; bad[1, ] <- c(.5, .4)
  expect_error(cbma_fit(X, list(bad)), "do not sum to 1")
})
