# Determinism and failure behaviour (SPEC Step 5).
source(testthat::test_path("..", "..", "R", "cbma.R"))

truth <- rep(1:2, each = 10)
X <- cbind(c(0, 6)[truth], c(0, 6)[truth]) + 0.2 * cbind(sin(1:20), cos(1:20))
A1 <- cbma_hard_to_prob(truth)
A2 <- cbma_hard_to_prob(truth, K = 3)

test_that("cbma_fit: same seed gives identical output, other seed gives other values", {
  a <- cbma_fit(X, list(A1, A2), n_iter = 50, seed = 3)
  expect_identical(a, cbma_fit(X, list(A1, A2), n_iter = 50, seed = 3))
  expect_false(identical(a$alloc, cbma_fit(X, list(A1, A2), n_iter = 50, seed = 4)$alloc))
})

test_that("cbma_fit: the result does not depend on the global RNG", {
  set.seed(1); a <- cbma_fit(X, list(A1), n_iter = 20, seed = 3)
  set.seed(2); b <- cbma_fit(X, list(A1), n_iter = 20, seed = 3)
  expect_identical(a, b)
})

test_that("cbma_fit: leaves an existing global RNG state unchanged", {
  set.seed(99); before <- .Random.seed
  cbma_fit(X, list(A1), n_iter = 5)
  expect_identical(.Random.seed, before)
})

test_that("cbma_fit: creates no global RNG state when there was none", {
  saved <- if (exists(".Random.seed", envir = globalenv())) get(".Random.seed", envir = globalenv())
  on.exit(if (!is.null(saved)) assign(".Random.seed", saved, envir = globalenv()))
  if (!is.null(saved)) rm(".Random.seed", envir = globalenv())
  cbma_fit(X, list(A1), n_iter = 5)
  expect_false(exists(".Random.seed", envir = globalenv(), inherits = FALSE))
})

test_that("cbma_ssmf: restores the RNG state after an error too", {
  set.seed(99); before <- .Random.seed
  expect_error(cbma_ssmf(cbma_similarity(A1), 2, lambda = Inf, n_iter = 2), "not finite")
  expect_identical(.Random.seed, before)
})

test_that("mismatched N across A_list stops with a clear error", {
  expect_error(cbma_fit(X, list(A1, A2[-1, ])), "must have nrow\\(X\\) rows")
  expect_error(cbma_fit(X[-1, ], list(A1)), "must have nrow\\(X\\) rows")
  expect_error(cbma_consensus(list(A1, A2[-1, ]), c(.5, .5)), "same number of rows")
})

test_that("rows that do not sum to 1 within 1e-6 stop with a clear error", {
  off <- function(d) { B <- A1; B[1, 1] <- 1 + d; B }
  expect_error(cbma_fit(X, list(A1, off(2e-6)), n_iter = 2), "A_list\\[\\[2\\]\\] do not sum to 1")
  expect_error(cbma_fit(X, list(off(-2e-6)), n_iter = 2), "do not sum to 1")
  expect_silent(cbma_fit(X, list(off(5e-7)), n_iter = 2))
  B <- A1; B[1, 1] <- NA
  expect_error(cbma_fit(X, list(B), n_iter = 2), "do not sum to 1")
})

test_that("a model with one non-empty class stops rather than giving NaN weights", {
  one <- cbma_hard_to_prob(rep(1, 20))
  expect_error(cbma_fit(X, list(A1, one), n_iter = 2), "finite and positive")
})
