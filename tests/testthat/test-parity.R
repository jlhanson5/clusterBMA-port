# Parity with the original clusterBMA (SPEC Step 4), against the fixtures from
# tools/make_reference.R. Takes several minutes: each factorisation is 5000 steps.
# The numbers are printed so a run can be compared with NOTES.md.
source(testthat::test_path("..", "..", "R", "cbma.R"))

# Scenario D's tolerances come from the reference's own seed-to-seed floor (NOTES.md).
TOL <- list(A = list(diff = .02, cor = .95), B = list(diff = .02, cor = .95),
            C = list(diff = .02, cor = .95), D = list(diff = .035, cor = .83))
D_SEEDS <- 1:5        # D has two solutions, so one seed would not test the tolerance

fixture <- function(n) readRDS(testthat::test_path("fixtures", paste0(n, ".rds")))

# Align port clusters to the reference with solve_LSAP on the cross-tabulation.
# Probabilities are compared on the reference's non-empty columns, as the floor was;
# the empty columns have no label overlap to align them by.
compare <- function(f, r) {
  K <- f$meta$K_BMA
  tab <- table(factor(f$labels, seq_len(K)), factor(r$labels, seq_len(K)))
  map <- as.integer(clue::solve_LSAP(tab, maximum = TRUE))
  cols <- sort(unique(f$labels))
  c(ari = mclust::adjustedRandIndex(r$labels, f$labels),
    diff = mean(abs(f$alloc[, cols] - r$alloc[, map[cols]])),
    cor = cor(r$uncertainty, f$uncertainty))
}
fit <- function(f, seed) cbma_fit(f$X, unname(f$inputs), K_BMA = f$meta$K_BMA, seed = seed,
                                  keep_consensus = TRUE)

for (n in c("A", "B", "C", "D")) {
  f <- fixture(n)
  r <- fit(f, 1L)
  m <- compare(f, r)
  cat(sprintf("\n%s seed 1: civi %.1e, weights %.1e, consensus %.1e, ARI %.4f, diff %.4f, cor %.4f, empty %d (ref %d)",
              n, max(abs(r$civi - f$civi)), max(abs(r$weights - f$weights)),
              max(abs(r$consensus - f$consensus)), m[["ari"]], m[["diff"]], m[["cor"]],
              length(r$empty_clusters), f$meta$K_BMA - length(unique(f$labels))))

  test_that(sprintf("parity %s: CIVI and weights within 1e-6", n), {
    expect_lt(max(abs(r$civi - f$civi)), 1e-6)
    expect_lt(max(abs(r$weights - f$weights)), 1e-6)
  })
  test_that(sprintf("parity %s: consensus within 1e-8", n), {
    expect_lt(max(abs(r$consensus - f$consensus)), 1e-8)
  })
  test_that(sprintf("parity %s: allocations and uncertainty agree", n), {
    expect_gte(m[["ari"]], .95)
    expect_lt(m[["diff"]], TOL[[n]]$diff)
    expect_gte(m[["cor"]], TOL[[n]]$cor)
  })
  test_that(sprintf("parity %s: same number of empty clusters", n), {
    expect_equal(length(r$empty_clusters), f$meta$K_BMA - length(unique(f$labels)))
  })
}

test_that("parity D: every port seed is within the tolerances", {
  f <- fixture("D")
  for (s in D_SEEDS[-1]) {
    m <- compare(f, fit(f, s))
    cat(sprintf("\nD seed %d: ARI %.4f, diff %.4f, cor %.4f", s, m[["ari"]], m[["diff"]], m[["cor"]]))
    expect_gte(m[["ari"]], .95)
    expect_lt(m[["diff"]], TOL$D$diff)
    expect_gte(m[["cor"]], TOL$D$cor)
  }
})
