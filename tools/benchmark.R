# =============================================================================
# tools/benchmark.R: scale benchmark for the port (SPEC Step 6). Not a test.
#   /usr/bin/time -l Rscript --vanilla tools/benchmark.R <N> [results.csv]   # from the repository root
# Pass "force" as a third argument to run all 5000 steps even over the limits.
# Run one process per N, so /usr/bin/time's peak memory belongs to that N. Appends
# one row to the results file (default tools/benchmark_results.csv). Simulated data only.
#
# Six input models and K_BMA = 6. The factorisation first runs PILOT_STEPS steps to
# project the time for 5000. Above N_LIMIT it runs in full only if the projection is
# within the spec's limits (LIMIT_MINS, LIMIT_MB); otherwise it stops and reports.
# Peak memory here is R's heap high-water mark from gc(), which is read at garbage
# collections and can miss a short spike; /usr/bin/time gives the process peak.
# =============================================================================

source("R/cbma.R")
suppressPackageStartupMessages(library(mclust))   # Mclust() needs mclustBIC on the search path

args <- commandArgs(trailingOnly = TRUE)
N <- as.integer(args[1])
stopifnot(!is.na(N))
K_BMA <- 6
PILOT_STEPS <- 20
N_LIMIT <- 5000
LIMIT_MINS <- 30
LIMIT_MB <- 8000
OUT <- if (length(args) >= 2) args[2] else "tools/benchmark_results.csv"

measure <- function(expr) {
  invisible(gc(reset = TRUE))
  secs <- system.time(value <- expr)[["elapsed"]]
  m <- gc()
  list(value = value, secs = secs, peak_mb = sum(m[, ncol(m)]))   # last column: max used, Mb
}

# Four unequal clusters in four dimensions (like four waves), so K_BMA = 6 exceeds what
# the data support and clusters can empty. Unit noise; neighbouring centres about 3 SDs
# apart along their shared axis. Cleaner than real CBCL scores, which are skewed.
set.seed(20261106 + N)
centres <- rbind(c(0, 0, 0, 0), c(3, 3, 3, 3), c(6, 6, 6, 6), c(3, 0, 3, 0))
truth <- sample(4, N, replace = TRUE, prob = c(.5, .25, .15, .1))
X <- centres[truth, ] + matrix(rnorm(4 * N), N)

km <- function(K) cbma_hard_to_prob(kmeans(X, K, nstart = 5, iter.max = 100)$cluster, K)
inputs <- measure(list(km(2), km(3), km(4), km(6),
                       Mclust(X, G = 3, verbose = FALSE)$z,
                       Mclust(X, G = 4, verbose = FALSE)$z))
A_list <- inputs$value

civi <- measure(vapply(A_list, \(A) cbma_civi(X, A)$value, 0))
weights <- cbma_weights(civi$value, "maximize")
consensus <- measure(cbma_consensus(A_list, weights))
C <- consensus$value
pilot <- measure(cbma_ssmf(C, K_BMA, n_iter = PILOT_STEPS))
projected_mins <- pilot$secs / PILOT_STEPS * 5000 / 60
cat(sprintf("N = %d: inputs %.0f s, index %.1f s, consensus %.1f s (%.0f MB), pilot %.1f s (%.0f MB), projected %.1f min\n",
            N, inputs$secs, civi$secs, consensus$secs, consensus$peak_mb, pilot$secs, pilot$peak_mb, projected_mins))

over <- N > N_LIMIT && (projected_mins > LIMIT_MINS || pilot$peak_mb > LIMIT_MB) && !identical(args[3], "force")
full <- if (!over) measure(cbma_ssmf(C, K_BMA, n_iter = 5000))
if (over) cat(sprintf("Over the spec's limits (%d min, %d MB): not running 5000 steps at N = %d.\n",
                      LIMIT_MINS, LIMIT_MB, N))

labels <- if (!over) max.col(full$value$alloc, ties.method = "first")
row <- data.frame(
  N = N, date = format(Sys.time(), "%Y-%m-%d %H:%M"),
  weights = paste(round(weights, 3), collapse = "/"),
  inputs_secs = round(inputs$secs, 1), index_secs = round(civi$secs, 1),
  consensus_secs = round(consensus$secs, 1), consensus_peak_mb = round(consensus$peak_mb),
  pilot_secs = round(pilot$secs, 1), pilot_peak_mb = round(pilot$peak_mb),
  projected_mins = round(projected_mins, 1), full_run = !over,
  ssmf_mins = if (over) NA else round(full$secs / 60, 1),
  ssmf_peak_mb = if (over) NA else round(full$peak_mb),
  empty_clusters = if (over) NA else length(full$value$empty_clusters),
  ari_truth = if (over) NA else round(adjustedRandIndex(truth, labels), 3),
  smallest_class = if (over) NA else round(min(tabulate(labels, K_BMA)[tabulate(labels, K_BMA) > 0]) / N, 4),
  blas = basename(extSoftVersion()[["BLAS"]]))
write.table(row, OUT, sep = ",", row.names = FALSE, append = file.exists(OUT),
            col.names = !file.exists(OUT))
print(t(row), quote = FALSE)
