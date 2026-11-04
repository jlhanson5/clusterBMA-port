# =============================================================================
# tools/make_reference.R: reference fixtures from the original clusterBMA (Step 1).
#   tools/setup_reference_env.sh                 # once
#   Rscript --vanilla tools/make_reference.R     # from the repository root
# Writes tests/testthat/fixtures/<scenario>.rds. Simulated data only. Not part of
# the test suite. The original package is never a runtime dependency of the port.
#
# The source sets no TF seed, so one is set here before each factorisation. Each
# scenario's factorisation is then repeated with the same seed (is the reference
# reproducible?) and a different seed (how much does the reference vary by itself?
# Step 4's parity tolerances cannot be tighter than that).
# =============================================================================

root <- getwd()
stopifnot(file.exists("SPEC_cbma_port.md"))
env <- file.path(root, ".fixture_env")
.libPaths(c(file.path(env, "rlib"), .libPaths()))
Sys.setenv(RETICULATE_CONDA = file.path(env, "miniforge", "bin", "conda"))

suppressPackageStartupMessages({
  library(clusterBMA)
  library(tensorflow)
})
clusterBMA_use_condaenv()

COMMIT <- system2("git", c("-C", "reference/clusterBMA", "rev-parse", "HEAD"), stdout = TRUE)
INDEX <- "Calinski_Harabasz"

# The source runs in graph mode (R/SimplexClust.R:34). A fresh default graph makes
# the op-level seeds depend on the graph seed only.
seed_tf <- function(seed) {
  tf$compat$v1$disable_eager_execution()
  tf$compat$v1$reset_default_graph()
  tf$compat$v1$set_random_seed(as.integer(seed))
}

km <- function(X, K, nstart = 20) hard_to_prob_fn(kmeans(X, centers = K, nstart = nstart)$cluster, K)
gmm <- function(X, K, seed) {
  g <- ClusterR::GMM(X, gaussian_comps = K, seed = seed)
  ClusterR::predict_GMM(X, CENTROIDS = g$centroids, COVARIANCE = g$covariance_matrices,
                        WEIGHTS = g$weights)$cluster_proba
}
ward <- function(X, K) hard_to_prob_fn(cutree(hclust(dist(X), "ward.D2"), K), K)

# Gaussian clusters with unit noise around the given centres.
blobs <- function(centres, n_each) {
  truth <- rep(seq_len(nrow(centres)), each = n_each)
  X <- centres[truth, , drop = FALSE] + matrix(rnorm(length(truth) * ncol(centres)), length(truth))
  list(X = X, truth = truth)
}

# ---- Scenarios (SPEC Step 1) -----------------------------------------------------
scenario <- list(
  # The package vignette example (vignettes/clusterBMA_vignette.Rmd). genRandomClust
  # writes files to the working directory, so it runs in a temp folder.
  A = function(seed) {
    old <- setwd(tempdir()); on.exit(setwd(old))
    s <- clusterGeneration::genRandomClust(numClust = 5, sepVal = 0.15, numNonNoisy = 2, numNoisy = 0,
                                           clustSizes = rep(100, 5), numReplicate = 1, clustszind = 3)
    X <- as.matrix(s$datList$test_1)
    list(X = X, truth = s$memList$test_1, K_BMA = 5,
         inputs = list(`kmeans_5` = km(X, 5, nstart = 1), `gmm_5` = gmm(X, 5, seed)))
  },
  # Well separated: 3 clusters, 2 dimensions, N = 300.
  B = function(seed) {
    b <- blobs(rbind(c(0, 0), c(8, 0), c(4, 7)), 100)
    list(X = b$X, truth = b$truth, K_BMA = 4,
         inputs = list(`kmeans_3` = km(b$X, 3), `gmm_3` = gmm(b$X, 3, seed), `ward_4` = ward(b$X, 4)))
  },
  # Poorly separated: 4 clusters, 20 dimensions, N = 500. Centres ~ N(0, 0.4^2) per
  # dimension, so neighbouring clusters are about 2.5 noise SDs apart.
  C = function(seed) {
    b <- blobs(matrix(rnorm(4 * 20, 0, 0.4), 4), 125)
    list(X = b$X, truth = b$truth, K_BMA = 5,
         inputs = list(`kmeans_3` = km(b$X, 3), `kmeans_4` = km(b$X, 4), `kmeans_5` = km(b$X, 5),
                       `gmm_4` = gmm(b$X, 4, seed)))
  },
  # Different K (2, 3, 6) on the same data, K_BMA = 6, to exercise empty clusters:
  # 3 true clusters, so K_BMA exceeds what the data support. The spec does not name
  # the methods; one each of hard, soft and hierarchical.
  D = function(seed) {
    b <- blobs(rbind(c(0, 0), c(8, 0), c(4, 7)), 100)
    list(X = b$X, truth = b$truth, K_BMA = 6,
         inputs = list(`kmeans_2` = km(b$X, 2), `gmm_3` = gmm(b$X, 3, seed), `ward_6` = ward(b$X, 6)))
  })
SEEDS <- c(A = 20261101L, B = 20261102L, C = 20261103L, D = 20261104L)

ari <- function(x, y) mclust::adjustedRandIndex(x, y)
versions <- function() {
  cfg <- reticulate::py_config()
  list(package_commit = COMMIT, python = sub(" .*", "", cfg$version_string),
       tensorflow = as.character(tf$version$VERSION), r = R.version.string,
       clusterCrit = as.character(packageVersion("clusterCrit")),
       ClusterR = as.character(packageVersion("ClusterR")),
       generated = format(Sys.time(), "%Y-%m-%d %H:%M %Z"))
}

grDevices::pdf(NULL)   # the source draws a heatmap (R/consensus_matrix_fn.R:33)
dest <- file.path("tests", "testthat", "fixtures")
for (nm in names(scenario)) {
  seed <- SEEDS[[nm]]
  set.seed(seed)
  s <- scenario[[nm]](seed)
  for (A in s$inputs) stopifnot(nrow(A) == nrow(s$X), all(is.finite(A)),
                                max(abs(rowSums(A) - 1)) < 1e-6)

  labels <- as.data.frame(lapply(s$inputs, \(A) prob_to_hard_fn(A)[[1]]$alloc_vector))
  civi <- new_weight_fn(as.data.frame(s$X), labels, length(s$inputs), INDEX, "max")

  seed_tf(seed)
  t0 <- Sys.time()
  res <- clusterBMA(input_data = as.data.frame(s$X), cluster_prob_matrices = unname(s$inputs),
                    n_final_clust = s$K_BMA, wt_crit_name = INDEX, wt_crit_direction = "max")
  secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  hard <- res[[3]]

  seed_tf(seed)
  same <- SimplexClust(S = res[[1]], G = s$K_BMA)[[1]]
  seed_tf(seed + 1L)
  other <- SimplexClust(S = res[[1]], G = s$K_BMA)[[1]]

  # Seed-to-seed floor for Step 4: align non-empty columns by label overlap.
  cols <- sort(unique(hard$alloc_vector))
  versus_reference <- function(P) {
    lab <- max.col(P, "first")
    match_col <- vapply(cols, \(j) which.max(tabulate(lab[hard$alloc_vector == j], s$K_BMA)), 1L)
    c(ari = ari(lab, hard$alloc_vector),
      mean_abs_diff = mean(abs(res[[2]][, cols] - P[, match_col])),
      uncertainty_cor = cor(hard$alloc_uncertainty, 1 - apply(P, 1, max)))
  }
  other_vs <- versus_reference(other)

  # Scenario D varies by seed more than the spec's tolerances allow, so its floor
  # is measured across 20 seeds (the first is the "other seed" above).
  floor_d <- NULL
  if (nm == "D") {
    floor_d <- t(vapply(seed + seq_len(20), function(sd) {
      seed_tf(sd)
      versus_reference(SimplexClust(S = res[[1]], G = s$K_BMA)[[1]])
    }, numeric(3)))
    rownames(floor_d) <- seed + seq_len(20)
    print(round(floor_d, 4))
  }

  meta <- c(versions(), list(
    scenario = nm, K_BMA = s$K_BMA, index = INDEX, direction = "max",
    input_names = names(s$inputs), fit_secs = secs,
    nonempty_columns = sort(unique(hard$alloc_vector)),
    same_seed_max_abs_diff = max(abs(same - res[[2]])),
    same_seed_ari = ari(max.col(same, "first"), hard$alloc_vector),
    other_seed_ari = other_vs[["ari"]],
    other_seed_mean_abs_diff = other_vs[["mean_abs_diff"]],
    other_seed_uncertainty_cor = other_vs[["uncertainty_cor"]],
    seed_floor = floor_d))

  saveRDS(list(X = s$X, truth = s$truth, inputs = s$inputs, seed = seed,
               civi = civi[[INDEX]], weights = civi$W_m, weights_returned = res[[5]],
               consensus = res[[1]], alloc = res[[2]], labels = hard$alloc_vector,
               labels_ordered = hard$alloc_ordered, uncertainty = hard$alloc_uncertainty,
               meta = meta),
          file.path(dest, paste0(nm, ".rds")))
  cat(sprintf("%s: N=%d, weights %s, non-empty %d of %d, ARI truth %.3f, same-seed ARI %.3f (max diff %.2g), other-seed ARI %.3f, mean diff %.4f, uncertainty r %.3f, %.0f s\n",
              nm, nrow(s$X), paste(round(civi$W_m, 3), collapse = "/"), length(meta$nonempty_columns), s$K_BMA,
              ari(hard$alloc_vector, s$truth), meta$same_seed_ari, meta$same_seed_max_abs_diff,
              meta$other_seed_ari, meta$other_seed_mean_abs_diff, meta$other_seed_uncertainty_cor, secs))
}
