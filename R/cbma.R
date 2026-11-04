# =============================================================================
# R/cbma.R: native R port of clusterBMA (Forbes et al., 2023, PLOS ONE).
# Combines clustering solutions into one probabilistic allocation with model
# weights and per-person allocation uncertainty. Reference source: of2/clusterBMA
# commit 4fba136; file references below are to that source. See NOTES.md for where
# and why the port differs from it.
# Dependencies: base R, clusterCrit.
# =============================================================================

# Index directions, from clusterCrit::bestCriterion. The seven elbow-rule indices
# there pick the best value by second difference, so they have no direction and do
# not fit the normalised weight (paper Eq 13). cbma_civi rejects them.
CBMA_MAXIMIZE <- c("Calinski_Harabasz", "Dunn", "Gamma", "PBM", "Point_Biserial",
                   "Ratkowsky_Lance", "Silhouette", "Tau", "Wemmert_Gancarski",
                   paste0("GDI", rep(1:5, each = 3), 1:3))
CBMA_MINIMIZE <- c("Banfeld_Raftery", "C_index", "Davies_Bouldin", "G_plus", "McClain_Rao",
                   "Ray_Turi", "Scott_Symons", "SD_Scat", "SD_Dis", "S_Dbw", "Xie_Beni")
CBMA_ELBOW <- c("Ball_Hall", "Ksq_DetW", "Trace_W", "Trace_WiB", "Det_Ratio",
                "Log_Det_Ratio", "Log_SS_Ratio")

# Integer labels to an N x K one-hot matrix. Labels need not be contiguous; they
# map to 1..K in sorted order. A K above the number of labels adds empty columns.
cbma_hard_to_prob <- function(labels, K = NULL) {
  if (anyNA(labels)) stop("labels contain NA")
  lev <- sort(unique(labels))
  if (is.null(K)) K <- length(lev)
  if (K < length(lev)) stop(sprintf("K = %d is below the %d distinct labels", K, length(lev)))
  A <- matrix(0, length(labels), K)
  A[cbind(seq_along(labels), match(labels, lev))] <- 1
  A
}

# Validity index on the hard projection of A (first column wins a tie, as which.max
# does in R/prob_to_hard_fn.R:9). The value is NaN if only one class is non-empty.
cbma_civi <- function(X, A, index = "Calinski_Harabasz") {
  if (index %in% CBMA_ELBOW)
    stop(sprintf("%s uses an elbow rule and has no direction, so it cannot be a weight", index))
  if (!index %in% c(CBMA_MAXIMIZE, CBMA_MINIMIZE)) stop(sprintf("unknown index: %s", index))
  if (nrow(X) != nrow(A)) stop("X and A have different numbers of rows")
  labels <- as.vector(max.col(A, ties.method = "first"))
  storage.mode(X) <- "double"
  list(value = clusterCrit::intCriteria(as.matrix(X), as.integer(labels), index)[[1]],
       direction = if (index %in% CBMA_MAXIMIZE) "maximize" else "minimize")
}

# Paper Eq 13 with equal priors (Eq 14). The check on the values is a deliberate
# difference from R/new_weight_fn.R, which would return NaN or weights outside [0, 1].
cbma_weights <- function(civi_values, direction) {
  direction <- match.arg(direction, c("maximize", "minimize"))
  if (!all(is.finite(civi_values)) || any(civi_values <= 0))
    stop("index values must be finite and positive to be used as weights")
  w <- if (direction == "maximize") civi_values else 1 / civi_values
  w / sum(w)
}

# Paper Eq 4.
cbma_similarity <- function(A) {
  S <- tcrossprod(A)
  diag(S) <- 1
  S
}

# Weighted sum of similarity matrices (paper Eq 15), accumulated one at a time so
# the M matrices are never all in memory.
cbma_consensus <- function(A_list, weights) {
  if (length(A_list) != length(weights)) stop("A_list and weights have different lengths")
  N <- nrow(A_list[[1]])
  C <- matrix(0, N, N)
  for (m in seq_along(A_list)) {
    if (weights[m] != 0) C <- C + weights[m] * cbma_similarity(A_list[[m]])
  }
  C
}

# Objective and its gradient with respect to the unconstrained V, for C (N x N) and
# V (N x K). Follows R/SimplexClust.R: P = softplus(V), normalised by row; A = P P';
# binary cross-entropy over ordered off-diagonal pairs, plus lambda times the sum of
# column L2 norms. A term with a zero coefficient is left out, not evaluated, so a
# hard C (C_ij = 1) with A_ij = 1 contributes 0 rather than 0 * log(0). Anything else
# non-finite is left to reach the caller, which stops: there is no clipping.
cbma_ssmf_step <- function(V, C, lambda, gradient = TRUE) {
  P0 <- -plogis(-V, log.p = TRUE)       # softplus(V), stable for large |V|
  s <- rowSums(P0)
  P <- P0 / s
  A <- tcrossprod(P)
  on <- C > 0; diag(on) <- FALSE        # the C log(A) terms
  off <- C < 1; diag(off) <- FALSE      # the (1 - C) log(1 - A) terms
  norms <- sqrt(colSums(P^2))
  value <- -sum(C[on] * log(A[on])) - sum((1 - C[off]) * log1p(-A[off])) + lambda * sum(norms)
  if (!gradient) return(list(value = value, P = P))
  D <- matrix(0, nrow(C), ncol(C))
  D[on] <- -C[on] / A[on]
  D[off] <- D[off] + (1 - C[off]) / (1 - A[off])
  G <- 2 * (D %*% P) + lambda * sweep(P, 2, norms, "/")          # d objective / d P
  list(value = value, P = P, grad = (G - rowSums(G * P)) / s * plogis(V))
}

# Symmetric simplex matrix factorisation: C (N x N) to an N x K allocation matrix.
# Adam on a softplus parametrisation, as R/SimplexClust.R does in float32 through TF,
# here in doubles. Defaults are the source's (lambda = 1, lr = 0.01, 5000 steps) and
# TF's Adam constants. tol = NULL runs every step, as the source does; otherwise
# stops when the relative change in the objective falls below tol. The RNG state is
# restored on exit. An empty cluster is one that is no observation's top class.
cbma_ssmf <- function(C, K, lambda = 1, n_iter = 5000, lr = 0.01, tol = NULL, seed = 1L) {
  if (!is.matrix(C) || nrow(C) != ncol(C)) stop("C must be a square matrix")
  if (anyNA(C) || any(C < 0 | C > 1 + 1e-8)) stop("C must contain values in [0, 1]")
  N <- nrow(C)
  had_seed <- exists(".Random.seed", envir = globalenv(), inherits = FALSE)
  if (had_seed) old_seed <- get(".Random.seed", envir = globalenv(), inherits = FALSE)
  on.exit(if (had_seed) assign(".Random.seed", old_seed, envir = globalenv())
          else if (exists(".Random.seed", envir = globalenv(), inherits = FALSE))
            rm(".Random.seed", envir = globalenv()))
  set.seed(seed)
  V <- matrix(runif(N * K), N, K)

  beta1 <- 0.9; beta2 <- 0.999; eps <- 1e-8
  m <- v <- matrix(0, N, K)
  trace <- numeric(n_iter)
  used <- n_iter
  for (t in seq_len(n_iter)) {
    step <- cbma_ssmf_step(V, C, lambda)
    if (!is.finite(step$value) || anyNA(step$grad))
      stop(sprintf("objective is not finite at step %d", t))
    trace[t] <- step$value
    if (!is.null(tol) && t > 1 && abs(trace[t - 1] - trace[t]) < tol * abs(trace[t - 1])) {
      used <- t
      break
    }
    m <- beta1 * m + (1 - beta1) * step$grad
    v <- beta2 * v + (1 - beta2) * step$grad^2
    V <- V - lr * sqrt(1 - beta2^t) / (1 - beta1^t) * m / (sqrt(v) + eps)
  }
  final <- cbma_ssmf_step(V, C, lambda, gradient = FALSE)
  if (!is.finite(final$value)) stop("objective is not finite at the final step")
  list(alloc = final$P, objective_trace = trace[seq_len(used)], iterations = used,
       empty_clusters = setdiff(seq_len(K), max.col(final$P, ties.method = "first")))
}

# Per-person probability that the assigned cluster is wrong (R/prob_to_hard_fn.R:15).
cbma_uncertainty <- function(A_BMA) {
  1 - A_BMA[cbind(seq_len(nrow(A_BMA)), max.col(A_BMA, ties.method = "first"))]
}

# All inputs must describe the same N people, with rows that are probabilities.
cbma_check_inputs <- function(X, A_list) {
  if (length(A_list) == 0) stop("A_list is empty")
  if (any(vapply(A_list, nrow, 1L) != nrow(X))) stop("every matrix in A_list must have nrow(X) rows")
  for (m in seq_along(A_list)) {
    if (anyNA(A_list[[m]]) || max(abs(rowSums(A_list[[m]]) - 1)) > 1e-6)
      stop(sprintf("rows of A_list[[%d]] do not sum to 1 within 1e-6", m))
  }
  invisible(TRUE)
}

# The whole method. K_BMA defaults to the largest K across inputs, the paper's
# heuristic. The consensus matrix is returned only on request (it is N x N).
cbma_fit <- function(X, A_list, K_BMA = NULL, index = "Calinski_Harabasz",
                     keep_consensus = FALSE, lambda = 1, n_iter = 5000, lr = 0.01,
                     tol = NULL, seed = 1L) {
  X <- as.matrix(X)
  cbma_check_inputs(X, A_list)
  if (is.null(K_BMA)) K_BMA <- max(vapply(A_list, ncol, 1L))
  civi <- lapply(A_list, \(A) cbma_civi(X, A, index))
  values <- vapply(civi, `[[`, 0, "value")
  weights <- cbma_weights(values, civi[[1]]$direction)
  C <- cbma_consensus(A_list, weights)
  f <- cbma_ssmf(C, K_BMA, lambda = lambda, n_iter = n_iter, lr = lr, tol = tol, seed = seed)
  list(weights = weights, civi = values,
       consensus = if (keep_consensus) C,
       alloc = f$alloc,
       labels = max.col(f$alloc, ties.method = "first"),
       uncertainty = cbma_uncertainty(f$alloc),
       empty_clusters = f$empty_clusters,
       settings = list(K_BMA = K_BMA, index = index, direction = civi[[1]]$direction,
                       lambda = lambda, n_iter = n_iter, lr = lr, tol = tol, seed = seed,
                       iterations = f$iterations),
       objective_trace = f$objective_trace)
}
