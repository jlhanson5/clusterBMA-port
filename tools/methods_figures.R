# =============================================================================
# tools/methods_figures.R: two methods figures drawn on simulated toy data (no real data).
#   m_clusterbma.png  how clusterBMA combines many clustering solutions into one
#   m_stability.png   the subsample stability rule, on real-like and null data
#   Rscript tools/methods_figures.R          (from the repository root)
# Output: figures/. Needs ggplot2, patchwork, mclust, MASS.
# =============================================================================

suppressPackageStartupMessages({ library(ggplot2); library(patchwork); library(mclust); library(MASS) })
source("R/cbma.R")
dir.create("figures", showWarnings = FALSE)
set.seed(20261122)

INK <- "#0b0b0b"; INK2 <- "#52514e"; GRID <- "#ecebe8"
CL <- c("#0d3266", "#5b9be6", "#c0611f", "#3f8f5a", "#a33b5c", "#7a7975")
theme_set(theme_minimal(base_size = 10) +
  theme(panel.grid.minor = element_blank(), panel.grid.major = element_line(color = GRID, linewidth = 0.3),
        axis.text = element_blank(), axis.title = element_blank(),
        plot.title = element_text(color = INK, face = "bold", size = 10), plot.subtitle = element_text(color = INK2, size = 8.5),
        strip.text = element_text(color = INK, face = "bold", hjust = 0),
        legend.position = "none", plot.background = element_rect(fill = "white", color = NA)))

# ---- toy data: three groups, two of them overlapping -------------------------------------
mu <- rbind(c(0, 0), c(3.2, 0.4), c(1.4, 3.4))
X <- do.call(rbind, lapply(1:3, \(g) mvrnorm(60, mu[g, ], diag(c(0.9, 0.7)))))
X <- scale(X)
n <- nrow(X)

# ---- inputs: k-means at k = 2-4 and Ward at k = 3-4 --------------------------------------
km <- function(k) kmeans(X, k, nstart = 20)$cluster
wd <- function(k) cutree(hclust(dist(X), "ward.D2"), k)
inp <- list("k-means, k = 2" = km(2), "k-means, k = 3" = km(3), "k-means, k = 4" = km(4),
            "Ward, k = 3" = wd(3), "Ward, k = 4" = wd(4))
A_list <- lapply(inp, cbma_hard_to_prob)
civi <- lapply(A_list, \(A) cbma_civi(X, A, "Calinski_Harabasz"))
w <- cbma_weights(vapply(civi, `[[`, 0, "value"), civi[[1]]$direction)
C <- cbma_consensus(A_list, w)
f <- cbma_ssmf(C, K = 4, n_iter = 3000, seed = 1L)
lab <- max.col(f$alloc, ties.method = "first"); lab <- match(lab, unique(lab[order(X[, 1])]))
unc <- cbma_uncertainty(f$alloc)
ord <- order(lab, X[, 1])          # display order for the matrices

pts <- data.frame(x = X[, 1], y = X[, 2])
pA <- ggplot(pts, aes(x, y)) + geom_point(color = INK2, size = 1.1, alpha = 0.8) + coord_equal() +
  labs(title = "Data", subtitle = "Each child is a point")
inp_df <- do.call(rbind, lapply(names(inp), \(nm) data.frame(pts, cl = factor(inp[[nm]]),
  panel = sprintf("%s\nweight %.2f", nm, w[[nm]]))))
inp_df$panel <- factor(inp_df$panel, unique(inp_df$panel))
pB <- ggplot(inp_df, aes(x, y, color = cl)) + geom_point(size = 0.7) + facet_wrap(~ panel, nrow = 1) +
  scale_color_manual(values = CL) + coord_equal() +
  labs(title = "1. Fit many clustering solutions (methods x k)",
       subtitle = "Each solution is weighted by a validity index (here Calinski-Harabasz) or equally")
mat_df <- function(M, panel) {
  M <- M[ord, ord]
  data.frame(i = rep(seq_len(n), n), j = rep(seq_len(n), each = n), v = as.vector(M), panel = panel)
}
co <- do.call(rbind, lapply(names(inp), \(nm) mat_df(cbma_similarity(A_list[[nm]]), nm)))
co$panel <- factor(co$panel, names(inp))
pC <- ggplot(co, aes(i, -j, fill = v)) + geom_raster() + facet_wrap(~ panel, nrow = 1) +
  scale_fill_gradient(low = "white", high = "#0d3266") + coord_equal() +
  labs(title = "2. For each solution: which pairs of children share a cluster?",
       subtitle = "Co-assignment matrix: dark = same cluster. Children sorted by the final consensus class.")
pD <- ggplot(mat_df(C, "Consensus"), aes(i, -j, fill = v)) + geom_raster() +
  scale_fill_gradient(low = "white", high = "#0d3266") + coord_equal() +
  labs(title = "3. Weighted average", subtitle = "Consensus matrix: share of\n(weighted) solutions that agree")
fin <- data.frame(pts, cl = factor(lab), unc = unc)
pE <- ggplot(fin, aes(x, y, color = cl, alpha = 1 - unc)) + geom_point(size = 1.6) +
  scale_color_manual(values = CL) + scale_alpha(range = c(0.25, 1)) + coord_equal() +
  labs(title = "4. Factorise into classes", subtitle = sprintf("Each child gets class probabilities; faded =\nuncertain. %d of 4 allowed classes used.", length(unique(lab))))
unc_df <- data.frame(unc = unc)
pF <- ggplot(unc_df, aes(unc)) + geom_histogram(bins = 20, fill = "#5b9be6", color = "white") +
  labs(title = "Per-child uncertainty", subtitle = "1 - highest class probability") +
  theme(axis.text = element_text(color = INK2, size = 7), axis.title = element_text(color = INK2, size = 8)) +
  xlab("Uncertainty") + ylab("Children")
p <- (wrap_elements(pA) + wrap_elements(pB) + plot_layout(widths = c(1, 5))) / pC / (pD | pE | pF) +
  plot_layout(heights = c(1, 1, 1.25)) +
  plot_annotation(title = "How clusterBMA combines many clustering solutions",
                  subtitle = "Simulated data (180 points, three groups, two of them overlapping). Forbes et al. (2023), native R port.",
                  theme = theme(plot.title = element_text(face = "bold", size = 13), plot.subtitle = element_text(color = INK2)))
ggsave("figures/m_clusterbma.png", p, width = 13, height = 10.5, dpi = 300)

# ---- stability rule: real-like vs null data ------------------------------------------------
null <- mvrnorm(n, colMeans(X), cov(X))       # same means and covariance, no subgroups
stab <- function(D, k, R = 30, frac = 0.8) {
  sols <- lapply(seq_len(R), function(r) {
    idx <- sort(sample(nrow(D), round(frac * nrow(D))))
    v <- rep(NA_integer_, nrow(D)); v[idx] <- kmeans(D[idx, ], k, nstart = 10)$cluster; v
  })
  pr <- combn(R, 2)
  apply(pr, 2, \(p) { a <- sols[[p[1]]]; b <- sols[[p[2]]]; ok <- !is.na(a) & !is.na(b); adjustedRandIndex(a[ok], b[ok]) })
}
st <- do.call(rbind, lapply(c("Data with groups" = "real", "Null data (no groups)" = "null"), function(s)
  do.call(rbind, lapply(2:6, \(k) data.frame(data = s, k = k, ari = stab(if (s == "real") X else null, k))))))
st$data <- factor(ifelse(st$data == "real", "Data with groups", "Null data (no groups)"), c("Data with groups", "Null data (no groups)"))
med <- aggregate(ari ~ data + k, st, median)
sA <- ggplot(rbind(data.frame(pts, data = "Data with groups"), data.frame(x = null[, 1], y = null[, 2], data = "Null data (no groups)")),
             aes(x, y)) + geom_point(size = 0.8, color = INK2, alpha = 0.8) + facet_wrap(~ data) + coord_equal() +
  labs(title = "1. Real data and a null copy", subtitle = "Null: multivariate normal with the same means and covariances")
sB <- ggplot(st, aes(factor(k), ari, color = data)) +
  geom_hline(yintercept = 0.75, color = "#c0611f", linetype = "22") +
  geom_boxplot(outlier.size = 0.4, width = 0.6, position = position_dodge(0.7)) +
  scale_color_manual(values = c("#2563b8", "#7a7975"), name = NULL) +
  labs(title = "2. Refit on 80% subsamples; compare every pair of solutions",
       subtitle = "Adjusted Rand index between pairs of subsample solutions (k-means, 30 subsamples per k). Dashed: the inclusion rule, median >= .75 at every k", x = "k", y = "Pairwise ARI") +
  theme(axis.text = element_text(color = INK2), axis.title = element_text(color = INK2), legend.position = "top", legend.justification = "left")
p2 <- sA / sB + plot_layout(heights = c(1, 1.2)) +
  plot_annotation(title = "The stability rule for entering clusterBMA",
                  subtitle = "A method enters the consensus only if its classes come back in subsamples at every k. Null data show what stability looks like without groups.",
                  theme = theme(plot.title = element_text(face = "bold", size = 13), plot.subtitle = element_text(color = INK2)))
ggsave("figures/m_stability.png", p2, width = 10, height = 9, dpi = 300)
cat("Toy medians (ARI):\n"); print(reshape(med, idvar = "data", timevar = "k", direction = "wide"))
