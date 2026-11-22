# =============================================================================
# tools/methods_figures_graphic.R: illustrative methods figures on simulated data only.
#   m4_trajectories.png   clustering trajectories: spaghetti -> classes -> classes slice a continuum
#   m5_perturbations.png  the five stability perturbations, drawn on the same data
#   m6_network.png        the consensus as a network: children linked when solutions agree
#   m7_probabilities.png  soft allocation: each child's class probabilities, sorted
#   Rscript tools/methods_figures_graphic.R      (from the repository root)
# Output: figures/. Needs ggplot2, patchwork, mclust, MASS, igraph.
# =============================================================================

suppressPackageStartupMessages({ library(ggplot2); library(patchwork); library(mclust); library(MASS); library(igraph) })
source("R/cbma.R")
dir.create("figures", showWarnings = FALSE)

INK <- "#0b0b0b"; INK2 <- "#52514e"; GRID <- "#ecebe8"
CL <- c("#0d3266", "#5b9be6", "#c0611f", "#3f8f5a", "#a33b5c", "#7a7975")
base <- theme_minimal(base_size = 10) +
  theme(panel.grid.minor = element_blank(), panel.grid.major = element_line(color = GRID, linewidth = 0.3),
        axis.text = element_text(color = INK2), axis.title = element_text(color = INK2),
        plot.title = element_text(color = INK, face = "bold", size = 10.5), plot.subtitle = element_text(color = INK2, size = 8.5),
        strip.text = element_text(color = INK, face = "bold", hjust = 0), legend.position = "none",
        plot.background = element_rect(fill = "white", color = NA))
bare <- base + theme(axis.text = element_blank(), axis.title = element_blank())
ann <- function(title, subtitle) plot_annotation(title = title, subtitle = subtitle,
  theme = theme(plot.title = element_text(face = "bold", size = 14, color = INK), plot.subtitle = element_text(color = INK2, size = 10),
                plot.background = element_rect(fill = "white", color = NA)))

# ---- m4: trajectories ------------------------------------------------------------------
set.seed(4)
n <- 400; waves <- c("Baseline", "Year 1", "Year 2", "Year 3")
sev <- rgamma(n, shape = 1.4, rate = 0.07)                 # skewed, continuous severity: no true groups
slope <- rnorm(n, 0, 1.2)
Y <- sapply(0:3, \(t) pmax(0, sev + slope * t + rnorm(n, 0, 3)))
cls <- kmeans(Y, 4, nstart = 25)$cluster
cls <- match(cls, order(tapply(rowMeans(Y), cls, mean)))
long <- data.frame(id = rep(seq_len(n), 4), wave = factor(rep(waves, each = n), waves), y = as.vector(Y), cl = factor(rep(cls, 4)))
means <- aggregate(y ~ wave + cl, long, mean)
pa <- ggplot(long, aes(wave, y, group = id)) + geom_line(color = INK2, alpha = 0.15, linewidth = 0.3) + base +
  labs(title = "1. Every child's trajectory", subtitle = "400 simulated children, 4 waves", x = NULL, y = "Symptom score")
pb <- ggplot(long, aes(wave, y, group = id, color = cl)) + geom_line(alpha = 0.12, linewidth = 0.3) +
  geom_line(data = means, aes(group = cl), linewidth = 1.6) + geom_point(data = means, aes(group = cl), size = 2.2) +
  scale_color_manual(values = CL) + base +
  labs(title = "2. A clustering method sorts them into classes", subtitle = "k-means, k = 4: class means in bold", x = NULL, y = NULL)
md <- data.frame(m = rowMeans(Y), cl = factor(cls))
pc <- ggplot(md, aes(m, fill = cl)) + geom_histogram(binwidth = 2, boundary = 0, color = "white", linewidth = 0.2) +
  scale_fill_manual(values = CL) + base +
  labs(title = "3. But the classes can slice a continuum", subtitle = "Mean score per child, coloured by class: one skewed\ndistribution, no gaps between classes",
       x = "Mean symptom score", y = "Children")
p4 <- (pa | pb | pc) + ann("What clustering trajectories does",
  "Simulated data with one continuous, skewed severity dimension and no true subgroups: a method still returns classes.")
ggsave("figures/m4_trajectories.png", p4, width = 14, height = 4.8, dpi = 300)

# ---- shared 2-D toy data ---------------------------------------------------------------
set.seed(20261122)
mu <- rbind(c(0, 0), c(3.2, 0.4), c(1.4, 3.4))
X <- scale(do.call(rbind, lapply(1:3, \(g) mvrnorm(60, mu[g, ], diag(c(0.9, 0.7))))))
N <- nrow(X); site <- rep(1:6, length.out = N)
km3 <- function(D) { k <- kmeans(D, 3, nstart = 20)$cluster; match(k, unique(k[order(D[, 1])])) }

# ---- m5: perturbations -----------------------------------------------------------------
set.seed(5)
keep <- sort(sample(N, round(0.8 * N)))
half <- site %in% c(1, 3, 5)
noisy <- X + matrix(rnorm(2 * N, 0, 0.35), N)
null <- mvrnorm(N, colMeans(X), cov(X))
pan <- function(df, title, sub) ggplot(df, aes(x, y)) + coord_equal() + bare + labs(title = title, subtitle = sub)
d0 <- data.frame(x = X[, 1], y = X[, 2], cl = factor(km3(X)))
q1 <- pan(d0, "Original", "Fit once on all children") + geom_point(aes(color = cl), size = 1) + scale_color_manual(values = CL)
ds <- data.frame(d0, kept = seq_len(N) %in% keep); ds$cl2 <- NA; ds$cl2[ds$kept] <- km3(X[keep, ])
q2 <- pan(ds, "80% subsample", "Refit on a random 80%;\ngrey crosses left out") +
  geom_point(data = ds[!ds$kept, ], shape = 4, color = "#b0aea8", size = 1.2) +
  geom_point(data = ds[ds$kept, ], aes(color = factor(cl2)), size = 1) + scale_color_manual(values = CL)
dh <- data.frame(d0, half = ifelse(half, "Training sites", "Test sites"))
q3 <- pan(dh, "Site split-half", "Fit on half the sites,\npredict the other half") +
  geom_point(aes(shape = half, color = cl), size = 1.1) + scale_shape_manual(values = c(16, 1)) + scale_color_manual(values = CL)
dn <- data.frame(x = X[, 1], y = X[, 2], xn = noisy[, 1], yn = noisy[, 2], cl = factor(km3(noisy)))
q4 <- pan(dn, "Added noise", "Measurement error added\n(reliability .80 or .90)") +
  geom_segment(aes(xend = xn, yend = yn), color = "#b0aea8", linewidth = 0.25) +
  geom_point(aes(xn, yn, color = cl), size = 1) + scale_color_manual(values = CL)
dl <- data.frame(x = null[, 1], y = null[, 2], cl = factor(km3(null)))
q5 <- pan(dl, "Null data", "Same means and covariances,\nno subgroups: still split") +
  geom_point(aes(color = cl), size = 1) + scale_color_manual(values = CL)
ari <- c(sub = { a <- km3(X)[keep]; adjustedRandIndex(a, ds$cl2[ds$kept]) }, noise = adjustedRandIndex(km3(X), km3(noisy)))
p5 <- (q1 | q2 | q3 | q4 | q5) + ann("Five ways to perturb the data before refitting",
  sprintf("Each perturbed fit is compared with the others (ARI). Here: subsample vs original ARI %.2f, noise vs original %.2f. Simulated data; k-means, k = 3.", ari[["sub"]], ari[["noise"]]))
ggsave("figures/m5_perturbations.png", p5, width = 15, height = 4.4, dpi = 300)

# ---- consensus on the toy data (as in methods_figures.R) -------------------------------
km <- function(k) kmeans(X, k, nstart = 20)$cluster
wd <- function(k) cutree(hclust(dist(X), "ward.D2"), k)
inp <- list(km(2), km(3), km(4), wd(3), wd(4))
A_list <- lapply(inp, cbma_hard_to_prob)
civi <- lapply(A_list, \(A) cbma_civi(X, A, "Calinski_Harabasz"))
w <- cbma_weights(vapply(civi, `[[`, 0, "value"), civi[[1]]$direction)
C <- cbma_consensus(A_list, w)
f <- cbma_ssmf(C, K = 4, n_iter = 3000, seed = 1L)
lab0 <- max.col(f$alloc, ties.method = "first"); ordc <- unique(lab0[order(X[, 1])])
lab <- match(lab0, ordc); alloc <- f$alloc[, ordc, drop = FALSE]; alloc <- alloc / rowSums(alloc)
unc <- cbma_uncertainty(f$alloc)

# ---- m6: network -----------------------------------------------------------------------
Cs <- C; diag(Cs) <- 0; Cs[Cs < 0.6] <- 0
g <- graph_from_adjacency_matrix(Cs, mode = "undirected", weighted = TRUE)
set.seed(6); L <- cmdscale(as.dist(1 - C), k = 2)          # layout: children close when solutions agree
L <- L + matrix(rnorm(2 * N, 0, 0.025), N)                 # separate children with identical rows
el <- as_edgelist(g, names = FALSE)
edges <- data.frame(x = L[el[, 1], 1], y = L[el[, 1], 2], xend = L[el[, 2], 1], yend = L[el[, 2], 2], w = E(g)$weight)
nodes <- data.frame(x = L[, 1], y = L[, 2], cl = factor(lab), unc = unc)
r1 <- ggplot(data.frame(x = X[, 1], y = X[, 2], cl = factor(lab), unc = unc), aes(x, y, color = cl, alpha = 1 - unc)) +
  geom_point(size = 1.8) + scale_color_manual(values = CL) + scale_alpha(range = c(0.3, 1)) + coord_equal() + bare +
  labs(title = "In the data", subtitle = "Consensus classes; faded = uncertain")
r2 <- ggplot() +
  geom_segment(data = edges, aes(x, y, xend = xend, yend = yend, alpha = w), color = "#7a7975", linewidth = 0.2) +
  geom_point(data = nodes, aes(x, y, fill = cl), shape = 21, color = "white", size = 2.4, stroke = 0.3) +
  scale_fill_manual(values = CL) + scale_alpha(range = c(0.05, 0.5)) + coord_equal() + bare +
  labs(title = "As a network", subtitle = "Children placed by how often the (weighted) solutions agree, and linked\nwhen at least 60% put them together")
p6 <- (r1 | r2) + plot_layout(widths = c(1, 1.3)) + ann("The consensus as a network",
  "Simulated data, five inputs (k-means k = 2-4, Ward k = 3-4), Calinski-Harabasz weights. Children the solutions disagree about sit between communities.")
ggsave("figures/m6_network.png", p6, width = 12, height = 6, dpi = 300)

# ---- m7: probabilities -----------------------------------------------------------------
o <- order(lab, -apply(alloc, 1, max))
# one stacked bar per child, its own (modal) class at the bottom
pr <- do.call(rbind, lapply(seq_along(o), function(j) {
  i <- o[j]; cls_ord <- c(lab[i], setdiff(seq_len(ncol(alloc)), lab[i])); p <- alloc[i, cls_ord]
  data.frame(xmin = j - 1, xmax = j, ymin = cumsum(p) - p, ymax = cumsum(p), cl = factor(cls_ord, seq_len(ncol(alloc))))
}))
s1 <- ggplot(pr, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = cl)) + geom_rect() + scale_fill_manual(values = CL, name = "Class") +
  scale_y_continuous(labels = scales::percent, expand = c(0, 0)) + scale_x_continuous(expand = c(0, 0)) +
  base + theme(legend.position = "top", legend.justification = "left", panel.grid = element_blank(), axis.text.x = element_blank()) +
  labs(title = "Each child's class probabilities", subtitle = "One bar per child, sorted by class, then by certainty; the child's own class at the bottom. Mixed bars: children between classes.",
       x = "Children", y = "Probability")
s2 <- ggplot(data.frame(unc = unc, cl = factor(lab)), aes(cl, unc, color = cl)) +
  geom_jitter(width = 0.2, height = 0, size = 1.2, alpha = 0.8) + scale_color_manual(values = CL) + base +
  labs(title = "Uncertainty by class", subtitle = "1 - highest probability", x = "Class", y = "Uncertainty")
p7 <- (s1 | s2) + plot_layout(widths = c(3, 1)) + ann("Soft allocation: clusterBMA gives probabilities, not just labels",
  "Simulated data, same consensus as the network figure.")
ggsave("figures/m7_probabilities.png", p7, width = 13, height = 4.8, dpi = 300)
cat("written\n")
