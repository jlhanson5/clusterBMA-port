# =============================================================================
# tools/forking_paths.R: a "garden of forking paths" figure. Each analytic decision is a
# ring of branches; each universe is a leaf on the outside, coloured by a value (e.g. its
# effect estimate). Input is one row per universe (aggregates only).
#   source("tools/forking_paths.R")
#   draw_forking(df, levels, value, labels, title, subtitle, file, value_name)
#     levels  column names, innermost decision first
#     labels  named list: for each level, a named character vector of display labels
# =============================================================================

suppressPackageStartupMessages(library(ggplot2))

draw_forking <- function(df, levels, value, labels = list(), title = NULL, subtitle = NULL, file,
                         value_name = value, label_depth = 1:2, ring_names = NULL, width = 11, height = 11.5,
                         breaks = waiver(), gap = 0) {
  INK <- "#0b0b0b"; INK2 <- "#52514e"
  df <- df[do.call(order, unname(as.list(df[levels]))), , drop = FALSE]
  L <- length(levels)
  # leaf positions, with a small gap between first-level branches
  first <- as.character(df[[levels[1]]])
  df$x <- seq_len(nrow(df)) + gap * (match(first, unique(first)) - 1)
  key <- function(d) if (d == 0) rep("root", nrow(df)) else do.call(paste, c(unname(as.list(df[levels[seq_len(d)]])), sep = "|"))
  nodes <- do.call(rbind, lapply(0:L, function(d) {
    k <- key(d); x <- tapply(df$x, k, mean)
    lab <- if (d == 0) "" else as.character(df[[levels[d]]][match(names(x), k)])
    parent <- if (d == 0) NA else key(d - 1)[match(names(x), k)]
    data.frame(key = names(x), depth = d, x = as.vector(x), lab = lab, parent = parent, n = as.vector(table(k)[names(x)]))
  }))
  edges <- merge(nodes[nodes$depth > 0, ], nodes[, c("key", "x", "depth")], by.x = "parent", by.y = "key", suffixes = c("", "_p"))
  # elbow: along the parent's ring to the child's angle, then outward
  arcs <- do.call(rbind, lapply(seq_len(nrow(edges)), function(i) {
    e <- edges[i, ]; xs <- seq(e$x_p, e$x, length.out = 20)
    data.frame(id = i, x = xs, y = e$depth_p, d = e$depth)
  }))
  radial <- data.frame(x = edges$x, y = edges$depth_p, yend = edges$depth, d = edges$depth)
  leaves <- data.frame(x = df$x, y = L + 0.55, v = df[[value]])
  lab_nodes <- nodes[nodes$depth %in% label_depth, ]
  lab_nodes$text <- vapply(seq_len(nrow(lab_nodes)), function(i) {
    lv <- levels[lab_nodes$depth[i]]; m <- labels[[lv]]; t <- lab_nodes$lab[i]
    if (!is.null(m) && t %in% names(m)) m[[t]] else t
  }, "")
  pad <- if (length(ring_names)) max(3 * gap, nrow(df) * 0.06) else max(1, nrow(df) * 0.01)
  lw <- c(1.1, 0.8, 0.55, 0.4, 0.3, 0.25, 0.2)[pmin(seq_len(L), 7)]
  p <- ggplot() +
    geom_path(data = arcs, aes(x, y, group = id, linewidth = factor(d)), color = "#7a8fae", lineend = "round") +
    geom_segment(data = radial, aes(x = x, xend = x, y = y, yend = yend, linewidth = factor(d)), color = "#7a8fae", lineend = "round") +
    geom_segment(data = leaves, aes(x = x, xend = x, y = L, yend = L + 0.9, color = v), linewidth = 0.7) +
    geom_label(data = lab_nodes, aes(x, depth, label = text), size = ifelse(lab_nodes$depth == 1, 3.4, 2.6),
               fontface = ifelse(lab_nodes$depth == 1, "bold", "plain"), linewidth = 0, fill = "white", color = INK,
               label.padding = unit(0.12, "lines")) +
    scale_linewidth_manual(values = setNames(lw, seq_len(L)), guide = "none") +
    scale_color_gradientn(colours = c("#c9ddf5", "#5b9be6", "#2563b8", "#0d3266", "#04101f"), name = value_name,
                          trans = "log", breaks = breaks) +
    coord_polar(start = 0) + scale_y_continuous(limits = c(-0.6, L + 1)) +
    scale_x_continuous(limits = c(0.5, max(df$x) + 0.5 + pad), expand = c(0, 0)) +
    labs(title = title, subtitle = subtitle) +
    (if (length(ring_names)) annotate("text", x = max(df$x) + 0.5 + pad / 2, y = c(seq_len(L), L + 0.45), label = c(ring_names, "universes"),
                                      size = 2.7, color = INK2, fontface = "italic") else NULL) +
    theme_void(base_size = 11) +
    theme(plot.title = element_text(face = "bold", size = 14, color = INK), plot.subtitle = element_text(color = INK2),
          legend.position = "bottom", legend.key.width = unit(2.2, "cm"), legend.title = element_text(color = INK2, vjust = 0.8),
          plot.background = element_rect(fill = "white", color = NA), plot.margin = margin(12, 12, 12, 12))
  ggsave(file, p, width = width, height = height, dpi = 300)
  invisible(p)
}
