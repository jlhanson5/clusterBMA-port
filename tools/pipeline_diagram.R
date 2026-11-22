# =============================================================================
# tools/pipeline_diagram.R: draw an analysis pipeline as a row of boxes with arrows and an
# optional band of checks underneath. Text only, no data.
#   source("tools/pipeline_diagram.R")
#   draw_pipeline(stages, checks, title, subtitle, file)
#     stages  list of list(title = , lines = c(...)), drawn left to right
#     checks  character vector, drawn as a band under the stages (or NULL)
# =============================================================================

suppressPackageStartupMessages(library(ggplot2))

draw_pipeline <- function(stages, checks = NULL, title = NULL, subtitle = NULL, file,
                          width = 15, height = 7.5, wrap = 30, fills = NULL) {
  INK <- "#0b0b0b"; INK2 <- "#52514e"
  # units are inches: the figure height is computed from the text
  n <- length(stages); gap <- 0.45; bw <- (width - 0.4 - gap * (n - 1)) / n
  fills <- fills %||% colorRampPalette(c("#e8f0fb", "#c9ddf5"))(n)
  body <- vapply(stages, \(s) paste(vapply(s$lines, \(l) paste(strwrap(l, wrap, prefix = "", initial = "• ", exdent = 2), collapse = "\n"), ""), collapse = "\n"), "")
  nl <- vapply(strsplit(body, "\n"), length, 0L)
  top <- 0; head_h <- 0.5; line_h <- 0.145
  bot <- top - head_h - max(nl) * line_h - 0.25
  x0 <- (seq_len(n) - 1) * (bw + gap)
  boxes <- data.frame(xmin = x0, xmax = x0 + bw, ymin = bot, ymax = top, fill = fills)
  heads <- data.frame(x = x0 + 0.1, y = top - 0.12, label = vapply(seq_len(n), \(i) sprintf("%d. %s", i, stages[[i]]$title), ""))
  texts <- data.frame(x = x0 + 0.1, y = top - head_h, label = body)
  arrows <- data.frame(x = x0[-n] + bw + 0.06, xend = x0[-1] - 0.06, y = (top + bot) / 2)
  p <- ggplot() +
    geom_rect(data = boxes, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax), fill = boxes$fill, color = "#2563b8", linewidth = 0.4) +
    geom_text(data = heads, aes(x, y, label = label), hjust = 0, vjust = 1, fontface = "bold", size = 3.6, color = INK, lineheight = 0.95) +
    geom_text(data = texts, aes(x, y, label = label), hjust = 0, vjust = 1, size = 2.85, color = INK, lineheight = 1.05) +
    geom_segment(data = arrows, aes(x = x, xend = xend, y = y, yend = y), arrow = arrow(length = unit(2.5, "mm"), type = "closed"),
                 color = "#2563b8", linewidth = 0.6)
  ylo <- bot
  if (length(checks)) {
    cb_top <- bot - 0.35
    ctext <- paste(vapply(checks, \(l) paste(strwrap(l, wrap * n * 0.5, initial = "• ", exdent = 2), collapse = "\n"), ""), collapse = "\n")
    half <- ceiling(length(checks) / 2)
    cols <- list(checks[seq_len(half)], checks[-seq_len(half)])
    ch <- max(vapply(cols, \(cc) sum(vapply(cc, \(l) length(strwrap(l, wrap * 2.2)), 0L)), 0L))
    cb_bot <- cb_top - 0.5 - ch * line_h - 0.15
    xr <- max(x0) + bw
    p <- p +
      annotate("rect", xmin = 0, xmax = xr, ymin = cb_bot, ymax = cb_top, fill = "#f6f5f3", color = "#7a7975", linewidth = 0.4) +
      annotate("text", x = 0.1, y = cb_top - 0.12, label = "Robustness checks", hjust = 0, vjust = 1, fontface = "bold", size = 3.6, color = INK)
    for (j in 1:2) if (length(cols[[j]])) {
      lab <- paste(vapply(cols[[j]], \(l) paste(strwrap(l, wrap * 2.2, initial = "• ", exdent = 2), collapse = "\n"), ""), collapse = "\n")
      p <- p + annotate("text", x = 0.1 + (j - 1) * xr / 2, y = cb_top - 0.5, label = lab, hjust = 0, vjust = 1, size = 2.85, color = INK, lineheight = 1.05)
    }
    ylo <- cb_bot
  }
  p <- p + coord_cartesian(xlim = c(-0.05, max(x0) + bw + 0.05), ylim = c(ylo - 0.05, top + 0.05), expand = FALSE) +
    labs(title = title, subtitle = subtitle) +
    theme_void(base_size = 11) +
    theme(plot.title = element_text(face = "bold", size = 14, color = INK, margin = margin(b = 4)),
          plot.subtitle = element_text(color = INK2, margin = margin(b = 10)),
          plot.background = element_rect(fill = "white", color = NA), plot.margin = margin(14, 14, 14, 14))
  height <- (top + 0.05) - (ylo - 0.05) + 1.25   # + title, subtitle and margins
  ggsave(file, p, width = width, height = height, dpi = 300)
  invisible(p)
}
