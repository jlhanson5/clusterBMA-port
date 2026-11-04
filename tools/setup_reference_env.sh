#!/bin/zsh
# Throwaway environment for tools/make_reference.R. Everything goes in .fixture_env/
# (gitignored): delete that folder to remove it. Your own R library is not touched.
# On arm64 macOS, TF 2.11 is published as tensorflow-macos, so that is what gets pinned.
# clusterBMA hard-codes the conda environment name (R/clusterBMA_use_condaenv.R:8).
set -e
cd "$(dirname "$0")/.."
E="$PWD/.fixture_env"; mkdir -p "$E/rlib"
curl -fsSL -o "$E/miniforge.sh" https://github.com/conda-forge/miniforge/releases/download/24.11.3-2/Miniforge3-24.11.3-2-MacOSX-arm64.sh
bash "$E/miniforge.sh" -b -p "$E/miniforge"
"$E/miniforge/bin/conda" create -y -q -n clusterBMA-pyenv_newpkgs python=3.10.9
"$E/miniforge/envs/clusterBMA-pyenv_newpkgs/bin/pip" install -q "tensorflow-macos==2.11.0" "numpy<1.24"
R_LIBS_USER="$E/rlib" Rscript --vanilla -e '
  lib <- Sys.getenv("R_LIBS_USER")
  install.packages(c("reticulate", "tensorflow", "Hmisc", "pheatmap", "ClusterR", "clusterGeneration"),
                   lib = lib, repos = "https://cloud.r-project.org")'
R_LIBS_USER="$E/rlib" R CMD INSTALL --no-docs --no-test-load -l "$E/rlib" reference/clusterBMA
