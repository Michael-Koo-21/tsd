#!/usr/bin/env Rscript
# Synthpop Generator - Standalone R Script
#
# Usage: Rscript synthpop_generate.R <input_csv> <output_csv> <n_samples> <seed>
#
# This script is called from Python via subprocess to generate synthetic data
# using the synthpop package.

# Load required library
suppressPackageStartupMessages({
  if (!require("synthpop", quietly = TRUE)) {
    cat("ERROR: synthpop package not installed\n")
    cat("Install with: install.packages('synthpop')\n")
    quit(status = 1)
  }
})

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 4) {
  cat("Usage: Rscript synthpop_generate.R <input_csv> <output_csv> <n_samples> <seed>\n")
  quit(status = 1)
}

input_file <- args[1]
output_file <- args[2]
n_samples <- as.integer(args[3])
seed <- as.integer(args[4])

cat(strrep("=", 70), "\n")
cat("Synthpop Generator (R)\n")
cat(strrep("=", 70), "\n\n")

cat("Parameters:\n")
cat(sprintf("  Input: %s\n", input_file))
cat(sprintf("  Output: %s\n", output_file))
cat(sprintf("  N samples: %d\n", n_samples))
cat(sprintf("  Random seed: %d\n", seed))
cat("\n")

# Set random seed for reproducibility
set.seed(seed)

# Read input data
cat("Reading training data...\n")
train_data <- read.csv(input_file, stringsAsFactors = TRUE)

cat(sprintf("  Loaded %d rows, %d columns\n", nrow(train_data), ncol(train_data)))
cat(sprintf("  Columns: %s\n", paste(colnames(train_data), collapse=", ")))
cat("\n")

# Generate synthetic data using synthpop
cat("Generating synthetic data with synthpop...\n")
cat("  Method: CART (default)\n")
cat("  Minbucket: 5\n")
cat("\n")

start_time <- Sys.time()

# Run synthpop
# Using method="cart" with minbucket=5 as specified in study design
synth_result <- syn(
  data = train_data,
  method = "cart",
  m = 1,  # Generate 1 synthetic dataset
  k = n_samples,  # Number of records to generate
  seed = seed,
  cart.minbucket = 5,
  print.flag = FALSE  # Suppress verbose output
)

end_time <- Sys.time()
elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))

cat(sprintf("  ✓ Generation complete in %.1f seconds\n", elapsed))
cat("\n")

# Extract synthetic data
synthetic_data <- synth_result$syn

cat("Validating synthetic data...\n")
cat(sprintf("  Generated %d rows, %d columns\n", nrow(synthetic_data), ncol(synthetic_data)))

# Check schema matches
if (!all(colnames(synthetic_data) == colnames(train_data))) {
  cat("  ✗ WARNING: Column mismatch!\n")
  cat(sprintf("    Expected: %s\n", paste(colnames(train_data), collapse=", ")))
  cat(sprintf("    Got: %s\n", paste(colnames(synthetic_data), collapse=", ")))
} else {
  cat("  ✓ Schema matches\n")
}

if (nrow(synthetic_data) == n_samples) {
  cat(sprintf("  ✓ Row count correct (%d)\n", n_samples))
} else {
  cat(sprintf("  ✗ WARNING: Expected %d rows, got %d\n", n_samples, nrow(synthetic_data)))
}

# Write output
cat("\nWriting synthetic data to CSV...\n")
write.csv(synthetic_data, output_file, row.names = FALSE)

cat(sprintf("  ✓ Saved to: %s\n", output_file))
cat("\n")

cat(strrep("=", 70), "\n")
cat("✓ Synthpop generation complete\n")
cat(strrep("=", 70), "\n")

# Exit successfully
quit(status = 0)
