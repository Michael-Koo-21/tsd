#!/usr/bin/env bash
# Download ACS PUMS 2024 1-Year California person file from Census Bureau.
#
# Usage:
#   bash scripts/download_data.sh
#
# The script downloads the CSV zip, extracts psam_p06.csv to data/raw/,
# and prints a checksum for verification.

set -euo pipefail

DATA_DIR="data/raw"
URL="https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/csv_pca.zip"
ZIP_FILE="${DATA_DIR}/csv_pca.zip"
TARGET_FILE="${DATA_DIR}/psam_p06.csv"

mkdir -p "${DATA_DIR}"

if [ -f "${TARGET_FILE}" ]; then
    echo "File already exists: ${TARGET_FILE}"
    echo "Delete it first if you want to re-download."
    exit 0
fi

echo "Downloading ACS PUMS 2024 1-Year California person file..."
curl -L -o "${ZIP_FILE}" "${URL}"

echo "Extracting psam_p06.csv..."
unzip -o "${ZIP_FILE}" "psam_p06.csv" -d "${DATA_DIR}"

rm -f "${ZIP_FILE}"

echo ""
echo "Download complete: ${TARGET_FILE}"
echo "SHA-256: $(shasum -a 256 "${TARGET_FILE}" | cut -d' ' -f1)"
echo "File size: $(du -h "${TARGET_FILE}" | cut -f1)"
