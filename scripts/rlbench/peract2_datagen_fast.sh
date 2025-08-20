# DATA_PATH=peract2_raw/
# ZARR_PATH=zarr_datasets/peract2/

# # Ultra-fast, just download our data!
# cd ${ZARR_PATH}
# wget https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/peract2.zip
# unzip peract2.zip
# rm -rf peract2.zip

# # Download the test seeds
# CURR_DIR=$(pwd)
# cd ${DATA_PATH}
# wget https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/peract2_test.zip
# unzip peract2_test.zip
# rm peract2_test.zip
# cd "$CURR_DIR"
# # Good to go!


#!/bin/bash

# Paths
DATA_PATH="peract2_raw"
ZARR_PATH="zarr_datasets/peract2"

# Step 1: Create required directories if they don't exist
mkdir -p "${ZARR_PATH}"
mkdir -p "${DATA_PATH}"

# Step 2: Download zipped Zarr dataset
cd "${ZARR_PATH}" || { echo "Failed to enter ${ZARR_PATH}"; exit 1; }

echo "Downloading peract2.zip from Hugging Face..."
wget --no-check-certificate https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/peract2.zip -O peract2.zip

if [ -f "peract2.zip" ]; then
    echo "Unzipping peract2.zip..."
    unzip -q peract2.zip
    rm -rf peract2.zip
else
    echo "Download failed: peract2.zip not found."
    exit 1
fi

# Step 3: Download the test seeds
cd "../../${DATA_PATH}" || { echo "Failed to enter ${DATA_PATH}"; exit 1; }

echo "Downloading peract2_test.zip from Hugging Face..."
wget --no-check-certificate https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/peract2_test.zip -O peract2_test.zip

if [ -f "peract2_test.zip" ]; then
    echo "Unzipping peract2_test.zip..."
    unzip -q peract2_test.zip
    rm peract2_test.zip
else
    echo "Download failed: peract2_test.zip not found."
    exit 1
fi

echo "✅ All data downloaded and extracted successfully. You're good to go!"
