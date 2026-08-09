#!/bin/bash

# Download ML-1M dataset
echo "Downloading ML-1M dataset..."
curl -L -o ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip

# Download Gowalla dataset
echo "Downloading Gowalla dataset..."
curl -L -o loc-gowalla_totalCheckins.txt.gz https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz
curl -L -o user_list.txt https://raw.githubusercontent.com/xiangwang1223/neural_graph_collaborative_filtering/master/Data/gowalla/user_list.txt
curl -L -o item_list.txt https://raw.githubusercontent.com/xiangwang1223/neural_graph_collaborative_filtering/master/Data/gowalla/item_list.txt

# Extract files
echo "Extracting files..."
unzip -o ml-1m.zip
gunzip -f loc-gowalla_totalCheckins.txt.gz

# Preprocess Gowalla data
echo "Preprocessing Gowalla data..."
python3 ../Code/preprocess.py

echo "All data is ready!"
