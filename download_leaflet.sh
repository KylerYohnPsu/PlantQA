#!/bin/bash
set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
RESET='\033[0m'

printf "${BLUE}Beginning data download.  This may take a while...${RESET}\n"
cd data
printf "${RED} Log into hugging face to be able to download LeafBranch dataset"
hf auth login
hf download enalis/LeafBench --repo-type dataset --local-dir PlantLeafLet
printf "${BLUE}Data download complete!${RESET}\n"
printf "${BLUE} Running python script to extract parquet files${RESET}\n"
cd ..
python src/read_parquets.py
printf "${BLUE}Finished extracting parquet files.${RESET}\n"