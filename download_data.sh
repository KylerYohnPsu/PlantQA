#!/bin/bash
set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
RESET='\033[0m'

printf "${BLUE}Beginning data download.  This may take a while...${RESET}\n"
cd data
hf download SyedNazmusSakib/PlantExpertVQA --repo-type dataset --local-dir PlantExpertVQA
printf "${BLUE}Data download complete!${RESET}\n"
