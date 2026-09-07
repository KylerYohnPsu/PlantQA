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
printf "${BLUE}Unzipping image folders...${RESET}\n"
cd PlantExpertVQA
unzip -o images_part1
unzip -o images_part2
unzip -o images_part3
unzip -o images_part4
printf "${BLUE}Finished unzipping the images.  Data should be good to go now.${RESET}\n"
