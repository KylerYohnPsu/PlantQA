#!/bin/bash

# stop executing the script if one of the steps fails
set -e

#colors to make printing nice and pretty for the setup file
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
RESET='\033[0m'


# setup venv
printf "${BLUE}Building python virtual enviornment...${RESET}\n"
python3 -m venv venv
source venv/bin/activate
printf "${GREEN}Successfully built python virtual environment!${RESET}\n"

# install python reqs
printf "${BLUE}Installing needed python libraries...${RESET}\n"
pip install -r requirements.txt
printf "${GREEN}Successfully installed python libraries!${RESET}\n"

# install library specific reqs here so we dont have to install them in the code itself
printf "${BLUE}Installing specific library requirements...${RESET}\n"
python3 -c "import nltk; nltk.download('stopwords')"
printf "${GREEN}Successfully installed specific library requirements!${RESET}\n"

# verify installation via pytest
printf "${BLUE}Running pytest to verify everything installed correctly...${RESET}\n"
python3 -m pytest -v
printf "${BLUE}Pytest finished running, please vrerify no major tests failed...${RESET}\n"
printf "${BLUE}If nothing failed in a major way, then the setup succeeded!${RESET}\n"

