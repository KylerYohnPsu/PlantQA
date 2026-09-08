'''
pull plant data down from EOL and save it to a local file
'''

import time

import requests
import selenium
from selenium.webdriver.common.by import By
from pathlib import Path


def find_path():
    """
    Find the path to the data directory
    """
    curr_dir = Path(__file__).parent
    parent_dir = curr_dir.parent
    data_dir = parent_dir / "data"
    return data_dir


def download_USDA_plant_list():
    """
    Utilize USDA list to get list of plant names store as csv in DATA
    """
    data_dir = find_path()
    options = selenium.webdriver.ChromeOptions()
    prefs = {"download.default_directory": str(data_dir)}
    options.add_experimental_option("prefs", prefs)
    driver = selenium.webdriver.Chrome(options=options)
    driver.get("https://plants.sc.egov.usda.gov/characteristics-search")
    time.sleep(5) 
    modal_button = driver.find_element(By.PARTIAL_LINK_TEXT, "Download ")
    assert modal_button is not None, "modal button not found"
    modal_button.click()
    time.sleep(5) 
    download_button = driver.find_element(By.PARTIAL_LINK_TEXT, "SearchResults.csv")
    assert download_button is not None, "download button not found"
    download_button.click()
    time.sleep(5) 


if __name__ == "__main__":
    download_USDA_plant_list()