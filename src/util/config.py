from configparser import ConfigParser
from pathlib import Path

class PlantVQAConfig:
    pass

def load_config(file_path: str | Path) -> PlantVQAConfig:
    """
    Load the specified configuration file into a config object
    PARAM:
        file_path: str, Path | The path to the config file to load
    RETURN:
        PlantVQAConfig: The configuration in an easy to use object
    """
    # read the config file
    parser= ConfigParser()
    parser.read(file_path)

    # Make the config object that we will return
    root= PlantVQAConfig()

    # iterate through the config file and read the sections into the config object
    # We do "setattr" that way we can customize the member variables of the config object
    # i.e. you can add a "hello" section to config.ini, then you can access it through config.hello
    for section in parser.sections():
        current_config= root

        for subsection in section.split("."):
            if not hasattr(current_config, subsection):
                setattr(current_config, subsection, PlantVQAConfig())

            current_config= getattr(current_config, subsection)

        for key, val in parser.items(section):
            setattr(current_config, key, val)

    return root

# Load the config file here, that way any file in the project can import it and use it
PROJECT_ROOT= Path(__file__).resolve().parents[2]
CONFIG_PATH= PROJECT_ROOT / "config.ini"

# load the config data, other files can import this object and use it
config= load_config(CONFIG_PATH)