import json
from MuMDIA.utilities.logger import log_info


def modify_config(key, value, config_file="config.json"):
    # Read the config.json file
    with open(config_file, "r") as file:
        config = json.load(file)

    # Modify the config based on the input
    config[key] = value

    # Write the modified config back to the file
    with open(config_file, "w") as file:
        json.dump(config, file, indent=4)

    log_info(f"Updated the config file with {key}: {value}")
