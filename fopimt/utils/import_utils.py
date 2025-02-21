import logging
import importlib
import subprocess
import sys
import requests
import json
import os


# File path where the mapping dictionary will be stored
MAPPING_FILE_PATH = os.path.join("fopimt" ,"utils" ,"module_to_pip_map.json")


# Load the mapping dictionary from a file if it exists, otherwise use a default mapping
def load_mapping():
    if os.path.exists(MAPPING_FILE_PATH):
        with open(MAPPING_FILE_PATH, 'r') as file:
            return json.load(file)
    else:
        # Default initial mapping
        return {}


# Save the updated mapping dictionary to a file
def save_mapping(mapping):
    with open(MAPPING_FILE_PATH, 'w') as file:
        json.dump(mapping, file, indent=4)


def dynamic_import(module_name, specific_part=None, alias=None, exec_globals=None):
    # Function for dynamic import of module into code
    # Purposely created for Python Solutions for tests and evaluators

    if exec_globals is None:
        exec_globals = globals()  # Default to global namespace if none provided

    try:
        if specific_part:
            # Import specific part of the module
            module = __import__(module_name, fromlist=[specific_part])
            exec_globals[alias if alias else specific_part] = getattr(module, specific_part)
            logging.debug(f"Utils:Import: Imported {specific_part} from {module_name} as {alias if alias else specific_part}")
        else:
            # Import the whole module
            exec_globals[alias if alias else module_name] = __import__(module_name)
            logging.debug(f"Utils:Import: Imported {module_name} as {alias if alias else module_name}")
    except ImportError:
        logging.error(f"Utils:Import: {module_name} could not be imported.")


def dynamic_install(module_name) -> bool:
    """Try to import a module, install if not found, and update the mapping."""

    # Initialize the mapping dictionary
    module_to_pip_map = load_mapping()

    try:
        # Try importing the module
        importlib.import_module(module_name)
        logging.debug(f"Utils:Install: {module_name} is already installed.")
    except ImportError:
        # Check if the module is in the mapping dictionary
        pip_package = module_to_pip_map.get(module_name)

        if not pip_package:
            # Query PyPI if the module is not in the known mappings
            pip_package = query_pypi(module_name)
            if pip_package:
                # If PyPI finds the package, add it to the mapping for future use
                module_to_pip_map[module_name] = pip_package

                logging.debug(f"Utils:Install: Found {module_name} on PyPI as {pip_package}. Adding to mappings.")
            else:
                # Fallback: assume module_name is the pip package name
                logging.debug(f"Utils:Install: {module_name} module not found in module mapping or PyPI. Is it spelled correctly?")
                raise ImportError(f"{module_name} module not found in module mapping or PyPI. Is it spelled correctly?")

        # Install the package via pip
        logging.debug(f"Utils:Install: Installing {pip_package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_package])
            save_mapping(module_to_pip_map)
        except Exception as e:
            logging.error(f"Utils:Install: {module_name} module cannot be installed. Is it spelled correctly? {repr(e)}")
            raise e



def query_pypi(module_name):
    """Query PyPI to find the pip package name for a given module."""
    url = f"https://pypi.org/pypi/{module_name}/json"
    response = requests.get(url)

    if response.status_code == 200:
        # Successfully found the package, return its official pip package name
        package_info = response.json()
        return package_info['info']['name']
    else:
        # Package not found on PyPI
        return None