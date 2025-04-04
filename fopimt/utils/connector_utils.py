import json
import os
import requests
import logging

# models = {
#     "llm.mock": [
#         {
#             "url": "",
#             "models": [
#                 "Meta: random search",
#                 "2048: left and slow",
#                 "ModernTV: video transitions"
#             ]
#         }
#     ],
#     "llm.ollama": [
#         {
#             "url": "http://localhost:11434",
#             "models": [
#                 "deepseek-r1:latest",
#                 "llama3.1:latest"
#             ]
#          },
#     ],
#     "llm.openai": [
#         {
#             "url": "https://api.openai.com/v1/models",
#             "models": [
#                 'gpt-4o',
#                 'chatgpt-4o-latest',
#                 'gpt-4o-mini',
#                 'o1',
#                 'o1-mini',
#                 'o3-mini'
#             ]
#         }
#     ],
#     "llm.openaiimage": [
#         {
#             "url": "",
#             "models": [
#                 'dall-e-3',
#                 'dall-e-2'
#             ]
#         }
#     ],
#     "llm.openaivision": [
#         {
#             "url": "",
#             "models": [
#                 'gpt-4o',
#                 'gpt-4o-mini',
#                 'gpt-4-turbo',
#                 'o1'
#             ]
#         }
#     ],
#     "llm.claude": [
#         {
#             "url": "https://api.anthropic.com/v1/models",
#             "models": [
#                 'claude-3-7-sonnet-20250219',
#                 'claude-3-5-sonnet-20241022',
#                 'claude-3-5-haiku-20241022'
#                 'claude-3-opus-20240229',
#                 'claude-3-haiku-20240307'
#             ]
#         }
#     ]
# }

#######################################################
# All services
#######################################################


def write_json(data):
    with open(os.path.join("fopimt", "utils", "available_models.json"), 'w') as json_file:
        json.dump(data, json_file, indent=2)


def get_available_models(connector_shortname: str) -> dict:
    """
    Lists all available models for specified llmconnector by its short name
    Reads the data from available_models.json
    The output is a dict with two keys:
        - model_names
        - model_longnames (for enum description in parameters, contains url of the model as well)
    """

    with open(os.path.join("fopimt", "utils", "available_models.json"), "r") as f:
        models = {}
        data = json.load(f)

        model_longnames = []
        model_names = []

        for model_instance in data[connector_shortname]:

            instance_url = model_instance["url"]
            instance_model_names = model_instance["models"]

        for model_name in instance_model_names:
            model_names.append(model_name)
            model_longnames.append(f"{model_name} (url: {instance_url})")

        models = {
            "model_names": model_names,
            "model_longnames": model_longnames
        }

        return models


def update_service_models(connector_shortname: str, connector_getupdate, api_key: str = None):
    """
    Updates models for single connector defined by its shortname, connector_getupdate  function and with optional api_key
    """
    with open(os.path.join("fopimt", "utils", "available_models.json"), "r") as f:
        data = json.load(f)
        ollama_models = data[connector_shortname]
        for model in ollama_models:
            url = model["url"]
            model["models"] = connector_getupdate(url=url, api_key=api_key)

        write_json(data)


def update_all_models():
    """
    This function updates the json config file for all services (if their update function is called below)
    """
    update_ollama_models() #Ollama models update


#######################################################
# Ollama
#######################################################


def get_ollama(url: str, api_key: str = None) -> list[str]:
    url = f"{url}/api/tags"
    try:
        response = requests.get(url, timeout=3)
        response.raise_for_status()
        data = response.json()
        models = [model["name"] for model in data.get("models", [])]
        return models
    except (requests.RequestException, ValueError, KeyError) as e:
        logging.error(f"Could not fetch models from API. Reason: {e}")
        return []


def update_ollama_models():
    update_service_models(connector_shortname="llm.ollama", connector_getupdate=get_ollama)



#######################################################
# OpenAI - Unfinished for inexistence of model typization
#######################################################


def get_openai(url: str, api_key: str = None) -> list[str]:

    headers = {
        "Authorization": f"Bearer {api_key or os.getenv('OPENAI_API_KEY')}"
    }

    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        print(data)
        models = [model["id"] for model in data.get("data", [])]
        return models
    except (requests.RequestException, ValueError, KeyError) as e:
        print(f"Error fetching OpenAI models: {e}")
        return []


def update_openai_models(api_key: str = None):
    update_service_models(connector_shortname="llm.openai", connector_getupdate=get_openai, api_key=api_key)

#######################################################
# Anthropic Claude
#######################################################