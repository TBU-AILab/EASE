import json
from typing import Optional


class ResourceProvider:

    def __init__(self, filepath: Optional[str] = None):
        """
        Resource provider class.
        Loads prompts from DB or disk file (specified by name).
        Loads configs for Magic.
        Arguments:
            filepath: str       -- (Primary resource) Path to the config file. Supports only JSON.
        """
        self._data: dict = {}
        if filepath is not None:
            try:
                f = open(filepath)
                self._data = json.load(f)
            except OSError:
                print("Could not open/read file:", filepath)
        else:
            raise RuntimeError('Resource provider not configured.')

    ####################################################################
    #########  Public functions
    ####################################################################
    def get_prompt_init(self) -> str:
        return self._data.get('prompts', {}).get('init', "")

    def get_prompt_system(self) -> str:
        return self._data.get('prompts', {}).get('system', "")

    def get_prompt_repeating(self) -> dict:
        return self._data.get('prompts', {}).get('repeating', {})

    def get_llm_type(self) -> dict:
        return self._data.get('llm', {})

    def get_specific(self, keyword: str) -> dict:
        return self._data.get(keyword, {})
    ####################################################################
    #########  Private functions
    ####################################################################

