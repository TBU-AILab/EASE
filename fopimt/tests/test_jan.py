from .test import Test
from ..solutions.solution import Solution
from ..utils.import_utils import dynamic_install
import re
from typing import Dict

import ast


class TestJanSOC(Test):
    """
    Test designed for correct Jan SOC output.
    """
    
    def _init_params(self):
        super()._init_params()
        self._error_msg = ""
        self._user_msg = ""
        self._error_msg_template = \
            "The solution does not fulfill the template. This is the reason: {0}."
        self._user_msg_template = \
            "The solution does not fulfill the template. This is the reason: {0}."

    ####################################################################
    #########  Public functions
    ####################################################################
    def test(self, solution: Solution) -> bool:
        """
        This function tests the proposed solution for Jans' super duper security task
        :param solution: Solution to test
        :return: Returns test result (True = test was OK, False = test was not OK)
        """
        self._result = True

        text = solution.get_input()
        if not isinstance(text, str) or not text.strip():
            self._result = False
            self._error_msg = self._error_msg_template.format("Generated text must be a non-empty string.")
            self._user_msg = self._user_msg_template.format("Generated text must be a non-empty string.")
            return self._result

        out: Dict[str, str] = {}

        for tag in ("SYSTEM_PROMPT", "SUMMARY_PROMPT", "MITIGATE_PROMPT"):
            pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
            matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)

            if len(matches) == 0:
                self._result = False
                self._error_msg += self._error_msg_template.format(f"Missing tag <{tag}>...</{tag}> in master output.")
                self._user_msg += self._user_msg_template.format(f"Missing tag <{tag}>...</{tag}> in master output.")
            if len(matches) > 1:
                self._result = False
                self._error_msg += self._error_msg_template.format(f"Duplicate tag <{tag}> found ({len(matches)} occurrences).")
                self._user_msg += self._user_msg_template.format(f"Duplicate tag <{tag}> found ({len(matches)} occurrences).")

            content = matches[0].strip()
            if not content:
                self._result = False
                self._error_msg += self._error_msg_template.format(f"Tag <{tag}> is present but empty.")
                self._user_msg += self._user_msg_template.format(f"Tag <{tag}> is present but empty.")

            key = tag.replace("_PROMPT", "").lower()  # system/summary/mitigate
            out[key] = content

        if self._result:
            self._error_msg = "Test:Jan: OK"
            self._user_msg = "Test:Jan: OK"

        return self._result


    @classmethod
    def get_short_name(cls) -> str:
        return "test.jan"

    @classmethod
    def get_long_name(cls) -> str:
        return "Jans' super duper security task test"

    @classmethod
    def get_description(cls) -> str:
        return "Test the solution for Jans' super duper security task."

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': {'text'},
            'output': set()
        }