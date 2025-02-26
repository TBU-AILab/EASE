import os
import json
import numpy as np
from typing import Optional, Any
import datetime

import pandas as pd
from pydantic import BaseModel

from ..modul import Modul
from ..message import Message
from ..loader import Parameter, PrimitiveType


def serialize_unserializable(obj):
    """
    Function for problems with serialization of various objects
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, bytes):
        return obj.decode("utf-8")
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)  # Fallback: convert to string if no other cases match


class SolutionAPI(BaseModel):
    msg_id: str
    fitness: float | None
    feedback: str | None
    metadata: Optional[dict[str, str]]


class Solution(Modul):

    def _init_params(self):
        """
        General Solution class. Defines _input and _fitness.
        """
        self._input = None
        self._fitness: float = -1
        self._prefix = ""
        self._suffix = ""
        self._path = ""
        self._path_meta = ""
        self._metadata = {}
        self._feedback = ""
        self._msg_id: str = ""

    def __str__(self):
        # TODO make it more pretty?
        return str(self._input)

    ####################################################################
    #########  Public functions
    ####################################################################
    def get_API(self) -> SolutionAPI:

        # TODO - fix this shit
        new_meta = dict()
        for m in self._metadata.keys():
            new_meta[m] = str(self._metadata[m])

        return SolutionAPI(
            msg_id=self._msg_id,
            fitness=self._fitness,
            feedback=self._feedback,
            metadata=new_meta,
        )

    def export(self, dir: str, id: str) -> None:
        # export solution itself (code, text, ...)
        file_name = self._prefix + id + self._suffix
        self._path = os.path.join(dir, file_name)
        file = open(self._path, "w", encoding="utf-8", newline="")
        file.write(self._input)
        file.close()

    def export_meta(self) -> None:
        # export metadata of the solution
        self._path_meta = self._path + ".dat"
        with open(self._path_meta, "w") as outfile:
            # json.dump(self._metadata, outfile, default=lambda df: json.loads(df.to_json()))
            json.dump(
                self._metadata, outfile, default=serialize_unserializable
            )

    def set_fitness(self, fitness: float) -> None:
        """
        Set the fitness value of the solution.
        Arguments:
            fitness: float  -- Fitness value to be stored.
        """
        self._fitness = fitness

    def get_fitness(self) -> float:
        """
        Returns the fitness value of the solution.
        """
        return self._fitness

    def get_input(self) -> Optional[str]:
        """
        Returns input of the solution.
        Type not defined.
        """
        return self._input

    def get_path(self) -> str:
        """
        Returns path of the exported solution.
        """
        return self._path

    def get_path_metadata(self) -> str:
        return self._path_meta

    def get_input_from_msg(self, msg: Message):
        """
        Extracts input of the solution from Message.
        Arguments:
            msg: Message    -- Message from which input will be extracted.
        """
        raise NotImplementedError(
            "The function for getting an input from message content has to be implemented"
        )

    def set_suffix(self, suffix: str) -> None:
        self._suffix = suffix

    def set_prefix(self, prefix: str) -> None:
        self._prefix = prefix

    def get_metadata(self):
        return self._metadata

    def add_metadata(self, name: str, value):
        self._metadata[name] = value

    def get_feedback(self) -> str:
        """

        :return:
        """
        return self._feedback

    def set_feedback(self, msg: str):
        """

        :param msg:
        :return:
        """
        self._feedback = msg

    def set_message_id(self, msg_id: str):
        self._msg_id = msg_id

    def get_message_id(self) -> str:
        return self._msg_id

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        return {
            "prefix": Parameter(
                short_name="prefix",
                default="",
                description="Prefix for generated solutions.",
                type=PrimitiveType.str,
                required=False
            )
        }

    ####################################################################
    #########  Private functions
    ####################################################################
