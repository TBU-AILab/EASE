from ...modul import Modul
import os


class Poster(Modul):

    @classmethod
    def get_short_name(cls) -> str:
        return "resource.poster"

    @classmethod
    def get_long_name(cls) -> str:
        return "Poster dataset"

    @classmethod
    def get_description(cls) -> str:
        return ("Data class for posterization providing multiple datasets with images.\nCurrently:\n - crime - for "
                "Crime shows\n - talkshow - for Talk shows\n - soapopera - for Soap operas")

    @staticmethod
    def crime() -> str:
        return os.path.abspath(r".\fopimt\resource\data\crime")

    @staticmethod
    def talkshow() -> str:
        return os.path.abspath(r".\fopimt\resource\data\talkshow")

    @staticmethod
    def soapopera() -> str:
        return os.path.abspath(r".\fopimt\resource\data\soapopera")

    @staticmethod
    def sample() -> str:
        return os.path.abspath(r".\fopimt\resource\data\sample")
