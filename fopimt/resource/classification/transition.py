import csv
import json
from pathlib import Path
from typing import Dict, List, Union
from ...modul import Modul
import os

PathLike = Union[str, Path]


class TransitionDetectionDataset(Modul):
    """
    Simple dataset loader returning a dictionary of transition dataset

    X = path to .ts file
    y = label
    """

    @classmethod
    def get_short_name(cls) -> str:
        return "resource.transition"

    @classmethod
    def get_long_name(cls) -> str:
        return "Transition detection dataset"

    @classmethod
    def get_description(cls) -> str:
        return "Transition detection dataset from ModernTV project"

    @staticmethod
    def full() -> dict:
        """
        Returns full Transition prediction dataset.
        """
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        samples_csv = Path(os.path.join(BASE_DIR, "transition_data", "samples.csv"))
        splits_json = Path(os.path.join(BASE_DIR, "transition_data", "splits.json"))

        def _build_split(sampless: dict, ids: list[int]) -> Dict[str, List]:
            data: List[str] = []
            target: List[int] = []

            for sid in ids:
                if sid not in sampless:
                    raise KeyError(f"Sample id {sid} not found in CSV")

                s = sampless[sid]
                x = s["path"]

                data.append(x)
                target.append(s["label"])

            return {"data": data, "target": target}

        # Load samples
        samples = {}

        with samples_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            required = {"id", "path", "label"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"CSV must contain columns {sorted(required)}")

            for row in reader:
                sid = int(row["id"])
                samples[sid] = {
                    "path": row["path"],
                    "label": int(row["label"]),
                }

                if "basename" in row and row["basename"]:
                    samples[sid]["basename"] = row["basename"]

        train = test = val = {}

        # Load splits
        with splits_json.open("r", encoding="utf-8") as f:
            splits = json.load(f)

            train = _build_split(samples, splits["train"])
            test = _build_split(samples, splits["test"])
            val = _build_split(samples, splits["val"])

        return {
            "train_data": {"X": train["data"], "y": train["target"]},
            "test_data": {"X": test["data"], "y": test["target"]},
            "val_data": {"X": val["data"], "y": val["target"]},
        }




