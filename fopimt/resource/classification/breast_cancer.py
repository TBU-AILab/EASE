from sklearn.model_selection import train_test_split
from ...modul import Modul
from sklearn.datasets import load_breast_cancer


class BreastCancer(Modul):

    @classmethod
    def get_short_name(cls) -> str:
        return "resource.breast_cancer"

    @classmethod
    def get_long_name(cls) -> str:
        return "BREAST CANCER dataset"

    @classmethod
    def get_description(cls) -> str:
        return "BREAST CANCER dataset from sklearn"

    @staticmethod
    def full() -> dict:
        """
        Returns full BREAST CANCER dataset.
        """
        breast_cancer = load_breast_cancer()
        x_train, x_test, y_train, y_test = train_test_split(
            breast_cancer.data, breast_cancer.target, test_size=0.2, random_state=42
        )

        train = {"data": x_train, "target": y_train}
        test = {"data": x_test, "target": y_test}

        return {
            "train_data": {"X": train["data"], "y": train["target"]},
            "test_data": {"X": test["data"], "y": test["target"]},
        }

