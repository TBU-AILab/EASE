from ...modul import Modul
import os
import glob
import random
from sklearn.model_selection import train_test_split


class Transition(Modul):

    @classmethod
    def get_short_name(cls) -> str:
        return "resource.transition"

    @classmethod
    def get_long_name(cls) -> str:
        return "Video transition dataset"

    @classmethod
    def get_description(cls) -> str:
        return "Video transition dataset for Modern TV"

    @staticmethod
    def sample() -> dict:

        """
            Loads .ts files from dataset_path, where files are stored in '0' and '1' subfolders.
            Splits them into train and test sets.

            Args:
                dataset_path (str): Path to the dataset folder.
                test_size (float): Proportion of the dataset to include in the test split.
                random_seed (int): Random seed for reproducibility.

            Returns:
                dict: train_data with keys 'X' and 'y'
                dict: test_data with keys 'X' and 'y'
            """
        test_size = 0.2
        random_seed = 42
        dataset_path = r"fopimt/resource/data/transitions"
        data = []
        labels = []

        for label in ["0", "1"]:
            folder_path = os.path.join(dataset_path, label)
            if not os.path.exists(folder_path):
                print("folder_path")
                continue
            ts_files = glob.glob(os.path.join(folder_path, "*.ts"))

            for ts_file in ts_files[:10]:
                data.append(ts_file)
                labels.append(int(label))  # Convert folder name to integer label (0 or 1)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=test_size, random_state=random_seed, stratify=labels
        )

        return {
            "train_data": {"X": X_train, "y": y_train},
            "test_data": {"X": X_test, "y": y_test},
        }

    @staticmethod
    def full() -> dict:

        """
            Loads .ts files from dataset_path, where files are stored in '0' and '1' subfolders.
            Splits them into train and test sets.

            Args:
                dataset_path (str): Path to the dataset folder.
                test_size (float): Proportion of the dataset to include in the test split.
                random_seed (int): Random seed for reproducibility.

            Returns:
                dict: train_data with keys 'X' and 'y'
                dict: test_data with keys 'X' and 'y'
            """
        test_size = 0.2
        random_seed = 42
        dataset_path = r"fopimt/resource/data/transitions"
        data = []
        labels = []

        for label in ["0", "1"]:
            folder_path = os.path.join(dataset_path, label)
            if not os.path.exists(folder_path):
                continue
            ts_files = glob.glob(os.path.join(folder_path, "*.ts"))

            for ts_file in ts_files:
                data.append(ts_file)
                labels.append(int(label))  # Convert folder name to integer label (0 or 1)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=test_size, random_state=random_seed, stratify=labels
        )

        return {
            "train_data": {"X": X_train, "y": y_train},
            "test_data": {"X": X_test, "y": y_test},
        }

