import json
import logging
import os.path
import pickle

import subprocess
from typing import Optional, cast

from pydantic import BaseModel

class PackageInstallationError(Exception):
    pass

class PythonPackage(BaseModel):
    name: str
    version: str
    system: bool


class PackageManager:
    def __init__(self):
        """
        Package Manager class.
        Managing user-installed Python packages at runtime using front-end (FE) and system Python packages.
        At initialization, the currently installed Python packages are frozen and treated as system packages,
        if the dump file not exists on the disk.
        The user cannot modify the system Python packages.
        """

        self._file_name = os.path.join("fopimt", "utils", "package_manager_dump.pkl")
        self._packages: list[PythonPackage] = []
        if not os.path.exists(self._file_name):
            self._packages = self._get_all_packages(system=True)
            logging.debug('PackageManager: located system python packages: ' + str(len(self._packages)))
            with open(self._file_name, 'wb') as f:
                pickle.dump(self._packages, f)
        else:
            with open(self._file_name, 'rb') as f:
                self._packages = pickle.load(f)

    ####################################################################
    #########  Public functions
    ####################################################################
    def add(self, packs: list[PythonPackage]) -> None:
        for p in packs:
            # locate the package in installed
            pkg = next((pkg for pkg in self._packages if cast(PythonPackage, pkg).name == p.name), None)
            if pkg is not None:
                logging.warning(f"PackageManager: Package {p.name} already installed.")
                continue

            p.system = False    # user cant add system package at runtime
            package_str = f"{p.name}=={p.version}"
            result = subprocess.run(['pip', 'install', package_str], capture_output=True, text=True)

            logging.debug("PackageManager: " + result.stdout)
            logging.error("PackageManager: " + result.stderr)
            if result.returncode != 0:
                # some kind of error
                raise PackageInstallationError(
                    f"Failed to install {package_str}:\n{result.stderr}"
                )

            self._packages.append(p)
            with open(self._file_name, 'wb') as f:
                pickle.dump(self._packages, f)


    def delete(self, packs: list[PythonPackage]) -> None:
        # pip uninstall -y package1 package2 package3
        for p in packs:
            # locate the package in installed and check if not system package
            pkg = next((pkg for pkg in self._packages if cast(PythonPackage, pkg).name == p.name), None)
            if pkg is None:
                logging.warning(f"PackageManager: Package {p.name} not found.")
                continue
            if cast(PythonPackage, pkg).system is True:
                logging.warning(f"PackageManager: Package {p.name} is system. I am unable to uninstall it.")
                continue

            package_str = f"{cast(PythonPackage, pkg).name}=={cast(PythonPackage, pkg).version}"
            result = subprocess.run(['pip', 'uninstall', '-y', package_str], capture_output=True, text=True)
            logging.debug("PackageManager: " + result.stdout)
            logging.error("PackageManager: " + result.stderr)
            if result.returncode != 0:
                # some kind of error
                raise PackageInstallationError(
                    f"Failed to uninstall {package_str}:\n{result.stderr}"
                )

            cast(list[PythonPackage], self._packages).remove(pkg)
            with open(self._file_name, 'wb') as f:
                pickle.dump(self._packages, f)

    def get_packages(self) -> list[PythonPackage]:
        return self._packages

    ####################################################################
    #########  Private functions
    ####################################################################
    def _get_all_packages(self, system: bool = False) -> list[PythonPackage]:
        """
        Private function. Returns the current state of Python packages using 'pip' as a collection of PythonPackage.
        Arguments:
            system: bool    -- Set the flag PythonPackage._system on each loaded package.
        """
        result = subprocess.run(['pip', 'freeze'], stdout=subprocess.PIPE, text=True)
        packages = result.stdout.strip().split('\n')
        out = []
        for line in packages:
            row = line.split('==')
            out.append(
                PythonPackage(name=row[0], version=row[1], system=system)
            )
        return out

