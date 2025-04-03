import os
import logging
import importlib.util
from enum import Enum
from typing import Any, Union, Type, Optional

from pydantic import BaseModel, Field


class PackageType(Enum):
    """
    Defines types of known Packages in Core
    """
    Analysis = 0
    Evaluator = 1
    LLMConnector = 2
    Solution = 3
    StoppingCondition = 4
    Test = 5
    Stat = 6


class PrimitiveType(Enum):
    int = 'int'
    float = 'float'
    str = 'str'
    bool = 'bool'
    bytes = 'bytes'
    enum = 'enum'
    markdown = 'markdown'


class Parameter(BaseModel):
    short_name: str
    long_name: Optional[str] = None
    description: Optional[str] = None
    type: PrimitiveType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    enum_options: Optional[list[Union[str, "ModulAPI"]]] = None
    enum_descriptions: Optional[list[str]] = None
    enum_long_names: Optional[list[str]] = None
    default: Optional[Any] = None
    readonly: Optional[bool] = False
    required: Optional[bool] = True

class ParameterInstance(Parameter):
    value: Any

class ModulAPI(BaseModel):
    """
    Base class for all Modul loaded and offered by Loader class, or rather Package.
    Contains short_name (unique ID), long_name (human-readable name) and description (for tooltip purposes).
    tags are mainly for compatibility purposes between Moduls.
    package_type is a shortcut to identify parent Package.
    """
    short_name: str
    long_name: str
    description: str
    package_type: PackageType = Field(...,
                                      description=(
                                          "The type of Package:\n"
                                          "- `0` Analysis.\n"
                                          "- `1` Evaluator.\n"
                                          "- `2` LLM connector.\n"
                                          "- `3` Solution.\n"
                                          "- `4` Stopping condition.\n"
                                          "- `5` Test.\n"
                                          "- `6` Statistic."
                                      )
                                      )
    parameters: dict[str, Parameter]

    class Config:
        # Allows assignment to fields that are not declared in the Pydantic model
        arbitrary_types_allowed = True

    def __init__(self, short_name: str, long_name: str, tags: dict, description: str,
                 package_type: PackageType, parameters: dict[str, Parameter]):
        super().__init__(short_name=short_name, long_name=long_name, description=description,
                         package_type=package_type, parameters=parameters)
        self._tags: dict = tags

    def __repr__(self):
        return 'M:' + self.short_name

    def __eq__(self, other):
        if isinstance(other, ModulAPI):
            return other.short_name == self.short_name
        return False

    def __hash__(self):
        if not hasattr(self, 'short_name'):
            self.short_name = ''
        return hash(self.short_name)


Parameter.update_forward_refs()
ModulAPI.update_forward_refs()


class Package:
    def __init__(self, name: str, directory: str, base_name: str, package_type: PackageType):
        """
        Low-level Package Class\n
        Describe the package name and location. Contains the classes inside the package.\n
        :param name: Name of the package\n
        :param directory: Directory name of the package\n
        :param package_type (PackageType): Package type
        """
        self._app_name: str = 'fopimt'
        self._name: str = name
        self._directory: str = directory
        self._base_name: str = base_name
        self._package_type: PackageType = package_type
        self._classes: list = []
        self._moduls: list[ModulAPI] = []
        self._moduls_imported: dict[str, Type[Any]] = dict()
        self._load_classes()

    ####################################################################
    #########  Public functions
    ####################################################################
    def get_name(self) -> str:
        """
        Returns name of the package.
        :return: String name of the package.
        """
        return self._name

    def get_package_type(self) -> PackageType:
        """
        Returns type of the package.
        :return: PackageType of the package
        """
        return self._package_type

    def get_moduls(self) -> list[ModulAPI]:
        """
        Returns list of all loaded moduls
        :return: List of all Moduls
        """
        return self._moduls

    def get_modul(self, short_name: str) -> Union[ModulAPI, None]:
        for modul in self._moduls:
            if short_name == modul.short_name:
                return modul
        return None

    def get_modul_imported(self, short_name: str) -> Type[Any]:

        if short_name not in self._moduls_imported.keys():
            raise KeyError(f"Modul with name [{short_name}] not in imported moduls.")

        return self._moduls_imported[short_name]

    ####################################################################
    #########  Private functions
    ####################################################################
    def _load_classes(self):
        """
        Load the classes inside the package.
        """
        mypath = os.path.join(os.path.dirname(__file__), self._directory)
        files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
        # Exclude everything that not start with _base_name + '_'
        self._classes = [x for x in files if x.startswith(self._base_name + '_')]
        for c in self._classes:
            self._load_module(c)

    def _load_module(self, path: str) -> Any:
        package_name = self._app_name + '.' + self._directory
        module_name = os.path.splitext(path)[0]
        module = importlib.import_module(f'.{module_name}', package_name)

        names = dir(module)
        names.remove(self._package_type.name)
        for n in names:
            if str(n).startswith(self._package_type.name):
                modul_class = getattr(module, n)
                short_name = getattr(modul_class, 'get_short_name')()
                long_name = getattr(modul_class, 'get_long_name')()
                description = getattr(modul_class, 'get_description')()
                parameters = getattr(modul_class, 'get_parameters')()
                tags = getattr(modul_class, 'get_tags')()
                self._moduls.append(ModulAPI(short_name=short_name,
                                             long_name=long_name,
                                             description=description,
                                             tags=tags,
                                             package_type=self.get_package_type(),
                                             parameters=parameters
                                             ))
                self._moduls_imported[short_name] = modul_class
                logging.debug('Loaded package: ' + short_name)


class Loader:
    def __init__(self, package_type_list_in: tuple[PackageType] = tuple()):
        """
        Class responsible for loading the classes (LLMConnectors, StoppingConditions, Evaluators, ...)\n
        It is also possible to add new custom class using this loader.\n
        Loader is also responsible for compatibility checks between different classes.\n
        """
        if len(package_type_list_in) == 0:
            package_type_list = PackageType
        else:
            package_type_list = package_type_list_in

        self._packages = dict()
        for package_type in package_type_list:
            match package_type:
                case PackageType.LLMConnector:
                    self._packages[package_type] = Package('LLM connectors', 'llmconnectors', 'llmconnector',
                                                           PackageType.LLMConnector)
                case PackageType.Analysis:
                    self._packages[package_type] = Package('Analysis', 'analysis', 'analysis',
                                                           PackageType.Analysis)
                case PackageType.Evaluator:
                    self._packages[package_type] = Package('Evaluators', 'evaluators', 'evaluator',
                                                           PackageType.Evaluator)
                case PackageType.Solution:
                    self._packages[package_type] = Package('Solutions', 'solutions', 'solution',
                                                           PackageType.Solution)
                case PackageType.StoppingCondition:
                    self._packages[package_type] = Package('Stopping conditions', 'stoppingconditions',
                                                           'stopping_condition',
                                                           PackageType.StoppingCondition)
                case PackageType.Test:
                    self._packages[package_type] = Package('Tests', 'tests', 'test',
                                                           PackageType.Test)
                case PackageType.Stat:
                    self._packages[package_type] = Package('Stats', 'stats', 'stat',
                                                           PackageType.Stat)

                case _:
                    logging.error(f'Unexpected package type: {package_type}')

    ####################################################################
    #########  Public functions
    ####################################################################
    def get_package(self, package_type: PackageType) -> Package:
        return self._packages[package_type]

    def get_modul_by_name(self, short_name: str) -> (Any, PackageType):
        for package in self._packages.values():
            try:
                return package.get_modul_imported(short_name), package.get_package_type()
            except KeyError:
                continue
        return None

    # @deprecated("Please use get_package()")
    def get_package_analysis(self) -> Package:
        return self._packages[PackageType.Analysis]

    # @deprecated("Please use get_package()")
    def get_package_evaluators(self) -> Package:
        return self._packages[PackageType.Evaluator]

    # @deprecated("Please use get_package()")
    def get_package_llmconnectors(self) -> Package:
        return self._packages[PackageType.LLMConnector]

    # @deprecated("Please use get_package()")
    def get_package_solutions(self) -> Package:
        return self._packages[PackageType.Solution]

    # @deprecated("Please use get_package()")
    def get_package_stoppingconditions(self) -> Package:
        return self._packages[PackageType.StoppingCondition]

    # @deprecated("Please use get_package()")
    def get_package_tests(self) -> Package:
        return self._packages[PackageType.Test]

    # @deprecated("Please use get_package()")
    def get_package_stats(self) -> Package:
        return self._packages[PackageType.Stat]

    ####################################################################
    #########  Private functions
    ####################################################################
