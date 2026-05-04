from enum import Enum
from typing import Any, Optional, Union

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
    int = "int"
    float = "float"
    str = "str"
    bool = "bool"
    bytes = "bytes"
    enum = "enum"
    markdown = "markdown"
    list = "list"
    time = "time"


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
    value: Optional[Any] = None
    sensitive: Optional[bool] = False


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
    package_type: PackageType = Field(
        ...,
        description=(
            "The type of Package:\n"
            "- `0` Analysis.\n"
            "- `1` Evaluator.\n"
            "- `2` LLM connector.\n"
            "- `3` Solution.\n"
            "- `4` Stopping condition.\n"
            "- `5` Test.\n"
            "- `6` Statistic."
        ),
    )
    parameters: dict[str, Parameter]

    class Config:
        # Allows assignment to fields that are not declared in the Pydantic model
        arbitrary_types_allowed = True

    def __init__(
        self,
        short_name: str,
        long_name: str,
        tags: dict,
        description: str,
        package_type: PackageType,
        parameters: dict[str, Parameter],
    ):
        super().__init__(
            short_name=short_name,
            long_name=long_name,
            description=description,
            package_type=package_type,
            parameters=parameters,
        )
        self._tags: dict = tags

    def __repr__(self):
        return "M:" + self.short_name

    def __eq__(self, other):
        if isinstance(other, ModulAPI):
            return other.short_name == self.short_name
        return False

    def __hash__(self):
        if not hasattr(self, "short_name"):
            self.short_name = ""
        return hash(self.short_name)


Parameter.update_forward_refs()
ModulAPI.update_forward_refs()
