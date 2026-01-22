import enum
from typing import Any, Optional, List, Dict, Type
from pydantic import BaseModel

from core.game_state import GameState, Player, GameParams


# RemoteConstructable base class marker (optional)
class RemoteConstructable(BaseModel):
    pass

# RemoteInvocationRequest and Response
class RemoteInvocationRequest(RemoteConstructable):
    requestType: str
    target: str
    className: Optional[str] = None
    method: Optional[str] = None
    objectId: Optional[str] = None
    args: List[Any] = []

class RemoteInvocationResponse(RemoteConstructable):
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None

# Map method names to expected argument types (can be extended)
METHOD_ARG_TYPES: Dict[str, List[Type[Any]]] = {
    "get_action": [GameState],
    "prepare_to_play_as": [Player, GameParams, str],
    "process_game_over": [GameState],
}

# Deserialize an argument using Pydantic based on method signature
def deserialize_argument_old(method: str, arg_json: Any, index: int) -> Any:
    expected = METHOD_ARG_TYPES.get(method, [])
    if index >= len(expected):
        return arg_json  # fallback
    typ = expected[index]
    if issubclass(typ, BaseModel):
        return typ.model_validate(arg_json)
    else:
        return arg_json

def deserialize_argument(method: str, arg_json: Any, index: int) -> Any:
    expected = METHOD_ARG_TYPES.get(method, [])
    if index >= len(expected):
        return arg_json  # fallback
    typ = expected[index]
    if issubclass(typ, BaseModel):
        return typ.model_validate(arg_json)
    elif isinstance(typ, type) and issubclass(typ, Enum):
        return typ(arg_json)
    else:
        return arg_json


def strip_type_field(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: strip_type_field(v) for k, v in obj.items() if k != "type"}
    elif isinstance(obj, list):
        return [strip_type_field(item) for item in obj]
    else:
        return obj

# Deserialize all args
def deserialize_args(method: str, args: List[Any]) -> List[Any]:
    clean_args = [strip_type_field(arg) for arg in args]
    return [deserialize_argument(method, arg, i) for i, arg in enumerate(clean_args)]

def deserialize_alt_old(method_name: str, raw_args: list) -> list:
    return [deserialize_argument(method_name, arg, i) for i, arg in enumerate(raw_args)]


from enum import Enum
from typing import get_args, get_origin

def deserialize_args_old(method_name: str, raw_args: list) -> list:
    expected_types = METHOD_ARG_TYPES.get(method_name, [])
    deserialized = []
    for expected_type, arg in zip(expected_types, raw_args):
        if isinstance(arg, dict) and hasattr(expected_type, "from_dict"):
            deserialized.append(expected_type.from_dict(arg))
        elif isinstance(expected_type, type) and issubclass(expected_type, Enum):
            deserialized.append(expected_type(arg))  # convert string to Enum
        else:
            deserialized.append(arg)
    return deserialized

# Serialize a result using Pydantic
def serialize_result_old(result: Any) -> Any:
    if isinstance(result, BaseModel):
        return result.model_dump()
    elif isinstance(result, enum.Enum):
        return result.value
    elif isinstance(result, (str, int, float, bool)) or result is None:
        return result
    else:
        raise ValueError(f"Cannot serialize result of type {type(result)}")


def serialize_result(result: Any) -> Any:
    if isinstance(result, BaseModel):
        return result.model_dump(by_alias=True, mode="json")
    elif isinstance(result, enum.Enum):
        return result.value
    elif isinstance(result, (str, int, float, bool)) or result is None:
        return result
    else:
        raise ValueError(f"Cannot serialize result of type {type(result)}")

