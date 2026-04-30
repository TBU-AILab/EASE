import importlib
import inspect
import json
import logging
import os
from enum import Enum
from typing import Optional

from ..modul import Modul


class Resource:
    """Config-driven resource discovery and resolution utility.

    Resource metadata is loaded from ``resource_config.json``. Each resource
    type defines a module path and a ``discovery_mode``:

    - ``methods``: exposes callable members using short-name format
      ``resource.<name>.<function>``.
    - ``classes``: exposes the concrete resource class using short-name format
      ``resource.<name>``.
    """

    config_path = os.path.join(os.path.dirname(__file__), "resource_config.json")

    @classmethod
    def _load_resource_config(cls, resource_type: str) -> dict:
        """Load configuration for a specific resource type.

        Args:
            resource_type: Resource type key from ``resource_config.json``.

        Returns:
            Resource-type configuration dictionary.

        Raises:
            ValueError: If the resource type is not present in configuration.
        """
        with open(cls.config_path) as f:
            config = json.load(f)
        resources = config["resources"]
        if resource_type not in resources:
            raise ValueError(
                f"Resource type '{resource_type}' is not defined in the configuration."
            )
        return resources[resource_type]

    @classmethod
    def get_resources(cls, resource_type: str) -> dict:
        """List available resources for UI selection.

        For ``methods`` discovery mode, one item is returned per discovered
        callable function using short-name format ``resource.<name>.<function>``.
        For ``classes`` mode, one item is returned per class using
        ``resource.<name>``.

        Args:
            resource_type: Resource type key from configuration.

        Returns:
            Dictionary with parallel arrays:
            ``short_names``, ``long_names``, ``descriptions``.
        """
        out = {"short_names": [], "long_names": [], "descriptions": []}
        resource_config = cls._load_resource_config(resource_type)
        path = resource_config["path"]
        discovery_mode = resource_config.get("discovery_mode", "methods")
        files = cls._get_files(path)
        for file in files:
            resource = cls._load_resource(path, file)
            desc = cls._get_description(resource)
            if discovery_mode == "methods":
                for f in desc["functions"]:
                    out["short_names"].append(desc["short_name"] + "." + f)
                    out["long_names"].append(desc["long_name"] + " " + f)
                    out["descriptions"].append(desc["description"])
            elif discovery_mode == "classes":
                out["short_names"].append(desc["short_name"])
                out["long_names"].append(desc["long_name"])
                out["descriptions"].append(desc["description"])
            else:
                raise ValueError(
                    f"Unsupported discovery_mode '{discovery_mode}' for resource type '{resource_type}'."
                )
        return out

    @classmethod
    def get_resource_target(cls, short_name: str, resource_type: str):
        """Resolve and return the runtime target for a resource short name.

        The returned target depends on the resource configuration's
        ``discovery_mode``:
        - ``methods``: returns the function (or callable attribute) specified by
          ``short_name`` in format ``resource.<name>.<function>``.
        - ``classes``: returns the resolved resource class itself (to be
          instantiated by the caller).

        Args:
            short_name: Resource identifier in short-name convention
                (``resource.<name>`` or ``resource.<name>.<function>``).
            resource_type: Resource type key defined in
                ``resource_config.json``.

        Returns:
            A callable method/attribute for ``methods`` discovery mode, or a
            resource class for ``classes`` discovery mode.

        Raises:
            ValueError: If ``short_name`` format is invalid or
                ``discovery_mode`` is unsupported.
            LookupError: If no matching resource file can be resolved.
            AttributeError: If the requested function does not exist on the
                resolved resource class in ``methods`` mode.
        """
        resource_config = cls._load_resource_config(resource_type)
        path = resource_config["path"]
        discovery_mode = resource_config.get("discovery_mode", "methods")

        parts = short_name.split(".")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid short_name '{short_name}'. Expected format 'resource.<name>' or 'resource.<name>.<function>'."
            )

        # Existing short_name convention: resource.<resource_name>[.<function_name>]
        resource_name = parts[1]
        files = cls._get_files(path, resource_name)
        if not files:
            raise LookupError(
                f"No resource file found for short_name '{short_name}' in '{path}'."
            )

        resource = cls._load_resource(path, files[0])

        if discovery_mode == "methods":
            if len(parts) < 3:
                raise ValueError(
                    f"Invalid short_name '{short_name}' for methods discovery. Expected 'resource.<name>.<function>'."
                )
            function_name = parts[2]
            if not hasattr(resource, function_name):
                raise AttributeError(
                    f"Resource '{resource.__name__}' has no function '{function_name}'."
                )
            return getattr(resource, function_name)

        if discovery_mode == "classes":
            return resource

        raise ValueError(
            f"Unsupported discovery_mode '{discovery_mode}' for resource type '{resource_type}'."
        )

    @classmethod
    def get_resource_function(cls, short_name: str, resource_type: str):
        """Backward-compatible alias for :meth:`get_resource_target`."""
        return cls.get_resource_target(short_name, resource_type)

    @staticmethod
    def _get_files(base: str, name: Optional[str] = None) -> list[str]:
        """List importable module filenames in a directory.

        Files are returned without extension and private files prefixed with
        ``_`` are ignored. If ``name`` is provided, only filenames containing
        that substring are kept.
        """
        files = [
            os.path.splitext(f)[0]
            for f in os.listdir(base)
            if os.path.isfile(os.path.join(base, f)) and not f.startswith("_")
        ]
        # logging.info(files)
        if name is not None:
            files = [f for f in files if name in f]
        return files

    @staticmethod
    def _load_resource(_dir: str, file: str):
        """Import a resource module and pick a concrete ``Modul`` subclass.

        Selection rules:
        - subclass of ``Modul``
        - not ``Modul`` itself
        - defined in the target module (not re-imported)
        - non-abstract class

        If multiple classes match, the first one is used and a warning is
        logged.
        """
        _package = _dir.replace("/", ".")
        module = importlib.import_module(f".{file}", _package)

        candidates = []
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if not issubclass(cls, Modul):
                continue
            if cls is Modul:
                continue
            # ignore imported base classes from other modules
            if cls.__module__ != module.__name__:
                continue
            # ignore abstract base classes
            if inspect.isabstract(cls):
                continue
            candidates.append(cls)

        if not candidates:
            raise LookupError(
                f"No concrete resource class found in module '{module.__name__}'"
            )

        if len(candidates) > 1:
            logging.warning(
                "Multiple resource classes found in '%s': %s. Using first.",
                module.__name__,
                [c.__name__ for c in candidates],
            )

        return candidates[0]

    @staticmethod
    def _get_description(resource):
        """Build a description payload for a discovered resource class."""
        functions = [
            f
            for f in dir(resource)
            if not f.startswith(
                (
                    "_",
                    "get_short_name",
                    "get_long_name",
                    "get_description",
                    "get_tags",
                    "get_parameters",
                    "get_order",
                    "render_html",
                    "render_latex",
                )
            )
        ]
        return {
            "short_name": getattr(resource, "get_short_name")(),
            "long_name": getattr(resource, "get_long_name")(),
            "description": getattr(resource, "get_description")(),
            "functions": functions,
        }
