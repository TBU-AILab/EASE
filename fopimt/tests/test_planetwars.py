import ast
from typing import Any

from ..loader_dto import Parameter, PrimitiveType
from ..solutions.solution import Solution
from .test import Test, TestResult


class TestPlanetWars(Test):
    """
    Static template/interface test for generated Planet Wars Python agents.

    It checks that the solution:
    - is valid Python code,
    - defines the expected class, usually MyAgent,
    - inherits from PlanetWarsPlayer,
    - implements get_action(self, game_state),
    - implements get_agent_type(self),
    - has a zero-argument constructor, or no explicit constructor,
    - does not override prepare_to_play_as incorrectly,
    - does not contain obvious unsafe imports/calls,
    - does not contain top-level executable code.

    This test intentionally does not execute generated code. Execution is handled
    by the separate PlanetWars evaluator service.
    """

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        return {
            "class_name": Parameter(
                short_name="class_name",
                type=PrimitiveType.str,
                long_name="Required agent class name",
                description="Name of the generated Planet Wars agent class.",
                default="MyAgent",
            ),
            "require_planetwars_inheritance": Parameter(
                short_name="require_planetwars_inheritance",
                type=PrimitiveType.bool,
                long_name="Require PlanetWarsPlayer inheritance",
                description=(
                    "If true, the class must inherit from PlanetWarsPlayer. "
                    "This follows the recommended Python agent pattern."
                ),
                default=True,
            ),
            "require_game_state_argument_name": Parameter(
                short_name="require_game_state_argument_name",
                type=PrimitiveType.bool,
                long_name="Require game_state argument name",
                description=(
                    "If true, get_action must use the template signature "
                    "get_action(self, game_state). If false, only the argument count "
                    "is checked."
                ),
                default=True,
            ),
            "require_action_reference": Parameter(
                short_name="require_action_reference",
                type=PrimitiveType.bool,
                long_name="Require Action reference",
                description=(
                    "If true, the code must reference Action somewhere. "
                    "This helps ensure get_action returns a Planet Wars Action object."
                ),
                default=True,
            ),
            "allow_top_level_helpers": Parameter(
                short_name="allow_top_level_helpers",
                type=PrimitiveType.bool,
                long_name="Allow top-level helper functions",
                description=(
                    "If true, top-level helper functions are allowed. "
                    "If false, only imports, constants, and the agent class are allowed."
                ),
                default=True,
            ),
            "allowed_import_roots": Parameter(
                short_name="allowed_import_roots",
                type=PrimitiveType.str,
                long_name="Allowed import roots",
                description=(
                    "Comma-separated list of allowed top-level import roots. "
                    "Keep this narrow because generated code is later executed "
                    "in the evaluator service."
                ),
                default=(
                    "agents,core,typing,math,random,collections,dataclasses,"
                    "functools,heapq,itertools,statistics"
                ),
            ),
            "forbidden_import_roots": Parameter(
                short_name="forbidden_import_roots",
                type=PrimitiveType.str,
                long_name="Forbidden import roots",
                description="Comma-separated list of always-forbidden import roots.",
                default=(
                    "os,sys,subprocess,socket,requests,urllib,pathlib,shutil,"
                    "glob,multiprocessing,threading,asyncio,importlib,ctypes,"
                    "inspect,builtins,pickle"
                ),
            ),
            "forbidden_call_names": Parameter(
                short_name="forbidden_call_names",
                type=PrimitiveType.str,
                long_name="Forbidden function calls",
                description=(
                    "Comma-separated list of function calls that should not appear "
                    "in generated agent code."
                ),
                default=(
                    "open,exec,eval,compile,input,__import__,breakpoint,"
                    "globals,locals,vars"
                ),
            ),
        }

    def _init_params(self):
        super()._init_params()

        self._class_name = self.parameters.get("class_name", "MyAgent")
        self._require_planetwars_inheritance = bool(
            self.parameters.get("require_planetwars_inheritance", True)
        )
        self._require_game_state_argument_name = bool(
            self.parameters.get("require_game_state_argument_name", True)
        )
        self._require_action_reference = bool(
            self.parameters.get("require_action_reference", True)
        )
        self._allow_top_level_helpers = bool(
            self.parameters.get("allow_top_level_helpers", True)
        )

        self._allowed_import_roots = self._parse_csv_set(
            self.parameters.get(
                "allowed_import_roots",
                (
                    "agents,core,typing,math,random,collections,dataclasses,"
                    "functools,heapq,itertools,statistics"
                ),
            )
        )

        self._forbidden_import_roots = self._parse_csv_set(
            self.parameters.get(
                "forbidden_import_roots",
                (
                    "os,sys,subprocess,socket,requests,urllib,pathlib,shutil,"
                    "glob,multiprocessing,threading,asyncio,importlib,ctypes,"
                    "inspect,builtins,pickle"
                ),
            )
        )

        self._forbidden_call_names = self._parse_csv_set(
            self.parameters.get(
                "forbidden_call_names",
                (
                    "open,exec,eval,compile,input,__import__,breakpoint,"
                    "globals,locals,vars"
                ),
            )
        )

        self._error_msg = "Test:PlanetWars: OK"
        self._user_msg = "Test:PlanetWars: OK"

    def test(self, solution: Solution) -> TestResult:
        source = solution.get_input() or ""

        errors: list[str] = []
        warnings: list[str] = []

        try:
            compile(source, "planetwars_candidate.py", "exec")
            tree = ast.parse(source)
        except Exception as exc:
            self._test_result = False
            self._error_msg = self._user_msg = (
                "Test:PlanetWars: The generated code is not valid Python. "
                f"The error was: {repr(exc)}"
            )
            return self._result(errors=[self._error_msg], warnings=warnings)

        import_aliases = self._collect_import_aliases(tree)

        errors.extend(self._check_imports(tree))
        errors.extend(self._check_forbidden_calls(tree, import_aliases))
        errors.extend(self._check_top_level_structure(tree))

        class_defs = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == self._class_name
        ]

        if len(class_defs) == 0:
            errors.append(
                f"The solution must define a class named '{self._class_name}'."
            )
            return self._finish(errors, warnings, tree)

        if len(class_defs) > 1:
            errors.append(
                f"The solution defines class '{self._class_name}' more than once."
            )
            return self._finish(errors, warnings, tree)

        agent_class = class_defs[0]

        errors.extend(self._check_class(agent_class, import_aliases))
        errors.extend(self._check_required_names(tree, import_aliases))

        return self._finish(errors, warnings, tree)

    @classmethod
    def get_short_name(cls) -> str:
        return "test.planetwars"

    @classmethod
    def get_long_name(cls) -> str:
        return "Planet Wars agent template test"

    @classmethod
    def get_description(cls) -> str:
        return (
            "Static test for generated Planet Wars Python agents. "
            "Checks the expected class template, required methods, constructor, "
            "prepare_to_play_as signature, imports, and obvious unsafe operations."
        )

    @classmethod
    def get_tags(cls) -> dict:
        return {"input": {"python"}, "output": set()}

    def _finish(
        self,
        errors: list[str],
        warnings: list[str],
        tree: ast.AST,
    ) -> TestResult:
        self._test_result = len(errors) == 0

        metadata = {
            "error_msg": "",
            "user_msg": "",
            "errors": errors,
            "warnings": warnings,
            "class_name": self._class_name,
            "imports": self._extract_imports(tree),
        }

        if self._test_result:
            self._error_msg = "Test:PlanetWars: OK"
            self._user_msg = "Test:PlanetWars: OK"
        else:
            joined_errors = "\n".join(f"- {err}" for err in errors)
            self._error_msg = self._user_msg = (
                "Test:PlanetWars: The generated agent does not match the required "
                "Planet Wars template/interface:\n"
                f"{joined_errors}"
            )

        metadata["error_msg"] = self._error_msg
        metadata["user_msg"] = self._user_msg

        return TestResult(
            class_ref=type(self),
            passed=self._test_result,
            metadata=metadata,
        )

    def _result(
        self,
        errors: list[str],
        warnings: list[str],
    ) -> TestResult:
        return TestResult(
            class_ref=type(self),
            passed=False,
            metadata={
                "error_msg": self._error_msg,
                "user_msg": self._user_msg,
                "errors": errors,
                "warnings": warnings,
                "class_name": self._class_name,
                "imports": [],
            },
        )

    def _check_class(
        self,
        node: ast.ClassDef,
        import_aliases: dict[str, str],
    ) -> list[str]:
        errors: list[str] = []

        if self._require_planetwars_inheritance:
            base_names = {
                self._normalize_name(self._node_name(base), import_aliases)
                for base in node.bases
            }

            valid_bases = {
                "PlanetWarsPlayer",
                "agents.planet_wars_agent.PlanetWarsPlayer",
            }

            if not any(
                base in valid_bases or base.endswith(".PlanetWarsPlayer")
                for base in base_names
            ):
                errors.append(
                    f"Class '{self._class_name}' must inherit from PlanetWarsPlayer."
                )

        methods = {
            item.name: item
            for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        if "get_action" not in methods:
            errors.append(
                f"Class '{self._class_name}' must implement get_action(self, game_state)."
            )
        else:
            errors.extend(self._check_get_action_signature(methods["get_action"]))
            if not self._function_has_return(methods["get_action"]):
                errors.append("get_action must contain at least one return statement.")

        if "get_agent_type" not in methods:
            errors.append(
                f"Class '{self._class_name}' must implement get_agent_type(self)."
            )
        else:
            errors.extend(
                self._check_no_required_args_after_self(
                    methods["get_agent_type"],
                    "get_agent_type",
                )
            )
            if not self._function_has_return(methods["get_agent_type"]):
                errors.append("get_agent_type must contain at least one return statement.")

        if "__init__" in methods:
            errors.extend(
                self._check_no_required_args_after_self(
                    methods["__init__"],
                    "__init__",
                )
            )

        if "prepare_to_play_as" in methods:
            errors.extend(
                self._check_prepare_to_play_as_signature(
                    methods["prepare_to_play_as"]
                )
            )

        return errors

    def _check_get_action_signature(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[str]:
        errors: list[str] = []

        args = list(node.args.posonlyargs) + list(node.args.args)

        if len(args) < 2:
            errors.append("get_action must have signature get_action(self, game_state).")
            return errors

        if args[0].arg != "self":
            errors.append("The first argument of get_action must be self.")

        if self._require_game_state_argument_name and args[1].arg != "game_state":
            errors.append(
                "The second argument of get_action should be named game_state."
            )

        required_after_self = self._required_args_after_self(node)
        if len(required_after_self) > 1:
            errors.append(
                "get_action must not require additional positional arguments beyond "
                "game_state."
            )

        return errors

    def _check_no_required_args_after_self(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        method_name: str,
    ) -> list[str]:
        required_after_self = self._required_args_after_self(node)

        if required_after_self:
            return [
                f"{method_name} must not require arguments beyond self. "
                f"Required arguments found: {required_after_self}"
            ]

        return []

    def _check_prepare_to_play_as_signature(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[str]:
        args = list(node.args.posonlyargs) + list(node.args.args)
        arg_names = [arg.arg for arg in args]

        has_opponent = "opponent" in arg_names
        has_kwargs = node.args.kwarg is not None

        if not has_opponent and not has_kwargs:
            return [
                "If prepare_to_play_as is overridden, it must accept an opponent "
                "parameter, e.g. prepare_to_play_as(self, player, params, opponent=None)."
            ]

        return []

    def _required_args_after_self(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[str]:
        positional_args = list(node.args.posonlyargs) + list(node.args.args)
        default_count = len(node.args.defaults)
        required_count = max(0, len(positional_args) - default_count)
        required_args = positional_args[:required_count]

        return [
            arg.arg
            for arg in required_args
            if arg.arg != "self"
        ]

    def _function_has_return(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> bool:
        return any(isinstance(child, ast.Return) for child in ast.walk(node))

    def _check_required_names(
            self,
            tree: ast.AST,
            import_aliases: dict[str, str],
    ) -> list[str]:
        errors: list[str] = []

        if self._require_action_reference:
            raw_names = {
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name)
            }

            normalized_names = {
                self._normalize_name(name, import_aliases)
                for name in raw_names
            }

            all_names = raw_names | normalized_names

            has_action_reference = any(
                name == "Action" or name.endswith(".Action")
                for name in all_names
            )

            if not has_action_reference:
                errors.append(
                    "The generated code should reference Action, because get_action "
                    "must return a core.game_state.Action."
                )

        return errors

    def _check_top_level_structure(self, tree: ast.Module) -> list[str]:
        errors: list[str] = []

        allowed_top_level_types = (
            ast.Import,
            ast.ImportFrom,
            ast.ClassDef,
            ast.Assign,
            ast.AnnAssign,
        )

        if self._allow_top_level_helpers:
            allowed_top_level_types = allowed_top_level_types + (ast.FunctionDef,)

        for node in tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(getattr(node, "value", None), ast.Constant):
                    # Module docstring or top-level literal.
                    continue

                errors.append(
                    "Top-level executable expression is not allowed. "
                    f"Unexpected expression type: {type(node.value).__name__}."
                )
                continue

            if not isinstance(node, allowed_top_level_types):
                errors.append(
                    "Top-level executable code is not allowed. "
                    f"Unexpected top-level statement: {type(node).__name__}."
                )

        return errors

    def _check_imports(self, tree: ast.AST) -> list[str]:
        errors: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in self._forbidden_import_roots:
                        errors.append(f"Forbidden import: {alias.name}")
                    elif root not in self._allowed_import_roots:
                        errors.append(
                            f"Import '{alias.name}' is not allowed for Planet Wars agents."
                        )

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                root = module.split(".")[0]

                if root in self._forbidden_import_roots:
                    errors.append(f"Forbidden import from: {module}")
                elif root not in self._allowed_import_roots:
                    errors.append(
                        f"Import from '{module}' is not allowed for Planet Wars agents."
                    )

        return errors

    def _check_forbidden_calls(
        self,
        tree: ast.AST,
        import_aliases: dict[str, str],
    ) -> list[str]:
        errors: list[str] = []

        forbidden_roots = self._forbidden_import_roots

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            call_name = self._node_name(node.func)
            normalized = self._normalize_name(call_name, import_aliases)

            short_name = normalized.split(".")[-1]
            root_name = normalized.split(".")[0]

            if short_name in self._forbidden_call_names:
                errors.append(f"Forbidden function call: {normalized}")

            if root_name in forbidden_roots:
                errors.append(f"Forbidden call through restricted module: {normalized}")

        return errors

    def _collect_import_aliases(self, tree: ast.AST) -> dict[str, str]:
        aliases: dict[str, str] = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    key = alias.asname or root
                    aliases[key] = root

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    key = alias.asname or alias.name

                    if module:
                        aliases[key] = f"{module}.{alias.name}"
                    else:
                        aliases[key] = alias.name

        return aliases

    def _extract_imports(self, tree: ast.AST) -> list[dict[str, Any]]:
        imports: list[dict[str, Any]] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(
                        {
                            "type": "import",
                            "module": alias.name,
                            "name": None,
                            "alias": alias.asname,
                        }
                    )

            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports.append(
                        {
                            "type": "from",
                            "module": node.module,
                            "name": alias.name,
                            "alias": alias.asname,
                        }
                    )

        return imports

    def _node_name(self, node: ast.AST | None) -> str:
        if node is None:
            return ""

        if isinstance(node, ast.Name):
            return node.id

        if isinstance(node, ast.Attribute):
            prefix = self._node_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr

        if isinstance(node, ast.Call):
            return self._node_name(node.func)

        if isinstance(node, ast.Subscript):
            return self._node_name(node.value)

        return type(node).__name__

    def _normalize_name(
        self,
        name: str,
        import_aliases: dict[str, str],
    ) -> str:
        if not name:
            return name

        parts = name.split(".")
        first = parts[0]

        if first in import_aliases:
            replacement = import_aliases[first].split(".")
            parts = replacement + parts[1:]

        return ".".join(parts)

    @staticmethod
    def _parse_csv_set(value: Any) -> set[str]:
        if value is None:
            return set()

        if isinstance(value, str):
            return {
                part.strip()
                for part in value.split(",")
                if part.strip()
            }

        # Fallback for programmatic calls, while still avoiding PrimitiveType.list
        # in the UI-facing parameter definition.
        try:
            return {
                str(part).strip()
                for part in value
                if str(part).strip()
            }
        except TypeError:
            return {str(value).strip()} if str(value).strip() else set()