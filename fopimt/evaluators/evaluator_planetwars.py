import copy
import json
import logging
import os
from typing import Any

import requests

from ..loader_dto import Parameter, PrimitiveType
from ..solutions.solution import Solution
from ..task_dto import OptimizationGoal
from .evaluator import Evaluator, EvaluatorResult


class EvaluatorPlanetWars(Evaluator):
    """
    Evaluator for Planet Wars agents.

    The generated solution is expected to be Python code defining:

        class MyAgent(PlanetWarsPlayer):
            def get_action(self, game_state):
                ...
            def get_agent_type(self):
                return "MyAgent"

    The evaluator does not execute generated code directly.
    It sends candidate code to a separate PlanetWars evaluation service.

    This evaluator also stores an in-memory history of previously evaluated
    solutions. If enabled, selected historical solutions are sent to the
    PlanetWars service as additional code-based opponents.
    """

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        return {
            "service_url": Parameter(
                short_name="service_url",
                type=PrimitiveType.str,
                long_name="PlanetWars evaluator service URL",
                description=(
                    "URL of the external PlanetWars evaluation service. "
                    "Inside Docker Compose, use http://planetwars-evaluator:8090."
                ),
                default="http://planetwars-evaluator:8090",
            ),
            "class_name": Parameter(
                short_name="class_name",
                type=PrimitiveType.str,
                long_name="Generated agent class name",
                description="Name of the generated class to instantiate and evaluate.",
                default="MyAgent",
            ),
            "n_games": Parameter(
                short_name="n_games",
                type=PrimitiveType.int,
                long_name="Number of games",
                description=(
                    "Number of games per opponent and per side. "
                    "If play_both_sides is true and two opponents are used, "
                    "total games = n_games * 2 * number_of_opponents."
                ),
                default=10,
                min_value=1,
                max_value=1000,
            ),
            "opponents": Parameter(
                short_name="opponents",
                type=PrimitiveType.str,
                long_name="Baseline opponents",
                description=(
                    "Comma-separated list of built-in opponent names. "
                    "Supported by the MVP service: "
                    "PureRandomAgent, CarefulRandomAgent, GreedyHeuristicAgent."
                ),
                default="CarefulRandomAgent,GreedyHeuristicAgent",
            ),
            "use_history_as_opponents": Parameter(
                short_name="use_history_as_opponents",
                type=PrimitiveType.bool,
                long_name="Use historical solutions as opponents",
                description=(
                    "If true, previously evaluated generated agents are also used "
                    "as additional opponents."
                ),
                default=False,
            ),
            "history_opponent_strategy": Parameter(
                short_name="history_opponent_strategy",
                type=PrimitiveType.enum,
                long_name="Historical opponent selection strategy",
                description="Determines which previous solutions are used as opponents.",
                enum_options=["best", "last", "all"],
                default="best",
            ),
            "max_history_opponents": Parameter(
                short_name="max_history_opponents",
                type=PrimitiveType.int,
                long_name="Maximum number of historical opponents",
                description=(
                    "Maximum number of previous generated agents sent as opponents. "
                    "Evaluation cost grows quickly with this value."
                ),
                default=3,
                min_value=0,
                max_value=50,
            ),
            "store_failed_solutions_in_history": Parameter(
                short_name="store_failed_solutions_in_history",
                type=PrimitiveType.bool,
                long_name="Store failed solutions in history",
                description=(
                    "If false, only successfully evaluated solutions are stored "
                    "as possible future opponents."
                ),
                default=False,
            ),
            "partial_observability": Parameter(
                short_name="partial_observability",
                type=PrimitiveType.bool,
                long_name="Use partial observability",
                description="Whether to evaluate the agent in partial-observation mode.",
                default=False,
            ),
            "play_both_sides": Parameter(
                short_name="play_both_sides",
                type=PrimitiveType.bool,
                long_name="Play both sides",
                description=(
                    "If true, the candidate is evaluated both as Player1 and Player2."
                ),
                default=True,
            ),
            "seed": Parameter(
                short_name="seed",
                type=PrimitiveType.int,
                long_name="Base random seed",
                description="Base seed used by the evaluator service.",
                default=12345,
            ),
            "timeout_sec": Parameter(
                short_name="timeout_sec",
                type=PrimitiveType.int,
                long_name="Evaluation timeout",
                description="Timeout passed to the PlanetWars evaluation service.",
                default=90,
                min_value=1,
                max_value=3600,
            ),
            "request_timeout_sec": Parameter(
                short_name="request_timeout_sec",
                type=PrimitiveType.int,
                long_name="HTTP request timeout",
                description="Timeout for the HTTP call from EASE to the evaluator service.",
                default=120,
                min_value=1,
                max_value=3600,
            ),
            "num_planets": Parameter(
                short_name="num_planets",
                type=PrimitiveType.int,
                long_name="Number of planets",
                description="Planet Wars map size parameter.",
                default=20,
                min_value=2,
                max_value=200,
            ),
            "max_ticks": Parameter(
                short_name="max_ticks",
                type=PrimitiveType.int,
                long_name="Maximum game ticks",
                description="Maximum number of ticks before the game is stopped.",
                default=2000,
                min_value=1,
                max_value=100000,
            ),
            "new_map_each_run": Parameter(
                short_name="new_map_each_run",
                type=PrimitiveType.bool,
                long_name="New map each run",
                description="Whether to generate a new map for each game.",
                default=True,
            ),
            "fitness_metric": Parameter(
                short_name="fitness_metric",
                type=PrimitiveType.enum,
                long_name="Fitness metric",
                description="Metric from the evaluation response used as scalar fitness.",
                enum_options=[
                    "win_rate",
                    "win_rate_minus_failed_action_rate",
                    "mean_ship_diff",
                ],
                default="win_rate",
            ),
            "feedback_msg_template": Parameter(
                short_name="feedback_msg_template",
                type=PrimitiveType.markdown,
                long_name="Template for feedback message",
                description="Feedback message for evaluation. Can use {keywords}.",
                default=(
                    "The Planet Wars agent was evaluated with the following results:\n"
                    "{fitness}"
                    "{win_rate}"
                    "{wins}"
                    "{losses}"
                    "{draws}"
                    "{mean_ticks}"
                    "{mean_ship_diff}"
                    "{failed_action_rate}"
                    "{opponent_count}"
                    "{history_opponent_count}"
                    "{by_opponent}"
                    "{errors}"
                ),
            ),
            "init_msg_template": Parameter(
                short_name="init_msg_template",
                type=PrimitiveType.markdown,
                long_name="Template for an initial message",
                description="Initial instruction for generating Planet Wars agents.",
                default="""Implement a Python agent for the Planet Wars RTS game.

The generated solution will be evaluated automatically over many games. The scalar fitness is primarily based on win rate. The agent may be evaluated against built-in baseline agents and, later, against previously generated agents.

The solution must define exactly one class named MyAgent.

Use this required structure:

from agents.planet_wars_agent import PlanetWarsPlayer
from core.game_state import Action, Player

class MyAgent(PlanetWarsPlayer):
    def get_action(self, game_state):
        # Analyze the current game state and return one legal Action.
        return Action.do_nothing()

    def get_agent_type(self):
        return "MyAgent"

Your agent should make one decision at a time from the current game_state. It should select a source planet, a destination planet, and a number of ships to send, or return Action.do_nothing() when no useful legal action exists.

Useful strategic considerations:
- Prefer actions with clear advantage instead of attacking blindly.
- Expand to valuable neutral planets when the cost and distance are reasonable.
- Attack enemy planets that are weak, nearby, or strategically valuable.
- Defend or preserve strong planets instead of emptying them.
- Prefer high-growth planets as targets when they can be captured safely.
- Avoid sending ships from planets with too few ships.
- Avoid repeated invalid actions.
- Keep enough ships behind for defense.
- Balance expansion, attack, and consolidation.

Implementation requirements:
- The code must be valid Python.
- Define class MyAgent.
- MyAgent must inherit from PlanetWarsPlayer.
- Implement get_action(self, game_state).
- Implement get_agent_type(self).
- Return core.game_state.Action from get_action.
- Use Action.do_nothing() only as a fallback.
- Do not include explanations, comments outside the code, or test code.
- Do not print debug output.
- Do not use external libraries.
- Do not use file I/O, network access, subprocesses, multiprocessing, threading, exec, eval, compile, dynamic imports, or reflection.
- The class must have a zero-argument constructor, or no explicit constructor.
- The decision logic must be fast and robust.

Generate only the Python code for the agent.
""",
                readonly=True,
            ),
            "keywords": Parameter(
                short_name="keywords",
                type=PrimitiveType.enum,
                long_name="Feedback keywords",
                description="Available feedback keywords.",
                enum_options=[
                    "fitness",
                    "win_rate",
                    "wins",
                    "losses",
                    "draws",
                    "mean_ticks",
                    "mean_ship_diff",
                    "failed_action_rate",
                    "opponent_count",
                    "history_opponent_count",
                    "by_opponent",
                    "errors",
                ],
                readonly=True,
            ),
        }

    def _init_params(self):
        super()._init_params()

        self._service_url = self.parameters.get(
            "service_url",
            os.getenv("PLANETWARS_EVALUATOR_URL", "http://planetwars-evaluator:8090"),
        ).rstrip("/")

        self._class_name = self.parameters.get("class_name", "MyAgent")
        self._n_games = int(self.parameters.get("n_games", 10))

        self._opponents = self._parse_opponents(
            self.parameters.get(
                "opponents",
                "CarefulRandomAgent,GreedyHeuristicAgent",
            )
        )

        self._use_history_as_opponents = bool(
            self.parameters.get("use_history_as_opponents", False)
        )
        self._history_opponent_strategy = self.parameters.get(
            "history_opponent_strategy",
            "best",
        )
        self._max_history_opponents = int(
            self.parameters.get("max_history_opponents", 3)
        )
        self._store_failed_solutions_in_history = bool(
            self.parameters.get("store_failed_solutions_in_history", False)
        )

        self._partial_observability = bool(
            self.parameters.get("partial_observability", False)
        )
        self._play_both_sides = bool(self.parameters.get("play_both_sides", True))
        self._seed = int(self.parameters.get("seed", 12345))
        self._timeout_sec = int(self.parameters.get("timeout_sec", 90))
        self._request_timeout_sec = int(
            self.parameters.get("request_timeout_sec", 120)
        )

        self._num_planets = int(self.parameters.get("num_planets", 20))
        self._max_ticks = int(self.parameters.get("max_ticks", 2000))
        self._new_map_each_run = bool(self.parameters.get("new_map_each_run", True))

        self._fitness_metric = self.parameters.get("fitness_metric", "win_rate")

        # In-memory history for this evaluator instance.
        # This resets when the backend restarts or the evaluator instance is recreated.
        self._solution_history: list[dict[str, Any]] = []

    def evaluate(
        self,
        solution: Solution,
        opt_goal: OptimizationGoal = OptimizationGoal.MAXIMIZATION,
    ) -> EvaluatorResult:
        agent_code = solution.get_input() or ""

        historical_opponents = self._select_historical_opponents()
        payload = self._build_payload(agent_code, historical_opponents)

        try:
            response = requests.post(
                f"{self._service_url}/evaluate",
                json=payload,
                timeout=self._request_timeout_sec,
            )
            response.raise_for_status()
            evaluation = response.json()

        except Exception as exc:
            logging.error("Evaluator:PlanetWars: Service call failed: %r", exc)

            evaluation = {
                "ok": False,
                "fitness": 0.0,
                "aggregate": {},
                "by_opponent": {},
                "errors": [
                    "Could not call PlanetWars evaluation service.",
                    repr(exc),
                ],
            }

        fitness = self._extract_fitness(evaluation)
        feedback = self._build_feedback(evaluation, fitness, historical_opponents)

        solution.set_fitness(fitness)
        solution.set_feedback(feedback)
        solution.add_metadata("planetwars_payload", self._metadata_safe_payload(payload))
        solution.add_metadata("planetwars_result", evaluation)

        self._check_if_best(solution)
        self._maybe_store_solution_in_history(solution, evaluation)

        return EvaluatorResult(
            class_ref=type(self),
            fitness=fitness,
            metadata={
                "solution_id": solution.get_id(),
                "fitness_metric": self._fitness_metric,
                "history_size": len(self._solution_history),
                "historical_opponents_used": [
                    item["name"] for item in historical_opponents
                ],
                "evaluation": evaluation,
            },
        )

    @classmethod
    def get_short_name(cls) -> str:
        return "eval.planetwars"

    @classmethod
    def get_long_name(cls) -> str:
        return "Planet Wars"

    @classmethod
    def get_description(cls) -> str:
        return (
            "Evaluator for Python Planet Wars agents. "
            "It sends generated agent code to an external PlanetWars evaluation service "
            "and can use previously evaluated generated agents as additional opponents."
        )

    @classmethod
    def get_tags(cls) -> dict:
        return {"input": {"python"}, "output": {"planetwars_agent"}}

    def _build_payload(
        self,
        agent_code: str,
        historical_opponents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "agent_code": agent_code,
            "class_name": self._class_name,
            "n_games": self._n_games,
            "opponents": self._opponents,
            "custom_opponents": [
                {
                    "name": item["name"],
                    "agent_code": item["agent_code"],
                    "class_name": item.get("class_name", self._class_name),
                }
                for item in historical_opponents
            ],
            "partial_observability": self._partial_observability,
            "play_both_sides": self._play_both_sides,
            "seed": self._seed,
            "timeout_sec": self._timeout_sec,
            "game_params": {
                "num_planets": self._num_planets,
                "max_ticks": self._max_ticks,
                "new_map_each_run": self._new_map_each_run,
            },
        }

    @staticmethod
    def _parse_opponents(value: Any) -> list[str]:
        """
        Main UI format is comma-separated string.

        This also accepts list values for robustness if the evaluator is called
        programmatically.
        """
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]

        if value is None:
            return ["CarefulRandomAgent", "GreedyHeuristicAgent"]

        opponents = [part.strip() for part in str(value).split(",")]
        return [opponent for opponent in opponents if opponent]

    def _select_historical_opponents(self) -> list[dict[str, Any]]:
        if not self._use_history_as_opponents:
            return []

        if self._max_history_opponents <= 0:
            return []

        history = list(self._solution_history)
        if not history:
            return []

        if self._history_opponent_strategy == "best":
            history.sort(key=lambda item: item.get("fitness", 0.0), reverse=True)
            return history[: self._max_history_opponents]

        if self._history_opponent_strategy == "last":
            return history[-self._max_history_opponents :]

        if self._history_opponent_strategy == "all":
            return history[-self._max_history_opponents :]

        return history[: self._max_history_opponents]

    def _maybe_store_solution_in_history(
        self,
        solution: Solution,
        evaluation: dict[str, Any],
    ) -> None:
        ok = bool(evaluation.get("ok", False))

        if not ok and not self._store_failed_solutions_in_history:
            return

        code = solution.get_input() or ""
        if not code.strip():
            return

        fitness = float(solution.get_fitness() or 0.0)
        index = len(self._solution_history)

        self._solution_history.append(
            {
                "name": f"history_{index}_fitness_{fitness:.5f}",
                "agent_code": code,
                "class_name": self._class_name,
                "fitness": fitness,
                "solution_id": solution.get_id(),
                "ok": ok,
            }
        )

        # Keep memory bounded. This does not control how many are used as opponents;
        # that is controlled by max_history_opponents.
        max_stored_history = max(50, self._max_history_opponents * 5)
        if len(self._solution_history) > max_stored_history:
            self._solution_history = self._solution_history[-max_stored_history:]

    def _extract_fitness(self, evaluation: dict[str, Any]) -> float:
        if not evaluation.get("ok", False):
            return 0.0

        aggregate = evaluation.get("aggregate") or {}

        win_rate = float(
            aggregate.get("win_rate", evaluation.get("fitness", 0.0)) or 0.0
        )
        failed_action_rate = float(aggregate.get("failed_action_rate", 0.0) or 0.0)
        mean_ship_diff = float(aggregate.get("mean_ship_diff", 0.0) or 0.0)

        if self._fitness_metric == "win_rate":
            return win_rate

        if self._fitness_metric == "win_rate_minus_failed_action_rate":
            return max(0.0, win_rate - failed_action_rate)

        if self._fitness_metric == "mean_ship_diff":
            return mean_ship_diff

        return win_rate

    def _build_feedback(
        self,
        evaluation: dict[str, Any],
        fitness: float,
        historical_opponents: list[dict[str, Any]],
    ) -> str:
        aggregate = evaluation.get("aggregate") or {}
        by_opponent = evaluation.get("by_opponent") or {}
        errors = evaluation.get("errors") or []

        by_opponent_txt = self._format_by_opponent(by_opponent)
        errors_txt = self._format_errors(errors)

        opponent_count = len(self._opponents) + len(historical_opponents)
        history_opponent_count = len(historical_opponents)

        self._keys = {
            "fitness": f"The scalar fitness is = {fitness}\n",
            "win_rate": (
                f"The aggregate win rate is = {aggregate.get('win_rate', 0.0)}\n"
            ),
            "wins": f"The number of wins is = {aggregate.get('wins', 0)}\n",
            "losses": f"The number of losses is = {aggregate.get('losses', 0)}\n",
            "draws": f"The number of draws is = {aggregate.get('draws', 0)}\n",
            "mean_ticks": (
                f"The average game length in ticks is = "
                f"{aggregate.get('mean_ticks', 0.0)}\n"
            ),
            "mean_ship_diff": (
                f"The average final ship difference is = "
                f"{aggregate.get('mean_ship_diff', 0.0)}\n"
            ),
            "failed_action_rate": (
                f"The failed action rate is = "
                f"{aggregate.get('failed_action_rate', 0.0)}\n"
            ),
            "opponent_count": (
                f"The total number of opponent types used is = {opponent_count}\n"
            ),
            "history_opponent_count": (
                f"The number of historical generated opponents used is = "
                f"{history_opponent_count}\n"
            ),
            "by_opponent": by_opponent_txt,
            "errors": errors_txt,
        }

        template = self.get_feedback_msg_template()
        if not template:
            template = self.get_parameters()["feedback_msg_template"].default

        try:
            return template.format(**self._keys)
        except Exception:
            logging.exception("Evaluator:PlanetWars: Could not format feedback.")
            return (
                "The Planet Wars evaluation finished, but the feedback template "
                "could not be formatted.\n\n"
                f"Fitness: {fitness}\n\n"
                f"Raw result:\n{json.dumps(evaluation, indent=2, ensure_ascii=False)}"
            )

    @staticmethod
    def _format_by_opponent(by_opponent: dict[str, Any]) -> str:
        if not by_opponent:
            return "No per-opponent results are available.\n"

        lines = ["Per-opponent results:"]
        for opponent, metrics in by_opponent.items():
            if not isinstance(metrics, dict):
                lines.append(f"- {opponent}: {metrics}")
                continue

            lines.append(
                "- {opponent}: win_rate={win_rate}, wins={wins}, "
                "losses={losses}, draws={draws}, failed_action_rate={failed}".format(
                    opponent=opponent,
                    win_rate=metrics.get("win_rate", 0.0),
                    wins=metrics.get("wins", 0),
                    losses=metrics.get("losses", 0),
                    draws=metrics.get("draws", 0),
                    failed=metrics.get("failed_action_rate", 0.0),
                )
            )

        return "\n".join(lines) + "\n"

    @staticmethod
    def _format_errors(errors: list[Any]) -> str:
        if not errors:
            return "No evaluator errors were reported.\n"

        lines = ["Evaluator errors:"]
        for error in errors[:5]:
            lines.append(f"- {error}")

        if len(errors) > 5:
            lines.append(f"- ... and {len(errors) - 5} more errors")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _metadata_safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
        safe_payload = copy.deepcopy(payload)

        code = safe_payload.pop("agent_code", "")
        safe_payload["agent_code_chars"] = len(code)

        custom_opponents = safe_payload.get("custom_opponents", [])
        for item in custom_opponents:
            opponent_code = item.pop("agent_code", "")
            item["agent_code_chars"] = len(opponent_code)

        return safe_payload

    def _check_if_best(self, solution: Solution) -> bool:
        if self._best is None or solution.get_fitness() >= self._best.get_fitness():
            self._best = copy.deepcopy(solution)
            return True

        return False
