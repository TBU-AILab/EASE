#!/usr/bin/env python3
"""
pw_match.py - run ONE Planet Wars match and emit JSON.

Usage examples:

# Builtin vs builtin
python tools/pw_match.py \
  --agent-a builtin:GreedyHeuristicAgent \
  --agent-b builtin:CarefulRandomAgent \
  --seed 123 \
  --out /tmp/result.json

# Candidate python file vs builtin
python tools/pw_match.py \
  --agent-a /path/to/candidate_agent.py \
  --agent-b builtin:GreedyHeuristicAgent \
  --seed 123 \
  --out /tmp/result.json

Expected JSON output format (minimal):
{
  "winner": "A" | "B" | "Draw",
  "score_a": <float>,
  "score_b": <float>,
  "ticks": <int>,
  "seed": <int>,
  "meta": {...}
}

Notes:
- This runner uses the Python game implementation under app/src/main/python.
- For EASE, agent-a and agent-b can be:
    * builtin:PureRandomAgent
    * builtin:CarefulRandomAgent
    * builtin:GreedyHeuristicAgent
    * a filesystem path to a .py file (candidate agent)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union


# ----------------------------
# Path bootstrap
# ----------------------------

def _add_repo_python_path() -> Tuple[Path, Path]:
    """
    Adds <repo_root>/app/src/main/python to sys.path so we can import core/* and agents/*.
    Assumes this file is at <repo_root>/tools/pw_match.py
    """
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[1]
    py_src = repo_root / "app" / "src" / "main" / "python"
    if not py_src.exists():
        raise FileNotFoundError(f"Could not find python sources at: {py_src}")
    sys.path.insert(0, str(py_src))
    return repo_root, py_src


REPO_ROOT, PY_SRC = _add_repo_python_path()

# Now imports work
from core.game_state import Action, GameParams, GameState, Player  # noqa: E402
from core.forward_model import ForwardModel  # noqa: E402
from core.game_runner import GameRunner  # noqa: E402
from agents.planet_wars_agent import PlanetWarsAgent, PlanetWarsPlayer  # noqa: E402
from agents.random_agents import PureRandomAgent, CarefulRandomAgent  # noqa: E402
from agents.greedy_heuristic_agent import GreedyHeuristicAgent  # noqa: E402


# ----------------------------
# Agent loading / wrapping
# ----------------------------

BuiltinFactory = Callable[[], PlanetWarsAgent]


BUILTINS: Dict[str, BuiltinFactory] = {
    "PureRandomAgent": PureRandomAgent,
    "CarefulRandomAgent": CarefulRandomAgent,
    "GreedyHeuristicAgent": GreedyHeuristicAgent,
}


def _first_present(d: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[Any]:
    for k in keys:
        if k in d:
            return d[k]
    return None


def _coerce_action(
    res: Any,
    player: Player,
) -> Action:
    """
    Convert whatever candidate returned into a core.game_state.Action.
    Supported returns:
      - None -> do nothing
      - Action -> forced to this player
      - dict with ids + ships (snake_case or camelCase or short keys)
      - tuple/list: (source_id, destination_id, num_ships)
    """
    if res is None:
        return Action.do_nothing()

    # If candidate explicitly indicates "no-op"
    if isinstance(res, str) and res.strip().lower() in ("none", "noop", "no-op", "donothing", "do_nothing"):
        return Action.do_nothing()

    # Native Action
    if isinstance(res, Action):
        # enforce player_id correctness (avoid weird bugs)
        return Action(
            player_id=player,
            source_planet_id=int(res.source_planet_id),
            destination_planet_id=int(res.destination_planet_id),
            num_ships=float(res.num_ships),
        )

    # Tuple/list
    if isinstance(res, (tuple, list)) and len(res) >= 3:
        try:
            return Action(
                player_id=player,
                source_planet_id=int(res[0]),
                destination_planet_id=int(res[1]),
                num_ships=float(res[2]),
            )
        except Exception:
            return Action.do_nothing()

    # Dict
    if isinstance(res, dict):
        # accept multiple key styles
        src = _first_present(res, ("source_planet_id", "sourcePlanetId", "source", "src", "from", "from_id", "fromId"))
        dst = _first_present(res, ("destination_planet_id", "destinationPlanetId", "destination", "dst", "to", "to_id", "toId"))
        ships = _first_present(res, ("num_ships", "numShips", "ships", "n_ships", "amount"))

        if src is None or dst is None or ships is None:
            return Action.do_nothing()

        try:
            return Action(
                player_id=player,
                source_planet_id=int(src),
                destination_planet_id=int(dst),
                num_ships=float(ships),
            )
        except Exception:
            return Action.do_nothing()

    # Anything else -> no-op
    return Action.do_nothing()


class FunctionAgent(PlanetWarsPlayer):
    """
    Wraps a simple function:

        def get_action(state_dict: dict) -> dict|tuple|Action|None

    We pass a dict created via `game_state.model_dump(by_alias=True)` so keys are camelCase.
    """

    def __init__(self, fn: Callable[[Dict[str, Any]], Any], name: str = "FunctionAgent"):
        super().__init__()
        self._fn = fn
        self._name = name
        self.n_exceptions = 0

    def get_action(self, game_state: GameState) -> Action:
        try:
            payload = game_state.model_dump(by_alias=True)
            res = self._fn(payload)
            return _coerce_action(res, self.player)
        except Exception:
            self.n_exceptions += 1
            return Action.do_nothing()

    def get_agent_type(self) -> str:
        return self._name


def _load_module_from_file(py_file: Path):
    spec = importlib.util.spec_from_file_location(py_file.stem, str(py_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for: {py_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_agent(agent_spec: str) -> PlanetWarsAgent:
    """
    agent_spec:
      - builtin:<Name>  (PureRandomAgent, CarefulRandomAgent, GreedyHeuristicAgent)
      - /path/to/file.py  (candidate)
    """
    agent_spec = (agent_spec or "").strip()
    if not agent_spec:
        raise ValueError("Empty agent spec")

    if agent_spec.startswith("builtin:"):
        name = agent_spec.split(":", 1)[1].strip()
        if name not in BUILTINS:
            raise ValueError(f"Unknown builtin agent '{name}'. Valid: {sorted(BUILTINS.keys())}")
        return BUILTINS[name]()

    # treat as path
    p = Path(agent_spec).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Agent file not found: {p}")

    mod = _load_module_from_file(p)

    # Prefer make_agent()
    if hasattr(mod, "make_agent") and callable(getattr(mod, "make_agent")):
        agent = mod.make_agent()
        if not hasattr(agent, "get_action"):
            raise TypeError("make_agent() did not return an agent with get_action()")
        return agent

    # Prefer Agent class
    if hasattr(mod, "Agent") and callable(getattr(mod, "Agent")):
        agent = mod.Agent()
        if not hasattr(agent, "get_action"):
            raise TypeError("Agent class instance has no get_action()")
        return agent

    # Fallback: get_action function
    if hasattr(mod, "get_action") and callable(getattr(mod, "get_action")):
        return FunctionAgent(mod.get_action, name=p.stem)

    raise ValueError(
        f"Could not load agent from {p}. Provide one of:\n"
        "- def make_agent() -> PlanetWarsAgent\n"
        "- class Agent(PlanetWarsPlayer)\n"
        "- def get_action(state_dict) -> (Action|dict|tuple|None)\n"
    )


# ----------------------------
# Running one match
# ----------------------------

@dataclass
class MatchResult:
    winner: str           # "A" | "B" | "Draw"
    score_a: float
    score_b: float
    ticks: int
    seed: int
    meta: Dict[str, Any]


def run_one_match(agent_a: PlanetWarsAgent, agent_b: PlanetWarsAgent, seed: int, params: GameParams) -> MatchResult:
    # Ensure deterministic map + deterministic random agents (they use the global random module)
    random.seed(seed)

    # Reset global counters (they accumulate across runs otherwise)
    ForwardModel.n_actions = 0
    ForwardModel.n_failed_actions = 0
    ForwardModel.n_updates = 0

    runner = GameRunner(agent_a, agent_b, params)
    model = runner.run_game()

    ships_a = float(model.get_ships(Player.Player1))
    ships_b = float(model.get_ships(Player.Player2))
    ticks = int(model.state.game_tick)

    leader = model.get_leader()
    if leader == Player.Player1:
        winner = "A"
    elif leader == Player.Player2:
        winner = "B"
    else:
        winner = "Draw"

    meta: Dict[str, Any] = {
        "ships_player1": ships_a,
        "ships_player2": ships_b,
        "leader": leader.value,
        "n_actions": ForwardModel.n_actions,
        "n_failed_actions": ForwardModel.n_failed_actions,
        "n_updates": ForwardModel.n_updates,
        "max_ticks": params.max_ticks,
        "num_planets": params.num_planets,
    }

    # If the wrapped agent has exception counters, expose them
    if hasattr(agent_a, "n_exceptions"):
        meta["agent_a_exceptions"] = int(getattr(agent_a, "n_exceptions"))
    if hasattr(agent_b, "n_exceptions"):
        meta["agent_b_exceptions"] = int(getattr(agent_b, "n_exceptions"))

    return MatchResult(
        winner=winner,
        score_a=ships_a,
        score_b=ships_b,
        ticks=ticks,
        seed=seed,
        meta=meta,
    )


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run ONE Planet Wars match and emit JSON.")
    ap.add_argument("--agent-a", required=True, help="builtin:<Name> or path/to/agent.py")
    ap.add_argument("--agent-b", required=True, help="builtin:<Name> or path/to/agent.py")
    ap.add_argument("--seed", type=int, required=True, help="Random seed for map + agents")
    ap.add_argument("--out", required=True, help="Path to write JSON result")
    ap.add_argument("--params-json", default="", help="Optional JSON file with GameParams overrides")
    ap.add_argument("--params", default="", help="Optional inline JSON string with GameParams overrides")
    ap.add_argument("--num-planets", type=int, default=None, help="Override: number of planets")
    ap.add_argument("--max-ticks", type=int, default=None, help="Override: max ticks")
    ap.add_argument("--quiet", action="store_true", help="Only print JSON (no extra text)")
    return ap.parse_args()


def build_params(args: argparse.Namespace) -> GameParams:
    # Start with defaults from the repo
    params = GameParams()

    # Apply overrides from file/string if provided
    def apply_dict(d: Dict[str, Any]) -> None:
        nonlocal params
        # pydantic can validate/merge nicely
        merged = {**params.model_dump(by_alias=False), **d}
        params = GameParams.model_validate(merged)

    if args.params_json:
        p = Path(args.params_json).expanduser().resolve()
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("--params-json must contain an object/dict")
        apply_dict(data)

    if args.params:
        data = json.loads(args.params)
        if not isinstance(data, dict):
            raise ValueError("--params must be a JSON object/dict")
        apply_dict(data)

    # Explicit CLI overrides win last
    if args.num_planets is not None:
        params.num_planets = int(args.num_planets)
    if args.max_ticks is not None:
        params.max_ticks = int(args.max_ticks)

    return params


def main() -> int:
    args = parse_args()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        params = build_params(args)
        agent_a = load_agent(args.agent_a)
        agent_b = load_agent(args.agent_b)

        result = run_one_match(agent_a, agent_b, seed=int(args.seed), params=params)

        payload = {
            "winner": result.winner,
            "score_a": result.score_a,
            "score_b": result.score_b,
            "ticks": result.ticks,
            "seed": result.seed,
            "meta": result.meta,
        }

        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload))

        return 0

    except Exception as e:
        # Write a “safe” JSON so the evaluator can still parse it
        err_payload = {
            "winner": "Draw",
            "score_a": 0.0,
            "score_b": 0.0,
            "ticks": 0,
            "seed": int(args.seed) if hasattr(args, "seed") else 0,
            "error": repr(e),
            "traceback": traceback.format_exc()[-4000:],  # keep it bounded
        }
        try:
            out_path.write_text(json.dumps(err_payload, indent=2), encoding="utf-8")
        except Exception:
            pass
        print(json.dumps(err_payload))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
