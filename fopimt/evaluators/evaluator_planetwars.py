import copy
import json
import logging
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .evaluator import Evaluator
from ..solutions.solution import Solution
from ..loader import Parameter, PrimitiveType


@dataclass
class OpponentStats:
    name: str
    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    errors: int = 0
    score_diffs: List[float] = None  # candidate_score - opp_score

    def __post_init__(self):
        if self.score_diffs is None:
            self.score_diffs = []

    def add_result(self, winner: str, score_cand: Optional[float], score_opp: Optional[float], had_error: bool):
        self.games += 1
        if had_error:
            self.errors += 1

        w = (winner or "Draw").upper()
        if w == "A":      # candidate is A in our accounting
            self.wins += 1
        elif w == "B":
            self.losses += 1
        else:
            self.draws += 1

        if isinstance(score_cand, (int, float)) and isinstance(score_opp, (int, float)):
            self.score_diffs.append(float(score_cand) - float(score_opp))

    @property
    def points(self) -> float:
        # standard: win=1, draw=0.5, loss=0
        return float(self.wins) + 0.5 * float(self.draws)

    @property
    def win_rate(self) -> float:
        return self.points / self.games if self.games > 0 else 0.0

    @property
    def avg_score_diff(self) -> float:
        return float(np.mean(self.score_diffs)) if self.score_diffs else 0.0


class EvaluatorPlanetWars(Evaluator):
    """
    Planet Wars RTS evaluator that runs matches against:
      - all built-in python agents in the planet-wars-rts repo
      - the current best agent discovered so far in this EASE run (self._best)

    IMPORTANT: This evaluator uses variables/constants below — no config is read from params.
    """

    # ============================================================
    # 1) HARD-CODED CONFIG
    # ============================================================

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Build absolute path to planet-wars-rts
    PLANETWARS_DIR = os.path.join(
        BASE_DIR,
        "..",
        "resource",
        "planetwarsrts",
        "planet-wars-rts",
    )
    PLANETWARS_DIR = os.path.normpath(PLANETWARS_DIR)

    # Path to the match runner inside planet-wars-rts.
    MATCH_RUNNER = Path("tools") / "pw_match.py"

    # Built-in opponent agents (Python) available in the uploaded repo.
    # These correspond to builtin names accepted by pw_match.py.
    BUILTIN_OPPONENTS = [
        "builtin:PureRandomAgent",
        "builtin:CarefulRandomAgent",
        "builtin:GreedyHeuristicAgent",
    ]

    # Number of seeds (each seed is one game; if SWAP_SIDES=True, it's 2 games per seed).
    N_SEEDS = 6

    # If True, play each seed twice: candidate as Player1 and candidate as Player2.
    SWAP_SIDES = True

    # Timeout per match runner invocation (seconds)
    TIMEOUT_S = 60.0

    # Optional: override game params (maxTicks, numPlanets, etc.) by passing JSON to runner.
    # Leave empty string to use repo defaults.
    # Example:
    #   PARAMS_JSON_INLINE = '{"max_ticks": 600, "num_planets": 12}'
    PARAMS_JSON_INLINE = ""

    # Fitness weights:
    # - Builtins are always included equally
    # - "best(previous)" is included with this relative weight when it exists
    BEST_OPP_WEIGHT = 1.0

    # Tie-break weight by avg score diff (usually small)
    SCORE_DIFF_WEIGHT = 0.01

    # ============================================================
    # 2) TEMPLATES (still exposed as standard Evaluator fields)
    # ============================================================

    INIT_TEMPLATE = (
        "You are designing a **Planet Wars RTS** agent.\n\n"
        "Return a single Python file that defines:\n"
        "```python\n"
        "def get_action(state: dict):\n"
        "    # Return None to do nothing\n"
        "    # or {'source': int, 'target': int, 'ships': int}\n"
        "    ...\n"
        "```\n\n"
        "Rules:\n"
        "- Be fast (called every tick).\n"
        "- No file I/O, no networking, no prints.\n"
        "- Be defensive: use state.get(...) and return None if unsure.\n"
    )

    FEEDBACK_TEMPLATE = (
        "### Planet Wars evaluation\n"
        "{summary}\n\n"
        "#### Opponent breakdown\n"
        "{table}\n\n"
        "{notes}"
    )

    # ------------------------------------------------------------
    # Evaluator API glue
    # ------------------------------------------------------------

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        # Keep these for compatibility with the framework UI, but we don't use them for config.
        return {
            "feedback_msg_template": Parameter(
                short_name="feedback_msg_template",
                type=PrimitiveType.markdown,
                long_name="Template for a feedback message",
                description="Feedback message for evaluation. Can use {keywords}",
                default=cls.FEEDBACK_TEMPLATE,
                readonly=True,
            ),
            "init_msg_template": Parameter(
                short_name="init_msg_template",
                type=PrimitiveType.markdown,
                long_name="Template for an initial message",
                description="Initial message for evaluation.",
                default=cls.INIT_TEMPLATE,
                readonly=True,
            ),
            "keywords": Parameter(
                short_name="keywords",
                type=PrimitiveType.enum,
                long_name="Feedback keywords",
                description="Keyword-based feedback sentences (unused here).",
                enum_options=[],
                readonly=True,
            ),
        }

    def _init_params(self):
        # Run base init (sets _best, templates, etc.)
        super()._init_params()

        # Force our templates and ignore external params for evaluation settings.
        self._init_msg_template = self.INIT_TEMPLATE
        self._feedback_msg_template = self.FEEDBACK_TEMPLATE
        self._feedback_keywords = []
        self._keys = {}

    @classmethod
    def get_short_name(cls) -> str:
        return "eval.planetwars"

    @classmethod
    def get_long_name(cls) -> str:
        return "Planet Wars RTS"

    @classmethod
    def get_description(cls) -> str:
        return "Evaluates Planet Wars agents against built-ins + current best agent."

    @classmethod
    def get_tags(cls) -> dict:
        return {
            "input": {"python"},
            "output": {"planetwars_agent"},
        }

    # ------------------------------------------------------------
    # Core evaluation logic
    # ------------------------------------------------------------

    def evaluate(self, solution: Solution) -> float:
        """
        Runs matches:
          candidate vs each builtin opponent
          candidate vs previous best (if available and different)

        Fitness: weighted mean win-rate across opponents * 100 + small tie-break by avg score diff.
        """
        # 1) Validate environment paths early with a helpful error
        pw_dir = Path(self.PLANETWARS_DIR)
        runner_path = (pw_dir / self.MATCH_RUNNER).resolve()

        if not pw_dir.exists():
            raise FileNotFoundError(
                f"Planet Wars repo not found at {pw_dir}\n"
                f"Edit EvaluatorPlanetWars.PLANETWARS_DIR to point to your local planet-wars-rts checkout."
            )
        if not runner_path.exists():
            raise FileNotFoundError(
                f"Match runner not found at {runner_path}\n"
                f"Expected: {self.MATCH_RUNNER} inside {pw_dir}\n"
                f"Create it (tools/pw_match.py) as discussed, or update MATCH_RUNNER."
            )

        # 2) Syntax check candidate
        try:
            compile(solution.get_input(), "candidate_agent.py", "exec")
        except Exception as e:
            feedback = (
                "### Planet Wars evaluation\n"
                "Your submitted Python code does not compile.\n\n"
                f"**Error:** `{repr(e)}`\n\n"
                "Fix syntax errors and try again."
            )
            solution.set_feedback(feedback)
            solution.set_fitness(float("-inf"))
            return float("-inf")

        # 3) Prepare opponents list (builtins + previous best)
        opponents: List[Tuple[str, str]] = []  # (label, spec)
        for spec in self.BUILTIN_OPPONENTS:
            opponents.append((spec.replace("builtin:", ""), spec))

        # Add previous best agent (if exists and different)
        prev_best_code: Optional[str] = None
        if self._best is not None:
            try:
                prev_best_code = self._best.get_input()
            except Exception:
                prev_best_code = None

        include_prev_best = bool(prev_best_code and prev_best_code.strip() and prev_best_code != solution.get_input())

        # 4) Run matches in a temp workspace
        with tempfile.TemporaryDirectory(prefix="ease_pw_eval_") as td:
            td_path = Path(td)

            cand_path = td_path / "candidate_agent.py"
            cand_path.write_text(solution.get_input(), encoding="utf-8")

            best_path: Optional[Path] = None
            if include_prev_best:
                best_path = td_path / "best_agent_prev.py"
                best_path.write_text(prev_best_code, encoding="utf-8")
                opponents.append(("best(previous)", str(best_path)))

            # 5) Evaluate per opponent
            per_opp: Dict[str, OpponentStats] = {}
            for label, spec in opponents:
                per_opp[label] = OpponentStats(name=label)

            # Base seeds
            base_seed = int(solution.get_metadata().get("seed", 0)) if isinstance(solution.get_metadata(), dict) else 0

            for label, opp_spec in opponents:
                stats = per_opp[label]

                for i in range(self.N_SEEDS):
                    seed = base_seed + i

                    # candidate as A
                    r1 = self._run_one(
                        runner_path=runner_path,
                        cwd=pw_dir,
                        agent_a=str(cand_path),
                        agent_b=str(opp_spec),
                        seed=seed,
                        out_json=td_path / f"res_{label}_A_{seed}.json",
                    )
                    stats.add_result(
                        winner=r1.get("winner", "Draw"),
                        score_cand=r1.get("score_a"),
                        score_opp=r1.get("score_b"),
                        had_error=bool(r1.get("error")),
                    )

                    # candidate as B
                    if self.SWAP_SIDES:
                        seed2 = seed + 10_000_000  # keep deterministic but distinct
                        r2 = self._run_one(
                            runner_path=runner_path,
                            cwd=pw_dir,
                            agent_a=str(opp_spec),
                            agent_b=str(cand_path),
                            seed=seed2,
                            out_json=td_path / f"res_{label}_B_{seed2}.json",
                        )
                        # Here candidate is B, so invert A/B for accounting
                        # winner == "B" => candidate win
                        winner2 = (r2.get("winner", "Draw") or "Draw").upper()
                        if winner2 == "B":
                            w = "A"  # candidate win in our accounting
                        elif winner2 == "A":
                            w = "B"
                        else:
                            w = "Draw"

                        # score diff should be candidate_score - opp_score => score_b - score_a
                        sa = r2.get("score_a")
                        sb = r2.get("score_b")
                        stats.add_result(
                            winner=w,
                            score_cand=sb,
                            score_opp=sa,
                            had_error=bool(r2.get("error")),
                        )

            # 6) Compute fitness
            # Weighted mean of win rates across opponents:
            # - all builtins weight=1
            # - best(previous) weight=BEST_OPP_WEIGHT
            weights = []
            win_rates = []
            score_diffs = []

            for label, _spec in opponents:
                st = per_opp[label]
                w = self.BEST_OPP_WEIGHT if label == "best(previous)" else 1.0
                weights.append(w)
                win_rates.append(st.win_rate)
                score_diffs.append(st.avg_score_diff)

            wsum = float(np.sum(weights)) if weights else 1.0
            overall_win_rate = float(np.sum(np.array(weights) * np.array(win_rates)) / wsum) if weights else 0.0
            overall_avg_score_diff = float(np.mean(score_diffs)) if score_diffs else 0.0

            fitness = 100.0 * overall_win_rate + self.SCORE_DIFF_WEIGHT * overall_avg_score_diff

            # 7) Feedback message
            summary = (
                f"- **Opponents:** {len(opponents)} "
                f"({' + '.join([lbl for lbl, _ in opponents])})\n"
                f"- **Seeds:** {self.N_SEEDS} "
                f"({'2 games/seed (swap sides)' if self.SWAP_SIDES else '1 game/seed'})\n"
                f"- **Overall win rate (weighted):** {overall_win_rate:.3f}\n"
                f"- **Overall avg score diff:** {overall_avg_score_diff:.3f}\n"
                f"- **Fitness:** {fitness:.3f}\n"
            )

            table = self._format_table(per_opp, opponents)

            notes_parts = []
            if include_prev_best:
                notes_parts.append("- Included **best(previous)** as an additional opponent.\n")
            else:
                notes_parts.append("- No previous best agent available yet (or candidate equals best).\n")

            notes_parts.append(
                "- Fitness uses win-rate (win=1, draw=0.5, loss=0) averaged across opponents, "
                "with a small score-difference tie-breaker.\n"
            )

            feedback = self.get_feedback_msg_template().format(
                summary=summary,
                table=table,
                notes="".join(notes_parts),
            )

            solution.set_feedback(feedback)
            solution.set_fitness(fitness)

            # 8) Update best
            self._check_if_best(solution)

            return fitness

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def _run_one(
        self,
        runner_path: Path,
        cwd: Path,
        agent_a: str,
        agent_b: str,
        seed: int,
        out_json: Path,
    ) -> Dict[str, Any]:
        """
        Runs exactly one match via pw_match.py and returns parsed JSON.
        If the runner fails or times out, returns a JSON dict with "error".
        """
        out_json.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(runner_path),
            "--agent-a", agent_a,
            "--agent-b", agent_b,
            "--seed", str(int(seed)),
            "--out", str(out_json),
        ]
        if self.PARAMS_JSON_INLINE.strip():
            cmd += ["--params", self.PARAMS_JSON_INLINE.strip()]

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=float(self.TIMEOUT_S),
            )
        except subprocess.TimeoutExpired:
            return {"winner": "Draw", "error": "timeout", "seed": int(seed)}

        if proc.returncode != 0:
            return {
                "winner": "Draw",
                "error": "runner_failed",
                "returncode": proc.returncode,
                "seed": int(seed),
                "stdout_tail": (proc.stdout or "")[-800:],
                "stderr_tail": (proc.stderr or "")[-800:],
            }

        # Prefer reading the JSON file the runner wrote
        try:
            if out_json.exists():
                return json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:
            pass

        # Fallback: try parse stdout last line as JSON
        s = (proc.stdout or "").strip()
        if not s:
            return {"winner": "Draw", "error": "empty_stdout", "seed": int(seed)}
        last = s.splitlines()[-1].strip()
        try:
            return json.loads(last)
        except Exception:
            return {"winner": "Draw", "error": "could_not_parse_json", "seed": int(seed), "stdout_tail": last[:400]}

    def _format_table(self, per_opp: Dict[str, OpponentStats], opponents: List[Tuple[str, str]]) -> str:
        """
        Returns a markdown table with per-opponent results.
        """
        header = "| Opponent | Games | W | D | L | WinRate | AvgScoreDiff | Errors |\n"
        sep = "|---|---:|---:|---:|---:|---:|---:|---:|\n"
        rows = []
        for label, _spec in opponents:
            st = per_opp[label]
            rows.append(
                f"| {st.name} | {st.games} | {st.wins} | {st.draws} | {st.losses} | "
                f"{st.win_rate:.3f} | {st.avg_score_diff:.3f} | {st.errors} |"
            )
        return header + sep + "\n".join(rows)

    def _check_if_best(self, solution: Solution) -> bool:
        """
        Same pattern as evaluator_2048.py: store best by fitness.
        """
        if self._best is None or solution.get_fitness() >= self._best.get_fitness():
            self._best = copy.deepcopy(solution)
            return True
        return False
