import copy
import logging

import numpy as np
import pandas as pd

from ..loader import Parameter, PrimitiveType
from ..resource.metahuristic.metaheuristic_runner import Runner
from ..resource.resource import Resource
from ..solutions.solution import Solution
from ..task_dto import OptimizationGoal
from ..utils.history_context import (
    escape_delimiters,
    fill_template,
    format_algorithm_id,
    format_score,
    parse_tags,
)
from ..utils.import_utils import dynamic_import
from ..utils.token_utils import count_common_tokens
from .evaluator import Evaluator, EvaluatorResult

CODE_CONTEXTS = ["none", "last_best", "all"]
FITNESS_STATS = ["mean", "min", "median"]
FUNCTIONS = ["resource.gnbg.f_24", "resource.cec2017.f_30", "resource.bbob.f_24"]

DEFAULT_ESCAPE_TAGS = (
    "SOURCE_CODE_CONTEXT, SOURCE_CODE, DEVELOPMENT_SCORE, ALGORITHM_HISTORY, ALGORITHM, SUMMARY_CONTEXT, "
    "INVALID_SUMMARY"
)

DEFAULT_CODE_BLOCK = """<SOURCE_CODE_CONTEXT>
{records}
</SOURCE_CODE_CONTEXT>"""

DEFAULT_CODE_RECORD = """  <ALGORITHM id="{id}" iteration="{iteration}" role="{role}">
    <DEVELOPMENT_SCORE>{score}</DEVELOPMENT_SCORE>
    <SOURCE_CODE>
{code}
    </SOURCE_CODE>
  </ALGORITHM>"""

DEFAULT_INIT_MSG = """You are an expert in continuous numerical optimization and Python
programming. Design an effective algorithm for minimizing an unknown,
single-objective, box-constrained black-box function. You may adapt or
combine existing optimization ideas, but the result must be a complete and
operational implementation.

Return exactly one self-contained Python program that defines this function:

def run(func, dim, bounds, max_time):
    # algorithm body
    return best

Contract and restrictions:
- func(x) returns one objective value for a NumPy vector x; lower is better.
- dim is the number of decision variables.
- bounds is an array-like object of shape (dim, 2), with one [lower, upper]
  pair per variable.
- max_time is the maximum permitted wall-clock time in seconds.
- Access the objective only by calling func. Do not inspect, identify, or
  make assumptions about its implementation, benchmark name, optimum,
  gradients, or analytical properties.
- Evaluate only finite vectors inside the supplied bounds.
- Return the best finite objective value that the algorithm has actually
  observed before max_time expires.
- Use only NumPy and the Python standard library. Do not use files, the
  network, subprocesses, or multiprocessing.
- Do not include benchmark functions, tests, statistical analyses, example
  executions, Markdown fences, or explanatory prose in the response.

The following minimal random-search example illustrates the required
interface, not the expected algorithmic quality:

import time
import numpy as np

def run(func, dim, bounds, max_time):
    start = time.perf_counter()
    bounds = np.asarray(bounds, dtype=float)
    best = float("inf")
    while time.perf_counter() - start < max_time:
        x = np.random.uniform(bounds[:, 0], bounds[:, 1], size=dim)
        value = float(func(x))
        if np.isfinite(value) and value < best:
            best = value
    return best

Now produce a substantially more capable optimizer under the same interface
and restrictions. Return only its complete Python source code."""


class EvaluatorPaperContextSummarizer(Evaluator):
    """
    Evaluator for the Paper Context Summarizer experiments. Based on the metaheuristic evaluator: the solution must be
    Python code following the template run(func, dim, bounds, max_time) defined by the Runner class.

    The feedback contains the source-code context: previous valid algorithms with their development scores
    (none | last_best | all). The LLM summary of the history is provided by the analysis module anal.historysummary,
    whose feedback follows the feedback of the evaluator. The instructions around the context (iteration
    instruction, untrusted-context preamble, closing instruction) belong to the repeated message of the Task.
    All texts are parameters.
    """

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        return {
            "code_context": Parameter(
                short_name="code_context",
                type=PrimitiveType.enum,
                long_name="Code context",
                description="Previous valid algorithms attached as source code with their development scores: "
                "all = whole history (chronological), last_best = the last and the best algorithm (sent twice "
                "when they are the same).",
                enum_options=CODE_CONTEXTS,
                default="last_best",
            ),
            "function": Parameter(
                short_name="function",
                type=PrimitiveType.enum,
                long_name="Benchmark function",
                description="Benchmark function (D=30): GNBG-II f24, CEC 2017 F30 (Composition Function 10), "
                "BBOB f24 (Lunacek bi-Rastrigin, instance 1).",
                enum_options=FUNCTIONS,
                default="resource.gnbg.f_24",
            ),
            "time": Parameter(
                short_name="time",
                type=PrimitiveType.int,
                long_name="Max time [s]",
                description="Max time of one run of the algorithm in seconds.",
                default=30,
            ),
            "runs": Parameter(
                short_name="runs",
                type=PrimitiveType.int,
                long_name="Runs",
                description="Number of independent runs per algorithm. 0 = use the value of the benchmark resource.",
                default=0,
            ),
            "fitness_stat": Parameter(
                short_name="fitness_stat",
                type=PrimitiveType.enum,
                long_name="Fitness statistic",
                description="Statistic over runs used as the development score (fitness) of the algorithm.",
                enum_options=FITNESS_STATS,
                default="mean",
            ),
            "code_block": Parameter(
                short_name="code_block",
                type=PrimitiveType.markdown,
                long_name="Code block template",
                description="Source-code context block. Placeholder: {records}.",
                default=DEFAULT_CODE_BLOCK,
            ),
            "code_record": Parameter(
                short_name="code_record",
                type=PrimitiveType.markdown,
                long_name="Code record template",
                description="One algorithm of the code block. Placeholders: {id}, {iteration}, {role} (history, "
                "last, best), {score}, {code}.",
                default=DEFAULT_CODE_RECORD,
            ),
            "id_format": Parameter(
                short_name="id_format",
                type=PrimitiveType.str,
                long_name="Algorithm id format",
                description="Identifier of the k-th valid algorithm, placeholder {index}. Must match "
                "anal.historysummary.",
                default="A{index}",
            ),
            "score_format": Parameter(
                short_name="score_format",
                type=PrimitiveType.str,
                long_name="Score format",
                description="Python format specification of the scores, e.g. .6e.",
                default=".6e",
            ),
            "escape_tags": Parameter(
                short_name="escape_tags",
                type=PrimitiveType.str,
                long_name="Escaped delimiters",
                description="Comma separated prompt delimiters that are escaped inside the code.",
                default=DEFAULT_ESCAPE_TAGS,
            ),
            "error_msg": Parameter(
                short_name="error_msg",
                type=PrimitiveType.markdown,
                long_name="Error message",
                description="Feedback for an algorithm that could not be evaluated. Placeholder: {reason}.",
                default="Error during solution evaluation: {reason}\n. Try to fix it.",
            ),
            "feedback_msg_template": Parameter(
                short_name="feedback_msg_template",
                type=PrimitiveType.markdown,
                long_name="Template for a feedback message",
                description="Feedback message for evaluation. Placeholder: {context} (the code block, empty for "
                "code_context = none).",
                default="{context}",
            ),
            "init_msg_template": Parameter(
                short_name="init_msg_template",
                type=PrimitiveType.markdown,
                long_name="Template for an initial message",
                description="Initial message for evaluation. Specific for each evaluator.",
                default=DEFAULT_INIT_MSG,
                readonly=True,
            ),
            "keywords": Parameter(
                short_name="keywords",
                type=PrimitiveType.enum,
                long_name="Feedback keywords",
                description="Feedback keyword-based sentences",
                enum_options=["context"],
                readonly=True,
            ),
        }

    def _init_params(self):
        super()._init_params()
        defaults = self.get_parameters()

        def param(name):
            return self.parameters.get(name, defaults[name].default)

        self._code_context = param("code_context")
        if self._code_context not in CODE_CONTEXTS:
            raise ValueError(f"Unknown code_context '{self._code_context}'")
        self._fitness_stat = param("fitness_stat")
        self._max_time = int(param("time"))
        self._runs = int(param("runs"))
        self._code_block = param("code_block")
        self._code_record = param("code_record")
        self._id_format = param("id_format")
        self._score_format = param("score_format")
        self._tags = parse_tags(param("escape_tags"))
        self._error_msg = param("error_msg")

        self._function_name = param("function")
        self.function = Resource.get_resource_function(
            self._function_name, "metabenchmark"
        )()
        if self._runs > 0:
            self.function["runs"] = self._runs

        self._solution_history: list[
            Solution
        ] = []  # valid solutions in generation order
        self._best_index: int | None = (
            None  # index of the best solution in _solution_history
        )

    ####################################################################
    #########  Public functions
    ####################################################################
    def evaluate(
        self,
        solution: Solution,
        opt_goal: OptimizationGoal = OptimizationGoal.MINIMIZATION,
    ) -> EvaluatorResult:
        """
        Evaluates the algorithm on the benchmark function and composes the source-code context.
        Solutions that cannot be executed are marked as invalid (metadata 'invalid') and not added to the history.
        """
        try:
            algorithm = self._load_algorithm(solution)
            new_row, exceptions = self._process_function(self.function, algorithm)
        except Exception as e:
            logging.error(
                "Evaluator:PaperContextSummarizer: Error during solution evaluation: "
                + repr(e)
            )
            return self._invalid(solution, repr(e))

        if any(f is None for f in new_row["Result_fitness"]):
            # At least one run did not evaluate the function at all (e.g. crashed immediately)
            reasons = [v for exc in exceptions[1] for v in exc.values() if v]
            return self._invalid(
                solution,
                "; ".join(map(str, reasons))
                or "Algorithm did not evaluate the function.",
            )

        fitness = float(new_row[self._fitness_stat.capitalize()])
        solution.set_fitness(fitness)

        df_results = pd.DataFrame([new_row])
        solution.add_metadata("results", df_results)
        solution.add_metadata("exceptions", {exceptions[0]: exceptions[1]})

        self._check_if_best(solution)
        self._solution_history.append(copy.deepcopy(solution))

        code_txt = self._build_code_block()
        self._keys = {"context": code_txt}
        feedback = fill_template(self.get_feedback_msg_template(), **self._keys)
        solution.set_feedback(feedback)
        solution.add_metadata(
            "context",
            {
                "code_context": self._code_context,
                "history_size": len(self._solution_history),
                "feedback_chars": len(feedback),
                # common-tokenizer size of the code block (sent in the next generator call)
                "code_tokens_common": count_common_tokens(code_txt),
            },
        )

        return EvaluatorResult(
            class_ref=type(self),
            fitness=fitness,
            metadata={"results": df_results, "exceptions": exceptions},
        )

    @classmethod
    def get_short_name(cls) -> str:
        return "eval.papercontextsummarizer"

    @classmethod
    def get_long_name(cls) -> str:
        return "Paper context summarizer"

    @classmethod
    def get_description(cls) -> str:
        return (
            "Evaluator for the Paper Context Summarizer experiments. Based on the metaheuristic evaluator. Assuming "
            "generated Python code with template code of the runner. Feedback = source-code context "
            "(none/last_best/all); the summary of the history is provided by anal.historysummary."
        )

    @classmethod
    def get_tags(cls) -> dict:
        return {"input": {"python"}, "output": {"metaheuristic"}}

    ####################################################################
    #########  Private functions
    ####################################################################
    def _invalid(self, solution: Solution, reason: str) -> EvaluatorResult:
        fitness = -1
        solution.set_fitness(fitness)
        solution.set_feedback(fill_template(self._error_msg, reason=reason))
        return EvaluatorResult(
            class_ref=type(self),
            fitness=fitness,
            metadata={"invalid": True, "error": reason},
        )

    @staticmethod
    def _load_algorithm(solution: Solution):
        local_scope = {}

        # Dynamic import of solution-specific libraries
        exec_globals = {}
        if "modules" in solution.get_metadata().keys():
            for module_name, specific_part, alias in solution.get_metadata()["modules"]:
                dynamic_import(module_name, specific_part, alias, exec_globals)

        compile(solution.get_input(), "solution_to_evaluate.py", "exec")
        exec(solution.get_input(), exec_globals, local_scope)

        # Merge exec_globals and exec_locals to ensure functions can access each other
        combined_scope = {**exec_globals, **local_scope}

        # Rebind the global scope for all functions defined in the script
        for key, value in combined_scope.items():
            if callable(value) and not isinstance(value, type):
                try:
                    value.__globals__.update(combined_scope)
                except Exception as e:
                    logging.error("Evaluator:PaperContextSummarizer: " + repr(e))

        return combined_scope["run"]

    def _build_code_block(self) -> str:
        """
        Source-code context: complete code + development score per selected algorithm.
        all = whole history in chronological order; last_best = the last and the best algorithm
        (the same algorithm is listed twice when it is both).
        """
        history = self._solution_history
        if self._code_context == "all":
            selected = [(i, "history") for i in range(len(history))]
        elif self._code_context == "last_best":
            selected = [(len(history) - 1, "last"), (self._best_index, "best")]
        else:
            return ""
        records = [
            fill_template(
                self._code_record,
                id=format_algorithm_id(i + 1, self._id_format),
                iteration=i + 1,
                role=role,
                score=format_score(history[i].get_fitness(), self._score_format),
                code=escape_delimiters(history[i].get_input().rstrip(), self._tags),
            )
            for i, role in selected
        ]
        return fill_template(self._code_block, records="\n".join(records))

    # Benchmark evaluation
    def _process_run(self, algorithm, run, func, dim, max_fes, max_time):
        logging.info(f"Evaluator:PaperContextSummarizer:Run{run}:F{func}:D{dim}")
        a = Runner(
            copy.deepcopy(algorithm),
            copy.deepcopy(func),
            dim,
            func.get_bounds(),
            max_fes,
            max_time,
        )
        result = a.run()

        exception_array = [
            {f"{run}:{key}": result.get(key, False)}
            for key in (
                "maxtimeexception",
                "maxevalexception",
                "dimexception",
                "unexpectedexception",
                "outofboundsexception",
            )
        ]
        best = dict(result["best"])
        best["evals_total"] = result.get("evals")
        best["runtime_s"] = result.get("runtime_s")
        best["termination"] = result.get("termination")
        return best, exception_array

    def _process_function(self, fdict, algorithm):
        func = fdict["func"]
        dim = fdict["dim"]
        runs = fdict["runs"]
        max_fes = fdict["max_fes"]

        result_array = []
        exception_array = []
        for run in range(runs):
            result, exceptions_a = self._process_run(
                algorithm, run, func, dim, max_fes, self._max_time
            )
            result_array.append(result)
            exception_array.extend(exceptions_a)

        fitness = [res["fitness"] for res in result_array]
        valid = [f for f in fitness if f is not None]
        new_row = {
            "Function": str(func),
            "Dimension": dim,
            "Runs": runs,
            "Max_fes": max_fes,
            "Max_time": self._max_time,
            "Result_fitness": fitness,
            "Result_params": [res["params"] for res in result_array],
            "Result_eval": [res["eval_num"] for res in result_array],
            "Result_evals_total": [res["evals_total"] for res in result_array],
            "Result_runtime_s": [res["runtime_s"] for res in result_array],
            "Result_best_time_s": [res["time_s"] for res in result_array],
            "Result_termination": [res["termination"] for res in result_array],
            "Min": min(valid) if valid else None,
            "Max": max(valid) if valid else None,
            "Mean": float(np.mean(valid)) if valid else None,
            "Median": float(np.median(valid)) if valid else None,
            "STD": float(np.std(valid)) if valid else None,
        }
        return new_row, [str(func), exception_array]

    def _check_if_best(self, solution: Solution) -> bool:
        """
        Saves the solution as _best if it is better (minimization) than the current best (ties: the later one,
        as in the previous experiment). Must be called before the solution is appended to the history.
        """
        if self._best is None or solution.get_fitness() <= self._best.get_fitness():
            self._best = copy.deepcopy(solution)
            self._best_index = len(self._solution_history)
            return True
        return False
