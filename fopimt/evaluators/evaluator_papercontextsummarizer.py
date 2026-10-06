import copy
import json
import logging
import time

import numpy as np
import pandas as pd

from ..loader import Loader, Parameter, PrimitiveType
from ..loader_dto import PackageType
from ..message import Message
from ..resource.metahuristic.metaheuristic_runner import Runner
from ..resource.resource import Resource
from ..solutions.solution import Solution
from ..task_dto import OptimizationGoal
from ..utils.import_utils import dynamic_import
from ..utils.token_utils import count_common_tokens, llm_call_record
from .evaluator import Evaluator, EvaluatorResult

CODE_CONTEXTS = ["none", "last_best", "all"]
SUMMARY_TYPES = ["none", "free", "structured"]
FITNESS_STATS = ["min", "mean", "median"]
FUNCTIONS = ["resource.gnbg.f_24", "resource.cec2017.f_30", "resource.bbob.f_24"]

# Family labels and feature flags follow the annotation schema of the Context paper (Viktorin et al., CSR 2027)
FAMILIES = [
    "DE",
    "PSO",
    "GA",
    "ES",
    "CMA-ES",
    "SA",
    "ACO",
    "EDA",
    "Memetic",
    "Tabu",
    "Other",
]
FEATURES = [
    "metaheuristic",
    "population_based",
    "stochastic",
    "local_search",
    "adaptation",
    "initialization",
    "restart",
    "surrogate",
    "elitism",
    "archive",
    "niching_or_diversity",
    "hybridized",
]

SUMMARY_PROMPT_FREE = """You will be given the Python source code of {n} optimization algorithms that were generated one after another. They are labelled A1 (first generated) to A{n} (most recent). Their performance is not given to you.

Write a concise natural-language summary of this history. For each algorithm, describe what it actually implements (main search paradigm, key operators and mechanisms, parameter settings or their adaptation, restarts, local search) and how it differs from the previous algorithms. Analyze the actual implemented behavior, not comments or claimed intent. Refer to the algorithms only by their labels. Do not include any code and do not guess performance. Use at most {max_words} words in total.

{codes}"""

SUMMARY_PROMPT_STRUCTURED = """You will be given the Python source code of {n} optimization algorithms that were generated one after another. They are labelled A1 (first generated) to A{n} (most recent). Their performance is not given to you.

Classify each algorithm into a strict JSON object with the following fields:
- "id": the label of the algorithm (e.g. "A1")
- "family": the PRIMARY original idea, one of {families}
- {features_list}: true/false
- "key_mechanisms": the main operators and mechanisms, at most {max_words} words

Rules:
1. Analyze the ACTUAL implemented behavior, not comments or claimed intent.
2. If uncertain, choose the most plausible class.
3. initialization means special/problem-aware/non-random initialization (pure uniform random initialization is false; Latin hypercube, opposition-based, seeding, greedy or heuristic initialization is true).
4. adaptation means that parameters or strategy components change during the run.
5. local_search must be true only if there is a recognizable explicit local search or refinement step.
6. surrogate must be true only if there is a predictive/model-based approximation.
7. hybridized should be true when the method combines distinct paradigms in a meaningful way.
8. Output ONLY a valid JSON array with exactly one object per algorithm, in the order A1 to A{n}, and nothing else.

{codes}"""


class EvaluatorPaperContextSummarizer(Evaluator):
    """
    Evaluator for the Paper Context Summarizer experiments. Based on the metaheuristic evaluator: the solution must be
    Python code following the template run(func, dim, bounds, max_time) defined by the Runner class.

    The feedback context is composed of two independent components:
    - code_context: which previous algorithms are attached as code with their output values (none | last_best | all),
    - summary: an LLM-generated summary of the whole history of valid algorithms (none | free | structured).
      The summary is regenerated from scratch in every iteration by a separate LLM (summarizer). The summarizer never
      sees the output values; they are attached to the summary deterministically by the evaluator.
    """

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        llms = (
            Loader((PackageType.LLMConnector,))
            .get_package(PackageType.LLMConnector)
            .get_moduls()
        )
        return {
            "llm": Parameter(
                short_name="llm",
                type=PrimitiveType.enum,
                long_name="Summarizer LLM",
                description="LLM connector used for summarization (only used if summary is not 'none').",
                enum_options=llms,
                required=False,
            ),
            "code_context": Parameter(
                short_name="code_context",
                type=PrimitiveType.enum,
                long_name="Code context",
                description="Previous algorithms attached as code with their output values.",
                enum_options=CODE_CONTEXTS,
                default="last_best",
            ),
            "summary": Parameter(
                short_name="summary",
                type=PrimitiveType.enum,
                long_name="Summary of history",
                description="Type of the LLM summary of all previously generated valid algorithms.",
                enum_options=SUMMARY_TYPES,
                default="none",
            ),
            "summary_max_words": Parameter(
                short_name="summary_max_words",
                type=PrimitiveType.int,
                long_name="Summary words per algorithm",
                description="Free summary: word limit per summarized algorithm (total = limit * count). "
                "Structured summary: word limit of the key_mechanisms field.",
                default=80,
            ),
            "summary_max_algorithms": Parameter(
                short_name="summary_max_algorithms",
                type=PrimitiveType.int,
                long_name="Max summarized algorithms",
                description="Maximum number of most recent valid algorithms included in the summary.",
                default=10,
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
                description="Statistic over runs used as the output value (fitness) of the algorithm.",
                enum_options=FITNESS_STATS,
                default="min",
            ),
            "feedback_msg_template": Parameter(
                short_name="feedback_msg_template",
                type=PrimitiveType.markdown,
                long_name="Template for a feedback message",
                description="Feedback message for evaluation. Can use {keywords}. {context} is composed "
                "according to code_context and summary.",
                default="{context}",
            ),
            "init_msg_template": Parameter(
                short_name="init_msg_template",
                type=PrimitiveType.markdown,
                long_name="Template for an initial message",
                description="Initial message for evaluation. Specific for each evaluator.",
                default="""Your task is to propose an algorithm to find a set of input parameter values that lead to minimum output value in a limited time. The template for the algorithm is given below. Deliver Python code that is fully operational and self-contained, requiring no external libraries or modifications post-delivery

Glossary:
func - function that returns output value (float) for an array of input parameter values (np.array).
dim - (int) dimension of the input vector.
bounds - (list) specified lower and upper bounds for the input vector values. A pair for each dimension.
max_time - (int) maximum acceptable time in seconds to return a result.

Template:

def run(func, dim, bounds, max_time):

    [Algorithm body]

    # return fitness of the best found solution
    return best

Example implementation of a random search algorithm in the given template:

import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')

    # Algorithm body
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time):
            return best

        params = [np.random.uniform(low, high) for low, high in bounds]
        fitness = func(params)
        if best is None or fitness <= best:
            best = fitness
        """,
                readonly=True,
            ),
            "keywords": Parameter(
                short_name="keywords",
                type=PrimitiveType.enum,
                long_name="Feedback keywords",
                description="Feedback keyword-based sentences",
                enum_options=["context", "last", "best", "last_best", "all", "summary"],
                readonly=True,
            ),
        }

    def _init_params(self):
        super()._init_params()
        defaults = self.get_parameters()

        def param(name):
            return self.parameters.get(name, defaults[name].default)

        self._code_context = param("code_context")
        self._summary_type = param("summary")
        if self._code_context not in CODE_CONTEXTS:
            raise ValueError(f"Unknown code_context '{self._code_context}'")
        if self._summary_type not in SUMMARY_TYPES:
            raise ValueError(f"Unknown summary '{self._summary_type}'")
        self._summary_max_words = int(param("summary_max_words"))
        self._summary_max_algorithms = int(param("summary_max_algorithms"))
        self._fitness_stat = param("fitness_stat")
        self._max_time = int(param("time"))
        self._runs = int(param("runs"))

        self._llmconnector = None
        if self._summary_type != "none":
            llm = self.parameters.get("llm", None)
            if not llm:
                raise ValueError(
                    "Summarizer LLM ('llm') must be set when summary is used."
                )
            self._llmconnector = (
                Loader((PackageType.LLMConnector,))
                .get_package(PackageType.LLMConnector)
                .get_modul_imported(llm["short_name"])(llm["parameters"])
            )

        self._function_name = param("function")
        self.function = Resource.get_resource_function(
            self._function_name, "metabenchmark"
        )()
        if self._runs > 0:
            self.function["runs"] = self._runs

        self._solution_history: list[
            Solution
        ] = []  # valid solutions in generation order

    ####################################################################
    #########  Public functions
    ####################################################################
    def evaluate(
        self,
        solution: Solution,
        opt_goal: OptimizationGoal = OptimizationGoal.MINIMIZATION,
    ) -> EvaluatorResult:
        """
        Evaluates the algorithm on the benchmark function and composes the feedback context.
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

        # Feedback context
        summary_info = None
        summary_txt = ""
        if self._summary_type != "none":
            summary_info = self._summarize(
                self._solution_history[-self._summary_max_algorithms :]
            )
            summary_txt = summary_info["text"]
            solution.add_metadata("summary", summary_info)

        last_txt = self._last_txt()
        best_txt = self._best_txt()
        last_best_txt = last_txt + "\n" + best_txt
        all_txt = self._all_txt()
        code_txt = {"none": "", "last_best": last_best_txt, "all": all_txt}[
            self._code_context
        ]
        context = "\n\n".join(part for part in (summary_txt, code_txt) if part)

        self._keys = {
            "context": context,
            "last": last_txt,
            "best": best_txt,
            "last_best": last_best_txt,
            "all": all_txt,
            "summary": summary_txt,
        }
        feedback = self.get_feedback_msg_template().format(**self._keys)
        solution.set_feedback(feedback)
        solution.add_metadata(
            "context",
            {
                "code_context": self._code_context,
                "summary": self._summary_type,
                "history_size": len(self._solution_history),
                "feedback_chars": len(feedback),
                # common-tokenizer sizes of the parts of the feedback (sent in the next generator call)
                "feedback_tokens_common": count_common_tokens(feedback),
                "summary_tokens_common": count_common_tokens(summary_txt),
                "code_tokens_common": count_common_tokens(code_txt),
            },
        )

        return EvaluatorResult(
            class_ref=type(self),
            fitness=fitness,
            metadata={
                "results": df_results,
                "exceptions": exceptions,
                # LLM calls made by the evaluator, collected by Task into the usage ledger
                "llm_calls": summary_info["llm_calls"] if summary_info else [],
            },
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
            "generated Python code with template code of the runner. Feedback context combines attached code "
            "(none/last_best/all) with an LLM summary of the history (none/free/structured)."
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
        solution.set_feedback(
            f"Error during solution evaluation: {reason}\n. Try to fix it."
        )
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

    # Code context texts (wording identical to the Context paper)
    def _last_txt(self) -> str:
        sol = self._solution_history[-1]
        return (
            f"The output value of the last generated algorithm is: {sol.get_fitness()}\n\n"
            f" The last generated algorithm code:\n{sol.get_input()}\n"
        )

    def _best_txt(self) -> str:
        return (
            f"The output value of the best generated algorithm is: {self._best.get_fitness()}\n\n"
            f" The best generated algorithm code:\n{self._best.get_input()}\n"
        )

    def _all_txt(self) -> str:
        txt = "The output values and codes for the last generated algorithms are as follows:\n"
        for index, sol in enumerate(reversed(self._solution_history), start=1):
            txt += f"{index}. output value is: {sol.get_fitness()}\n\n {index}. algorithm code is:\n{sol.get_input()}\n\n"
        return txt

    # Summary
    def _summarize(self, history: list[Solution]) -> dict:
        """
        Summarizes the given history of valid solutions (in generation order) with the summarizer LLM.
        Output values are attached by the evaluator, the summarizer does not see them.
        """
        n = len(history)
        labels = [f"A{i}" for i in range(1, n + 1)]
        codes = "\n\n".join(
            f"{label}:\n```python\n{sol.get_input()}\n```"
            for label, sol in zip(labels, history)
        )
        if self._summary_type == "free":
            prompt = SUMMARY_PROMPT_FREE.format(
                n=n, max_words=self._summary_max_words * n, codes=codes
            )
        else:
            prompt = SUMMARY_PROMPT_STRUCTURED.format(
                n=n,
                families=json.dumps(FAMILIES),
                features_list=", ".join(f'"{f}"' for f in FEATURES),
                max_words=self._summary_max_words,
                codes=codes,
            )

        llm_calls = []
        raw = []
        records = None
        attempts = 2 if self._summary_type == "structured" else 1
        for _ in range(attempts):
            response = self._send_summarizer(prompt)
            llm_calls.append(
                llm_call_record(
                    "summarizer",
                    self._llmconnector.get_short_name(),
                    {"prompt": prompt},
                    response,
                )
            )
            raw.append(response.get_content())
            if self._summary_type == "free":
                break
            records = self._parse_structured(response.get_content(), labels)
            if records is not None:
                break

        header = (
            f"Summary of all {n} previously generated algorithms "
            f"(A1 = first generated, A{n} = most recent):\n"
        )
        values = [sol.get_fitness() for sol in history]
        if records is not None:
            for record, value in zip(records, values):
                record["output_value"] = value
            body = "\n".join(
                json.dumps(
                    {
                        "id": r["id"],
                        "output_value": r["output_value"],
                        **{
                            k: v
                            for k, v in r.items()
                            if k not in ("id", "output_value")
                        },
                    }
                )
                for r in records
            )
            text = header + body + "\n"
        else:
            # free summary, or structured summary that could not be parsed
            text = (
                header
                + raw[-1].strip()
                + "\n\nThe output values of the summarized algorithms are:\n"
                + "\n".join(f"{label}: {value}" for label, value in zip(labels, values))
                + "\n"
            )

        return {
            "type": self._summary_type,
            "n_algorithms": n,
            "text": text,
            "raw": raw,
            "parse_failed": self._summary_type == "structured" and records is None,
            "llm_calls": llm_calls,
        }

    def _send_summarizer(self, prompt: str, max_attempts: int = 5) -> Message:
        msg = Message(role=self._llmconnector.get_role_user(), message=prompt)
        for attempt in range(1, max_attempts + 1):
            try:
                return self._llmconnector.send([msg]).response
            except Exception as e:
                logging.error(
                    f"Evaluator:PaperContextSummarizer: summarizer call failed ({attempt}/{max_attempts}): {repr(e)}"
                )
                if attempt == max_attempts:
                    raise RuntimeError(
                        f"Summarizer LLM failed {max_attempts} times: {repr(e)}"
                    ) from e
                time.sleep(min(60, 2**attempt))

    @staticmethod
    def _parse_structured(content: str, labels: list[str]) -> list[dict] | None:
        start, end = content.find("["), content.rfind("]")
        if start < 0 or end <= start:
            return None
        try:
            records = json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return None
        if not isinstance(records, list) or len(records) != len(labels):
            return None
        if not all(isinstance(r, dict) for r in records):
            return None
        for record, label in zip(records, labels):
            record["id"] = label
        return records

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
            "Min": min(valid) if valid else None,
            "Max": max(valid) if valid else None,
            "Mean": float(np.mean(valid)) if valid else None,
            "Median": float(np.median(valid)) if valid else None,
            "STD": float(np.std(valid)) if valid else None,
        }
        return new_row, [str(func), exception_array]

    def _check_if_best(self, solution: Solution) -> bool:
        """
        Saves the solution as _best if it is better (minimization) than the current best.
        """
        if self._best is None or solution.get_fitness() <= self._best.get_fitness():
            self._best = copy.deepcopy(solution)
            return True
        return False
