"""Create the complete Task matrix from the context-summarizer paper.

The default plan is the full 1,485-chain design:

* S1: 3 generators x 9 context cells x 3 benchmarks x 11 repetitions = 891
* S2: 3 generators x 6 summary cells x 3 benchmarks x 11 repetitions = 594

By default the command only prints a plan. Pass ``--create`` to persist the
Tasks through the EASE REST API, and add ``--run`` only when the selected
batch should start immediately. API tokens come either from the frontEASE
wrapper (``--tokens-stdin``) or from the provider environment variables used
by ``create_smoke_tasks.py``.
"""

import argparse
import os
import sys
from types import SimpleNamespace

import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.create_smoke_tasks import (  # noqa: E402
    CELLS,
    PROVIDERS,
    read_frontease_tokens,
    task_config,
)

GENERATORS = ("anthropic", "openai", "google")
BENCHMARKS = (
    "resource.gnbg.f_24",
    "resource.bbob.f_24",
    "resource.cec2017.f_30",
)
SCENARIOS = ("s1", "s2")
SUMMARY_CELLS = tuple(cell for cell, (_, summary) in CELLS.items() if summary)

PAPER_REPETITIONS = 11
PAPER_ITERATIONS = 10
PAPER_RUNS = 30
PAPER_TIME_SECONDS = 30
PAPER_MAX_FES = 0


def parse_token_selections(values: list[str]) -> dict[str, str]:
    """Parse repeatable ``PROVIDER=TOKEN_NAME`` selections."""
    selections = {}
    for value in values:
        provider, separator, name = value.partition("=")
        if not separator or provider not in PROVIDERS or not name:
            raise ValueError(
                f"Invalid token selection '{value}'; expected PROVIDER=TOKEN_NAME."
            )
        if provider in selections and selections[provider] != name:
            raise ValueError(f"Token for provider '{provider}' was selected twice.")
        selections[provider] = name
    return selections


def _planning_tokens(args, token_names) -> dict[str, list[tuple[str, str]]]:
    """Use non-secret placeholders when only rendering a local plan."""
    providers = set(args.providers)
    if "s2" in args.scenarios:
        providers.add("cerit")
    return {
        PROVIDERS[provider][0]: [
            (token_names.get(provider, "plan"), "not-used-in-plan-mode")
        ]
        for provider in providers
    }


def _task_args(args, provider, scenario, function, token_names):
    summarizer_provider = None if scenario == "s1" else "cerit"
    return SimpleNamespace(
        provider=provider,
        model=None,
        summarizer_provider=summarizer_provider,
        summarizer_model=None,
        token=token_names.get(provider),
        summarizer_token=(
            token_names.get(summarizer_provider) if summarizer_provider else None
        ),
        function=function,
        iterations=args.iterations,
        runs=args.runs,
        time=args.time,
        max_fes=args.max_fes,
        author=args.author,
        name_prefix=args.name_prefix,
    )


def build_plan(args, tokens=None) -> list[dict]:
    """Build and validate every Task before any REST mutation is attempted."""
    token_names = parse_token_selections(args.token)
    plan = []
    for provider in args.providers:
        generator_model = PROVIDERS[provider][2]
        for scenario in args.scenarios:
            cells = tuple(CELLS) if scenario == "s1" else SUMMARY_CELLS
            for function in args.functions:
                task_args = _task_args(
                    args, provider, scenario, function, token_names
                )
                for cell in cells:
                    for repetition in range(1, args.repetitions + 1):
                        config = task_config(task_args, cell, tokens)
                        # The paper has no total-attempt cap; it stops after five
                        # consecutive invalid outputs or ten valid algorithms.
                        config["modules"] = [
                            module
                            for module in config["modules"]
                            if module["short_name"] != "stop.condmaxiters"
                        ]
                        summarizer = (
                            f" | sum=cerit:{PROVIDERS['cerit'][2]}"
                            if scenario == "s2"
                            else ""
                        )
                        config["name"] = (
                            f"{args.name_prefix} {scenario.upper()} R{repetition:02d} "
                            f"| {cell} | gen={provider}:{generator_model}{summarizer} "
                            f"| {function}"
                        )
                        plan.append(
                            {
                                "scenario": scenario,
                                "provider": provider,
                                "function": function,
                                "cell": cell,
                                "repetition": repetition,
                                "config": config,
                            }
                        )
    return plan


def create_tasks(plan: list[dict], url: str, run: bool = False) -> list[str]:
    """Persist a prevalidated plan and optionally start every created Task."""
    created = []
    for index, item in enumerate(plan, start=1):
        response = requests.post(f"{url}/task")
        response.raise_for_status()
        task_id = response.json()["id"]

        response = requests.put(f"{url}/task/{task_id}", json=item["config"])
        if response.status_code != 200:
            raise RuntimeError(
                f"Task {task_id} could not be initialized: "
                f"{response.status_code} {response.text}"
            )
        if run:
            requests.patch(f"{url}/task/{task_id}/run").raise_for_status()
        created.append(task_id)
        state = "started" if run else "created"
        print(f"[{index}/{len(plan)}] {state}: {task_id}  {item['config']['name']}")
    return created


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or create the full context-summarizer paper experiment."
    )
    parser.add_argument("--url", default="http://localhost:8086")
    parser.add_argument("--author", default="")
    parser.add_argument(
        "--providers", nargs="+", choices=GENERATORS, default=list(GENERATORS)
    )
    parser.add_argument(
        "--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS)
    )
    parser.add_argument(
        "--functions", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS)
    )
    parser.add_argument("--repetitions", type=int, default=PAPER_REPETITIONS)
    parser.add_argument("--iterations", type=int, default=PAPER_ITERATIONS)
    parser.add_argument("--runs", type=int, default=PAPER_RUNS)
    parser.add_argument("--time", type=int, default=PAPER_TIME_SECONDS)
    parser.add_argument(
        "--max-fes",
        type=int,
        default=PAPER_MAX_FES,
        help=(
            "Maximum objective-function evaluations per run "
            "(default: 0 = unlimited)."
        ),
    )
    parser.add_argument("--name-prefix", default="PAPER")
    parser.add_argument(
        "--tokens-stdin",
        action="store_true",
        help="Read frontEASE tokens from stdin (normally via the shell wrapper).",
    )
    parser.add_argument(
        "--token",
        action="append",
        default=[],
        metavar="PROVIDER=TOKEN_NAME",
        help="Select a named frontEASE token; repeat for multiple providers.",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Persist the planned Tasks. Without this flag only the plan is printed.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Start each created Task immediately (requires --create).",
    )
    return parser


def _positive(parser, name, value, allow_zero=False):
    minimum = 0 if allow_zero else 1
    if value < minimum:
        parser.error(f"{name} must be >= {minimum}.")


def main(argv=None) -> list[str]:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.run and not args.create:
        parser.error("--run requires --create.")
    _positive(parser, "--repetitions", args.repetitions)
    _positive(parser, "--iterations", args.iterations)
    _positive(parser, "--runs", args.runs)
    _positive(parser, "--time", args.time)
    _positive(parser, "--max-fes", args.max_fes, allow_zero=True)

    try:
        token_names = parse_token_selections(args.token)
    except ValueError as error:
        parser.error(str(error))
    args.token = [f"{provider}={name}" for provider, name in token_names.items()]

    tokens = read_frontease_tokens(sys.stdin) if args.tokens_stdin else None
    if not args.create and tokens is None:
        tokens = _planning_tokens(args, token_names)
    plan = build_plan(args, tokens)
    counts = {
        scenario.upper(): sum(item["scenario"] == scenario for item in plan)
        for scenario in args.scenarios
    }
    count_text = ", ".join(f"{key}={value}" for key, value in counts.items())
    print(
        f"Planned {len(plan)} Tasks ({count_text}); "
        f"{args.iterations} valid algorithms x {args.runs} runs x {args.time} s, "
        f"max_fes={args.max_fes}."
    )

    if not args.create:
        print("Plan only; pass --create to persist these Tasks.")
        return []
    return create_tasks(plan, args.url, args.run)


if __name__ == "__main__":
    main()
