"""
Creates smoke-test Tasks of the context-summarizer experiment through the REST API of the EASE core
(the same way as frontEASE does). The Tasks are small (few valid iterations, short runs) and are meant to verify
the whole pipeline with real LLM providers: generation, evaluation, code context, history summary and the usage
ledger (usage_calls.csv / usage_iterations.csv in out_task/t_<id>/).

API keys are taken either from the tokens stored in frontEASE (Management > Tokens) or from environment variables
(CERIT_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY).

frontEASE tokens (recommended; run on the server in the frontEASE directory with docker-compose.yml):
    src/FoP_IMT.Core/tools/create_tasks_frontease.sh you@example.com --provider openai --summarizer-provider cerit
The wrapper reads the tokens of the user from the frontEASE database and pipes them to this script (--tokens-stdin),
so a token never appears on a command line or in the shell history. Every LLM connector of a Task (generator,
summarizer) gets the token stored for its connector type; if the user has several tokens for one connector type,
choose one by its name with --token (generator) / --summarizer-token (summarizer).

Environment variables (inside the core container):
    docker compose exec -e CERIT_API_KEY=<key> backend-core \
        python tools/create_smoke_tasks.py --author you@example.com --provider cerit

frontEASE imports Tasks that exist only in the core at its start (InitialTaskSyncJob, or "Trigger now" in the
Hangfire dashboard); --author must be the e-mail of the frontEASE user, otherwise the Task is assigned to a superadmin.
"""

import argparse
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fopimt.evaluators.evaluator_papercontextsummarizer import DEFAULT_INIT_MSG  # noqa: E402, I001

PROVIDERS = {
    # name: (connector short name, environment variable with the API key, default model)
    "cerit": ("llm.cerit", "CERIT_API_KEY", "kimi-k3"),
    "openai": ("llm.openai", "OPENAI_API_KEY", "gpt-6.1-sol"),
    "anthropic": ("llm.anthropic", "ANTHROPIC_API_KEY", "claude-opus-5-5"),
    "google": ("llm.google", "GOOGLE_API_KEY", "gemini-3.8-flash"),
    "mock": ("llm.mock", None, "Meta: random search"),
}

# The 9 cells of the experiment: name -> (code_context, summary_type or None)
CELLS = {
    "nocontext": ("none", None),
    "sum": ("none", "free"),
    "sum_structured": ("none", "structured"),
    "all": ("all", None),
    "all+sum": ("all", "free"),
    "all+sum_structured": ("all", "structured"),
    "last_best": ("last_best", None),
    "last_best+sum": ("last_best", "free"),
    "last_best+sum_structured": ("last_best", "structured"),
}
DEFAULT_CELLS = ["nocontext", "last_best+sum", "all+sum_structured"]

# Repeated messages as in the previous (context) experiment
REPEATED_NOCONTEXT = "Generate an algorithm."
REPEATED_CONTEXT = "Improve."


def read_frontease_tokens(stream) -> dict[str, list[tuple[str, str]]]:
    """
    Tokens of the user exported from the frontEASE database by create_tasks_frontease.sh:
    one line per token and connector type, "connector_type<TAB>name<TAB>token", ordered by priority.
    :return: connector type -> [(name, token), ...]
    """
    tokens: dict[str, list[tuple[str, str]]] = {}
    for line in stream.read().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            sys.exit("Unexpected format of the frontEASE token export.")
        connector_type, name, token = parts
        tokens.setdefault(connector_type, []).append((name, token))
    return tokens


def pick_frontease_token(
    tokens: dict[str, list[tuple[str, str]]], short_name: str, name: str | None
) -> str:
    """Token stored in frontEASE for the connector type; by name if given, otherwise the only one."""
    candidates = tokens.get(short_name, [])
    names = [n for n, _ in candidates]
    if name is not None:
        candidates = [c for c in candidates if c[0] == name]
    if len(candidates) == 1:
        return candidates[0][1]
    if not names:
        sys.exit(f"No frontEASE token is stored for the connector type {short_name}.")
    if name is not None:
        sys.exit(
            f"{len(candidates)} frontEASE tokens named '{name}' for {short_name} (available: {names})."
        )
    sys.exit(
        f"Several frontEASE tokens for {short_name} (available: {names}); choose one with "
        "--token (generator) or --summarizer-token (summarizer)."
    )


def connector(
    provider: str,
    model: str | None,
    tokens: dict[str, list[tuple[str, str]]] | None = None,
    token_name: str | None = None,
) -> dict:
    short_name, env, default_model = PROVIDERS[provider]
    if provider == "mock":
        return {
            "short_name": short_name,
            "parameters": {"response": model or default_model},
        }
    if tokens is not None:
        token = pick_frontease_token(tokens, short_name, token_name)
    else:
        token = os.environ.get(env, "")
        if not token:
            sys.exit(f"Missing API key: set the environment variable {env}.")
    return {
        "short_name": short_name,
        "parameters": {"token": token, "model": model or default_model},
    }


def task_config(args, cell: str, tokens=None) -> dict:
    code_context, summary_type = CELLS[cell]
    generator = connector(args.provider, args.model, tokens, args.token)
    modules = [
        generator,
        {"short_name": "sol.codepython", "parameters": {}},
        {"short_name": "test.psyntax", "parameters": {}},
        {"short_name": "test.pimports", "parameters": {}},
        {
            "short_name": "eval.papercontextsummarizer",
            "parameters": {
                "code_context": code_context,
                "function": args.function,
                "time": args.time,
                "runs": args.runs,
                "fitness_stat": "mean",
            },
        },
        {"short_name": "stop.condvaliditers", "parameters": {"value": args.iterations}},
        {"short_name": "stop.condconsinvaliditers", "parameters": {"value": 5}},
        {
            "short_name": "stop.condmaxiters",
            "parameters": {"value": 3 * args.iterations},
        },
    ]
    if summary_type is not None:
        summarizer_provider = args.summarizer_provider or args.provider
        summarizer = connector(
            summarizer_provider,
            args.summarizer_model
            or (args.model if not args.summarizer_provider else None),
            tokens,
            # the same provider as the generator (scenario S1) uses the generator token unless chosen otherwise
            args.summarizer_token
            or (args.token if summarizer_provider == args.provider else None),
        )
        modules.append(
            {
                "short_name": "anal.historysummary",
                "parameters": {
                    "llm": summarizer,
                    "summary_type": summary_type,
                    "iterations": args.iterations,
                },
            }
        )
    model_name = generator["parameters"].get("model") or generator["parameters"].get(
        "response"
    )
    return {
        "name": f"{args.name_prefix} {cell} | {args.provider}:{model_name} | {args.function}",
        "author": args.author,
        "max_context_size": 0,
        "feedback_from_solution": True,
        "initial_message": DEFAULT_INIT_MSG,
        "repeated_message": {
            "type": 0,  # SINGLE
            "msgs": [REPEATED_NOCONTEXT if cell == "nocontext" else REPEATED_CONTEXT],
            "weights": [1.0],
        },
        "optimization_goal": 0,  # MINIMIZATION
        "modules": modules,
    }


def main(argv=None) -> list[str]:
    parser = argparse.ArgumentParser(
        description="Create smoke-test Tasks of the context-summarizer experiment."
    )
    parser.add_argument(
        "--url", default="http://localhost:8086", help="URL of the EASE core REST API."
    )
    parser.add_argument(
        "--author",
        default="",
        help="E-mail of the frontEASE user that will own the Tasks.",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDERS),
        default="cerit",
        help="Generator provider.",
    )
    parser.add_argument(
        "--model", default=None, help="Generator model (default per provider)."
    )
    parser.add_argument(
        "--summarizer-provider",
        choices=sorted(PROVIDERS),
        default=None,
        help="Summarizer provider (default: the generator provider).",
    )
    parser.add_argument(
        "--summarizer-model",
        default=None,
        help="Summarizer model (default: the generator model).",
    )
    parser.add_argument(
        "--tokens-stdin",
        action="store_true",
        help="Read the frontEASE tokens of the user from stdin (used by create_tasks_frontease.sh).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Name of the frontEASE token of the generator (needed only if there are several for its connector).",
    )
    parser.add_argument(
        "--summarizer-token",
        default=None,
        help="Name of the frontEASE token of the summarizer (default: the generator token for the same provider).",
    )
    parser.add_argument(
        "--cells", nargs="+", choices=sorted(CELLS), default=DEFAULT_CELLS
    )
    parser.add_argument(
        "--function",
        default="resource.gnbg.f_24",
        choices=["resource.gnbg.f_24", "resource.cec2017.f_30", "resource.bbob.f_24"],
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Valid iterations (algorithms) per Task.",
    )
    parser.add_argument("--runs", type=int, default=3, help="Runs per algorithm.")
    parser.add_argument(
        "--time", type=int, default=5, help="Time limit of one run [s]."
    )
    parser.add_argument("--name-prefix", default="SMOKE")
    parser.add_argument(
        "--run", action="store_true", help="Start the Tasks right away."
    )
    args = parser.parse_args(argv)
    tokens = read_frontease_tokens(sys.stdin) if args.tokens_stdin else None

    # all configurations first, so that a missing token stops the script before any Task is created
    configs = [(cell, task_config(args, cell, tokens)) for cell in args.cells]
    created = []
    for cell, config in configs:
        info = requests.post(f"{args.url}/task").json()
        task_id = info["id"]
        r = requests.put(f"{args.url}/task/{task_id}", json=config)
        if r.status_code != 200:
            sys.exit(
                f"Task {task_id} ({cell}) could not be initialized: {r.status_code} {r.text}"
            )
        if args.run:
            requests.patch(f"{args.url}/task/{task_id}/run").raise_for_status()
        created.append(task_id)
        print(f"{'started' if args.run else 'created'}: {task_id}  {config['name']}")

    print(
        "\nOutputs: out_task/t_<id>/ (messages.csv, usage_calls.csv, usage_iterations.csv, solution/, anal/).\n"
        "frontEASE imports the Tasks at its start or via Hangfire > Recurring jobs > InitialTaskSyncJob > Trigger now."
    )
    return created


if __name__ == "__main__":
    main()
