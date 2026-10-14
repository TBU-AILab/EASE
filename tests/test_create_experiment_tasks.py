import unittest

from tools.create_experiment_tasks import (
    BENCHMARKS,
    GENERATORS,
    PAPER_ITERATIONS,
    PAPER_MAX_FES,
    PAPER_REPETITIONS,
    PAPER_RUNS,
    PAPER_TIME_SECONDS,
    SUMMARY_CELLS,
    _parser,
    build_plan,
    parse_token_selections,
)


TOKENS = {
    "llm.anthropic": [("paper", "anthropic-secret")],
    "llm.openai": [("paper", "openai-secret")],
    "llm.google": [("paper", "google-secret")],
    "llm.cerit": [("paper", "cerit-secret")],
}


def _args(*extra):
    return _parser().parse_args(list(extra))


def _module(config, short_name):
    return next(
        module for module in config["modules"] if module["short_name"] == short_name
    )


class ExperimentTaskPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = build_plan(_args(), TOKENS)

    def test_full_paper_design_has_1485_independent_chains(self):
        self.assertEqual(1485, len(self.plan))
        self.assertEqual(
            891, sum(item["scenario"] == "s1" for item in self.plan)
        )
        self.assertEqual(
            594, sum(item["scenario"] == "s2" for item in self.plan)
        )

    def test_default_dimensions_match_the_paper(self):
        expected = (
            len(GENERATORS)
            * (9 + len(SUMMARY_CELLS))
            * len(BENCHMARKS)
            * PAPER_REPETITIONS
        )
        self.assertEqual(expected, len(self.plan))

        for item in self.plan:
            config = item["config"]
            evaluator = _module(config, "eval.papercontextsummarizer")["parameters"]
            self.assertEqual(PAPER_RUNS, evaluator["runs"])
            self.assertEqual(PAPER_TIME_SECONDS, evaluator["time"])
            self.assertEqual(PAPER_MAX_FES, evaluator["max_fes"])
            self.assertEqual("mean", evaluator["fitness_stat"])
            self.assertEqual(
                PAPER_ITERATIONS,
                _module(config, "stop.condvaliditers")["parameters"]["value"],
            )
            module_names = {module["short_name"] for module in config["modules"]}
            self.assertIn("stop.condconsinvaliditers", module_names)
            self.assertNotIn("stop.condmaxiters", module_names)

    def test_s2_contains_only_summary_cells_and_uses_kimi(self):
        s2 = [item for item in self.plan if item["scenario"] == "s2"]
        self.assertTrue(s2)
        for item in s2:
            self.assertIn(item["cell"], SUMMARY_CELLS)
            analysis = _module(item["config"], "anal.historysummary")
            self.assertEqual("llm.cerit", analysis["parameters"]["llm"]["short_name"])
            self.assertEqual(
                "kimi-k3", analysis["parameters"]["llm"]["parameters"]["model"]
            )

    def test_s1_uses_the_generator_as_summarizer(self):
        summarized = [
            item
            for item in self.plan
            if item["scenario"] == "s1" and item["cell"] in SUMMARY_CELLS
        ]
        for item in summarized:
            generator = item["config"]["modules"][0]
            analysis = _module(item["config"], "anal.historysummary")
            self.assertEqual(generator, analysis["parameters"]["llm"])

    def test_repetitions_have_unique_names(self):
        names = [item["config"]["name"] for item in self.plan]
        self.assertEqual(len(names), len(set(names)))

    def test_subsets_support_phase_one_and_no_evaluation_cap(self):
        args = _args(
            "--providers",
            "anthropic",
            "--scenarios",
            "s1",
            "--functions",
            "resource.gnbg.f_24",
            "--repetitions",
            "1",
            "--max-fes",
            "0",
        )
        plan = build_plan(args, TOKENS)
        self.assertEqual(9, len(plan))
        for item in plan:
            evaluator = _module(item["config"], "eval.papercontextsummarizer")
            self.assertEqual(0, evaluator["parameters"]["max_fes"])

    def test_named_token_selections_are_parsed_per_provider(self):
        self.assertEqual(
            {"anthropic": "primary", "cerit": "local"},
            parse_token_selections(["anthropic=primary", "cerit=local"]),
        )
        with self.assertRaises(ValueError):
            parse_token_selections(["anthropic"])


if __name__ == "__main__":
    unittest.main()
