# Paper "Context summarizer" – decision log

Working notes for the follow-up of *Context matters in LLM-driven algorithm design* (Viktorin et al., Computer
Science Review 63 (2027) 101079). Branch `paper/summarizer` in EASE and frontEASE.

## Experiment design (agreed 2026-10-06)

- Factorial design, 9 cells: **code part** {none, last_best, all} × **summary of history** {none, free, structured}.
  none × none = nocontext.
- 11 repetitions per cell, 10 valid algorithms per repetition.
- Phase 1: one frontier generator LLM; later a second/third model (local or frontier, depending on budget).
- Summarizer: either the same model via a separate API call, or a local model (Ollama).
- The summary is regenerated from scratch in every iteration over **all valid algorithms so far** (max 10).
  The summarizer never sees fitness values; the evaluator attaches them deterministically.
- Goals: solution quality (primary), token usage (secondary), cost (cashflow) in post-processing.
- Functions (D=30): GNBG-II f24, CEC 2017 F30, BBOB f24 (see below).

## Decisions

| Date | Decision | Reason |
|---|---|---|
| 2026-10-06 | `fitness_stat = min` (minimum over 30 runs) | Same value the LLM saw in the previous experiment (the paper defines q as the mean, but the evaluator used min). |
| 2026-10-06 | `last_best` keeps both code blocks when the last algorithm is also the best | Same as the previous experiment. |
| 2026-10-06 | Google connector sends the conversation as stringified JSON (unchanged) | Comparability with the previous experiment. |
| 2026-10-06 | Token accounting: every LLM call is logged (`usage_calls.csv`, `usage_iterations.csv`) with provider-reported (billed) counts and counts from one common tokenizer (`o200k_base`) | Billed counts give the cost; the common tokenizer makes context sizes comparable across models. |
| 2026-10-06 | CEC function: **CEC 2017 F30** (Composition Function 10), D=30 | See "Benchmark selection". |
| 2026-10-06 | BBOB function: **f24 Lunacek bi-Rastrigin**, instance 1, D=30 | See "Benchmark selection". |

## Benchmark selection

Difficulty = how hard it was for optimization algorithms to find the global optimum.

### CEC: CEC 2017 F30 (Composition Function 10), D=30

**Suite.** The latest editions of the CEC competition on single objective bound constrained optimization
(CEC 2024, 2025 and 2026) all use the **29 functions of the CEC 2017 suite in D=30**, Max_FEs = 10000·D,
search range [-100, 100]^D. CEC 2022 (12 functions, D=10/20) is an older side line, not used anymore.

**Why 29 and not 30 functions.** F2 (*Shifted and Rotated Sum of Different Power Function*,
f(z) = Σ|z_i|^(i+1)) was officially excluded because of numerical instability: unstable behavior especially in higher
dimensions and significantly different results of the same algorithm implemented in C and Matlab. The numbering of
F3–F30 was kept. (Illustration: BlockEA in CEC 2024 still evaluated F2 and had an initial error of ~3·10^48.)

**Data.** Final errors of all competitors from the official repositories (P-N-Suganthan/2024-CEC, 2025-CEC, 2026-CEC):
6 algorithms in 2024, 1 in 2025 (RDEx, the only bound constrained single-objective submission), 7 in 2026
(14 in total, 25–52 runs per function). The result files use mixed function numbering (most use F1..F29 without F2;
mLSHADE_LR and jSOa 2024 use the original numbering; BlockEA 2024 includes F2 and lacks F30). The mapping of every
algorithm was verified by correlating its initial errors per function with the verified Python port
(r ≈ 0.99 for the correct mapping vs 0.3–0.6 otherwise).

**Results.** 18–19 of 29 functions were never solved (error < 1e-8) by any run of any algorithm, including all 10
composition functions. Success rate therefore does not separate them; they were ranked by how far from the optimum
the algorithms ended:

| Function | Criterion | 2024 | 2025 | 2026 | pooled |
|---|---|---|---|---|---|
| **F30** Composition 10 | median over algorithms (typical algorithm) | **#1** (1926) | #2 | **#1** (1981) | **#1** (1970) |
| | median of the best algorithm | #4 | #2 | #2 | #4 |
| F26 Composition 6 | median over algorithms | #3 | #1 | #3 | #3 (896) |
| | median of the best algorithm | **#1** (501) | **#1** | **#1** (652) | **#1** (501) |
| F27 Composition 7 | best run | **#1** | **#1** | #2 | **#1** |

(values = final error f − f*)

**Decision: F30.** It is the hardest function for a *typical* state-of-the-art algorithm (median error ~1950, twice
the next function) in 2024, 2026 and pooled, and it is in the top 4 by every criterion in every edition. It is also
the most complex structure of the suite (composition of three hybrid functions). Alternative: F26 is the hardest for
the *best* algorithm. Note: errors of composition functions are "quantized" by the component biases (100, 200, ...),
so the fine ranking among them is indicative only.

**Implementation.** `resource.cec2017.f_30` – numpy port of the official C code (`cec17_test_func.cpp`) including its
side effects on the shared work buffers; verified against the compiled official code for all 30 functions at D=30
(max relative error 5·10^-14). Only D=30 input data are bundled. `evaluate()` returns f(x) − 100·k.

### BBOB: f24 Lunacek bi-Rastrigin, instance 1, D=30

**Why.** Consistently the hardest function of the noiseless bbob suite in the literature and in the COCO data:
the virtual best algorithm of 2009–2016 (`best09-16-bbob` reference data shipped with `cocopp`) needs an expected
running time of ~9·10^5·D evaluations at 20D and ~3.3·10^6·D at 40D to reach 1e-8, roughly 8–14× more than the
next hardest function (f19). Two funnels, the deceptive one covers ~70% of the search space.

**D=30.** BBOB is officially benchmarked at 2, 3, 5, 10, 20, 40, but the functions are scalable; D=30 keeps all
three functions in the same dimension. f24 is the hardest at both 20D and 40D.

**Implementation.** `resource.bbob.f_24` via IOHexperimenter (`ioh`), because the official `cocoex` does not provide
D=30. `ioh` was verified against `cocoex` for all 24 functions × 3 instances at 20D and 40D (max relative difference
≤ 1e-11). `evaluate()` returns f(x) − f(x*).

### Protocol notes

- Evaluation speed differs a lot (BBOB f24 ~5 µs, CEC F30 ~300 µs per evaluation in Python), so with the 30 s time
  limit the number of function evaluations differs per function (cap 10^6 for all). Competition difficulty was
  measured with other budgets (CEC: 3·10^5 FEs).
- Error scales differ (random search: ~10^2 on GNBG f24, ~10^8 on CEC F30); cross-function analysis needs
  normalization or ranks.

## Open TODOs

- [ ] Smoke test against real LLM APIs (OpenAI, Anthropic, Google, Ollama) to verify provider usage fields and the
      usage ledger – waiting for API keys.
