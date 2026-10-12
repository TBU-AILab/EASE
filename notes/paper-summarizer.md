# Paper "Context summarizer" – decision log

Working notes for the follow-up of *Context matters in LLM-driven algorithm design* (Viktorin et al., Computer
Science Review 63 (2027) 101079). Branch `paper/summarizer` in EASE and frontEASE.

## Experiment design (agreed 2026-10-06)

- Factorial design, 9 cells: **code part** {none, last_best, all} × **summary of history** {none, free, structured}.
  none × none = nocontext.
- 11 repetitions per cell, 10 valid algorithms per repetition.
- Phase 1: one frontier generator LLM; the other two models follow (all three in the end).
- Summarizer: scenario 1 = the same model via a separate API call; scenario 2 = a local model (Ollama).
- The summary is regenerated from scratch in every iteration over **all valid algorithms so far** (max 10);
  no summary is generated after the last (10th) valid algorithm.
  The summarizer gets the code **and the development score** of every algorithm (since 2026-10-12; before, the
  scores were attached after summarization). The free-form summarizer decides itself whether and how it reports
  the scores; the structured one copies them into the field `development_score`.
- Feedback = code block first, summary block second.
- Goals: solution quality (primary), token usage (secondary), cost (cashflow) in post-processing.
- Functions (D=30): GNBG-II f24, CEC 2017 F30, BBOB f24 (see below).

## Decisions

| Date | Decision | Reason |
|---|---|---|
| 2026-10-07 | `fitness_stat = mean` (mean over the 30 runs) – **replaces the 2026-10-06 decision `min`** | The intended idea of the previous experiment (q = mean of the runs). |
| 2026-10-06 | `last_best` keeps both code blocks when the last algorithm is also the best | Same as the previous experiment. |
| 2026-10-06 | Google connector sends the conversation as stringified JSON (unchanged) | Comparability with the previous experiment. |
| 2026-10-06 | Token accounting: every LLM call is logged (`usage_calls.csv`, `usage_iterations.csv`) with provider-reported (billed) counts and counts from one common tokenizer (`o200k_base`) | Billed counts give the cost; the common tokenizer makes context sizes comparable across models. |
| 2026-10-06 | Per run of a generated algorithm the evaluator records: final error, number of function evaluations, evaluation and time at which the best value was found, wall-clock runtime and termination reason (`returned`, `max_time`, `max_evals`, `dim_error`, `exception`) – columns `Result_*` in the solution metadata | The previous experiment did not log runtimes consistently (limitation in the paper); algorithm parameters can be read from the generated code. |
| 2026-10-06 | CEC function: **CEC 2017 F30** (Composition Function 10), D=30 | See "Benchmark selection". |
| 2026-10-06 | BBOB function: **f24 Lunacek bi-Rastrigin**, instance 1, D=30 | See "Benchmark selection". |
| 2026-10-07 | The experiment description agreed in the conversation is authoritative; the paper draft is adapted to it | The first paper draft is a draft. |
| 2026-10-07 | Prompts: taken (approximately) from the paper draft – initial prompt, feedback template (untrusted-context preamble, `<SOURCE_CODE_CONTEXT>` / `<SUMMARY_CONTEXT>` blocks, closing instruction), free and structured summary prompts; parts referring to scores are removed from the summary prompts. Final wording can be adjusted in frontEASE before the experiment. | Single source of prompts for the paper and the experiment. |
| 2026-10-07 | Code block: complete code + development score (scientific notation `%.6e`) per algorithm; `all` = chronological, `last_best` = last then best | Paper draft serialization. |
| 2026-10-07 | ~~Scores are attached to the summary deterministically: free = list `A<k>: score` after the summary, structured = field `score` after `iteration` in each record~~ – **replaced 2026-10-12** (scores go into the summarizer) | The summarizer never sees the scores. |
| 2026-10-07 | Structured summary schema adapted from the draft without score-dependent fields (no `score`, `strengths/weaknesses_supported_by_results`; synthesis = `design_evolution`, `recurring_mechanisms`, `abandoned_mechanisms`) | Score-dependent fields cannot be filled without scores. |
| 2026-10-07 | No length limit of summaries | Agreed. |
| 2026-10-07 | Structured summary validated analytically (one JSON object, keys, types, ids and iterations in chronological order); at most 2 repair requests (original input + invalid summary + diagnostic); then the repetition stops (summary failure). Free summary: only non-empty. | Paper draft. |
| 2026-10-07 | No seeding of the experiment (BBOB uses the fixed instance 1), no holdout re-evaluation, no randomized interleaving of runs, no FE budget (the limit is time), out-of-bounds clipping and `func` calls as in `Runner`, imports via `test.pimports`; baseline algorithms will be added, not decided yet | Agreed; the paper draft has to be adapted. |
| 2026-10-07 | Summary moved from the evaluator to the Analysis module `anal.historysummary`; all texts (prompts, templates, schema, formats) are module parameters editable in frontEASE; preamble and closing instruction moved to the repeated message | Modular setup (summarizer configured as an analysis with its own LLM); experiment texts not hard-coded. |
| 2026-10-08 | Local models via the CERIT-SC AI-as-a-Service API (e-INFRA CZ, OpenAI compatible, `https://llm.ai.e-infra.cz/v1`): new connector `llm.cerit` = copy of `llm.openai` with `base_url` and `timeout` (1800 s, the service limits non-streaming requests to 30 min) parameters; models in `available_models.json` (editable in frontEASE) | Same library as OpenAI, different base URL and models. |
| 2026-10-08 | Smoke Tasks are created through the core REST API by `tools/create_smoke_tasks.py` (run inside the core container, API keys from environment variables; all 9 cells available, default 3 cells × 3 valid iterations × 3 runs × 5 s); frontEASE imports them by `InitialTaskSyncJob` (author = e-mail of the frontEASE user) | Reproducible smoke test without pickled files or keys in files. |
| 2026-10-08 | Usage of the LLM calls of an iteration is recorded even when the analysis fails (summary failure) | The failed calls were paid for. |
| 2026-10-08 | Duplicate stopping condition `stop.condmaxvaliditers` removed; the existing `stop.condvaliditers` is used | Same behaviour already existed in the core. |
| 2026-10-08 | `llm.cerit`: response cache of the service (LiteLLM proxy) disabled per request (`cache: {no-cache, no-store}`, parameter `disable_cache`, default on) | Smoke test: in `nocontext` iterations 2 and 3 got a byte-identical algorithm (identical tokens, 0.1 s response) – identical requests were answered from the cache, which would make repetitions dependent. |
| 2026-10-08 | Task: the same message is never sent twice in one request. Fixes P0 sent twice in the first request (no system message, `max_context_size` = 0; ~400 input tokens per repetition) and the system message sent twice when the history window reaches the beginning (system message with `max_context_size` 1–3). Verified on all combinations of system message × `max_context_size` ∈ {0, 1, 2, 3, unlimited}: only the duplicates were removed. | Requested; differs from the previous experiment only by the removed duplicate. |
| 2026-10-09 | Generators: **G1 = `claude-opus-5-5`** (Anthropic, phase 1; $4/$20 per 1M tokens), **G2 = `gpt-6.1-sol`** (OpenAI; $2/$10), **G3 = `gemini-3.8-flash`** (Google; $0.75/$3.75 until 2026-12-31, then $1.50/$7.50). S2 local summarizer = `llm.cerit` / `kimi-k3` (Moonshot Kimi K3, 2.8T MoE, 1M context, 48k output, fp8). Exact model names, no aliases. Provider defaults for all inference settings (Opus 5.5: adaptive thinking always on, effort `medium`; no server-side `fallbacks`, so a refusal is an invalid attempt and the model never changes). | Chosen by the user. Note: CERIT exact names are retired when a model is replaced – a replacement during the experiment would require a decision (aliases are not used because they would change the model silently). |
| 2026-10-09 | Model lists in `available_models.json` updated to the current offers (OpenAI GPT-6 family, Anthropic Claude 5.x, Gemini 3.x, CERIT exact names without aliases); default model of the connectors = first model of the list | Lists were outdated (e.g. `gemini-3-pro-preview` of the previous experiment is shut down). |
| 2026-10-12 | New initial prompt P0 (default of `init_msg_template`, used by the smoke script): text by the user – explicit import allowlist (numpy, math, random, time), externally seeded module-level generators, shorter black-box rule, "Now produce an optimizer …" – plus an instruction to comment the code and a commented random-search example | Requested by the user. Open: the evaluator does not seed the generators and the import allowlist is not enforced (`test.pimports` installs any import). |
| 2026-10-12 | Repeated message as in the previous experiment: C1 (nocontext) `Generate an algorithm.`, other cells `Improve.`; the untrusted-context preamble and the closing instruction are no longer used | Requested by the user; nocontext has no context, its requests are only P0 + the instruction. |
| 2026-10-12 | Free summary prompt loosened: no list of required topics (the draft asked to cover the same content as the structured summary), the summarizer decides what to keep, how to present it and how long it is; kept: untrusted-data rule, no scores (changed 2026-10-12, see below), refer to the algorithms by their identifiers, no invented performance, no copied code, no new algorithm | The free summary was too similar to the structured one (smoke test); the two conditions should differ in content freedom, not only in format. |
| 2026-10-12 | **Development scores go into the summarizer** – replaces the 2026-10-07 attachment of scores after summarization. The history sent to the summarizer has `<DEVELOPMENT_SCORE>` in every record (same serialization `%.6e` as in the code block). Free: the summarizer decides whether and how it uses or reports the scores; no score list is appended to the summary (`summary_block_free` = summary only; parameter `score_line` removed). Structured: new schema field type `score`, field `"development_score": "<string>"` directly after `algorithm_id`; the validator requires the exact input string (wrong or missing score → repair request); parameter `score_field` removed. Logged only: `scores_verbatim` = number of scores copied verbatim into a free summary. The other schema fields are unchanged (no score-dependent fields such as strengths/weaknesses). | Requested by the user: the summarizer sees the same evidence as the generator in the code block (code + score as one unit) and the free form keeps full freedom including the scores. Consequence: in `sum` (no code block) the generator may get no scores at all if the free summary omits them – part of the summary policy, measurable via `scores_verbatim`; a summarizer may also misreport a score in the free form (cannot be validated). |
| 2026-10-07 | BBOB via the official COCO implementation `cocoex.BareProblem` (any dimension), replacing IOHexperimenter | Official implementation in D=30; identical values to ioh (max rel. diff 1e-11). |

## EASE setup of one experiment cell (2026-10-07)

The summary is an **Analysis module**, the source-code context is produced by the evaluator. All texts are module
parameters (defaults = adapted paper-draft prompts) and can be overwritten in frontEASE for every Task.

| Module | Setting |
|---|---|
| LLM | generator model |
| Solution | `sol.codepython` |
| Tests | `test.psyntax`, `test.pimports` (+ `test.meta`) |
| Evaluator | `eval.papercontextsummarizer`: `code_context` none / last_best / all, `function`, `time`=30, `fitness_stat`=mean; texts `code_block`, `code_record`, `error_msg`, `id_format`, `score_format`, `escape_tags` |
| Analysis | `anal.historysummary` (only in the 6 summary cells): `summary_type` free / structured, `llm` = summarizer (same model or local), `iterations`=10, `repairs`=2; texts `prompt_free`, `prompt_structured`, `structured_schema` (JSON field types, used for the prompt and the validation), `history_record` (with the development score), `repair_prompt_*`, `summary_block_*`, `id_format`, `score_format`, `escape_tags` |
| Stopping | `stop.condvaliditers`=10, `stop.condconsinvaliditers` |
| Task | `max_context_size`=0, no system message, initial message = P0, feedback from solution on |

Message to the generator = repeated message + evaluator feedback (code block) + analysis feedback (summary block).
Every request = [P0, message]; the first request of a repetition = [P0] only. Repeated message (as in the previous
experiment, since 2026-10-12):

- `nocontext` (C1): `Generate an algorithm.` – the request is P0 + this instruction, there is no context;
- all other cells (C2–C9): `Improve.` followed by the code block and/or the summary block.

(Until 2026-10-12 the repeated message also contained an untrusted-context preamble and a closing instruction taken
from the paper draft; removed – P0 already requires returning only the source code.)

The structured-summary check is part of the analysis module (with the repair loop), not a Test module: Tests check
the generated code before evaluation and a failure makes the algorithm invalid and is reported to the generator,
whereas a summary failure is not an invalid algorithm and must be repaired by the summarizer.

## Cost estimate (2026-10-09)

Order-of-magnitude estimate of the API cost of **one generator** (45 configurations × 11 repetitions = 495 Tasks),
including the S1 summarizer (the same model). S2 (CERIT `kimi-k3`) is free of API charges.

Assumptions (from the sizes observed in the smoke tests):
- generator: ~5 200 calls (4 950 valid algorithms + ~5 % invalid attempts), on average ~5k input and ~5k output tokens
  per call (output includes reasoning; the `all` context grows to ~15k input tokens at the 10th algorithm);
- S1 summarizer: 1 782 calls (6 summary configurations × 3 functions × 11 repetitions × 9 summaries), on average
  ~8k input and ~1.5k output tokens;
- total ≈ 40M input + 29M output tokens per generator; list prices, no prompt-caching or batch discount.
Frontier models with long reasoning may produce more output tokens – treat the numbers as an order of magnitude.

| Provider | Model | Price per 1M tokens (input / output) | Estimate per generator |
|---|---|---|---|
| OpenAI | `gpt-6-astra` | $10 / $50 | ~$1 840 |
| OpenAI | **`gpt-6.1-sol` (G2)** | $2 / $10 | ~$370 |
| OpenAI | `gpt-6-luna` | $0.10 / $0.50 | ~$20 |
| Anthropic | `claude-fable-5-1` | $10 / $50 | ~$1 840 |
| Anthropic | **`claude-opus-5-5` (G1)** | $4 / $20 | ~$740 |
| Anthropic | `claude-sonnet-5-5` | $2 / $10 | ~$370 |
| Google | **`gemini-3.8-flash` (G3)** | $0.75 / $3.75 (until 2026-12-31), $1.50 / $7.50 (from 2027-01-01) | ~$140 (2026), ~$280 (2027) |
| Google | `gemini-3.1-pro-preview` | $2 / $12 (prompts ≤ 200k tokens) | ~$430 |
| CERIT | **`kimi-k3` (S2)** | – | $0 |

Chosen set (G1 + G2 + G3): ≈ **$1 250** (G3 run in 2026) to ≈ $1 390 (G3 run in 2027). Phase 1 (G1 only): ≈ $740.
Actual costs will be computed from `usage_calls.csv` (billed tokens per call) with the price list valid at the run date.

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
the next function) in 2024, 2026 and pooled, and it is in the top 4 by every criterion in 2024, 2026 and pooled (in 2025, with a single
algorithm, it is 6th by best run). It is also
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

**Implementation.** `resource.bbob.f_24` via the official COCO implementation: `cocoex.BareProblem("bbob", 24, 30, 1)`.
`cocoex.Suite` offers only the standard dimensions, `BareProblem` instantiates the functions in any dimension
(problem id `bbob_f024_i01_d30`). Cross-checked against IOHexperimenter for all 24 functions × 3 instances in D=30
(and IOHexperimenter against `cocoex.Suite` at 20D/40D): max relative difference ≤ 1e-11, f24 i1 bit-identical.
f* is taken from `best_value()`. `evaluate()` returns f(x) − f(x*).

### Protocol notes

- The limit of a run is time (30 s), not a number of function evaluations. Evaluation speed differs (BBOB f24
  ~5 µs, CEC F30 ~300 µs per evaluation in Python), so the number of evaluations differs per function; it is
  recorded per run.
- The 30 s limit is **not a hard limit**: it is enforced only when the algorithm evaluates the function after the
  limit (MaxTimeException). An algorithm that keeps computing without evaluations after the limit is not stopped;
  this is visible in `Result_runtime_s` (> time) with termination `returned`.
- Error scales differ (random search: ~10^2 on GNBG f24, ~10^8 on CEC F30); cross-function analysis needs
  normalization or ranks.

## Paper draft – status

2026-10-12: the draft (`Paper_Context_Summarizer (2).zip`) was updated to the experiment as implemented
(`Paper_Context_Summarizer (3).zip`): models (Opus 5.5, GPT-6.1 Sol, Gemini 3.8 Flash; Kimi K3 for S2), summarizer
scenarios S1/S2 (abstract, highlights, introduction, RQ4, design, counts: 1 485 chains, 14 850 algorithms,
10 692 summary calls, 445 500 runs), current prompts (P0, `Generate an algorithm.` / `Improve.`, code and summary
blocks, free and structured summary prompts without scores), scores attached after summarization, summary validation
as implemented, invalid-algorithm handling as implemented (syntax/import/execution, diagnostic to the model), `last_best`
duplicates, evaluation as implemented (no seeds, time limit enforced by the objective wrapper, clipping, run-level
records, FE cap 10^6), repetition-best score as the primary outcome, token accounting, benchmark verification;
removed: holdout, preregistration, randomized interleaving, seed banks, watchdog/isolated process, +inf for
out-of-bounds, vectorized calls, model-identity and finish-reason checks, compatibility pilot, sensitivity analysis,
named baselines (CMA-ES, jSO, DE); workflow figure without holdout.

2026-10-12: examples of assembled feedback messages added (Boxes 1–4, as Box 3 of the previous paper); then the
paper was switched to development scores in the summarizer input (no attached score list; structured field
`development_score`; Boxes regenerated from the module defaults).

Still open in the paper: P0 says the generators are "seeded externally by the evaluator" and lists an import
allowlist, while the evaluator does not seed and does not enforce the allowlist (`test.pimports`); the statistical
plan (primary outcome = repetition-best score, planned contrasts, Holm) and the human summary audit were kept from
the draft and need the authors' confirmation; baselines are not decided.

## Open TODOs

- [x] CERIT: the service honours `cache: {no-cache: true}` – verified 2026-10-08 by a `nocontext` smoke run: identical
      requests in iterations 2 and 3 produced different algorithms (normal response times, different token counts).
      `cached_input_tokens` still appear: this is the prompt-prefix (KV) cache of the inference server, not a cached
      response. The same run confirms P0 is sent once (first request 495 instead of 927 input tokens).
- [x] The first request of every repetition contained the initial message (P0) twice – fixed 2026-10-08 (see the
      decision table).

- [ ] Smoke test against real LLM APIs (OpenAI, Anthropic, Google, Ollama) to verify provider usage fields and the
      usage ledger – waiting for API keys.
- [x] CERIT (`llm.cerit`): real model IDs checked 2026-10-08 via `GET /v1/models`; `available_models.json` lists the
      concrete chat models only (no embeddings, rerankers, Whisper, and no aliases such as `mini`, `coder`,
      `thinker`, `glm`, `kimi`, `deepseek`, `auto-llm`, which may point to different models over time).
- [x] CERIT: chat completion responses contain `usage` (checked 2026-10-08: prompt/completion tokens, cached and
      created cache tokens). `reasoning_tokens` is reported as 0 even for reasoning models; the reasoning is included
      in `completion_tokens`, so the billed output is correct but the reasoning share is not available.
