# Experiment checklist – context summarizer

Tick a configuration when all 11 repetitions finished successfully (10 valid algorithms each). Keep the count of
finished repetitions (`0/11`) and the Task ids / notes on the line. Decisions and protocol: `notes/paper-summarizer.md`.

## Fixed protocol (all configurations)

| Setting | Value |
|---|---|
| Repetitions per configuration | 11 |
| Valid algorithms per repetition | 10 (`stop.condvaliditers` = 10, `stop.condconsinvaliditers` = 5) |
| Runs per algorithm / time limit | 30 runs × 30 s (`runs` = 0 → resource default 30, `time` = 30) |
| Development score | mean over the 30 runs (`fitness_stat` = mean) |
| Dimension | D = 30 |
| Task | `max_context_size` = 0, no system message, initial message = P0 (evaluator default), feedback from solution on |
| Repeated message | C1: `Generate another optimizer independently.` + closing; other cells: `Improve the optimizer using the evidence below.` + preamble + closing |
| Tests | `test.psyntax`, `test.pimports` |
| Summary (`anal.historysummary`) | `iterations` = 10, `repairs` = 2; no summary after the 10th algorithm |
| CERIT connector | `disable_cache` = on |

Task name pattern: `SUM | <G> | <F> | <cell> | <S> | rep <nn>` (e.g. `SUM | G1 | F1 | last_best+sum | S1 | rep 03`).

## Variables to fill in before the start

- [x] G1 = `llm.anthropic` / `claude-opus-5-5` (Claude Opus 5.5) – phase 1
- [x] G2 = `llm.openai` / `gpt-6.1-sol` (GPT-6.1 Sol)
- [x] G3 = `llm.google` / `gemini-3.8-flash` (Gemini 3.8 Flash)
- [x] S1 summarizer = the generator model itself (G1 → `claude-opus-5-5`, G2 → `gpt-6.1-sol`, G3 → `gemini-3.8-flash`), separate API call
- [x] S2 local summarizer = `llm.cerit` / `kimi-k3` (Kimi K3, CERIT-SC; `disable_cache` on)

Summarizer scenarios: **S1** = the same model as the generator (separate API call), **S2** = local model `kimi-k3` (CERIT).
Cells without a summary (C1, C4, C7) are run only once (they do not depend on the summarizer).

## Size

| | per generator × function | per generator | total (3 generators) |
|---|---|---|---|
| Configurations | 15 (9 × S1 + 6 × S2) | 45 | 135 |
| Tasks (× 11 repetitions) | 165 | 495 | 1 485 |
| Valid algorithms (× 10) | 1 650 | 4 950 | 14 850 |
| Summarizer calls (summary Tasks × 9, without repairs) | 1 188 | 3 564 | 10 692 |
| Evaluation CPU time (algorithms × 30 runs × 30 s) | ~412 h | ~1 237 h | ~3 712 h |

## Pre-flight

- [x] Smoke test CERIT (`gpt-oss-120b`): nocontext, last_best+sum, all+sum_structured – 2026-10-08
- [x] CERIT response cache disabled and verified – 2026-10-08
- [ ] Smoke test of G1 `claude-opus-5-5` (API key, usage fields in `usage_calls.csv`)
- [ ] Smoke test of G2 `gpt-6.1-sol`
- [ ] Smoke test of G3 `gemini-3.8-flash`
- [ ] Smoke test of the S2 summarizer `kimi-k3` (structured summary validates)
- [ ] Core image rebuilt from the current `requirements.txt` (`coco-experiment`)
- [ ] All three functions evaluated once on the server (GNBG, CEC 2017 F30, BBOB)
- [ ] Final texts (prompts, repeated messages) frozen in frontEASE and recorded in the decision log

## G1 – Claude Opus 5.5 (`llm.anthropic` / `claude-opus-5-5`) – phase 1

### G1 × F1 – GNBG-II f24 (`resource.gnbg.f_24`)

Summarizer S1 (same model):

- [ ] `G1-F1-C1-S1` **nocontext** (code `none`) — 0/11 — tasks: 
- [ ] `G1-F1-C2-S1` **sum** (code `none`, summary `free`) — 0/11 — tasks: 
- [ ] `G1-F1-C3-S1` **sum_structured** (code `none`, summary `structured`) — 0/11 — tasks: 
- [ ] `G1-F1-C4-S1` **all** (code `all`) — 0/11 — tasks: 
- [ ] `G1-F1-C5-S1` **all+sum** (code `all`, summary `free`) — 0/11 — tasks: 
- [ ] `G1-F1-C6-S1` **all+sum_structured** (code `all`, summary `structured`) — 0/11 — tasks: 
- [ ] `G1-F1-C7-S1` **last_best** (code `last_best`) — 0/11 — tasks: 
- [ ] `G1-F1-C8-S1` **last_best+sum** (code `last_best`, summary `free`) — 0/11 — tasks: 
- [ ] `G1-F1-C9-S1` **last_best+sum_structured** (code `last_best`, summary `structured`) — 0/11 — tasks: 

Summarizer S2 (local model):

- [ ] `G1-F1-C2-S2` **sum** (code `none`, summary `free`) — 0/11 — tasks: 
- [ ] `G1-F1-C3-S2` **sum_structured** (code `none`, summary `structured`) — 0/11 — tasks: 
- [ ] `G1-F1-C5-S2` **all+sum** (code `all`, summary `free`) — 0/11 — tasks: 
- [ ] `G1-F1-C6-S2` **all+sum_structured** (code `all`, summary `structured`) — 0/11 — tasks: 
- [ ] `G1-F1-C8-S2` **last_best+sum** (code `last_best`, summary `free`) — 0/11 — tasks: 
- [ ] `G1-F1-C9-S2` **last_best+sum_structured** (code `last_best`, summary `structured`) — 0/11 — tasks: 

### G1 × F2 – CEC 2017 F30 (`resource.cec2017.f_30`)

Summarizer S1 (same model):

- [ ] `G1-F2-C1-S1` **nocontext** (code `none`) — 0/11 — tasks: 
- [ ] `G1-F2-C2-S1` **sum** (code `none`, summary `free`) — 0/11 — tasks: 
- [ ] `G1-F2-C3-S1` **sum_structured** (code `none`, summary `structured`) — 0/11 — tasks: 
- [ ] `G1-F2-C4-S1` **all** (code `all`) — 0/11 — tasks: 
- [ ] `G1-F2-C5-S1` **all+sum** (code `all`, summary `free`) — 0/11 — tasks: 
- [ ] `G1-F2-C6-S1` **all+sum_structured** (code `all`, summary `structured`) — 0/11 — tasks: 
- [ ] `G1-F2-C7-S1` **last_best** (code `last_best`) — 0/11 — tasks: 
- [ ] `G1-F2-C8-S1` **last_best+sum** (code `last_best`, summary `free`) — 0/11 — tasks: 
- [ ] `G1-F2-C9-S1` **last_best+sum_structured** (code `last_best`, summary `structured`) — 0/11 — tasks: 

Summarizer S2 (local model):

- [ ] `G1-F2-C2-S2` **sum** (code `none`, summary `free`) — 0/11 — tasks: 
- [ ] `G1-F2-C3-S2` **sum_structured** (code `none`, summary `structured`) — 0/11 — tasks: 
- [ ] `G1-F2-C5-S2` **all+sum** (code `all`, summary `free`) — 0/11 — tasks: 
- [ ] `G1-F2-C6-S2` **all+sum_structured** (code `all`, summary `structured`) — 0/11 — tasks: 
- [ ] `G1-F2-C8-S2` **last_best+sum** (code `last_best`, summary `free`) — 0/11 — tasks: 
- [ ] `G1-F2-C9-S2` **last_best+sum_structured** (code `last_best`, summary `structured`) — 0/11 — tasks: 

### G1 × F3 – BBOB f24 (i1) (`resource.bbob.f_24`)

Summarizer S1 (same model):

- [ ] `G1-F3-C1-S1` **nocontext** (code `none`) — 0/11 — tasks: 
- [ ] `G1-F3-C2-S1` **sum** (code `none`, summary `free`) — 0/11 — tasks: 
- [ ] `G1-F3-C3-S1` **sum_structured** (code `none`, summary `structured`) — 0/11 — tasks: 
- [ ] `G1-F3-C4-S1` **all** (code `all`) — 0/11 — tasks: 
- [ ] `G1-F3-C5-S1` **all+sum** (code `all`, summary `free`) — 0/11 — tasks: 
- [ ] `G1-F3-C6-S1` **all+sum_structured** (code `all`, summary `structured`) — 0/11 — tasks: 
- [ ] `G1-F3-C7-S1` **last_best** (code `last_best`) — 0/11 — tasks: 
- [ ] `G1-F3-C8-S1` **last_best+sum** (code `last_best`, summary `free`) — 0/11 — tasks: 
- [ ] `G1-F3-C9-S1` **last_best+sum_structured** (code `last_best`, summary `structured`) — 0/11 — tasks: 

Summarizer S2 (local model):

- [ ] `G1-F3-C2-S2` **sum** (code `none`, summary `free`) — 0/11 — tasks: 
- [ ] `G1-F3-C3-S2` **sum_structured** (code `none`, summary `structured`) — 0/11 — tasks: 
- [ ] `G1-F3-C5-S2` **all+sum** (code `all`, summary `free`) — 0/11 — tasks: 
- [ ] `G1-F3-C6-S2` **all+sum_structured** (code `all`, summary `structured`) — 0/11 — tasks: 
- [ ] `G1-F3-C8-S2` **last_best+sum** (code `last_best`, summary `free`) — 0/11 — tasks: 
- [ ] `G1-F3-C9-S2` **last_best+sum_structured** (code `last_best`, summary `structured`) — 0/11 — tasks: 

## G2 – GPT-6.1 Sol (`llm.openai` / `gpt-6.1-sol`)

### G2 × F1 – GNBG-II f24 (`resource.gnbg.f_24`)

Summarizer S1 (same model):

- [ ] `G2-F1-C1-S1` **nocontext** (code `none`) — 0/11 — tasks: 
- [ ] `G2-F1-C2-S1` **sum** (code `none`, summary `free`) — 0/11 — tasks: 
- [ ] `G2-F1-C3-S1` **sum_structured** (code `none`, summary `structured`) — 0/11 — tasks: 
- [ ] `G2-F1-C4-S1` **all** (code `all`) — 0/11 — tasks: 
- [ ] `G2-F1-C5-S1` **all+sum** (code `all`, summary `free`) — 0/11 — tasks: 
- [ ] `G2-F1-C6-S1` **all+sum_structured** (code `all`, summary `structured`) — 0/11 — tasks: 
- [ ] `G2-F1-C7-S1` **last_best** (code `last_best`) — 0/11 — tasks: 
- [ ] `G2-F1-C8-S1` **last_best+sum** (code `last_best`, summary `free`) — 0/11 — tasks: 
- [ ] `G2-F1-C9-S1` **last_best+sum_structured** (code `last_best`, summary `structured`) — 0/11 — tasks: 

Summarizer S2 (local model):

- [ ] `G2-F1-C2-S2` **sum** (code `none`, summary `free`) — 0/11 — tasks: 
- [ ] `G2-F1-C3-S2` **sum_structured** (code `none`, summary `structured`) — 0/11 — tasks: 
- [ ] `G2-F1-C5-S2` **all+sum** (code `all`, summary `free`) — 0/11 — tasks: 
- [ ] `G2-F1-C6-S2` **all+sum_structured** (code `all`, summary `structured`) — 0/11 — tasks: 
- [ ] `G2-F1-C8-S2` **last_best+sum** (code `last_best`, summary `free`) — 0/11 — tasks: 
- [ ] `G2-F1-C9-S2` **last_best+sum_structured** (code `last_best`, summary `structured`) — 0/11 — tasks: 

### G2 × F2 – CEC 2017 F30 (`resource.cec2017.f_30`)

Summarizer S1 (same model):

- [ ] `G2-F2-C1-S1` **nocontext** (code `none`) — 0/11 — tasks: 
- [ ] `G2-F2-C2-S1` **sum** (code `none`, summary `free`) — 0/11 — tasks: 
- [ ] `G2-F2-C3-S1` **sum_structured** (code `none`, summary `structured`) — 0/11 — tasks: 
- [ ] `G2-F2-C4-S1` **all** (code `all`) — 0/11 — tasks: 
- [ ] `G2-F2-C5-S1` **all+sum** (code `all`, summary `free`) — 0/11 — tasks: 
- [ ] `G2-F2-C6-S1` **all+sum_structured** (code `all`, summary `structured`) — 0/11 — tasks: 
- [ ] `G2-F2-C7-S1` **last_best** (code `last_best`) — 0/11 — tasks: 
- [ ] `G2-F2-C8-S1` **last_best+sum** (code `last_best`, summary `free`) — 0/11 — tasks: 
- [ ] `G2-F2-C9-S1` **last_best+sum_structured** (code `last_best`, summary `structured`) — 0/11 — tasks: 

Summarizer S2 (local model):

- [ ] `G2-F2-C2-S2` **sum** (code `none`, summary `free`) — 0/11 — tasks: 
- [ ] `G2-F2-C3-S2` **sum_structured** (code `none`, summary `structured`) — 0/11 — tasks: 
- [ ] `G2-F2-C5-S2` **all+sum** (code `all`, summary `free`) — 0/11 — tasks: 
- [ ] `G2-F2-C6-S2` **all+sum_structured** (code `all`, summary `structured`) — 0/11 — tasks: 
- [ ] `G2-F2-C8-S2` **last_best+sum** (code `last_best`, summary `free`) — 0/11 — tasks: 
- [ ] `G2-F2-C9-S2` **last_best+sum_structured** (code `last_best`, summary `structured`) — 0/11 — tasks: 

### G2 × F3 – BBOB f24 (i1) (`resource.bbob.f_24`)

Summarizer S1 (same model):

- [ ] `G2-F3-C1-S1` **nocontext** (code `none`) — 0/11 — tasks: 
- [ ] `G2-F3-C2-S1` **sum** (code `none`, summary `free`) — 0/11 — tasks: 
- [ ] `G2-F3-C3-S1` **sum_structured** (code `none`, summary `structured`) — 0/11 — tasks: 
- [ ] `G2-F3-C4-S1` **all** (code `all`) — 0/11 — tasks: 
- [ ] `G2-F3-C5-S1` **all+sum** (code `all`, summary `free`) — 0/11 — tasks: 
- [ ] `G2-F3-C6-S1` **all+sum_structured** (code `all`, summary `structured`) — 0/11 — tasks: 
- [ ] `G2-F3-C7-S1` **last_best** (code `last_best`) — 0/11 — tasks: 
- [ ] `G2-F3-C8-S1` **last_best+sum** (code `last_best`, summary `free`) — 0/11 — tasks: 
- [ ] `G2-F3-C9-S1` **last_best+sum_structured** (code `last_best`, summary `structured`) — 0/11 — tasks: 

Summarizer S2 (local model):

- [ ] `G2-F3-C2-S2` **sum** (code `none`, summary `free`) — 0/11 — tasks: 
- [ ] `G2-F3-C3-S2` **sum_structured** (code `none`, summary `structured`) — 0/11 — tasks: 
- [ ] `G2-F3-C5-S2` **all+sum** (code `all`, summary `free`) — 0/11 — tasks: 
- [ ] `G2-F3-C6-S2` **all+sum_structured** (code `all`, summary `structured`) — 0/11 — tasks: 
- [ ] `G2-F3-C8-S2` **last_best+sum** (code `last_best`, summary `free`) — 0/11 — tasks: 
- [ ] `G2-F3-C9-S2` **last_best+sum_structured** (code `last_best`, summary `structured`) — 0/11 — tasks: 

## G3 – Gemini 3.8 Flash (`llm.google` / `gemini-3.8-flash`)

### G3 × F1 – GNBG-II f24 (`resource.gnbg.f_24`)

Summarizer S1 (same model):

- [ ] `G3-F1-C1-S1` **nocontext** (code `none`) — 0/11 — tasks: 
- [ ] `G3-F1-C2-S1` **sum** (code `none`, summary `free`) — 0/11 — tasks: 
- [ ] `G3-F1-C3-S1` **sum_structured** (code `none`, summary `structured`) — 0/11 — tasks: 
- [ ] `G3-F1-C4-S1` **all** (code `all`) — 0/11 — tasks: 
- [ ] `G3-F1-C5-S1` **all+sum** (code `all`, summary `free`) — 0/11 — tasks: 
- [ ] `G3-F1-C6-S1` **all+sum_structured** (code `all`, summary `structured`) — 0/11 — tasks: 
- [ ] `G3-F1-C7-S1` **last_best** (code `last_best`) — 0/11 — tasks: 
- [ ] `G3-F1-C8-S1` **last_best+sum** (code `last_best`, summary `free`) — 0/11 — tasks: 
- [ ] `G3-F1-C9-S1` **last_best+sum_structured** (code `last_best`, summary `structured`) — 0/11 — tasks: 

Summarizer S2 (local model):

- [ ] `G3-F1-C2-S2` **sum** (code `none`, summary `free`) — 0/11 — tasks: 
- [ ] `G3-F1-C3-S2` **sum_structured** (code `none`, summary `structured`) — 0/11 — tasks: 
- [ ] `G3-F1-C5-S2` **all+sum** (code `all`, summary `free`) — 0/11 — tasks: 
- [ ] `G3-F1-C6-S2` **all+sum_structured** (code `all`, summary `structured`) — 0/11 — tasks: 
- [ ] `G3-F1-C8-S2` **last_best+sum** (code `last_best`, summary `free`) — 0/11 — tasks: 
- [ ] `G3-F1-C9-S2` **last_best+sum_structured** (code `last_best`, summary `structured`) — 0/11 — tasks: 

### G3 × F2 – CEC 2017 F30 (`resource.cec2017.f_30`)

Summarizer S1 (same model):

- [ ] `G3-F2-C1-S1` **nocontext** (code `none`) — 0/11 — tasks: 
- [ ] `G3-F2-C2-S1` **sum** (code `none`, summary `free`) — 0/11 — tasks: 
- [ ] `G3-F2-C3-S1` **sum_structured** (code `none`, summary `structured`) — 0/11 — tasks: 
- [ ] `G3-F2-C4-S1` **all** (code `all`) — 0/11 — tasks: 
- [ ] `G3-F2-C5-S1` **all+sum** (code `all`, summary `free`) — 0/11 — tasks: 
- [ ] `G3-F2-C6-S1` **all+sum_structured** (code `all`, summary `structured`) — 0/11 — tasks: 
- [ ] `G3-F2-C7-S1` **last_best** (code `last_best`) — 0/11 — tasks: 
- [ ] `G3-F2-C8-S1` **last_best+sum** (code `last_best`, summary `free`) — 0/11 — tasks: 
- [ ] `G3-F2-C9-S1` **last_best+sum_structured** (code `last_best`, summary `structured`) — 0/11 — tasks: 

Summarizer S2 (local model):

- [ ] `G3-F2-C2-S2` **sum** (code `none`, summary `free`) — 0/11 — tasks: 
- [ ] `G3-F2-C3-S2` **sum_structured** (code `none`, summary `structured`) — 0/11 — tasks: 
- [ ] `G3-F2-C5-S2` **all+sum** (code `all`, summary `free`) — 0/11 — tasks: 
- [ ] `G3-F2-C6-S2` **all+sum_structured** (code `all`, summary `structured`) — 0/11 — tasks: 
- [ ] `G3-F2-C8-S2` **last_best+sum** (code `last_best`, summary `free`) — 0/11 — tasks: 
- [ ] `G3-F2-C9-S2` **last_best+sum_structured** (code `last_best`, summary `structured`) — 0/11 — tasks: 

### G3 × F3 – BBOB f24 (i1) (`resource.bbob.f_24`)

Summarizer S1 (same model):

- [ ] `G3-F3-C1-S1` **nocontext** (code `none`) — 0/11 — tasks: 
- [ ] `G3-F3-C2-S1` **sum** (code `none`, summary `free`) — 0/11 — tasks: 
- [ ] `G3-F3-C3-S1` **sum_structured** (code `none`, summary `structured`) — 0/11 — tasks: 
- [ ] `G3-F3-C4-S1` **all** (code `all`) — 0/11 — tasks: 
- [ ] `G3-F3-C5-S1` **all+sum** (code `all`, summary `free`) — 0/11 — tasks: 
- [ ] `G3-F3-C6-S1` **all+sum_structured** (code `all`, summary `structured`) — 0/11 — tasks: 
- [ ] `G3-F3-C7-S1` **last_best** (code `last_best`) — 0/11 — tasks: 
- [ ] `G3-F3-C8-S1` **last_best+sum** (code `last_best`, summary `free`) — 0/11 — tasks: 
- [ ] `G3-F3-C9-S1` **last_best+sum_structured** (code `last_best`, summary `structured`) — 0/11 — tasks: 

Summarizer S2 (local model):

- [ ] `G3-F3-C2-S2` **sum** (code `none`, summary `free`) — 0/11 — tasks: 
- [ ] `G3-F3-C3-S2` **sum_structured** (code `none`, summary `structured`) — 0/11 — tasks: 
- [ ] `G3-F3-C5-S2` **all+sum** (code `all`, summary `free`) — 0/11 — tasks: 
- [ ] `G3-F3-C6-S2` **all+sum_structured** (code `all`, summary `structured`) — 0/11 — tasks: 
- [ ] `G3-F3-C8-S2` **last_best+sum** (code `last_best`, summary `free`) — 0/11 — tasks: 
- [ ] `G3-F3-C9-S2` **last_best+sum_structured** (code `last_best`, summary `structured`) — 0/11 — tasks:
