import logging
import copy
import json
import os
import time
import urllib.parse
import urllib.request
import urllib.error
import re
from typing import Dict
import uuid

from .evaluator import Evaluator
from ..solutions.solution import Solution
from ..loader import Parameter, PrimitiveType


class EvaluatorJan(Evaluator):

    """
    SOC pipeline task

    System prompt:

    You are a prompt-engineering assistant for a SOC workflow.

Goal:
Produce exactly three reusable prompts for a smaller “slave” LLM:
1) a SYSTEM prompt,
2) a SUMMARY prompt,
3) a MITIGATE prompt.

The slave LLM will receive a SOC event descriptor in JSON format (fields such as event_id, timestamp, source, user, mitre_tag, risk_score, confidence, cloud_service, resource_id, etc.). The slave’s job is to help a SOC operator by summarizing the incident and proposing mitigation/detections.

Critical constraints:
- Output MUST follow the exact template below so it can be parsed automatically.
- Do NOT include any extra commentary, headings, markdown, code fences, or additional fields.
- Prompts must be concise, unambiguous, and operational for SOC use.
- The slave must treat the provided mitre_tag as the primary technique and center the analysis on it.
- High precision: do not invent details not present in the event descriptor. If the descriptor lacks evidence, the slave must state limitations and ask targeted questions.
- The slave may propose additional MITRE techniques only as “candidates” with explicit rationale and lower confidence.
- The slave output must be deterministic and easy to parse (fixed section headers, consistent bullets).

Template (output exactly this structure and nothing else):

<SYSTEM_PROMPT>
...text...
</SYSTEM_PROMPT>
<SUMMARY_PROMPT>
...text...
</SUMMARY_PROMPT>
<MITIGATE_PROMPT>
...text...
</MITIGATE_PROMPT>


    Initial prompt:

Create the three reusable prompts for the slave LLM.

Behavior requirements for the slave LLM:
- Input will be a SOC event descriptor in JSON.
- The slave must produce a compact operator-facing summary and recommended actions.
- The slave must focus primarily on the event’s mitre_tag:
  - Interpret what the technique means at a high level (without guessing specifics).
  - Explain why the event might map to that mitre_tag using only fields in the descriptor.
  - If evidence is weak (e.g., low confidence), explicitly say so and list what extra telemetry is needed.
- The slave must output exactly these sections, with these exact headers:
  SUMMARY
  EVENT_FIELDS
  PRIMARY_TECHNIQUE
  CANDIDATE_TECHNIQUES
  MISSING_INFO
  MITIGATIONS
  DETECTIONS
  NEXT_STEPS
- In PRIMARY_TECHNIQUE: include the T-code from mitre_tag, plus: Evidence (quotes/field values), Rationale, Confidence(0-100).
- In CANDIDATE_TECHNIQUES: list at most 3 additional techniques only if justified; otherwise write "None".
- In MITIGATIONS/DETECTIONS/NEXT_STEPS: map each bullet explicitly back to the primary technique (and candidates if used).
- Keep it short: prefer bullets; no long paragraphs; no speculative storytelling.

Additional constraints:
- The slave must not invent IOCs, commands, processes, affected hosts, or attack paths that are not in the JSON.
- When the event_type is "cloud" and cloud_service exists, include cloud-specific checks and telemetry suggestions (IAM changes, audit logs, resource policy review, etc.) while staying technique-mapped.
- Use the event’s risk_score to set urgency (e.g., High/Medium/Low) with a simple thresholding of your choice.

Output MUST match the exact parseable template described in the system instruction.

    """



    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        return {
            'llmmodel': Parameter(short_name="llmmodel", type=PrimitiveType.enum,
                                   long_name="LLM model",
                                   description="Which model shoudl be used for generating the output.",
                               enum_options=['llama3.2:1b', 'llama3.1:8b', 'llama3.1:80b'],
                                   default='llama3.2:1b'
                                   ),
            'model_temp': Parameter(short_name="model_temp", type=PrimitiveType.float,
                                 long_name="LLM model temperature",
                                 description="LLM model temperature.",
                                 default=0.2
                                 ),
            'n_total': Parameter(short_name="n_total", type=PrimitiveType.int,
                                   long_name="Number of incidents",
                                   description="Number of security incidents to be analyzed.",
                                   default=200
                                   ),
            'feedback_msg_template': Parameter(short_name="feedback_msg_template", type=PrimitiveType.markdown,
                                               long_name="Template for a feedback message",
                                               description="Feedback message for evaluation. Can use {keywords}",
                                               default="The evaluation metrics for the current prompts are:\nAvg. "
                                                       "Score: {avg_score}\nAvg. Tokens: {avg_tokens}\nAvg. Latency: "
                                                       "{avg_latency}\nComposite: {composite}"
                                               ),
            'init_msg_template': Parameter(short_name="init_msg_template", type=PrimitiveType.markdown,
                                           long_name="Template for an initial message",
                                           description="Initial message for evaluation. Specific for each evaluator.",
                                           default="""Create the three reusable prompts for the slave LLM.
                                           Behavior requirements for the slave LLM:
- Input will be a SOC event descriptor in JSON.
- The slave must produce a compact operator-facing summary and recommended actions.
- The slave must focus primarily on the event’s mitre_tag:
  - Interpret what the technique means at a high level (without guessing specifics).
  - Explain why the event might map to that mitre_tag using only fields in the descriptor.
  - If evidence is weak (e.g., low confidence), explicitly say so and list what extra telemetry is needed.
- The slave must output exactly these sections, with these exact headers:
  SUMMARY
  EVENT_FIELDS
  PRIMARY_TECHNIQUE
  CANDIDATE_TECHNIQUES
  MISSING_INFO
  MITIGATIONS
  DETECTIONS
  NEXT_STEPS
- In PRIMARY_TECHNIQUE: include the T-code from mitre_tag, plus: Evidence (quotes/field values), Rationale, Confidence(0-100).
- In CANDIDATE_TECHNIQUES: list at most 3 additional techniques only if justified; otherwise write "None".
- In MITIGATIONS/DETECTIONS/NEXT_STEPS: map each bullet explicitly back to the primary technique (and candidates if used).
- Keep it short: prefer bullets; no long paragraphs; no speculative storytelling.

Additional constraints:
- The slave must not invent IOCs, commands, processes, affected hosts, or attack paths that are not in the JSON.
- When the event_type is "cloud" and cloud_service exists, include cloud-specific checks and telemetry suggestions (IAM changes, audit logs, resource policy review, etc.) while staying technique-mapped.
- Use the event’s risk_score to set urgency (e.g., High/Medium/Low) with a simple thresholding of your choice.

Output MUST match the exact parseable template described in the system instruction.""",
                                           readonly=True),

            'keywords': Parameter(short_name="keywords", type=PrimitiveType.enum, long_name='Feedback keywords',
                                  description="Feedback keyword-based sentences",
                                  enum_options=['llmmodel', 'model_temp', 'n_total', 'experiment_id', 'avg_score',
                                                'avg_tokens', 'avg_latency', 'composite', 'prompt_system', 'prompt_summary', 'prompt_mitigate'], readonly=True),

        }

    def _init_params(self):
        super()._init_params()
        self._llmmodel = self.parameters.get('llmmodel', self.get_parameters().get('llmmodel').default)
        self._llmmodeltemp = self.parameters.get('model_temp', self.get_parameters().get('model_temp').default)
        self._ntotal = self.parameters.get('n_total', self.get_parameters().get('n_total').default)
        self._TAGS = ("SYSTEM_PROMPT", "SUMMARY_PROMPT", "MITIGATE_PROMPT")
        self._keys = {
            'llmmodel': self._llmmodel,
            'model_temp': self._llmmodeltemp,
            'n_total': self._ntotal,
            'experiment_id': None,
            'avg_score': None,
            'avg_tokens': None,
            'avg_latency': None,
            'composite': None,
            'prompt_system': None,
            'prompt_summary': None,
            'prompt_mitigate': None
        }

    def evaluate(self, solution: Solution) -> float:

        base = os.getenv("EASE_API_BASE_URL", "http://tunnel:10000").rstrip("/")

        # Get prompts from the solution
        try:
            prompts = self._parse_master_prompts(solution.get_input())
            prompt_system = prompts['system']
            prompt_summary = prompts['summary']
            prompt_mitigate = prompts['mitigate']

        except Exception as e:
            logging.error("Eval:Jan:", repr(e))
            logging.info("Eval:Jan: Fallback to default prompts.")
            prompt_system = "You are a SOC analyst."
            prompt_summary = "Summarize this incident."
            prompt_mitigate = "Suggest mitigations."

        self._keys['prompt_system'] = prompt_system
        self._keys['prompt_summary'] = prompt_summary
        self._keys['prompt_mitigate'] = prompt_mitigate

        experiment_id = str(uuid.uuid4())

        # --- placeholders from your curl (you'll change manually) ---
        payload = {
            "experiment_id": experiment_id,
            "model": self._llmmodel,
            "temperature": self._llmmodeltemp,
            "n_total": self._ntotal,
            "prompt_system": prompt_system,
            "prompt_summary": prompt_summary,
            "prompt_mitigate": prompt_mitigate,
        }

        print(payload)

        self._keys['experiment_id'] = experiment_id
        expected = int(payload.get("n_total", 2))

        # polling settings
        poll_interval_s = float(os.getenv("EASE_STATUS_POLL_S", "5.0"))
        max_wait_s = float(os.getenv("EASE_STATUS_MAX_WAIT_S", "7200.0"))

        # Where we store everything you asked for
        results_dict = {}

        def _http_json(method: str, url: str, body_obj=None, timeout_s: float = 30.0):
            data = None
            headers = {"Accept": "application/json"}
            if body_obj is not None:
                data = json.dumps(body_obj).encode("utf-8")
                headers["Content-Type"] = "application/json"

            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    status = getattr(resp, "status", 200)
                    raw = resp.read().decode("utf-8", errors="replace")
                    try:
                        return status, json.loads(raw), raw
                    except Exception:
                        return status, None, raw
            except urllib.error.HTTPError as e:
                raw = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
                try:
                    return e.code, json.loads(raw), raw
                except Exception:
                    return e.code, None, raw

        err = None
        try:
            # 1) POST /run
            run_url = f"{base}/run"
            status, run_json, run_raw = _http_json("POST", run_url, body_obj=payload, timeout_s=60.0)
            results_dict["run"] = {"http_status": status, "json": run_json, "raw": run_raw}

            if status < 200 or status >= 300:
                raise RuntimeError(f"POST /run failed with HTTP {status}: {run_raw[:2000]}")

            # 2) Poll /status/<experiment_id>?expected=<expected> until done==true
            q = urllib.parse.urlencode({"expected": expected})
            status_url = f"{base}/status/{urllib.parse.quote(experiment_id)}?{q}"

            deadline = time.time() + max_wait_s
            last_status_json = None

            while True:
                if time.time() > deadline:
                    raise TimeoutError(f"Timed out waiting for done=true after {max_wait_s}s")

                s_status, s_json, s_raw = _http_json("GET", status_url, body_obj=None, timeout_s=30.0)
                results_dict["status_last"] = {"http_status": s_status, "json": s_json, "raw": s_raw}

                print(results_dict)

                if s_status != 200 or not isinstance(s_json, dict):
                    # keep trying a little, but record what we saw
                    last_status_json = s_json
                else:
                    last_status_json = s_json
                    if bool(s_json.get("done")) is True:
                        break

                time.sleep(poll_interval_s)

            # 3) GET /result/<experiment_id>?expected=<expected>
            result_url = f"{base}/result/{urllib.parse.quote(experiment_id)}?{q}"
            r_status, r_json, r_raw = _http_json("GET", result_url, body_obj=None, timeout_s=30.0)
            results_dict["result"] = {"http_status": r_status, "json": r_json, "raw": r_raw}

            if r_status != 200 or not isinstance(r_json, dict):
                raise RuntimeError(f"GET /result failed with HTTP {r_status}: {r_raw[:2000]}")

            # Save final result json directly in a dict (your request)
            results_dict["parsed"] = r_json

            # Fitness: "avg_score"
            if "avg_score" in r_json and isinstance(r_json["avg_score"], (int, float)):
                fitness = float(r_json["avg_score"])
            else:
                fitness = 0.0

            # Attach results to solution metadata
            solution.add_metadata("experiment_id", experiment_id)
            solution.add_metadata("expected", expected)
            solution.add_metadata("api_base", base)
            solution.add_metadata("run_status", results_dict["run"]["http_status"])
            solution.add_metadata("status_last", last_status_json)
            solution.add_metadata("result", r_json)

            if "composite" in r_json and isinstance(r_json["composite"], (int, float)):
                self._keys['composite'] = float(r_json["composite"])
            if "avg_score" in r_json and isinstance(r_json["avg_score"], (int, float)):
                self._keys['avg_score'] = float(r_json["avg_score"])
            if "avg_tokens" in r_json and isinstance(r_json["avg_tokens"], (int, float)):
                self._keys['avg_tokens'] = float(r_json["avg_tokens"])
            if "avg_latency" in r_json and isinstance(r_json["avg_latency"], (int, float)):
                self._keys['avg_latency'] = float(r_json["avg_latency"])

        except Exception as e:
            err = str(e)
            fitness = 0.0
            solution.add_metadata("error", err)

        solution.set_fitness(fitness)
        solution.add_metadata("results_dict", results_dict)
        feedback = self.get_feedback_msg_template().format(**self._keys)
        solution.set_feedback(feedback)

        self._check_if_best(solution)

        return fitness

    @classmethod
    def get_short_name(cls) -> str:
        return "eval.jan"

    @classmethod
    def get_long_name(cls) -> str:
        return "Jans' super duper cybersec awesomness"

    @classmethod
    def get_description(cls) -> str:
        return "Evolves prompts for the super duper cybersec task."

    @classmethod
    def get_tags(cls) -> dict:
        return {'input': {'text'}, 'output': set()}

    def _parse_master_prompts(self, text: str) -> Dict[str, str]:
        """
        Parses master LLM output of the form:

        <SYSTEM_PROMPT>...</SYSTEM_PROMPT>
        <SUMMARY_PROMPT>...</SUMMARY_PROMPT>
        <MITIGATE_PROMPT>...</MITIGATE_PROMPT>

        Returns:
            {"system": "...", "summary": "...", "mitigate": "..."}
        Raises:
            ValueError if any tag is missing or duplicated.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input must be a non-empty string.")

        out: Dict[str, str] = {}

        for tag in self._TAGS:
            pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
            matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)

            if len(matches) == 0:
                raise ValueError(f"Missing tag <{tag}>...</{tag}> in master output.")
            if len(matches) > 1:
                raise ValueError(f"Duplicate tag <{tag}> found ({len(matches)} occurrences).")

            content = matches[0].strip()
            if not content:
                raise ValueError(f"Tag <{tag}> is present but empty.")

            key = tag.replace("_PROMPT", "").lower()  # system/summary/mitigate
            out[key] = content

        return out

    def _check_if_best(self, solution: Solution) -> bool:
        if self._best is None or solution.get_fitness() >= self._best.get_fitness():
            self._best = copy.deepcopy(solution)
            return True
        return False
