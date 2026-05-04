import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template, select_autoescape

from fopimt.modul_dto import (
    AnalysisResult,
    EvaluatorResult,
    LLMConnectorResult,
    SolutionResult,
    StatResult,
    StoppingConditionResult,
    TestResult,
)


def latex_escape(text: object) -> str:
    if text is None:
        return ""

    s = str(text)
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "%": r"\%",
        "$": r"\$",
        "&": r"\&",
        "#": r"\#",
        "_": r"\_",
        "^": r"\textasciicircum{}",
        "~": r"\textasciitilde{}",
    }

    return "".join(replacements.get(ch, ch) for ch in s)


def _get_templates_root_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "templates"


def _get_template(template_relative_path: str, output_format: str = "html") -> Template:
    normalized_format = (output_format or "html").strip().lower()
    if normalized_format not in {"html", "latex"}:
        raise ValueError(
            f"Unsupported output format: {output_format}. Supported formats are: html, latex"
        )

    template_root = _get_templates_root_dir() / normalized_format
    template_path = template_root / template_relative_path
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path.as_posix()}")

    if normalized_format == "html":
        environment = Environment(
            loader=FileSystemLoader(str(template_root)),
            autoescape=select_autoescape(["html", "xml"]),
        )
    else:
        environment = Environment(
            loader=FileSystemLoader(str(template_root)),
            autoescape=False,
        )
        environment.filters["latex_escape"] = latex_escape

    return environment.get_template(template_relative_path)


def _safe_get(getter, fallback):
    try:
        value = getter()
    except Exception:
        return fallback
    return fallback if value is None else value


def _safe_json(value) -> str:
    try:
        return json.dumps(value, indent=2, sort_keys=True)
    except Exception:
        return "{}"


class DefaultLLMConnectorRenderer:
    @staticmethod
    def render_template(
        module_result: LLMConnectorResult,
        output_format: str = "html",
    ) -> str:
        class_ref = getattr(module_result, "class_ref", None)
        short_name = class_ref.get_short_name() if class_ref else ""
        long_name = class_ref.get_long_name() if class_ref else ""
        description = class_ref.get_description() if class_ref else ""
        class_ref_name = (
            f"{class_ref.__module__}.{class_ref.__name__}" if class_ref else ""
        )

        message = getattr(module_result, "response", None)
        response_text = _safe_get(lambda: message.get_content(), "") if message else ""

        message_created_at_str = ""
        if message:
            message_created_at_str = _safe_get(
                lambda: message.get_timestamp().isoformat(), ""
            )

        message_metadata = (
            _safe_get(lambda: message.get_metadata(), {}) if message else {}
        )
        message_metadata_text = _safe_json(message_metadata)

        metadata = getattr(module_result, "metadata", {}) or {}
        metadata_text = _safe_json(metadata)

        llmconnector_template_data = {
            # --------------------------------------------------
            # Base Viz — Main Result
            # --------------------------------------------------
            "main_result_label": "LLM response summary",
            "main_result_value": long_name,
            "main_result_note": "See module and message info in pills below",
            # --------------------------------------------------
            # Base Viz — About this module
            # --------------------------------------------------
            "description": description or "",
            # --------------------------------------------------
            # Base Viz — Module metadata
            # --------------------------------------------------
            "metadata_text": metadata_text,
            "metadata_lang": "json",
            "metadata_open": False,
            # --------------------------------------------------
            # LLM Connector — Module info pills
            # --------------------------------------------------
            "module_name": short_name,
            "module_class_ref": class_ref_name,
            # --------------------------------------------------
            # LLM Connector — Message info pills
            # --------------------------------------------------
            "message_role": _safe_get(lambda: message.get_role(), "unknown")
            if message
            else "unknown",
            "message_model": _safe_get(lambda: message.get_model(), "unknown")
            if message
            else "unknown",
            "message_tokens": _safe_get(lambda: message.get_tokens(), "unknown")
            if message
            else "unknown",
            "message_created_at": message_created_at_str,
            "message_id": _safe_get(lambda: message.get_id(), "unknown")
            if message
            else "unknown",
            # --------------------------------------------------
            # LLM Connector — Response section
            # --------------------------------------------------
            "response_text": response_text,
            "response_open": False,
            # --------------------------------------------------
            # LLM Connector — Message metadata
            # --------------------------------------------------
            "message_metadata_text": message_metadata_text,
            "message_metadata_lang": "json",
            "message_metadata_open": False,
        }

        template = _get_template("llm_connector.jinja2", output_format)

        return template.render(**llmconnector_template_data)


class DefaultEvaluatorRenderer:
    @staticmethod
    def render_template(
        module_result: EvaluatorResult,
        output_format: str = "html",
    ) -> str:
        class_ref = getattr(module_result, "class_ref", None)
        description = class_ref.get_description() if class_ref else ""

        fitness_value = _safe_get(lambda: module_result.fitness, None)
        fitness_display = "—" if fitness_value is None else f"{fitness_value:.4f}"

        metadata = getattr(module_result, "metadata", {}) or {}
        metadata_text = _safe_json(metadata)

        evaluator_template_data = {
            # --------------------------------------------------
            # Base Viz — Main Result
            # --------------------------------------------------
            "main_result_label": "Fitness",
            "main_result_value": fitness_display,
            "main_result_note": "Evaluator defines wether higher or lower is better.",
            # --------------------------------------------------
            # Base Viz — About this module
            # --------------------------------------------------
            "description": description or "",
            # --------------------------------------------------
            # Base Viz — Module metadata
            # --------------------------------------------------
            "metadata_text": metadata_text,
            "metadata_lang": "json",
            "metadata_open": False,
        }

        template = _get_template("shared/base_viz.jinja2", output_format)

        return template.render(**evaluator_template_data)


class DefaultAnalysisRenderer:
    @staticmethod
    def render_template(
        module_result: AnalysisResult,
        output_format: str = "html",
    ) -> str:
        class_ref = getattr(module_result, "class_ref", None)
        short_name = (
            _safe_get(lambda: class_ref.get_short_name(), "") if class_ref else ""
        )
        long_name = (
            _safe_get(lambda: class_ref.get_long_name(), "") if class_ref else ""
        )
        description = (
            _safe_get(lambda: class_ref.get_description(), "") if class_ref else ""
        )
        class_ref_name = (
            _safe_get(lambda: f"{class_ref.__module__}.{class_ref.__name__}", "")
            if class_ref
            else ""
        )

        metadata = getattr(module_result, "metadata", {}) or {}
        metadata_text = _safe_json(metadata)

        data = {
            "main_result_label": "Analysis result summary",
            "main_result_value": f"{long_name or short_name or 'Analysis'}",
            "main_result_note": "This analysis module does not provide a numeric summary.",
            "description": description or "",
            "metadata_text": metadata_text,
            "metadata_lang": "json",
            "metadata_open": False,
            "short_name": short_name,
            "class_ref": class_ref_name,
        }

        template = _get_template("shared/base_viz.jinja2", output_format)

        return template.render(**data)


class DefaultStatRenderer:
    @staticmethod
    def render_template(module_result: StatResult, output_format: str = "html") -> str:
        class_ref = getattr(module_result, "class_ref", None)
        short_name = (
            _safe_get(lambda: class_ref.get_short_name(), "") if class_ref else ""
        )
        long_name = (
            _safe_get(lambda: class_ref.get_long_name(), "") if class_ref else ""
        )
        description = (
            _safe_get(lambda: class_ref.get_description(), "") if class_ref else ""
        )
        class_ref_name = (
            _safe_get(lambda: f"{class_ref.__module__}.{class_ref.__name__}", "")
            if class_ref
            else ""
        )

        metadata = getattr(module_result, "metadata", {}) or {}
        metadata_text = _safe_json(metadata)

        data = {
            "main_result_label": "Stat result summary",
            "main_result_value": f"{long_name or short_name or 'Stat'}",
            "main_result_note": "This stat module does not provide a numeric summary.",
            "description": description or "",
            "metadata_text": metadata_text,
            "metadata_lang": "json",
            "metadata_open": False,
            "short_name": short_name,
            "class_ref": class_ref_name,
        }

        template = _get_template("shared/base_viz.jinja2", output_format)

        return template.render(**data)


class DefaultStoppingConditionRenderer:
    @staticmethod
    def render_template(
        module_result: StoppingConditionResult,
        output_format: str = "html",
    ) -> str:
        class_ref = getattr(module_result, "class_ref", None)
        short_name = (
            _safe_get(lambda: class_ref.get_short_name(), "") if class_ref else ""
        )
        description = (
            _safe_get(lambda: class_ref.get_description(), "") if class_ref else ""
        )
        class_ref_name = (
            _safe_get(lambda: f"{class_ref.__module__}.{class_ref.__name__}", "")
            if class_ref
            else ""
        )

        metadata = getattr(module_result, "metadata", {}) or {}
        metadata_text = _safe_json(metadata)

        is_satisfied = _safe_get(
            lambda: getattr(module_result, "is_satisfied", False), False
        )
        main_result_value = "Satisfied" if is_satisfied else "Not satisfied"

        data = {
            "main_result_label": "Stopping condition",
            "main_result_value": main_result_value,
            "main_result_note": "More info in module and metadata pills below.",
            "description": description or "",
            "metadata_text": metadata_text,
            "metadata_lang": "json",
            "metadata_open": False,
            "short_name": short_name,
            "class_ref": class_ref_name,
        }

        template = _get_template("shared/base_viz.jinja2", output_format)

        return template.render(**data)


class DefaultTestRenderer:
    @staticmethod
    def render_template(module_result: TestResult, output_format: str = "html") -> str:
        class_ref = getattr(module_result, "class_ref", None)
        short_name = (
            _safe_get(lambda: class_ref.get_short_name(), "") if class_ref else ""
        )
        description = (
            _safe_get(lambda: class_ref.get_description(), "") if class_ref else ""
        )
        class_ref_name = (
            _safe_get(lambda: f"{class_ref.__module__}.{class_ref.__name__}", "")
            if class_ref
            else ""
        )

        metadata = getattr(module_result, "metadata", {}) or {}
        metadata_text = _safe_json(metadata)

        passed = _safe_get(lambda: getattr(module_result, "passed", False), False)
        main_result_value = "Passed" if passed else "Failed"

        data = {
            "main_result_label": "Test result",
            "main_result_value": main_result_value,
            "main_result_note": "More info in module and metadata pills below.",
            "description": description or "",
            "metadata_text": metadata_text,
            "metadata_lang": "json",
            "metadata_open": False,
            "short_name": short_name,
            "class_ref": class_ref_name,
        }

        template = _get_template("shared/base_viz.jinja2", output_format)

        return template.render(**data)


class DefaultSolutionRenderer:
    @staticmethod
    def render_template(
        module_result: SolutionResult,
        output_format: str = "html",
    ) -> str:
        class_ref = getattr(module_result, "class_ref", None)
        short_name = (
            _safe_get(lambda: class_ref.get_short_name(), "") if class_ref else ""
        )
        long_name = (
            _safe_get(lambda: class_ref.get_long_name(), "") if class_ref else ""
        )
        description = (
            _safe_get(lambda: class_ref.get_description(), "") if class_ref else ""
        )

        metadata = getattr(module_result, "metadata", {}) or {}
        metadata_text = _safe_json(metadata)

        evaluator_input_serialized = _safe_get(
            lambda: getattr(module_result, "evaluator_input_serialized", None), None
        )
        if evaluator_input_serialized is None:
            evaluator_input = _safe_get(
                lambda: getattr(module_result, "evaluator_input", {}), {}
            )
            evaluator_input_serialized = _safe_json(evaluator_input)

        data = {
            "main_result_label": "Solution summary",
            "main_result_value": f"{long_name or short_name or 'Solution'}",
            "main_result_note": "See evaluator input and module metadata below.",
            "description": description or "",
            "metadata_text": metadata_text,
            "metadata_lang": "json",
            "metadata_open": False,
            "evaluator_input_serialized": evaluator_input_serialized,
        }

        template = _get_template("solution.jinja2", output_format)
        return template.render(**data)
