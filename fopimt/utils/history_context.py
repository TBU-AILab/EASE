"""
Helpers shared by the modules that put the history of generated algorithms into the LLM context
(eval.papercontextsummarizer: source-code block, anal.historysummary: LLM summary of the history).
"""

import json

# Field types of a structured-summary schema
#   "id"        algorithm identifier (string, must match the input record)
#   "iteration" iteration number (integer, must match the input record)
#   "string"    string
#   "integer"   integer
#   "list"      list of strings
#   {...}       nested object
# A top-level entry containing an "id" field is a list with one record per input algorithm,
# other top-level entries are objects.
FIELD_PLACEHOLDERS = {
    "id": "<string>",
    "iteration": 0,
    "string": "<string>",
    "integer": 0,
    "list": ["<string>"],
}


def fill_template(template: str, **values) -> str:
    """
    Replaces {name} placeholders by values. Unlike str.format, other braces in the template (e.g. a JSON example
    written by the user in frontEASE) are kept as they are.
    """
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def escape_delimiters(text: str, tags: list[str]) -> str:
    """Escapes prompt delimiters occurring inside a payload, so that the payload cannot alter the prompt structure."""
    for tag in tags:
        text = text.replace(f"</{tag}", f"&lt;/{tag}").replace(f"<{tag}", f"&lt;{tag}")
    return text


def parse_tags(tags: str) -> list[str]:
    """Comma separated list of delimiter tags -> list (longer tags first, so that prefixes do not break them)."""
    return sorted(
        (t.strip() for t in tags.split(",") if t.strip()), key=len, reverse=True
    )


def format_score(value: float, score_format: str) -> str:
    return format(value, score_format)


def format_algorithm_id(index: int, id_format: str) -> str:
    return id_format.format(index=index)


def parse_schema(schema_text: str) -> dict:
    """Parses and checks a structured-summary schema (see FIELD_PLACEHOLDERS for the field types)."""
    schema = json.loads(schema_text)
    if not isinstance(schema, dict) or not schema:
        raise ValueError("Structured summary schema must be a non-empty JSON object.")

    def check(fields: dict, where: str):
        for key, kind in fields.items():
            if isinstance(kind, dict):
                check(kind, f"{where}.{key}")
            elif kind not in FIELD_PLACEHOLDERS:
                raise ValueError(
                    f"Unknown field type '{kind}' of {where}.{key} in structured summary schema."
                )

    for key, fields in schema.items():
        if not isinstance(fields, dict):
            raise ValueError(f"Schema entry '{key}' must be an object of fields.")
        check(fields, key)
    if not any(_is_record_list(fields) for fields in schema.values()):
        raise ValueError(
            'Structured summary schema needs one entry with an "id" field (per-algorithm records).'
        )
    return schema


def _is_record_list(fields: dict) -> bool:
    return "id" in fields.values()


def schema_example(schema: dict) -> str:
    """Renders the schema as the JSON example shown to the summarizer."""

    def render(fields: dict) -> dict:
        return {
            k: (render(v) if isinstance(v, dict) else FIELD_PLACEHOLDERS[v])
            for k, v in fields.items()
        }

    return json.dumps(
        {
            key: ([render(fields)] if _is_record_list(fields) else render(fields))
            for key, fields in schema.items()
        },
        indent=2,
    )


def _check_fields(obj, fields: dict, where: str) -> str:
    if not isinstance(obj, dict):
        return f"{where} must be a JSON object."
    if set(obj) != set(fields):
        missing = sorted(set(fields) - set(obj))
        unknown = sorted(set(obj) - set(fields))
        return f"{where} has wrong keys (missing: {missing}, unknown: {unknown})."
    for key, kind in fields.items():
        value = obj[key]
        if isinstance(kind, dict):
            err = _check_fields(value, kind, f'{where}["{key}"]')
            if err:
                return err
        elif kind in ("string", "id") and not isinstance(value, str):
            return f'{where}["{key}"] must be a string.'
        elif kind in ("integer", "iteration") and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            return f'{where}["{key}"] must be an integer.'
        elif kind == "list" and not (
            isinstance(value, list) and all(isinstance(v, str) for v in value)
        ):
            return f'{where}["{key}"] must be a list of strings.'
    return ""


def validate_structured_summary(
    content: str, schema: dict, records: list[tuple[str, int]]
) -> tuple[dict | None, str]:
    """
    Deterministic validation of a structured summary against the schema.
    :param content: Raw summarizer response.
    :param schema: Parsed schema (parse_schema).
    :param records: Expected (algorithm_id, iteration) pairs in chronological order.
    :return: (parsed summary, "") if valid, otherwise (None, diagnostic).
    """
    text = content.strip()
    if text.startswith("```"):  # permitted transport wrapper
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rstrip()
        if text.endswith("```"):
            text = text[:-3]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return (
            None,
            f"The response is not exactly one valid JSON object ({e.msg} at position {e.pos}).",
        )
    if not isinstance(data, dict):
        return None, "The response must be one JSON object."
    if set(data) != set(schema):
        return (
            None,
            f"The top-level object must contain exactly the keys {list(schema)}.",
        )

    for key, fields in schema.items():
        if not _is_record_list(fields):
            err = _check_fields(data[key], fields, f'"{key}"')
            if err:
                return None, err
            continue
        items = data[key]
        if not isinstance(items, list):
            return None, f'"{key}" must be a list.'
        if len(items) != len(records):
            return (
                None,
                f'"{key}" must contain exactly {len(records)} items, one per input record.',
            )
        id_key = next(k for k, v in fields.items() if v == "id")
        it_key = next((k for k, v in fields.items() if v == "iteration"), None)
        for i, (item, (alg_id, iteration)) in enumerate(zip(items, records)):
            err = _check_fields(item, fields, f'"{key}"[{i}]')
            if err:
                return None, err
            if item[id_key] != alg_id or (
                it_key is not None and item[it_key] != iteration
            ):
                return None, (
                    f'"{key}"[{i}] must describe algorithm "{alg_id}" (iteration {iteration}); '
                    "one item per input record, in chronological order."
                )
    return data, ""


def attach_scores(
    data: dict, schema: dict, scores: list[str], score_field: str
) -> dict:
    """Inserts the scores into the per-algorithm records, right after the iteration (or id) field."""
    for key, fields in schema.items():
        if not _is_record_list(fields):
            continue
        anchor = next((k for k, v in fields.items() if v == "iteration"), None) or next(
            k for k, v in fields.items() if v == "id"
        )
        for item, score in zip(data[key], scores):
            ordered = {}
            for k, v in item.items():
                ordered[k] = v
                if k == anchor:
                    ordered[score_field] = score
            item.clear()
            item.update(ordered)
    return data
