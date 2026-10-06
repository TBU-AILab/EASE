import logging

import tiktoken

# One provider-independent tokenizer, so that context sizes are comparable across models.
# Provider-reported (billed) counts are stored separately, see LLMConnector._set_usage.
COMMON_ENCODING = "o200k_base"

_encoder = None


def count_common_tokens(text: str | None) -> int:
    """
    Counts tokens of the text with the common tokenizer (COMMON_ENCODING).
    Falls back to ~4 characters per token if the tokenizer is not available.
    """
    global _encoder
    if not text:
        return 0
    if _encoder is None:
        try:
            _encoder = tiktoken.get_encoding(COMMON_ENCODING)
        except Exception as e:
            logging.warning(
                f"token_utils: common tokenizer not available, using estimate: {repr(e)}"
            )
            _encoder = False
    if _encoder is False:
        return len(text) // 4
    return len(_encoder.encode(text, disallowed_special=()))


def llm_call_record(
    role: str,
    module: str,
    input_texts: dict[str, str],
    response,
) -> dict:
    """
    Builds a normalized record of one LLMConnector.send() call for usage accounting.
    :param role: Purpose of the call, e.g. 'generator' or 'summarizer'.
    :param module: Short name of the LLMConnector.
    :param input_texts: Parts of the sent input by kind (e.g. {'init': ..., 'new_message': ...}).
    :param response: Response Message of the call.
    :return: dict with billed usage (provider-reported) and common-tokenizer counts.
    """
    breakdown = {k: count_common_tokens(v) for k, v in input_texts.items()}
    input_common = sum(breakdown.values())
    output_common = count_common_tokens(response.get_content())

    usage = response.get_usage()
    if usage is None:
        # Connector does not report usage: fall back to the common tokenizer, marked as estimated
        usage = {
            "provider": None,
            "model": None,
            "input_tokens": input_common,
            "output_tokens": output_common,
            "cached_input_tokens": 0,
            "cache_write_input_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": input_common + output_common,
            "duration_s": None,
            "calls": 1,
            "estimated": True,
        }

    return {
        "role": role,
        "module": module,
        **usage,
        "input_tokens_common": input_common,
        "output_tokens_common": output_common,
        "input_breakdown_common": breakdown,
    }
