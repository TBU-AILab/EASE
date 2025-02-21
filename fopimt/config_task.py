from enum import Enum


# OLD variant
class ConfigTask(Enum):
    """
    Enum of all possible config items for Task class.
    """
    STAT = 'stat'
    FEEDBACK_FROM_SOLUTION = 'send_feedback'
    SAVE_TO_DISK = 'save_to_disk'
    NAME = 'name'
    AUTHOR = 'author'
    MAX_CONTEXT_SIZE = 'max_context_size'
    SYSTEM_MESSAGE = "system_message"
    INIT_MESSAGE = "init_message"
    REP_MESSAGE = "rep_message"
    LLM = "llm"
    TEST = "test"
    ANALYSIS = "analysis"
    EVALUATOR = "evaluator"
    COND = "cond"
    SOLUTION = "solution"
