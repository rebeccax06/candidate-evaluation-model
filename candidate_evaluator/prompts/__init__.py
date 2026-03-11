"""Prompt templates for candidate evaluation"""

from candidate_evaluator.prompts.evaluation_prompts import (
    SYSTEM_PROMPT,
    EVALUATION_PROMPT_TEMPLATE,
    COMPARISON_PROMPT_TEMPLATE,
    HOLISTIC_RANKING_PROMPT_TEMPLATE,
    HOLISTIC_SELECTION_PROMPT_TEMPLATE,
    get_evaluation_prompt,
    get_comparison_prompt,
    get_holistic_ranking_prompt,
    get_holistic_selection_prompt,
)

__all__ = [
    "SYSTEM_PROMPT",
    "EVALUATION_PROMPT_TEMPLATE",
    "COMPARISON_PROMPT_TEMPLATE",
    "HOLISTIC_RANKING_PROMPT_TEMPLATE",
    "HOLISTIC_SELECTION_PROMPT_TEMPLATE",
    "get_evaluation_prompt",
    "get_comparison_prompt",
    "get_holistic_ranking_prompt",
    "get_holistic_selection_prompt",
]
