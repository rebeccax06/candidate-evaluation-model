"""Shared per-candidate job processing.

Both background workers (the local file-based one in ``background_worker.py``
and the Supabase/Railway one in ``worker.py``) process jobs the same way:
resolve the job's evaluation mode, run one candidate through the evaluator,
and serialize the result. This module is the single home for that logic so
the two workers cannot drift.
"""

import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Concurrent Claude calls per job. Kept modest so a user's API rate limits
# absorb bursts (the Anthropic SDK retries 429/5xx twice with backoff).
DEFAULT_EVAL_CONCURRENCY = 4


class EvaluationMode(str, Enum):
    """How a job evaluates each candidate."""

    CRITERIA = "criteria"
    HOLISTIC = "holistic"
    SCREEN = "screen"

    @classmethod
    def coerce(cls, value: Any, default: "EvaluationMode" = None) -> "EvaluationMode":
        """Parse a stored mode string, falling back to ``default`` (criteria)."""
        if default is None:
            default = cls.CRITERIA
        try:
            return cls(value)
        except ValueError:
            return default


def result_filename(mode: EvaluationMode, candidate_id: str) -> str:
    """Output filename for one candidate's result (local file-based worker)."""
    suffix = {
        EvaluationMode.SCREEN: "_screening",
        EvaluationMode.HOLISTIC: "_holistic_evaluation",
        EvaluationMode.CRITERIA: "_evaluation",
    }[mode]
    return f"{candidate_id}{suffix}.json"


def evaluate_candidate_for_mode(
    evaluator,
    mode: EvaluationMode,
    candidate_id: str,
    material_paths: List[str],
    *,
    role: Optional[str] = None,
    description: Optional[str] = None,
    candidate_name: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Run one candidate through the evaluator in the given mode.

    Returns ``(result, result_dict)``: the pydantic result object and a
    JSON-serializable dict of it.
    """
    if mode is EvaluationMode.SCREEN:
        result = evaluator.screen_candidate(
            candidate_id=candidate_id,
            material_paths=material_paths,
            description=description or "",
            candidate_name=candidate_name,
        )
    elif mode is EvaluationMode.HOLISTIC:
        result = evaluator.evaluate_candidate_holistic(
            candidate_id=candidate_id,
            material_paths=material_paths,
            candidate_name=candidate_name,
            role=role,
        )
    else:
        result = evaluator.evaluate_candidate(
            candidate_id=candidate_id,
            material_paths=material_paths,
            candidate_name=candidate_name,
            role=role,
        )
    return result, serialize_result(result)


def serialize_result(result) -> Dict[str, Any]:
    """``model_dump()`` a result into a JSON-serializable dict.

    Two fields need help: ``candidate.evaluation_date`` is a datetime, and
    criteria results carry ``EvaluationCriterion`` enum members in scores.
    """
    data = result.model_dump()
    candidate = data.get("candidate")
    if candidate and candidate.get("evaluation_date") is not None:
        candidate["evaluation_date"] = str(candidate["evaluation_date"])
    for score in data.get("scores") or []:
        criterion = score.get("criterion")
        if criterion is not None and not isinstance(criterion, str):
            score["criterion"] = getattr(criterion, "value", str(criterion))
    return data


def eval_concurrency() -> int:
    """Concurrent Claude calls per job, tunable via EVAL_CONCURRENCY (min 1)."""
    try:
        return max(1, int(os.environ.get("EVAL_CONCURRENCY", DEFAULT_EVAL_CONCURRENCY)))
    except ValueError:
        return DEFAULT_EVAL_CONCURRENCY


def evaluate_chunk_concurrently(
    evaluator,
    mode: EvaluationMode,
    chunk: List[Tuple[str, List[str]]],
    *,
    role: Optional[str] = None,
    description: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> List[Tuple[str, Any, Optional[Dict[str, Any]], Optional[str]]]:
    """Evaluate a chunk of ``(candidate_id, material_paths)`` concurrently.

    Only the Claude API evaluation runs in threads — callers keep all their
    storage/DB/file work on the main thread (the Supabase client is not
    guaranteed thread-safe). Concurrency does not change any prompt, model,
    or parameter, so per-candidate output is identical to a serial run; only
    completion order differs.

    Returns ``(candidate_id, result, result_dict, error_traceback)`` tuples in
    completion order; ``error_traceback`` is None on success, and the
    formatted traceback string on failure (with result/result_dict None).
    """
    max_workers = max_workers or eval_concurrency()
    outcomes = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                evaluate_candidate_for_mode,
                evaluator,
                mode,
                candidate_id,
                material_paths,
                role=role,
                description=description,
            ): candidate_id
            for candidate_id, material_paths in chunk
        }
        for future in as_completed(futures):
            candidate_id = futures[future]
            try:
                result, result_dict = future.result()
                outcomes.append((candidate_id, result, result_dict, None))
            except Exception:
                outcomes.append((candidate_id, None, None, traceback.format_exc()))
    return outcomes


def summary_entry(mode: EvaluationMode, candidate_id: str, result) -> Dict[str, Any]:
    """Per-candidate success row for a job's results list (local worker)."""
    entry = {
        "candidate_id": candidate_id,
        "status": "success",
        "evaluation_mode": mode.value,
    }
    if mode is EvaluationMode.SCREEN:
        entry["matches"] = result.matches
        entry["confidence"] = result.confidence
        entry["outcome"] = result.outcome()
    elif mode is EvaluationMode.HOLISTIC:
        entry["overall_score"] = result.overall_score
        entry["recommendation"] = result.recommendation
        entry["interview_decision"] = result.interview_decision
    else:
        entry["overall_score"] = result.overall_score
        entry["recommendation"] = result.recommendation
    return entry
