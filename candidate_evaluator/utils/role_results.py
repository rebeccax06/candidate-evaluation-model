"""Helpers for role-specific evaluation results, grouping, and ranking."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

ROLE_ORDER = ["clinician", "engineer", "phd"]

ROLE_LABELS = {
    "clinician": "Clinician",
    "engineer": "Engineer / Tech",
    "phd": "PhD / Research",
}

ROLE_DIMENSION_KEYS = {
    "clinician": ("challenge_complexity", "need_investigation_stage", "systems_thinking"),
    "engineer": ("build_stage", "ownership_clarity", "user_grounding"),
    "phd": ("research_evidence_level", "publication_strength", "research_ownership"),
}


def _assessment_as_dict(assessment: Any) -> Optional[Dict[str, Any]]:
    if assessment is None:
        return None
    if hasattr(assessment, "model_dump"):
        return assessment.model_dump()
    if isinstance(assessment, dict):
        return assessment
    return None


def eval_row_role(row: dict) -> Optional[str]:
    """Get canonical role from a DB evaluation row."""
    role = row.get("role")
    if role:
        return str(role).lower()
    result = row.get("result") or {}
    role = result.get("role")
    return str(role).lower() if role else None


def eval_row_role_score(row: dict) -> Optional[float]:
    """Get role-specific score from a DB evaluation row."""
    if row.get("role_specific_score") is not None:
        try:
            return float(row["role_specific_score"])
        except (TypeError, ValueError):
            pass
    rsa = _assessment_as_dict((row.get("result") or {}).get("role_specific_assessment"))
    if rsa and rsa.get("score") is not None:
        try:
            return float(rsa["score"])
        except (TypeError, ValueError):
            return None
    return None


def result_object_role(result: Any) -> Optional[str]:
    """Get role from an EvaluationResult or HolisticEvaluationResult."""
    role = getattr(result, "role", None)
    return str(role).lower() if role else None


def result_object_role_score(result: Any) -> Optional[float]:
    """Get role-specific score from a result object."""
    rsa = _assessment_as_dict(getattr(result, "role_specific_assessment", None))
    if rsa and rsa.get("score") is not None:
        try:
            return float(rsa["score"])
        except (TypeError, ValueError):
            return None
    return None


def role_dimension_summary(assessment: Any, role: Optional[str] = None) -> str:
    """Compact summary of role-specific dimensions for table display."""
    data = _assessment_as_dict(assessment)
    if not data:
        return "-"
    role = (role or data.get("role") or "").lower()
    parts = []
    for key in ROLE_DIMENSION_KEYS.get(role, ()):
        val = data.get(key)
        if val:
            parts.append(f"{key.replace('_', ' ').title()}: {str(val).replace('_', ' ')}")
    return "; ".join(parts) if parts else "-"


def sort_eval_rows_by_role_score(rows: List[dict]) -> List[dict]:
    """Sort evaluation DB rows by role-specific score (desc), then overall score."""
    return sorted(
        rows,
        key=lambda e: (
            eval_row_role_score(e) if eval_row_role_score(e) is not None else -1,
            e.get("overall_score") or 0,
        ),
        reverse=True,
    )


def sort_results_by_role_score(results: List[Any]) -> List[Any]:
    """Sort result objects by role-specific score (desc), then overall score."""
    return sorted(
        results,
        key=lambda r: (
            result_object_role_score(r) if result_object_role_score(r) is not None else -1,
            getattr(r, "overall_score", 0) or 0,
        ),
        reverse=True,
    )


def group_eval_rows_by_role(rows: List[dict]) -> Dict[str, List[dict]]:
    """Group DB evaluation rows by role (role-specific evaluations only)."""
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        role = eval_row_role(row)
        if role and role in ROLE_LABELS:
            grouped[role].append(row)
    return dict(grouped)


def group_results_by_role(results: List[Any]) -> Dict[str, List[Any]]:
    """Group result objects by role (role-specific evaluations only)."""
    grouped: Dict[str, List[Any]] = defaultdict(list)
    for result in results:
        role = result_object_role(result)
        if role and role in ROLE_LABELS:
            grouped[role].append(result)
    return dict(grouped)


def build_role_ranking_row_from_eval_row(rank: int, row: dict, n_in_group: int) -> dict:
    """Build a summary table row for role-specific ranking from a DB row."""
    result_data = row.get("result") or {}
    rsa = result_data.get("role_specific_assessment") or {}
    role = eval_row_role(row)
    role_score = eval_row_role_score(row)
    pct = 100 if n_in_group == 1 else round(100.0 * (n_in_group - rank) / (n_in_group - 1))
    eval_type = row.get("evaluation_type", "")
    return {
        "Rank": rank,
        "Percentile": f"{pct}th",
        "Candidate ID": row.get("candidate_id", ""),
        "Name": row.get("candidate_name") or "-",
        "Role Score": f"{role_score:.1f}/10" if role_score is not None else "N/A",
        "Overall Score": f"{row.get('overall_score', 0):.1f}/10" if row.get("overall_score") else "N/A",
        "Mode": eval_type.capitalize() if eval_type else "-",
        "Confidence": str(rsa.get("confidence", "-")).capitalize(),
        "Role Dimension": role_dimension_summary(rsa, role),
        "Date": str(row.get("created_at", ""))[:10],
    }


def count_role_evaluations_from_rows(rows: List[dict]) -> int:
    return sum(1 for row in rows if eval_row_role(row))


def count_role_evaluations_from_results(results: List[Any]) -> int:
    return sum(1 for result in results if result_object_role(result))


def build_role_ranking_row_from_result(rank: int, result: Any, n_in_group: int) -> dict:
    """Build a summary table row for role-specific ranking from a result object."""
    rsa = getattr(result, "role_specific_assessment", None)
    role = result_object_role(result)
    role_score = result_object_role_score(result)
    pct = 100 if n_in_group == 1 else round(100.0 * (n_in_group - rank) / (n_in_group - 1))
    data = _assessment_as_dict(rsa) or {}
    mode = "Holistic" if result.__class__.__name__ == "HolisticEvaluationResult" else "Criteria"
    return {
        "Rank": rank,
        "Percentile": f"{pct}th",
        "Candidate ID": result.candidate.candidate_id,
        "Name": result.candidate.name or "-",
        "Role Score": f"{role_score:.1f}/10" if role_score is not None else "N/A",
        "Overall Score": f"{result.overall_score:.1f}/10",
        "Mode": mode,
        "Confidence": str(data.get("confidence", "-")).capitalize(),
        "Role Dimension": role_dimension_summary(rsa, role),
        "Date": result.candidate.evaluation_date.strftime("%Y-%m-%d"),
    }
