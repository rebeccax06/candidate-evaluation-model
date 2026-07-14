"""Helpers for role-specific evaluation results, grouping, and ranking."""

from __future__ import annotations

import html as _html
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

ROLE_ORDER = ["combined"]

ROLE_LABELS = {
    "combined": "Combined (All Dimensions)",
}

# Role-prefixed dimension keys for the combined assessment.
_CLINICAL_DIMS = (
    "clinical_challenge_complexity",
    "clinical_need_investigation_stage",
    "clinical_systems_thinking",
)
_ENGINEERING_DIMS = (
    "engineering_build_stage",
    "engineering_ownership_clarity",
    "engineering_user_grounding",
)
_RESEARCH_DIMS = (
    "research_evidence_level",
    "research_publication_strength",
    "research_ownership",
)

ROLE_DIMENSION_KEYS = {
    "combined": _CLINICAL_DIMS + _ENGINEERING_DIMS + _RESEARCH_DIMS,
}

# All 9 combined dimension keys (clinical, engineering, research).
COMBINED_DIMENSION_KEYS = _CLINICAL_DIMS + _ENGINEERING_DIMS + _RESEARCH_DIMS

# Ordered worst -> best levels for each of the 9 dimensions. Index position drives
# the red (worst) -> green (best) heatmap color.
DIMENSION_LEVELS = {
    "clinical_challenge_complexity": ["not_shown", "limited", "moderate", "high", "exceptional"],
    "clinical_need_investigation_stage": [
        "witnessed_only",
        "need_identified",
        "investigated",
        "intervention_proposed",
        "tested_or_implemented",
        "measured",
        "sustained_impact",
    ],
    "clinical_systems_thinking": ["not_shown", "limited", "moderate", "strong"],
    "engineering_build_stage": [
        "idea_only",
        "designed",
        "prototype_built",
        "tested",
        "iterated",
        "deployed",
        "used_by_real_users",
        "scaled_or_sustained",
    ],
    "engineering_ownership_clarity": [
        "unclear",
        "supporting_contributor",
        "substantial_contributor",
        "primary_driver",
    ],
    "engineering_user_grounding": ["not_shown", "limited", "moderate", "strong"],
    "research_evidence_level": ["none", "limited", "moderate", "strong", "exceptional"],
    "research_publication_strength": [
        "none_or_not_shown",
        "limited",
        "moderate",
        "strong",
        "exceptional",
    ],
    "research_ownership": [
        "unclear",
        "supporting_contributor",
        "substantial_contributor",
        "primary_driver",
    ],
}

# Short, human-friendly labels shown in the heatmap (role indicated by the group).
DIMENSION_SHORT_LABELS = {
    "clinical_challenge_complexity": "Challenge Complexity",
    "clinical_need_investigation_stage": "Need Investigation Stage",
    "clinical_systems_thinking": "Systems Thinking",
    "engineering_build_stage": "Build Stage",
    "engineering_ownership_clarity": "Ownership Clarity",
    "engineering_user_grounding": "User Grounding",
    "research_evidence_level": "Evidence Level",
    "research_publication_strength": "Publication Strength",
    "research_ownership": "Research Ownership",
}

# Dimensions grouped by role for display.
DIMENSION_GROUPS = [
    ("Clinical", list(_CLINICAL_DIMS)),
    ("Engineering", list(_ENGINEERING_DIMS)),
    ("Research", list(_RESEARCH_DIMS)),
]

# Legacy (pre-combined) unprefixed keys, so evaluations already run on the
# cloud/web version with the old per-role prompts still map onto the heatmap.
_LEGACY_KEY_FALLBACK = {
    "clinical_challenge_complexity": "challenge_complexity",
    "clinical_need_investigation_stage": "need_investigation_stage",
    "clinical_systems_thinking": "systems_thinking",
    "engineering_build_stage": "build_stage",
    "engineering_ownership_clarity": "ownership_clarity",
    "engineering_user_grounding": "user_grounding",
    "research_publication_strength": "publication_strength",
}


def dimension_value(data: Dict[str, Any], key: str) -> Optional[Any]:
    """Read a dimension value, falling back to the legacy unprefixed key."""
    value = data.get(key)
    if value is None and key in _LEGACY_KEY_FALLBACK:
        value = data.get(_LEGACY_KEY_FALLBACK[key])
    return value


def dimension_level_score(dim_key: str, value: Any) -> Optional[float]:
    """Normalize a dimension level to a 0.0 (worst) -> 1.0 (best) score.

    Returns None when the value is missing or not a recognized level.
    """
    levels = DIMENSION_LEVELS.get(dim_key)
    if not levels or value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized not in levels:
        return None
    if len(levels) == 1:
        return 1.0
    return levels.index(normalized) / (len(levels) - 1)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def heatmap_color(score: Optional[float]) -> str:
    """Map a 0.0 -> 1.0 score to a red (worst) -> yellow -> green (best) hex color.

    Missing/unknown scores return a neutral gray.
    """
    if score is None:
        return "#e8e8e8"
    score = max(0.0, min(1.0, float(score)))
    # red #d73027 -> yellow #ffe066 -> green #1a9850
    if score <= 0.5:
        t = score / 0.5
        r = _lerp(215, 255, t)
        g = _lerp(48, 224, t)
        b = _lerp(39, 102, t)
    else:
        t = (score - 0.5) / 0.5
        r = _lerp(255, 26, t)
        g = _lerp(224, 152, t)
        b = _lerp(102, 80, t)
    return f"#{int(round(r)):02x}{int(round(g)):02x}{int(round(b)):02x}"


def build_role_dimension_heatmap_html(assessment: Any) -> str:
    """Build a 3x3 grid heatmap (HTML) of the 9 role dimensions.

    Each row is a role (Clinical, Engineering, Research). Each of the three cells in
    a row is an equal-size square colored red (worst) -> green (best), labeled with
    the dimension name. Hovering a square shows a tooltip with the candidate's rating.
    Dimensions without evidence render as a neutral "Not shown" gray square.
    """
    data = _assessment_as_dict(assessment) or {}
    rows_html: List[str] = []
    for group, keys in DIMENSION_GROUPS:
        cells = [
            '<div style="width:96px;min-width:96px;display:flex;align-items:center;'
            'font-weight:700;font-size:0.85rem;color:#444;">{}</div>'.format(_html.escape(group))
        ]
        for key in keys:
            raw = dimension_value(data, key)
            score = dimension_level_score(key, raw)
            color = heatmap_color(score)
            rating = str(raw).replace("_", " ").title() if raw else "Not shown"
            label = DIMENSION_SHORT_LABELS.get(key, key)
            tooltip = _html.escape(f"{label}: {rating}", quote=True)
            cells.append(
                '<div title="{tooltip}" style="flex:1;aspect-ratio:1 / 1;'
                "background:{color};border-radius:12px;display:flex;align-items:center;"
                "justify-content:center;text-align:center;padding:8px;box-sizing:border-box;"
                "color:#111111;font-size:0.8rem;font-weight:600;line-height:1.2;"
                'cursor:default;">{label}</div>'.format(
                    tooltip=tooltip, color=color, label=_html.escape(label)
                )
            )
        rows_html.append(
            '<div style="display:flex;gap:10px;align-items:stretch;">' + "".join(cells) + "</div>"
        )
    return (
        '<div style="display:flex;flex-direction:column;gap:10px;max-width:560px;">'
        + "".join(rows_html)
        + "</div>"
    )


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
    """Compact summary of role-specific dimensions for table display.

    Uses the 9 combined dimensions with legacy-key fallback so both new combined
    evaluations and historical per-role evaluations render.
    """
    data = _assessment_as_dict(assessment)
    if not data:
        return "-"
    parts = []
    for key in COMBINED_DIMENSION_KEYS:
        val = dimension_value(data, key)
        if val:
            parts.append(f"{DIMENSION_SHORT_LABELS.get(key, key)}: {str(val).replace('_', ' ')}")
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
    """Group DB evaluation rows under the single combined bucket.

    Any role-specific evaluation (new combined runs and legacy clinician/engineer/phd
    runs) is surfaced under "combined".
    """
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        if eval_row_role(row):
            grouped["combined"].append(row)
    return dict(grouped)


def group_results_by_role(results: List[Any]) -> Dict[str, List[Any]]:
    """Group result objects under the single combined bucket (see group_eval_rows_by_role)."""
    grouped: Dict[str, List[Any]] = defaultdict(list)
    for result in results:
        if result_object_role(result):
            grouped["combined"].append(result)
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
