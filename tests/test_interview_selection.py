"""Tests for the two-phase interview-selection flow."""

import json
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from candidate_evaluator.core.models import (
    CandidateProfile,
    HolisticEvaluationResult,
    InnovationPotential,
    ProgramFit,
)
from candidate_evaluator.prompts.evaluation_prompts import (
    get_holistic_ranking_prompt,
    get_holistic_selection_prompt,
)
from candidate_evaluator.utils.config import Config, APIConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_holistic_result(
    candidate_id: str,
    overall_score: float = 7.0,
    interview_decision: bool = True,
    innovation_level: str = "high",
    program_fit_level: str = "strong",
    red_flags: list = None,
) -> HolisticEvaluationResult:
    """Create a minimal HolisticEvaluationResult for testing."""
    return HolisticEvaluationResult(
        candidate=CandidateProfile(
            candidate_id=candidate_id,
            name=f"Test {candidate_id}",
            materials=[f"{candidate_id}.pdf"],
            evaluation_date=datetime(2026, 1, 1),
        ),
        overall_assessment=f"Assessment for {candidate_id}.",
        innovation_potential=InnovationPotential(
            level=innovation_level,
            confidence="high",
            reasoning="Strong evidence.",
        ),
        program_fit=ProgramFit(
            level=program_fit_level,
            confidence="high",
            detailed_analysis="Good fit.",
        ),
        overall_score=overall_score,
        score_justification="Justified.",
        recommendation="Strong fit",
        interview_decision=interview_decision,
        interview_decision_reasoning="Score above threshold.",
        red_flags=red_flags or [],
    )


# ---------------------------------------------------------------------------
# Prompt builder tests
# ---------------------------------------------------------------------------

class TestHolisticRankingPrompt:
    def test_contains_all_candidate_ids(self):
        evals = [_make_holistic_result(f"cand_{i}") for i in range(4)]
        prompt = get_holistic_ranking_prompt(evals)
        for i in range(4):
            assert f"cand_{i}" in prompt

    def test_n_candidates_in_prompt(self):
        evals = [_make_holistic_result(f"c{i}") for i in range(5)]
        prompt = get_holistic_ranking_prompt(evals)
        assert "5" in prompt

    def test_prompt_requests_json_output(self):
        prompt = get_holistic_ranking_prompt([_make_holistic_result("c1")])
        assert "ranking" in prompt
        assert "json" in prompt.lower()

    def test_custom_template_is_used(self):
        custom = "CUSTOM {n_candidates} {candidates_data}"
        prompt = get_holistic_ranking_prompt([_make_holistic_result("c1")], template=custom)
        assert prompt.startswith("CUSTOM 1")

    def test_includes_program_fit_detail(self):
        """Rich prompt should include program-fit reasoning, not just a level label."""
        ev = _make_holistic_result("c1", program_fit_level="strong")
        prompt = get_holistic_ranking_prompt([ev])
        assert "Program Fit" in prompt
        assert "STRONG" in prompt

    def test_includes_innovation_potential(self):
        ev = _make_holistic_result("c1", innovation_level="high")
        prompt = get_holistic_ranking_prompt([ev])
        assert "Innovation Potential" in prompt
        assert "HIGH" in prompt


class TestHolisticSelectionPrompt:
    def test_contains_n_interviews(self):
        evals = [_make_holistic_result(f"c{i}") for i in range(4)]
        prompt = get_holistic_selection_prompt(evals, n_interviews=2)
        assert "2" in prompt

    def test_prompt_requests_selected_key(self):
        prompt = get_holistic_selection_prompt([_make_holistic_result("c1")], n_interviews=1)
        assert "selected" in prompt

    def test_pool_size_shown(self):
        evals = [_make_holistic_result(f"c{i}") for i in range(6)]
        prompt = get_holistic_selection_prompt(evals, n_interviews=3)
        assert "6" in prompt

    def test_custom_template_is_used(self):
        custom = "SEL {n_pool} {n_interviews} {candidates_data}"
        prompt = get_holistic_selection_prompt(
            [_make_holistic_result("c1")], n_interviews=1, template=custom
        )
        assert prompt.startswith("SEL 1 1")

    def test_includes_notable_qualities(self):
        """Selection prompt should include notable qualities for richer comparison."""
        from candidate_evaluator.core.models import NotableQuality
        ev = _make_holistic_result("c1")
        ev.notable_qualities = [
            NotableQuality(
                quality="Systems thinking",
                confidence="high",
                evidence="Demonstrated across projects.",
                significance="Directly relevant to program goals.",
            )
        ]
        prompt = get_holistic_selection_prompt([ev], n_interviews=1)
        assert "Systems thinking" in prompt


# ---------------------------------------------------------------------------
# CandidateEvaluator.select_interviews_holistic (mocked API)
# ---------------------------------------------------------------------------

def _make_evaluator():
    """Return a CandidateEvaluator with a mocked Anthropic client."""
    config = Config(api=APIConfig(anthropic_api_key="test-key"))
    with patch("candidate_evaluator.core.evaluator.Anthropic"):
        from candidate_evaluator.core.evaluator import CandidateEvaluator
        evaluator = CandidateEvaluator(config)
    return evaluator


class TestSelectInterviewsHolistic:
    def _api_response(self, payload: dict) -> MagicMock:
        """Build a fake Anthropic API message response containing JSON."""
        msg = MagicMock()
        msg.content = [MagicMock()]
        msg.content[0].type = "text"
        msg.content[0].text = json.dumps(payload)
        return msg

    def test_raises_if_n_interviews_zero(self):
        evaluator = _make_evaluator()
        evals = [_make_holistic_result(f"c{i}") for i in range(3)]
        with pytest.raises(ValueError, match="n_interviews must be at least 1"):
            evaluator.select_interviews_holistic(evals, n_interviews=0)

    def test_raises_if_n_exceeds_pool(self):
        evaluator = _make_evaluator()
        evals = [_make_holistic_result(f"c{i}") for i in range(2)]
        with pytest.raises(ValueError, match="exceeds the number of candidates"):
            evaluator.select_interviews_holistic(evals, n_interviews=5)

    def test_phase1_skipped_when_pool_covers_all(self):
        """With only 3 candidates and n=2 (pool=max(4,4)=4 → capped at 3), Phase 1 is skipped."""
        evaluator = _make_evaluator()
        evals = [
            _make_holistic_result("c1", overall_score=9.0),
            _make_holistic_result("c2", overall_score=7.0),
            _make_holistic_result("c3", overall_score=5.0),
        ]
        # pool_size = min(3, max(2+2, int(2*2.0))) = min(3, 4) = 3 = total → skip Phase 1
        phase2_response = {
            "selected": ["c1", "c2"],
            "selection_rationale": "Top two scores.",
            "candidate_notes": {"c1": "Best", "c2": "Second"},
        }
        evaluator.client.messages.create.return_value = self._api_response(phase2_response)

        result = evaluator.select_interviews_holistic(evals, n_interviews=2)

        # Only one API call (Phase 2 only)
        assert evaluator.client.messages.create.call_count == 1
        assert result.selected_candidate_ids == ["c1", "c2"]
        assert "Score-based" in result.metadata["phase1_rationale"]

    def test_two_phase_flow_with_large_pool(self):
        """With many candidates, both phases should run (2 API calls)."""
        evaluator = _make_evaluator()
        evals = [_make_holistic_result(f"c{i}", overall_score=float(10 - i)) for i in range(10)]

        # Phase 1 returns a ranking
        phase1_response = {
            "ranking": [f"c{i}" for i in range(10)],
            "ranking_rationale": "Descending by score.",
        }
        # Phase 2 selects top 2
        phase2_response = {
            "selected": ["c0", "c1"],
            "selection_rationale": "Highest scores.",
            "candidate_notes": {},
        }
        evaluator.client.messages.create.side_effect = [
            self._api_response(phase1_response),
            self._api_response(phase2_response),
        ]

        result = evaluator.select_interviews_holistic(evals, n_interviews=2, pool_multiplier=2.0)

        assert evaluator.client.messages.create.call_count == 2
        assert result.selected_candidate_ids == ["c0", "c1"]
        assert result.n_interviews_requested == 2
        # Pool should be max(2+2, int(2*2))=4 → first 4 candidates
        assert len(result.pool_candidate_ids) == 4
        assert result.all_candidate_ids_ranked == [f"c{i}" for i in range(10)]

    def test_missing_ids_from_phase1_are_appended(self):
        """If the model omits some IDs in Phase 1, the code should add them."""
        evaluator = _make_evaluator()
        evals = [_make_holistic_result(f"c{i}") for i in range(6)]

        # Phase 1 returns only 4 of 6 IDs
        phase1_response = {
            "ranking": ["c0", "c1", "c2", "c3"],
            "ranking_rationale": "Only four ranked.",
        }
        phase2_response = {
            "selected": ["c0", "c1"],
            "selection_rationale": "Top two.",
            "candidate_notes": {},
        }
        evaluator.client.messages.create.side_effect = [
            self._api_response(phase1_response),
            self._api_response(phase2_response),
        ]

        result = evaluator.select_interviews_holistic(evals, n_interviews=2, pool_multiplier=2.0)

        # All 6 IDs must appear in the full ranking
        assert len(result.all_candidate_ids_ranked) == 6

    def test_fallback_to_top_n_when_model_returns_no_selected(self):
        """If Phase 2 returns an empty 'selected', fall back to top-N of the pool."""
        evaluator = _make_evaluator()
        evals = [_make_holistic_result(f"c{i}", overall_score=float(10 - i)) for i in range(5)]

        phase2_response = {
            "selected": [],
            "selection_rationale": "",
            "candidate_notes": {},
        }
        evaluator.client.messages.create.return_value = self._api_response(phase2_response)

        result = evaluator.select_interviews_holistic(evals, n_interviews=2)

        assert len(result.selected_candidate_ids) == 2
