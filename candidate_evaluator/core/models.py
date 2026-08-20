"""Data models for candidate evaluation"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum


class EvaluationCriterion(str, Enum):
    """Evaluation criteria enum"""
    CRITICAL_THINKING = "critical_thinking"
    COACHABILITY = "coachability"
    CURIOSITY = "curiosity"
    CREATIVITY = "creativity"
    COLLABORATION = "collaboration"
    FOLLOW_THROUGH = "follow_through"
    PROBLEM_SOLVING_MOTIVATION = "problem_solving_motivation"
    EVIDENCE_BASED = "evidence_based"
    DETAIL_ORIENTATION = "detail_orientation"
    COMMUNICATION = "communication"
    EXPERTISE_ENABLER = "expertise_enabler"

    @property
    def display_name(self) -> str:
        """Get human-readable name for criterion"""
        names = {
            "critical_thinking": "Critical Thinking / Logical Analysis",
            "coachability": "Coachability – Receptive to Feedback",
            "curiosity": "Curiosity",
            "creativity": "Demonstrated Creativity in Solution Development",
            "collaboration": "Collaborates with Others Effectively; Incorporates Inputs from Others",
            "follow_through": "Demonstrated Follow Through",
            "problem_solving_motivation": "Motivation to Solve Problems",
            "evidence_based": "Understands the Value of Evidence to Challenge Assumptions",
            "detail_orientation": "Works Toward Increasing Specificity / Detail Orientation",
            "communication": "Effective Communicator",
            "expertise_enabler": "Uses Own Expertise as an Enabler Rather Than a Limitation"
        }
        return names.get(self.value, self.value)

    @property
    def description(self) -> str:
        """Get description of what this criterion measures"""
        descriptions = {
            "critical_thinking": "Ability to analyze information objectively, identify patterns, and draw logical conclusions",
            "coachability": "Openness to feedback, willingness to learn, and ability to implement suggestions",
            "curiosity": "Drive to explore, ask questions, and seek deeper understanding",
            "creativity": "Ability to develop innovative solutions and think outside conventional approaches",
            "collaboration": "Capacity to work effectively with others and integrate diverse perspectives",
            "follow_through": "Consistency in completing tasks and following projects to completion",
            "problem_solving_motivation": "Intrinsic drive to tackle challenges and find solutions",
            "evidence_based": "Reliance on data and evidence rather than assumptions in decision-making",
            "detail_orientation": "Attention to specifics and commitment to thoroughness and accuracy",
            "communication": "Clarity, precision, and effectiveness in written and verbal expression",
            "expertise_enabler": "Leveraging subject matter knowledge to enable solutions rather than limit possibilities"
        }
        return descriptions.get(self.value, "")


class Evidence(BaseModel):
    """Evidence supporting a score"""
    quote: str = Field(description="Direct quote from materials")
    source: str = Field(description="Source document (e.g., 'resume.pdf', 'cover_letter.txt')")
    context: str = Field(description="Additional context about why this evidence is relevant")


class CriterionScore(BaseModel):
    """Score for a single evaluation criterion"""
    criterion: EvaluationCriterion
    score: int = Field(ge=1, le=10, description="Score from 1-10")
    reasoning: str = Field(description="Detailed reasoning for the score")
    evidence: List[Evidence] = Field(default_factory=list, description="Supporting evidence")
    confidence: str = Field(
        default="medium",
        description="Confidence level: low, medium, high"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional notes, caveats, or areas where evidence was limited"
    )

    @validator('confidence')
    def validate_confidence(cls, v):
        if v.lower() not in ['low', 'medium', 'high']:
            raise ValueError('Confidence must be low, medium, or high')
        return v.lower()


class CandidateProfile(BaseModel):
    """Profile of a candidate being evaluated"""
    candidate_id: str = Field(description="Unique identifier for the candidate")
    name: Optional[str] = Field(default=None, description="Candidate name if provided")
    materials: List[str] = Field(description="List of material file paths evaluated")
    evaluation_date: datetime = Field(default_factory=datetime.now)


class RoleSpecificEvidence(BaseModel):
    """Evidence item within a role-specific assessment."""
    quote: str = Field(default="", description="Exact quote from materials")
    source: str = Field(default="", description="Source document name")
    context: str = Field(default="", description="What this evidence demonstrates")


class RoleSpecificAssessment(BaseModel):
    """Structured role-specific assessment (clinician, engineer, phd, or combined)."""
    role: str = Field(description="clinician, engineer, phd, or combined")
    marker: str = Field(default="", description="Primary role-specific marker label")
    score: float = Field(ge=1, le=10, description="Role-specific score from 1-10")
    confidence: str = Field(default="medium", description="low, medium, or high")
    reasoning: str = Field(default="", description="Evidence-based role assessment")
    evidence: List[RoleSpecificEvidence] = Field(default_factory=list)
    evidence_gaps: List[str] = Field(default_factory=list)
    # Clinical dimensions
    clinical_challenge_complexity: Optional[str] = None
    clinical_need_investigation_stage: Optional[str] = None
    clinical_systems_thinking: Optional[str] = None
    # Engineering dimensions
    engineering_build_stage: Optional[str] = None
    engineering_ownership_clarity: Optional[str] = None
    engineering_user_grounding: Optional[str] = None
    # Research dimensions
    research_evidence_level: Optional[str] = None
    research_publication_strength: Optional[str] = None
    research_ownership: Optional[str] = None

    @validator('confidence')
    def validate_confidence(cls, v):
        if v is None:
            return "medium"
        v = str(v).lower()
        if 'high' in v:
            return 'high'
        if 'low' in v:
            return 'low'
        return 'medium'


def parse_role_specific_assessment(data: Any) -> Optional[RoleSpecificAssessment]:
    """Parse role_specific_assessment from API/DB data, tolerating partial responses."""
    if not data or not isinstance(data, dict):
        return None
    if not data.get("role") or data.get("score") is None:
        return None
    try:
        score = float(data["score"])
    except (TypeError, ValueError):
        return None
    evidence = []
    for ev in data.get("evidence") or []:
        if isinstance(ev, dict):
            evidence.append(RoleSpecificEvidence(
                quote=ev.get("quote", ""),
                source=ev.get("source", ""),
                context=ev.get("context", ""),
            ))
    def _dim(new_key: str, legacy_key: str) -> Optional[str]:
        # Prefer the role-prefixed key; fall back to the legacy unprefixed key so
        # historical per-role evaluations still populate.
        value = data.get(new_key)
        return value if value is not None else data.get(legacy_key)

    return RoleSpecificAssessment(
        role=str(data["role"]).lower(),
        marker=data.get("marker", ""),
        score=score,
        confidence=data.get("confidence", "medium"),
        reasoning=data.get("reasoning", ""),
        evidence=evidence,
        evidence_gaps=data.get("evidence_gaps") or [],
        clinical_challenge_complexity=_dim("clinical_challenge_complexity", "challenge_complexity"),
        clinical_need_investigation_stage=_dim("clinical_need_investigation_stage", "need_investigation_stage"),
        clinical_systems_thinking=_dim("clinical_systems_thinking", "systems_thinking"),
        engineering_build_stage=_dim("engineering_build_stage", "build_stage"),
        engineering_ownership_clarity=_dim("engineering_ownership_clarity", "ownership_clarity"),
        engineering_user_grounding=_dim("engineering_user_grounding", "user_grounding"),
        research_evidence_level=data.get("research_evidence_level"),
        research_publication_strength=_dim("research_publication_strength", "publication_strength"),
        research_ownership=data.get("research_ownership"),
    )


class EvaluationResult(BaseModel):
    """Complete evaluation result for a candidate"""
    candidate: CandidateProfile
    scores: List[CriterionScore]
    overall_score: float = Field(
        description="Weighted average of all criterion scores"
    )
    overall_assessment: str = Field(
        description="High-level summary assessment of the candidate"
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Key strengths identified"
    )
    areas_for_development: List[str] = Field(
        default_factory=list,
        description="Areas where evidence was limited or scores were lower"
    )
    recommendation: str = Field(
        description="Overall recommendation (e.g., 'Strong fit', 'Potential fit with development', etc.)"
    )
    role: Optional[str] = Field(
        default=None,
        description="Role-specific evaluation context (clinician, engineer, phd)"
    )
    role_specific_assessment: Optional[RoleSpecificAssessment] = Field(
        default=None,
        description="Structured role-specific assessment when role evaluation is used"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (model used, processing time, etc.)"
    )

    def get_score_by_criterion(self, criterion: EvaluationCriterion) -> Optional[CriterionScore]:
        """Get score for a specific criterion"""
        for score in self.scores:
            if score.criterion == criterion:
                return score
        return None

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary for quick reference"""
        return {
            "candidate_id": self.candidate.candidate_id,
            "candidate_name": self.candidate.name,
            "overall_score": self.overall_score,
            "evaluation_date": self.candidate.evaluation_date.isoformat(),
            "recommendation": self.recommendation,
            "overall_assessment": self.overall_assessment,
            "strengths": self.strengths,
            "areas_for_development": self.areas_for_development,
            "scores": {
                score.criterion.value: score.score
                for score in self.scores
            }
        }


class HolisticEvidence(BaseModel):
    """Structured evidence for holistic evaluation"""
    quote: str = Field(description="Exact quote from materials")
    source: str = Field(description="Source document name")
    context: str = Field(default="", description="What this evidence demonstrates")


class InnovationPotential(BaseModel):
    """Innovation potential assessment for holistic evaluation"""
    level: str = Field(description="high, medium, or low")
    confidence: str = Field(default="medium", description="high, medium, or low confidence")
    reasoning: str = Field(description="Detailed explanation of the assessment")
    # Support both old format (key_evidence as strings) and new format (structured evidence)
    key_evidence: List[str] = Field(default_factory=list, description="Legacy: Direct quotes")
    evidence: List[HolisticEvidence] = Field(default_factory=list, description="Structured evidence items")


class ProgramFit(BaseModel):
    """Program fit assessment for holistic evaluation"""
    level: str = Field(description="strong, moderate, or weak")
    confidence: str = Field(default="medium", description="high, medium, or low confidence")
    detailed_analysis: str = Field(default="", description="Detailed analysis of program fit")
    strengths_for_program: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)
    evidence: List[HolisticEvidence] = Field(default_factory=list, description="Structured evidence items")


class NotableQuality(BaseModel):
    """A notable quality identified in holistic evaluation"""
    quality: str = Field(description="Name of the quality")
    confidence: str = Field(default="medium", description="high, medium, or low confidence")
    # Support both old format (single string) and new format (list of structured evidence)
    evidence: Any = Field(description="Evidence from materials (string or list of HolisticEvidence)")
    significance: str = Field(description="Why this matters for the program")


class RedFlag(BaseModel):
    """A red flag or concern identified in holistic evaluation"""
    flag: str = Field(description="Description of the concern")
    severity: str = Field(default="medium", description="high, medium, or low severity")
    evidence: str = Field(default="", description="Supporting evidence or note about missing evidence")


class InterviewQuestion(BaseModel):
    """A suggested interview question with context"""
    category: str = Field(default="General", description="Question category/purpose")
    question: str = Field(description="The question to ask")
    purpose: str = Field(default="", description="What you're trying to learn")


class HolisticEvaluationResult(BaseModel):
    """Complete holistic (criteria-free) evaluation result for a candidate"""
    candidate: CandidateProfile
    overall_assessment: str = Field(description="3-4 paragraph holistic assessment")
    innovation_potential: InnovationPotential
    program_fit: ProgramFit
    notable_qualities: List[NotableQuality] = Field(default_factory=list)
    # Support both old format (list of strings) and new format (list of RedFlag)
    red_flags: Any = Field(default_factory=list, description="Red flags (strings or RedFlag objects)")
    # Support both old format (list of strings) and new format (list of InterviewQuestion)
    questions_for_interview: Any = Field(default_factory=list, description="Interview questions")
    overall_score: float = Field(ge=1, le=10, description="Overall score from 1-10")
    score_justification: str = Field(default="", description="Detailed justification for the score")
    recommendation: str = Field(description="Strong fit / Potential fit / Not recommended")
    interview_decision: bool = Field(description="Binary recommendation for interview")
    interview_decision_reasoning: str = Field(default="", description="Reasoning for interview decision")
    role: Optional[str] = Field(
        default=None,
        description="Role-specific evaluation context (clinician, engineer, phd)"
    )
    role_specific_assessment: Optional[RoleSpecificAssessment] = Field(
        default=None,
        description="Structured role-specific assessment when role evaluation is used"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (model used, processing time, etc.)"
    )

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary for quick reference"""
        # Handle both old and new red_flags format
        red_flags_count = len(self.red_flags) if isinstance(self.red_flags, list) else 0

        return {
            "candidate_id": self.candidate.candidate_id,
            "candidate_name": self.candidate.name,
            "overall_score": self.overall_score,
            "evaluation_date": self.candidate.evaluation_date.isoformat(),
            "recommendation": self.recommendation,
            "interview_decision": self.interview_decision,
            "innovation_potential": self.innovation_potential.level,
            "innovation_confidence": self.innovation_potential.confidence,
            "program_fit": self.program_fit.level,
            "program_fit_confidence": self.program_fit.confidence,
            "red_flags_count": red_flags_count,
        }


# Below this confidence the yes/no screening decision is not trusted and the
# candidate is surfaced for human review instead. Model confidence is not
# calibrated, so this is a review band, not a probability cutoff.
SCREENING_REVIEW_CONFIDENCE = 0.7

SCREENING_OUTCOME_MATCH = "match"
SCREENING_OUTCOME_REVIEW = "review"
SCREENING_OUTCOME_NO_MATCH = "no_match"

SCREENING_OUTCOME_LABELS = {
    SCREENING_OUTCOME_MATCH: "Match",
    SCREENING_OUTCOME_REVIEW: "Needs review",
    SCREENING_OUTCOME_NO_MATCH: "No match",
}


class ScreeningResult(BaseModel):
    """Result of screening a candidate against a natural-language target profile."""
    candidate: CandidateProfile
    description: str = Field(description="The target profile description used for screening")
    matches: bool = Field(description="Whether the candidate matches the full target profile")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence in the match decision, from 0.0 to 1.0"
    )
    reasoning: str = Field(default="", description="Short explanation of the decision")
    supporting_evidence: List[str] = Field(
        default_factory=list,
        description="Quotes or details from the materials supporting the decision"
    )
    disqualifiers: List[str] = Field(
        default_factory=list,
        description="Evidence that violates an exclusion or a missing required qualification"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (model used, processing time, etc.)"
    )

    def outcome(self, review_below: float = SCREENING_REVIEW_CONFIDENCE) -> str:
        """Three-way screening decision: match / no_match / review.

        Any decision below the review-confidence threshold is downgraded to
        'review' (borderline — a human should look) regardless of direction.
        """
        if self.confidence < review_below:
            return SCREENING_OUTCOME_REVIEW
        return SCREENING_OUTCOME_MATCH if self.matches else SCREENING_OUTCOME_NO_MATCH

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary for quick reference"""
        return {
            "candidate_id": self.candidate.candidate_id,
            "candidate_name": self.candidate.name,
            "matches": self.matches,
            "confidence": self.confidence,
            "outcome": self.outcome(),
            "reasoning": self.reasoning,
            "evaluation_date": self.candidate.evaluation_date.isoformat(),
        }


class InterviewSelectionResult(BaseModel):
    """Result of the two-phase, context-window-efficient interview selection process."""

    selected_candidate_ids: List[str] = Field(
        description="Final N candidate IDs to interview, ordered strongest-first"
    )
    pool_candidate_ids: List[str] = Field(
        description="Intermediate top-pool IDs passed into Phase 2 (superset of selected)"
    )
    all_candidate_ids_ranked: List[str] = Field(
        description="All candidate IDs ranked by Phase 1 (empty if Phase 1 was skipped)"
    )
    n_interviews_requested: int = Field(description="Number of interview slots requested")
    selection_rationale: str = Field(
        default="", description="Claude's explanation of the final selection"
    )
    candidate_notes: Dict[str, str] = Field(
        default_factory=dict,
        description="Per-candidate notes from the selection step"
    )
    pool_multiplier: float = Field(
        default=2.0, description="Multiplier used to size the intermediate pool"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ComparisonResult(BaseModel):
    """Comparison of multiple candidates"""
    candidates: List[EvaluationResult]
    comparison_date: datetime = Field(default_factory=datetime.now)
    ranking: List[str] = Field(
        description="Candidate IDs in ranked order (best to worst)"
    )
    comparison_matrix: Dict[str, Dict[str, Any]] = Field(
        description="Matrix comparing candidates across criteria"
    )
    insights: List[str] = Field(
        default_factory=list,
        description="Key insights from the comparison"
    )


class ExpertRating(BaseModel):
    """Expert/human rating for a candidate"""
    candidate_id: str = Field(description="Unique identifier matching AI evaluation")
    rater_id: Optional[str] = Field(default=None, description="Identifier for the expert rater")
    scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Criterion name -> score mapping"
    )
    overall_score: Optional[float] = Field(default=None, description="Overall expert score if provided")
    interview_decision: Optional[bool] = Field(
        default=None,
        description="Whether expert recommended for interview"
    )
    comments: Dict[str, str] = Field(
        default_factory=dict,
        description="Criterion name -> comment mapping"
    )
    rating_date: Optional[datetime] = Field(default=None)


class ExpertComparisonMetrics(BaseModel):
    """Metrics comparing AI evaluations to expert ratings"""
    total_candidates: int
    matched_candidates: int = Field(description="Candidates with both AI and expert ratings")

    # Per-criterion metrics
    criterion_mae: Dict[str, float] = Field(
        default_factory=dict,
        description="Mean Absolute Error per criterion"
    )
    criterion_correlation: Dict[str, float] = Field(
        default_factory=dict,
        description="Pearson correlation coefficient per criterion"
    )

    # Overall metrics
    overall_mae: float = Field(description="Mean Absolute Error for overall scores")
    overall_correlation: float = Field(description="Correlation for overall scores")

    # Classification metrics (for interview recommendations)
    sensitivity: Optional[float] = Field(
        default=None,
        description="True positive rate - % of expert-recommended candidates AI also recommended"
    )
    specificity: Optional[float] = Field(
        default=None,
        description="True negative rate - % of expert-rejected candidates AI also rejected"
    )
    cohens_kappa: Optional[float] = Field(
        default=None,
        description="Inter-rater agreement statistic"
    )

    # Disagreement analysis
    high_disagreement_candidates: List[str] = Field(
        default_factory=list,
        description="Candidate IDs where AI and expert strongly disagree"
    )
    ai_bias: Optional[str] = Field(
        default=None,
        description="Detected systematic bias (e.g., 'AI scores higher on average')"
    )


class DistributionAnalysis(BaseModel):
    """Analysis of score distributions across candidate groups"""
    group_name: str = Field(description="Name of the group (e.g., 'Top 50%', 'Bottom 50%')")
    candidate_count: int
    candidate_ids: List[str]

    # Overall statistics
    mean_overall_score: float
    median_overall_score: float
    std_overall_score: float
    min_overall_score: float
    max_overall_score: float

    # Per-criterion statistics
    criterion_stats: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Criterion -> {mean, median, std, min, max}"
    )

    # Common patterns
    common_strengths: List[str] = Field(default_factory=list)
    common_weaknesses: List[str] = Field(default_factory=list)


class AdmitPatternEvidence(BaseModel):
    """Evidence supporting a pattern finding"""
    pattern: str = Field(description="Description of the pattern")
    admitted_examples: List[str] = Field(
        default_factory=list,
        description="Examples from admitted candidates"
    )
    rejected_examples: List[str] = Field(
        default_factory=list,
        description="Counter-examples from rejected candidates"
    )
    confidence: str = Field(default="medium", description="high, medium, or low")


class AdmitPatternCategory(BaseModel):
    """A category of patterns distinguishing admitted from rejected candidates"""
    category_name: str = Field(description="Name of the pattern category")
    description: str = Field(description="Detailed description of patterns in this category")
    patterns: List[AdmitPatternEvidence] = Field(default_factory=list)
    importance: str = Field(default="medium", description="high, medium, or low importance")


class AdmitPatternAnalysisResult(BaseModel):
    """Complete result of admit pattern analysis"""
    analysis_date: datetime = Field(default_factory=datetime.now)
    total_candidates: int
    admitted_count: int
    rejected_count: int
    
    # Score comparison
    admitted_mean_score: float
    rejected_mean_score: float
    score_difference: float
    
    # Key distinguishing patterns
    key_patterns: List[AdmitPatternCategory] = Field(
        default_factory=list,
        description="Major pattern categories distinguishing groups"
    )
    
    # Specific findings
    admitted_strengths: List[str] = Field(
        default_factory=list,
        description="Common strengths in admitted candidates"
    )
    rejected_weaknesses: List[str] = Field(
        default_factory=list,
        description="Common weaknesses in rejected candidates"
    )
    
    # Surprising findings
    surprising_admits: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Admitted candidates with lower scores or unusual profiles"
    )
    surprising_rejects: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Rejected candidates with higher scores or strong profiles"
    )
    
    # Summary
    executive_summary: str = Field(description="High-level summary of findings")
    methodology_notes: str = Field(
        default="",
        description="Notes about the analysis methodology and limitations"
    )
    
    # Raw data for reference
    candidate_summaries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Summary data for each candidate with admit status"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
