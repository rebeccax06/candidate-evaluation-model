"""Pattern analysis and linguistic marker extraction"""

import re
import json
import logging
from typing import List, Dict, Set, Tuple, Any, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime

from anthropic import Anthropic

from candidate_evaluator.core.models import (
    EvaluationCriterion,
    EvaluationResult,
    HolisticEvaluationResult,
    AdmitPatternAnalysisResult,
    AdmitPatternCategory,
    AdmitPatternEvidence
)
from candidate_evaluator.prompts.evaluation_prompts import get_admit_pattern_analysis_prompt

logger = logging.getLogger(__name__)


@dataclass
class LinguisticMarker:
    """A linguistic marker found in candidate materials"""
    phrase: str
    context: str
    source: str
    associated_criteria: List[EvaluationCriterion]
    marker_type: str  # 'keyword', 'phrase_pattern', 'syntactic_pattern'
    confidence: float


@dataclass
class PatternFindings:
    """Findings from pattern analysis"""
    criterion: EvaluationCriterion
    positive_markers: List[LinguisticMarker]
    negative_markers: List[LinguisticMarker]
    phrase_frequency: Dict[str, int]
    pattern_summary: str


class LinguisticPatternAnalyzer:
    """Analyzes text for linguistic markers indicating various criteria"""

    # Keywords and phrases associated with each criterion
    CRITERION_MARKERS = {
        EvaluationCriterion.CRITICAL_THINKING: {
            'positive_keywords': [
                'analyze', 'analyzed', 'analysis', 'evaluate', 'assessed',
                'systematic', 'logic', 'logical', 'reasoning', 'rationale',
                'root cause', 'investigated', 'examined', 'compared',
                'weighed', 'considered', 'concluded', 'inferred',
                'synthesized', 'integrated', 'derived', 'determined'
            ],
            'positive_phrases': [
                r'root cause analysis',
                r'systematic approach',
                r'analyzed.*data',
                r'evaluated.*options',
                r'logical.*conclusion',
                r'critical.*thinking',
                r'reasoned.*approach',
                r'evidence.*suggests',
                r'based on.*analysis'
            ]
        },
        EvaluationCriterion.COACHABILITY: {
            'positive_keywords': [
                'feedback', 'learned', 'improved', 'adapted', 'incorporated',
                'mentorship', 'guidance', 'advice', 'suggestion', 'receptive',
                'open', 'flexible', 'adjusted', 'refined', 'iterated',
                'coaching', 'development', 'growth mindset'
            ],
            'positive_phrases': [
                r'received.*feedback',
                r'incorporated.*feedback',
                r'learned.*from',
                r'improved.*based on',
                r'adapted.*approach',
                r'mentored by',
                r'guidance from',
                r'open to.*feedback',
                r'receptive to.*suggestions',
                r'growth.*mindset'
            ]
        },
        EvaluationCriterion.CURIOSITY: {
            'positive_keywords': [
                'curious', 'explored', 'investigated', 'researched',
                'discovered', 'learned', 'studied', 'questioned',
                'wondered', 'inquired', 'probed', 'examined',
                'delved', 'dug deeper', 'fascinated', 'interested'
            ],
            'positive_phrases': [
                r'wanted to.*understand',
                r'curious.*about',
                r'explored.*options',
                r'researched.*alternatives',
                r'sought.*understanding',
                r'asked.*questions',
                r'investigated.*further',
                r'learned.*about',
                r'discovered.*that',
                r'deep dive'
            ]
        },
        EvaluationCriterion.CREATIVITY: {
            'positive_keywords': [
                'innovative', 'creative', 'novel', 'unique', 'original',
                'invented', 'designed', 'developed', 'pioneered',
                'unconventional', 'breakthrough', 'revolutionary',
                'reimagined', 'transformed', 'ingenious'
            ],
            'positive_phrases': [
                r'creative.*solution',
                r'innovative.*approach',
                r'novel.*method',
                r'unique.*way',
                r'out.*of.*the.*box',
                r'first.*to',
                r'pioneered.*approach',
                r'developed.*new',
                r'invented.*method',
                r'unconventional.*solution'
            ]
        },
        EvaluationCriterion.COLLABORATION: {
            'positive_keywords': [
                'collaborated', 'team', 'partnered', 'coordinated',
                'cooperated', 'worked with', 'cross-functional',
                'stakeholder', 'together', 'jointly', 'collective',
                'consensus', 'facilitated', 'aligned'
            ],
            'positive_phrases': [
                r'worked.*with.*team',
                r'collaborated.*with',
                r'cross-functional.*team',
                r'partnered.*with',
                r'facilitated.*discussion',
                r'stakeholder.*engagement',
                r'incorporated.*feedback.*from',
                r'coordinated.*with',
                r'team.*effort',
                r'collective.*decision'
            ]
        },
        EvaluationCriterion.FOLLOW_THROUGH: {
            'positive_keywords': [
                'completed', 'delivered', 'finished', 'achieved',
                'accomplished', 'executed', 'implemented', 'fulfilled',
                'finalized', 'concluded', 'maintained', 'sustained',
                'consistent', 'thorough', 'comprehensive'
            ],
            'positive_phrases': [
                r'completed.*project',
                r'delivered.*on.*time',
                r'followed.*through',
                r'saw.*through.*completion',
                r'achieved.*goals',
                r'maintained.*throughout',
                r'start.*to.*finish',
                r'executed.*plan',
                r'fulfilled.*commitments',
                r'comprehensive.*implementation'
            ]
        },
        EvaluationCriterion.PROBLEM_SOLVING_MOTIVATION: {
            'positive_keywords': [
                'challenge', 'problem', 'solve', 'solution', 'overcome',
                'tackle', 'address', 'resolve', 'fix', 'troubleshoot',
                'motivated', 'driven', 'passionate', 'eager'
            ],
            'positive_phrases': [
                r'solved.*problem',
                r'motivated.*to.*solve',
                r'tackled.*challenge',
                r'driven.*to.*find',
                r'passionate.*about.*solving',
                r'overcame.*obstacle',
                r'addressed.*issue',
                r'resolved.*problem',
                r'eager.*to.*solve',
                r'enjoyed.*solving'
            ]
        },
        EvaluationCriterion.EVIDENCE_BASED: {
            'positive_keywords': [
                'data', 'evidence', 'metrics', 'measured', 'quantified',
                'tracked', 'analyzed', 'statistics', 'results', 'validated',
                'tested', 'verified', 'proof', 'demonstrated', 'findings'
            ],
            'positive_phrases': [
                r'data.*driven',
                r'evidence.*based',
                r'measured.*impact',
                r'tracked.*metrics',
                r'analyzed.*data',
                r'based.*on.*data',
                r'validated.*through',
                r'tested.*hypothesis',
                r'quantified.*results',
                r'metrics.*showed'
            ]
        },
        EvaluationCriterion.DETAIL_ORIENTATION: {
            'positive_keywords': [
                'detailed', 'thorough', 'meticulous', 'precise',
                'accurate', 'specific', 'comprehensive', 'exhaustive',
                'granular', 'careful', 'rigorous', 'methodical'
            ],
            'positive_phrases': [
                r'attention.*to.*detail',
                r'detailed.*analysis',
                r'thorough.*review',
                r'meticulous.*approach',
                r'precise.*implementation',
                r'comprehensive.*documentation',
                r'carefully.*considered',
                r'methodical.*process',
                r'granular.*level',
                r'rigorous.*testing'
            ]
        },
        EvaluationCriterion.COMMUNICATION: {
            'positive_keywords': [
                'communicated', 'presented', 'articulated', 'explained',
                'documented', 'wrote', 'shared', 'conveyed', 'clarified',
                'discussed', 'facilitated', 'transparent', 'clear'
            ],
            'positive_phrases': [
                r'clearly.*communicated',
                r'effectively.*presented',
                r'articulated.*vision',
                r'explained.*complex',
                r'documented.*process',
                r'shared.*findings',
                r'facilitated.*discussion',
                r'transparent.*communication',
                r'conveyed.*information',
                r'clear.*explanation'
            ]
        },
        EvaluationCriterion.EXPERTISE_ENABLER: {
            'positive_keywords': [
                'expertise', 'knowledge', 'enabled', 'facilitated',
                'leveraged', 'applied', 'pragmatic', 'practical',
                'balanced', 'adapted', 'flexible', 'integrated'
            ],
            'positive_phrases': [
                r'leveraged.*expertise',
                r'applied.*knowledge',
                r'enabled.*team',
                r'expertise.*in.*\w+',
                r'used.*background.*to',
                r'knowledge.*helped',
                r'technical.*expertise.*enabled',
                r'pragmatic.*approach',
                r'balanced.*technical',
                r'adapted.*expertise'
            ]
        }
    }

    def __init__(self):
        """Initialize the pattern analyzer"""
        self.extracted_markers: Dict[EvaluationCriterion, List[LinguisticMarker]] = defaultdict(list)

    def analyze_text(self, text: str, source: str = "document") -> Dict[EvaluationCriterion, PatternFindings]:
        """
        Analyze text to extract linguistic markers for each criterion.

        Args:
            text: Text to analyze
            source: Source identifier (e.g., filename)

        Returns:
            Dictionary mapping criteria to their pattern findings
        """
        text_lower = text.lower()
        findings = {}

        for criterion in EvaluationCriterion:
            positive_markers = []
            phrase_counts = Counter()

            markers = self.CRITERION_MARKERS.get(criterion, {})

            # Extract keyword markers
            for keyword in markers.get('positive_keywords', []):
                # Find instances of this keyword
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)

                for match in matches:
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    context = text[start:end].strip()

                    marker = LinguisticMarker(
                        phrase=keyword,
                        context=context,
                        source=source,
                        associated_criteria=[criterion],
                        marker_type='keyword',
                        confidence=0.7
                    )
                    positive_markers.append(marker)
                    phrase_counts[keyword] += 1

            # Extract phrase pattern markers
            for phrase_pattern in markers.get('positive_phrases', []):
                matches = re.finditer(phrase_pattern, text_lower, re.IGNORECASE)

                for match in matches:
                    matched_text = match.group(0)
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    context = text[start:end].strip()

                    marker = LinguisticMarker(
                        phrase=matched_text,
                        context=context,
                        source=source,
                        associated_criteria=[criterion],
                        marker_type='phrase_pattern',
                        confidence=0.85
                    )
                    positive_markers.append(marker)
                    phrase_counts[matched_text] += 1

            # Generate summary
            top_phrases = phrase_counts.most_common(5)
            summary = self._generate_pattern_summary(criterion, positive_markers, top_phrases)

            findings[criterion] = PatternFindings(
                criterion=criterion,
                positive_markers=positive_markers,
                negative_markers=[],  # Could add negative indicators
                phrase_frequency=dict(phrase_counts),
                pattern_summary=summary
            )

        return findings

    def _generate_pattern_summary(
        self,
        criterion: EvaluationCriterion,
        markers: List[LinguisticMarker],
        top_phrases: List[Tuple[str, int]]
    ) -> str:
        """Generate a summary of patterns found"""
        if not markers:
            return f"No strong linguistic markers found for {criterion.display_name}."

        summary_parts = [
            f"Found {len(markers)} linguistic markers for {criterion.display_name}."
        ]

        if top_phrases:
            phrases_str = ", ".join([f"'{phrase}' ({count}x)" for phrase, count in top_phrases[:3]])
            summary_parts.append(f"Most frequent indicators: {phrases_str}.")

        # Analyze marker types
        keyword_count = sum(1 for m in markers if m.marker_type == 'keyword')
        pattern_count = sum(1 for m in markers if m.marker_type == 'phrase_pattern')

        summary_parts.append(
            f"Includes {keyword_count} keyword matches and {pattern_count} phrase patterns."
        )

        return " ".join(summary_parts)

    def analyze_multiple_files(
        self,
        file_contents: List[Dict[str, str]]
    ) -> Dict[EvaluationCriterion, PatternFindings]:
        """
        Analyze multiple files and aggregate findings.

        Args:
            file_contents: List of dicts with 'content' and 'metadata'

        Returns:
            Aggregated pattern findings across all files
        """
        all_findings = defaultdict(lambda: {
            'markers': [],
            'phrase_counts': Counter()
        })

        # Analyze each file
        for file_data in file_contents:
            content = file_data['content']
            source = file_data['metadata']['filename']

            file_findings = self.analyze_text(content, source)

            for criterion, findings in file_findings.items():
                all_findings[criterion]['markers'].extend(findings.positive_markers)
                all_findings[criterion]['phrase_counts'].update(findings.phrase_frequency)

        # Aggregate into final findings
        aggregated_findings = {}

        for criterion, data in all_findings.items():
            top_phrases = data['phrase_counts'].most_common(5)
            summary = self._generate_pattern_summary(
                criterion,
                data['markers'],
                top_phrases
            )

            aggregated_findings[criterion] = PatternFindings(
                criterion=criterion,
                positive_markers=data['markers'],
                negative_markers=[],
                phrase_frequency=dict(data['phrase_counts']),
                pattern_summary=summary
            )

        return aggregated_findings

    def get_innovation_indicators(
        self,
        findings: Dict[EvaluationCriterion, PatternFindings]
    ) -> Dict[str, any]:
        """
        Extract specific indicators of innovation program potential.

        Args:
            findings: Pattern findings from analysis

        Returns:
            Dictionary of innovation indicators
        """
        innovation_criteria = [
            EvaluationCriterion.CREATIVITY,
            EvaluationCriterion.CURIOSITY,
            EvaluationCriterion.PROBLEM_SOLVING_MOTIVATION,
            EvaluationCriterion.CRITICAL_THINKING
        ]

        total_markers = sum(
            len(findings[c].positive_markers)
            for c in innovation_criteria
            if c in findings
        )

        # Extract top innovation phrases
        innovation_phrases = Counter()
        for criterion in innovation_criteria:
            if criterion in findings:
                innovation_phrases.update(findings[criterion].phrase_frequency)

        return {
            'total_innovation_markers': total_markers,
            'top_innovation_phrases': innovation_phrases.most_common(10),
            'innovation_criteria_coverage': len([
                c for c in innovation_criteria
                if c in findings and len(findings[c].positive_markers) > 0
            ]),
            'innovation_potential_score': min(10, total_markers / 5)  # Rough score
        }


class AdmitPatternAnalyzer:
    """
    Analyzes patterns distinguishing admitted from rejected candidates.
    
    This class takes evaluation results with admit status labels and
    uses Claude to identify distinguishing patterns.
    """
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-5", max_tokens: int = 8192):
        """
        Initialize the admit pattern analyzer.
        
        Args:
            api_key: Anthropic API key
            model: Claude model to use
            max_tokens: Maximum tokens for response
        """
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
    
    def prepare_candidate_summary(
        self,
        evaluation: Any,
        admit_status: bool
    ) -> Dict[str, Any]:
        """
        Prepare a summary dictionary for a single candidate.
        
        Args:
            evaluation: EvaluationResult or HolisticEvaluationResult
            admit_status: True if admitted, False if rejected
            
        Returns:
            Summary dictionary for the candidate
        """
        summary = {
            'candidate_id': evaluation.candidate.candidate_id,
            'admit_status': admit_status,
            'overall_score': evaluation.overall_score,
            'recommendation': evaluation.recommendation,
        }
        
        # Handle both EvaluationResult and HolisticEvaluationResult
        if isinstance(evaluation, EvaluationResult):
            summary['scores'] = {
                score.criterion.value: score.score
                for score in evaluation.scores
            }
            summary['strengths'] = evaluation.strengths
            summary['weaknesses'] = evaluation.areas_for_development
        elif isinstance(evaluation, HolisticEvaluationResult):
            summary['innovation_potential'] = evaluation.innovation_potential.level
            summary['program_fit'] = evaluation.program_fit.level
            summary['interview_decision'] = evaluation.interview_decision
            summary['strengths'] = evaluation.program_fit.strengths_for_program
            summary['weaknesses'] = evaluation.program_fit.concerns
            
            # Extract red flags
            red_flags = []
            for rf in evaluation.red_flags:
                if hasattr(rf, 'flag'):
                    red_flags.append({'flag': rf.flag, 'severity': rf.severity})
                else:
                    red_flags.append(str(rf))
            summary['red_flags'] = red_flags
            
            # Extract notable qualities
            notable = []
            for q in evaluation.notable_qualities:
                notable.append(q.quality)
            summary['notable_qualities'] = notable
        
        return summary
    
    def analyze_patterns(
        self,
        candidate_summaries: List[Dict[str, Any]]
    ) -> AdmitPatternAnalysisResult:
        """
        Analyze patterns distinguishing admitted from rejected candidates.
        
        Args:
            candidate_summaries: List of candidate summary dictionaries with admit_status
            
        Returns:
            AdmitPatternAnalysisResult with findings
        """
        logger.info(f"Analyzing patterns for {len(candidate_summaries)} candidates")
        
        # Calculate basic statistics
        admitted = [c for c in candidate_summaries if c.get('admit_status')]
        rejected = [c for c in candidate_summaries if not c.get('admit_status')]
        
        admitted_scores = [c.get('overall_score', 0) for c in admitted if c.get('overall_score')]
        rejected_scores = [c.get('overall_score', 0) for c in rejected if c.get('overall_score')]
        
        admitted_mean = sum(admitted_scores) / len(admitted_scores) if admitted_scores else 0
        rejected_mean = sum(rejected_scores) / len(rejected_scores) if rejected_scores else 0
        
        # Generate prompt
        prompt = get_admit_pattern_analysis_prompt(candidate_summaries)
        
        # Call Claude API
        logger.info("Calling Claude API for pattern analysis...")
        response = self._call_claude_api(prompt)
        
        # Parse response
        logger.info("Parsing pattern analysis response...")
        analysis_data = self._parse_response(response)
        
        # Build result
        result = AdmitPatternAnalysisResult(
            total_candidates=len(candidate_summaries),
            admitted_count=len(admitted),
            rejected_count=len(rejected),
            admitted_mean_score=admitted_mean,
            rejected_mean_score=rejected_mean,
            score_difference=admitted_mean - rejected_mean,
            key_patterns=self._parse_key_patterns(analysis_data.get('key_patterns', [])),
            admitted_strengths=analysis_data.get('admitted_strengths', []),
            rejected_weaknesses=analysis_data.get('rejected_weaknesses', []),
            surprising_admits=analysis_data.get('surprising_admits', []),
            surprising_rejects=analysis_data.get('surprising_rejects', []),
            executive_summary=analysis_data.get('executive_summary', ''),
            methodology_notes=analysis_data.get('methodology_notes', ''),
            candidate_summaries=candidate_summaries,
            metadata={
                'model': self.model,
                'analysis_date': datetime.now().isoformat(),
                'score_analysis': analysis_data.get('score_analysis', {}),
                'predictive_factors': analysis_data.get('predictive_factors', [])
            }
        )
        
        logger.info("Pattern analysis complete")
        return result
    
    def _call_claude_api(self, prompt: str) -> str:
        """Call Claude API with the prompt."""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            raise
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse Claude's JSON response."""
        try:
            # Try to extract JSON from code block
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                if end > start:
                    json_str = response[start:end].strip()
                    return json.loads(json_str)
            
            # Try to find raw JSON object
            brace_start = response.find('{')
            if brace_start >= 0:
                brace_count = 0
                for i, char in enumerate(response[brace_start:], brace_start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = response[brace_start:i+1]
                            return json.loads(json_str)
            
            raise ValueError("No JSON found in response")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response preview: {response[:500]}...")
            # Return empty result on parse failure
            return {
                'executive_summary': 'Failed to parse analysis response.',
                'key_patterns': [],
                'admitted_strengths': [],
                'rejected_weaknesses': []
            }
    
    def _parse_key_patterns(self, patterns_data: List[Dict]) -> List[AdmitPatternCategory]:
        """Parse key patterns into model objects."""
        categories = []
        for pattern_data in patterns_data:
            patterns = []
            for p in pattern_data.get('patterns', []):
                patterns.append(AdmitPatternEvidence(
                    pattern=p.get('pattern', ''),
                    admitted_examples=p.get('admitted_examples', []),
                    rejected_examples=p.get('rejected_examples', []),
                    confidence=p.get('confidence', 'medium')
                ))
            
            categories.append(AdmitPatternCategory(
                category_name=pattern_data.get('category_name', 'Unknown'),
                description=pattern_data.get('description', ''),
                patterns=patterns,
                importance=pattern_data.get('importance', 'medium')
            ))
        
        return categories
    
    def calculate_basic_statistics(
        self,
        candidate_summaries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate basic statistics without calling Claude.
        
        Useful for quick analysis or when API is not available.
        """
        admitted = [c for c in candidate_summaries if c.get('admit_status')]
        rejected = [c for c in candidate_summaries if not c.get('admit_status')]
        
        # Score statistics
        admitted_scores = [c.get('overall_score', 0) for c in admitted if c.get('overall_score')]
        rejected_scores = [c.get('overall_score', 0) for c in rejected if c.get('overall_score')]
        
        stats = {
            'total_candidates': len(candidate_summaries),
            'admitted_count': len(admitted),
            'rejected_count': len(rejected),
            'admission_rate': len(admitted) / len(candidate_summaries) if candidate_summaries else 0,
        }
        
        if admitted_scores:
            stats['admitted_mean_score'] = sum(admitted_scores) / len(admitted_scores)
            stats['admitted_min_score'] = min(admitted_scores)
            stats['admitted_max_score'] = max(admitted_scores)
        
        if rejected_scores:
            stats['rejected_mean_score'] = sum(rejected_scores) / len(rejected_scores)
            stats['rejected_min_score'] = min(rejected_scores)
            stats['rejected_max_score'] = max(rejected_scores)
        
        if admitted_scores and rejected_scores:
            stats['score_difference'] = stats['admitted_mean_score'] - stats['rejected_mean_score']
            
            # Find potential threshold
            admitted_min = min(admitted_scores)
            rejected_max = max(rejected_scores)
            if admitted_min > rejected_max:
                stats['apparent_threshold'] = (admitted_min + rejected_max) / 2
            else:
                # There's overlap - find gray zone
                stats['gray_zone_min'] = max(min(admitted_scores), min(rejected_scores))
                stats['gray_zone_max'] = min(max(admitted_scores), max(rejected_scores))
        
        # Per-criterion analysis (for EvaluationResult-based summaries)
        criterion_stats = {}
        for c in candidate_summaries:
            scores = c.get('scores', {})
            for criterion, score in scores.items():
                if criterion not in criterion_stats:
                    criterion_stats[criterion] = {'admitted': [], 'rejected': []}
                
                if c.get('admit_status'):
                    criterion_stats[criterion]['admitted'].append(score)
                else:
                    criterion_stats[criterion]['rejected'].append(score)
        
        stats['criterion_differences'] = []
        for criterion, data in criterion_stats.items():
            if data['admitted'] and data['rejected']:
                admitted_mean = sum(data['admitted']) / len(data['admitted'])
                rejected_mean = sum(data['rejected']) / len(data['rejected'])
                diff = admitted_mean - rejected_mean
                stats['criterion_differences'].append({
                    'criterion': criterion,
                    'admitted_mean': admitted_mean,
                    'rejected_mean': rejected_mean,
                    'difference': diff
                })
        
        # Sort by difference
        stats['criterion_differences'].sort(key=lambda x: abs(x['difference']), reverse=True)
        
        return stats
